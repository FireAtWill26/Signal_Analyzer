from contextlib import redirect_stderr
import numpy as np
from pathlib import Path
from os.path import isfile, join, dirname
from os import listdir
from scipy.stats import rankdata
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import time
import os
import math
import shutil
from copy import deepcopy
from collections import Counter, defaultdict
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import train_test_split
from multiprocessing.shared_memory import SharedMemory
from lion_pytorch import Lion
# from muon import Muon, MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from scipy.stats import skew, kurtosis
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn, Task
from rich.console import Console

from prometheus.utils.utils import WeightedCorrNp, calc_fcst_weighted_ic, weighted_mse_wrap, weighted_mse_eval_wrap, filter_eligible_data, filter_eligible_data_v2, operations_by_group
from prometheus.utils.torch_utils import select_gpu_with_minimum_memory, early_stopping_func, WarmupLR, DecayingCosineWarmRestarts, SharedMemDataset, SharedMemSeqDataset
from prometheus.ops import create_op_process
from prometheus.modelpool.basemodel import BaseModel, create_model
from prometheus.utils.registry_factory import TRAINING_REGISTRY, MODEL_REGISTRY
from prometheus.utils.speedup_package import calculate_stock_mean_3d_parallel
from prometheus.utils.pearson import niocorr
from prometheus.utils.spearman import niocorr_spearman
from prometheus.modelpool.Adversarial import Adversarial
from prometheus.utils.data_processing import RollingStatistics
from prometheus.riskmodel.barra.factor_dict import *
from prometheus.utils.weight_calculation import calculate_weights_cython, calculate_turnover_cython

class train_dataset(Dataset):
    def __init__(self, x, y, vertical_norm=False, denormalizer=False):
        self.x = torch.from_numpy(x).to(torch.float32)
        if denormalizer:
            mean = torch.unsqueeze(torch.mean(self.x, dim=1), 1)
            std = torch.unsqueeze(torch.std(self.x, dim=1), 1)
            self.x = torch.cat([self.x, 2*torch.tanh(mean), 2*torch.tanh(std)], dim=1)
        if vertical_norm:
            self.x = F.normalize(self.x, p=2.0, dim=0)
        self.y = torch.from_numpy(y).to(torch.float32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]



@TRAINING_REGISTRY.register('Ensemble_test')
class training_test(BaseModel):
    def __init__(self, *args, **kwargs):
        # super().__init__(*args, **kwargs)
        pass

    def fit(self, cfg, indices, train_output, sample_stocks_date, sample_stocks_seccode, model_name, lock=None, *args, **kwargs):

        super().fit_helper(cfg, indices, train_output, sample_stocks_date=sample_stocks_date, model_name=model_name, *args, **kwargs)

        train_input_buffer = SharedMemory(model_name)
        self.train_input = np.ndarray((indices[0].shape[0], indices[1].shape[0], indices[1].shape[1]), dtype=np.float32, buffer=train_input_buffer.buf)
        self.train_output = train_output
        self.indices = indices
        self.model_name = model_name
        self.pid = os.getpid()

        Path.mkdir(Path(self.training_dic["model_path"]) / model_name, exist_ok=True, parents=True)

        self.model_path_list = [Path(self.training_dic["model_path"]) / model_name / f'final_model_{i}.pt' for i in range(self.training_dic["num_buckets"])]
        self.model_nd_path_list = [Path(self.training_dic["model_path"]) / model_name / f'final_model_nd_{i}.pt' for i in range(self.training_dic["num_buckets"])]
        self.optimizer_path_list = [Path(self.training_dic["model_path"]) / model_name / f'final_optimizer_{i}.pt' for i in range(self.training_dic["num_buckets"])]
        self.dict_path_list = [Path(self.training_dic["model_path"]) / model_name / f'training_dict_{i}.pt' for i in range(self.training_dic["num_buckets"])]
        self.log_path_list = [Path(self.training_dic["model_path"]) / model_name / f'training_log_{i}.csv' for i in range(self.training_dic["num_buckets"])]
        if "finetune" in self.training_dic and self.training_dic["finetune"]:
            self.log_path_finetune = Path(self.training_dic["model_path"]) / model_name / 'finetune_log.csv'
        self.summary_path = Path("/dfs/data/tensorBoard") / cfg.train_basics["model_path_name"] / model_name
        self.regressor_path = Path(self.training_dic["model_path"]) / model_name / 'final_regressor.pt'
        self.regressor_optimizer_path = Path(self.training_dic["model_path"]) / model_name / 'final_regressor_optimizer.pt'

        if "feature_select" in self.training_dic and self.training_dic["feature_select"]:
            self.training_dic["features_delete"] = self.feature_selection(cfg)
        
        num_workers = max(4, min(16, torch.cuda.device_count() * 4))  # 根据GPU数量动态调整
        
        if Path(self.summary_path).exists():
            shutil.rmtree(self.summary_path)

        Path.mkdir(self.summary_path, exist_ok=True, parents=True)

        self.feature_buckets, self.feature_list = self.feature_bucketing(cfg)

        days, stocks = self.indices[1].shape
        if not "eval_mode" in self.training_dic:
            self.training_dic["eval_mode"] = "random"        
        rows, cols = np.where(indices[1])
        if self.training_dic["eval_mode"] == "recent":
            train_days = math.ceil(days * (1 - self.training_dic["eval_size"]))
            train_indices = np.zeros(indices[1].shape).astype(bool)
            pos_holder = np.zeros(indices[1].shape).astype(bool)
            for i in range(train_days):
                train_indices[i] = indices[1][i]
                pos_holder[i] = indices[1][i]
            if "avoid_horizon" in self.training_dic and self.training_dic["avoid_horizon"]:
                for i in range(self.training_dic["horizon"]):
                    pos_holder[train_days + i] = indices[1][i]
            eval_indices = indices[1] & ~pos_holder
        elif self.training_dic["eval_mode"] == "by_date":
            all_dates = np.unique(rows)
            _, eval_dates = train_test_split(range(len(all_dates)), test_size=self.training_dic["eval_size"], random_state=0, shuffle=True)
            eval_indices = np.zeros(indices[1].shape).astype(bool)
            pos_holder = np.zeros(indices[1].shape).astype(bool)
            row_filter = np.unique(rows)[eval_dates]
            for i in range(indices[1].shape[0]):
                if i in row_filter:
                    eval_indices[i] = indices[1][i]
                    if "avoid_horizon" in self.training_dic and self.training_dic["avoid_horizon"]:
                        for j in range(max(0, i-self.training_dic["horizon"]), min(days, i+self.training_dic["horizon"])):
                            pos_holder[j] = indices[1][j]
                    else:
                        pos_holder[i] = indices[1][i]
            train_indices = indices[1] & ~pos_holder
        else:
            _, eval_idx = train_test_split(range(len(rows)), test_size=self.training_dic["eval_size"], random_state=0, shuffle=True)
            pos_holder = np.zeros(indices[1].shape).astype(bool)
            eval_indices = np.zeros(indices[1].shape).astype(bool)
            for i in eval_idx:
                eval_indices[rows[i]][cols[i]] = True
                if "avoid_horizon" in self.training_dic and self.training_dic["avoid_horizon"]:
                    for j in range(max(0, rows[i]-self.training_dic["horizon"]), min(days, rows[i]+self.training_dic["horizon"])):
                        pos_holder[j][cols[i]] = True
                else:
                    pos_holder[rows[i]][cols[i]] = True
            train_indices = indices[1] & ~pos_holder

        if "date_weight" in self.training_dic and self.training_dic["date_weight"]:
            date_weight = torch.arange(days)
            date_weight = torch.sigmoid(date_weight-(days-20*self.training_dic["date_weight_month"])) * self.training_dic["date_weight_pow"]
        else:
            date_weight = torch.zeros(days)

        self.train_dataset_list = [SharedMemDataset(model_name, self.train_input.shape, self.train_input.dtype, self.feature_buckets[i], train_indices, self.train_output,date_weight,feature_matrix=self.feature_matrix) for i in range(self.training_dic["num_buckets"])]
        self.eval_dataset_list = [SharedMemDataset(model_name, self.train_input.shape, self.train_input.dtype, self.feature_buckets[i], eval_indices, self.train_output,date_weight,feature_matrix=self.feature_matrix) for i in range(self.training_dic["num_buckets"])]
        self.train_dataset = SharedMemDataset(model_name, self.train_input.shape, self.train_input.dtype, self.indices[0], train_indices, self.train_output,date_weight,feature_matrix=self.feature_matrix)
        self.eval_dataset = SharedMemDataset(model_name, self.train_input.shape, self.train_input.dtype, self.indices[0], eval_indices, self.train_output,date_weight,feature_matrix=self.feature_matrix)

        self.train_loader_list = [DataLoader(self.train_dataset_list[i], batch_size=self.training_dic["batch_size"], shuffle=True, drop_last=True, prefetch_factor=4, pin_memory=True, num_workers=num_workers, persistent_workers=True) for i in range(self.training_dic["num_buckets"])]
        self.eval_loader_list = [DataLoader(self.eval_dataset_list[i], batch_size=self.training_dic["batch_size"], shuffle=False, drop_last=True, prefetch_factor=4, pin_memory=True, num_workers=num_workers, persistent_workers=True) for i in range(self.training_dic["num_buckets"])]
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.training_dic["batch_size"], shuffle=True, drop_last=True, prefetch_factor=4, pin_memory=True, num_workers=num_workers, persistent_workers=True)
        self.eval_loader = DataLoader(self.eval_dataset, batch_size=self.training_dic["batch_size"], shuffle=False, drop_last=True, prefetch_factor=4, pin_memory=True, num_workers=num_workers, persistent_workers=True)

        if ("finetune" in self.training_dic and self.training_dic["finetune"]) or ("discrimination" in self.training_dic and self.training_dic["discrimination"]):
            # recent_indices = (np.arange(days) > (days - (20 * self.training_dic["finetune_month"]))).reshape(-1,1)
            # recent_indices = np.repeat(recent_indices, stocks, axis=1)

            train_days, eval_days = train_test_split(range((days - (20 * self.training_dic["finetune_month"])), days), test_size=self.training_dic["eval_size"], random_state=0, shuffle=True)

            finetune_train_indices = np.zeros(indices[1].shape).astype(bool)
            for i in train_days:
                finetune_train_indices[i] = indices[1][i]
            finetune_eval_indices = np.zeros(indices[1].shape).astype(bool)
            for i in eval_days:
                finetune_eval_indices[i] = indices[1][i]

            finetune_train_dataset_list = [SharedMemDataset(model_name, self.train_input.shape, self.train_input.dtype, self.feature_buckets[i], finetune_train_indices, self.train_output,np.ones(days),feature_matrix=self.feature_matrix) for i in range(self.training_dic["num_buckets"])]
            finetune_eval_dataset_list = [SharedMemDataset(model_name, self.train_input.shape, self.train_input.dtype, self.feature_buckets[i], finetune_eval_indices, self.train_output,np.ones(days),feature_matrix=self.feature_matrix) for i in range(self.training_dic["num_buckets"])]

            self.finetune_train_loader_list = [DataLoader(finetune_train_dataset_list[i], batch_size=self.training_dic["batch_size"], shuffle=True, drop_last=True, prefetch_factor=4, pin_memory=True, num_workers=num_workers, persistent_workers=True) for i in range(self.training_dic["num_buckets"])]
            self.finetune_eval_loader_list = [DataLoader(finetune_eval_dataset_list[i], batch_size=self.training_dic["batch_size"], shuffle=False, drop_last=True, prefetch_factor=4, pin_memory=True, num_workers=num_workers, persistent_workers=True) for i in range(self.training_dic["num_buckets"])]

        
        train_days, eval_days = train_test_split(range((days - (20 * self.training_dic["finetune_month"])), days), test_size=self.training_dic["eval_size"], random_state=0, shuffle=True)
        regressor_train_indices = np.zeros(indices[1].shape).astype(bool)
        for i in train_days:
            regressor_train_indices[i] = indices[1][i]
        regressor_eval_indices = np.zeros(indices[1].shape).astype(bool)
        for i in eval_days:
            regressor_eval_indices[i] = indices[1][i]
        regressor_train_dataset = SharedMemDataset(model_name, self.train_input.shape, self.train_input.dtype, self.indices[0], regressor_train_indices, self.train_output,np.ones(days),feature_matrix=self.feature_matrix)
        regressor_eval_dataset = SharedMemDataset(model_name, self.train_input.shape, self.train_input.dtype, self.indices[0], regressor_eval_indices, self.train_output,np.ones(days),feature_matrix=self.feature_matrix)

        self.regressor_train_loader = DataLoader(regressor_train_dataset, batch_size=self.training_dic["batch_size"], shuffle=True, drop_last=True, prefetch_factor=4, pin_memory=True, num_workers=num_workers, persistent_workers=True)
        self.regressor_eval_loader = DataLoader(regressor_eval_dataset, batch_size=self.training_dic["batch_size"], shuffle=False, drop_last=True, prefetch_factor=4, pin_memory=True, num_workers=num_workers, persistent_workers=True)

        print(f"len of train_data: {len(self.train_dataset_list[0])}, len of eval_data: {len(self.eval_dataset_list[0])}, len of train_loader: {len(self.train_loader_list[0])}, len of eval_loader: {len(self.eval_loader_list[0])}")

        if lock is not None:
            with lock:
                time.sleep(30)
                super().gpu_helper(*args, **kwargs)

        if not "denormalizer" in self.training_dic:
            self.training_dic["denormalizer"] = False
        if not "step_num" in self.training_dic:
            self.training_dic["step_num"] = 30
        self.training_dic["eval_steps"] = len(self.train_loader) // self.training_dic["step_num"]

        self.cfg_list = [deepcopy(cfg) for i in range(self.training_dic["num_buckets"])]
        
        self.input_dim_list = [np.sum(self.feature_buckets[i]) for i in range(self.training_dic["num_buckets"])]
        if "extra_feature" in self.training_dic and self.training_dic["extra_feature"]:
            self.input_dim_list = [input_dim+self.feature_matrix.shape[0] for input_dim in self.input_dim_list]
        if "feature_insert_pos" in cfg.model.model_params and cfg.model.model_params["feature_insert_pos"] is not None:
            if "feature_insert" in cfg.model.model_params and cfg.model.model_params["feature_insert"] is not None:
                self.input_dim_list = [input_dim-self.cfg.model.model_params["feature_insert"].sum() for input_dim in self.input_dim_list]
        if self.training_dic["denormalizer"]:
            self.input_dim += 2
        if "FastKAN" in cfg.model.type:
            for i in range(len(cfg.model.model_params["layers_hidden"])-self.training_dic["plain_layer"]):
                for j in range(self.training_dic["num_buckets"]):
                    self.cfg_list[j].model.model_params["layers_hidden"][i] = math.ceil(self.input_dim_list[j] * self.cfg_list[j].model.model_params["layers_hidden"][i])
        # elif self.cfg.model.type == "KANUNet":
        #     for i in range(self.training_dic["mult_layer"]):
        #         self.cfg.model.model_params["layers_hidden"][i] *= self.input_dim
        for i in range(len(self.cfg_list)):
            self.cfg_list[i].model.model_params["device"] = self.device
            self.cfg_list[i].training.model_params["feature_chosen"] = self.feature_list[i]
        self.model_list = [create_model(self.cfg_list[i].model) for i in range(self.training_dic["num_buckets"])]

        self.regressor = Regressor(self.training_dic["num_buckets"], device=self.device)
        self.linear_regressor = LinearRegression(positive=True, fit_intercept=False)


        writer = SummaryWriter(self.summary_path / "run")
        writer.add_graph(self.model_list[0], torch.ones(self.input_dim_list[0], device=self.cfg_list[0].model.model_params["device"]))

        result = {}
        for i in range(self.training_dic["num_buckets"]):
            result[i] = {"train_loss": [], "eval_loss": [], "eval_mse_loss": [], "eval_correlation_loss": [], "eval_correlation_mask": [],  "good_points": [], "huber_delta": []}

        if "discrimination" in self.training_dic and self.training_dic["discrimination"]:
            for i in range(self.training_dic["num_buckets"]):
                result[i]["adv_loss"] = []

            self.adversarial_list = [Adversarial(self.cfg_list[i], self.device, self.input_dim_list[i]) for i in range(self.training_dic["num_buckets"])]

        # if only_eval set to true in config, call the quick_evaluation function which will not train the model but only do the evaluation.
        if self.training_dic["only_eval"]:
            return self.quick_evaluation()

        for i in range(self.training_dic["num_buckets"]):
            torch.save(self.cfg_list[i], self.dict_path_list[i])
        
        self.vertical_norm = self.training_dic["vertical_norm"]

        train_y = train_output[indices[1]]

        var = np.var(train_y)
        std = np.std(train_y)
        mean = np.mean(train_y)
        skewness = skew(train_y)
        kurt = kurtosis(train_y)

        self.mean = mean
        self.std = std

        print(f"target stats| Mean: {mean.item()}| Variance: {var.item()}| Skewness: {skewness}| Kurtosis: {kurt}")

        pbar = tqdm(range(self.training_dic['num_buckets']), desc='Training', ncols=160)

        if self.training_dic["loss_fn"] == "CCC":
            self.loss_fn = self.loss_fn_eval = lambda x, y, w: self.ccc(x,y,w)
        elif self.training_dic["loss_fn"] == "Correlation":
            self.loss_fn = self.loss_fn_eval = lambda x, y, w: self.correlation_loss(x,y,w)
        else:
            self.loss_fn = self.loss_fn_eval = lambda x, y, w: 0
        if self.training_dic["reg_loss"] == "Huber":
            self.reg_fn = nn.HuberLoss(reduction="none", delta=self.training_dic["huber_delta"])
        else:
            self.reg_fn = nn.MSELoss(reduction="none")

        self.correlation_ratio = self.training_dic["correlation_ratio"]
        lr = self.training_dic['lr']
        self.weight_decay = self.training_dic["weight_decay"]


        if self.training_dic['opt'] == "Adam":
            self.optimizer_list = [torch.optim.Adam(self.model_list[i].parameters(), lr=lr, weight_decay=self.weight_decay) for i in range(self.training_dic["num_buckets"])]
        elif self.training_dic['opt'] == "AdamW":
            self.optimizer_list = [torch.optim.AdamW(self.model_list[i].parameters(), lr=lr, weight_decay=self.weight_decay) for i in range(self.training_dic["num_buckets"])]
        elif self.training_dic['opt'] == "RMSprop":
            self.optimizer_list = [torch.optim.RMSprop(self.model_list[i].parameters(), lr=lr, weight_decay=self.weight_decay) for i in range(self.training_dic["num_buckets"])]
        elif self.training_dic["opt"] == "Lion":
            self.optimizer_list = [Lion(self.model_list[i].parameters(), lr=lr/10, weight_decay=self.weight_decay) for i in range(self.training_dic["num_buckets"])]
        # elif self.training_dic["opt"] == "Muon":
        #     muon_params = [p for p in self.model.parameters() if p.ndim >= 2]
        #     adamw_params = [p for p in self.model.parameters() if p.ndim < 2]
        #     self.optimizer = MuonWithAuxAdam(dict(params=muon_params, use_muon=True, lr=lr, weight_decay=self.weight_decay), dict(params=adamw_params, use_muon=False, lr=lr/10, weight_decay=self.weight_decay))
        elif self.training_dic['opt'] == "SGD":
            self.optimizer_list = [torch.optim.SGD(self.model_list[i].parameters(), lr=lr, weight_decay=self.weight_decay) for i in range(self.training_dic["num_buckets"])]
        # elif self.training_dic['opt'] == "LBFGS":
        #     self.optimizer_list = [torch.optim.LBFGS(self.model_list[i].parameters(), lr=lr, history_size=10, tolerance_grad=1e-32, tolerance_change=1e-32) for i in range(self.training_dic["num_buckets"])]
        else:
            raise ValueError("opt must be Adam, AdamW, SGD, LBFGS, or RMSprop.")

        
        # if self.training_dic["opt"] == "LBFGS":
        #     def closure():
        #         self.optimizer.zero_grad()
        #         pred = self.model.forward(data_batch)
        #         train_loss = self.loss_fn(pred, target_batch)
        #         train_loss.backward()
        #         return train_loss

        scheduler_list = [torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_list[i], "min", factor=self.training_dic["scheduler_factor"], patience=self.training_dic["scheduler_patience"], threshold=self.training_dic["scheduler_threshold"]) for i in range(self.training_dic["num_buckets"])]

        for model_idx in pbar:
            self.model = self.model_list[model_idx]
            self.optimizer = self.optimizer_list[model_idx]
            self.scheduler = scheduler_list[model_idx]

            total_step = 0
            best_eval_loss = float("inf")
            best_correlation_loss = -float("inf")
            if self.training_dic["eval_mask"]:
                best_correlation_mask_loss = -float("inf")
            retrain_cnt = 1
            overfit_cnt = 0
            pred_mean = 0
            pred_std = 0
            best_points_cnt = 0
            for epoch in range(self.training_dic["epochs"]):
                
                self.model.train()
                epoch_training_loss = 0
                batch_num = 0
                step_cnt = 0
                adv_loss = 0

                if "discrimination" in self.training_dic and self.training_dic["discrimination"] and self.training_dic["batch_size"] != cfg.dis_train.model_params["dis_target_size"]:
                    assert cfg.dis_train.model_params["dis_target_size"] % self.training_dic["batch_size"] == 0, f"Target size in discrimination must be divisible by batch size"
                    dis_batch_cnt = cfg.dis_train.model_params["dis_target_size"] // self.training_dic["batch_size"]
                    dis_cur_cnt = 0
                    dis_pred = []
                    dis_data = []
                    dis_target = []

                for data in self.train_loader_list[model_idx]:
                    # import ipdb; ipdb.set_trace()
                    step_cnt += 1
                    # 提前批量移动数据到设备
                    if "extra_feature" in self.training_dic and self.training_dic["extra_feature"]:
                        data_batch = torch.cat([data[0], data[-1]], dim=1).to(self.device, non_blocking=True, dtype=data[0].dtype)
                    else:                    
                        data_batch = data[0].to(self.device, non_blocking=True)
                    if self.training_dic["denormalizer"]:
                        batch_mean = torch.unsqueeze(torch.mean(data_batch, dim=1), 1)
                        batch_std = torch.unsqueeze(torch.std(data_batch, dim=1), 1)
                        data_batch = torch.cat([data_batch, batch_mean, batch_std.pow(0.5)], dim=1).to(self.device, non_blocking=True)
                    target_batch = data[1].to(self.device, non_blocking=True)
                    weight = torch.ones_like(target_batch)
                    weight += data[2].to(self.device, non_blocking=True)
                    if "horizontal_norm" in self.training_dic and self.training_dic["horizontal_norm"]:
                        data_batch = F.normalize(data_batch, p=2.0, dim=1)
                    if self.training_dic["opt"] == "LBFGS":
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        # train_loss = self.optimizer.step(closure)
                    else:
                        self.optimizer.zero_grad()
                        # with autocast():
                        pred = torch.squeeze(self.model.forward(data_batch))
                        train_mean = torch.mean(pred)
                        train_std = torch.std(pred)
                        if self.training_dic["ignore_negative_target"]:
                            mask = (target_batch <= self.training_dic["ignore_threshold"]) | (pred <= self.training_dic["ignore_threshold"])
                            pred = pred[mask]
                            target_batch = target_batch[mask]
                        if self.training_dic["weight_essential"]:
                            # mask1 = ((target_batch < mean + self.training_dic["weight_pos"] * std) & (pred > pred_mean + self.training_dic["weight_pos"] * pred_std)) 
                            # mask2 = (target_batch > mean + self.training_dic["weight_pos"] * std)
                            # # & (pred < pred_mean + self.training_dic["weight_pos"] * pred_std)
                            # weight = torch.where(mask1, (1 + mean + self.training_dic["weight_pos"] * std - target_batch).pow(self.training_dic["weight_pow_over"]), weight)
                            # weight = torch.where(mask2, (1 + target_batch - mean - self.training_dic["weight_pos"] * std).pow(self.training_dic["weight_pow_under"]), weight)
                            weight += torch.max(F.sigmoid(8 * (target_batch - mean - self.training_dic["weight_pos"] * std)), F.sigmoid(8 * (pred - pred_mean - self.training_dic["weight_pos"] * pred_std))) * self.training_dic["weight_pow_over"]
                        if self.training_dic["loss_fn"] == "CCC" or self.training_dic["loss_fn"] == "Correlation":
                            correlation_loss = self.loss_fn(pred, target_batch, weight)
                            reg_loss = self.reg_fn(pred, target_batch)
                            if self.training_dic["weight_essential"]:
                                reg_loss = (reg_loss * weight)
                            reg_loss = reg_loss.mean()
                            train_loss = -self.correlation_ratio * correlation_loss + (1 - self.correlation_ratio) * reg_loss
                        else:
                            reg_loss = self.reg_fn(pred, target_batch)
                            if self.training_dic["weight_essential"]:
                                reg_loss = (reg_loss * weight)
                            reg_loss = torch.mean(reg_loss)
                            train_loss = reg_loss
                        # assert torch.isnan(train_loss).sum() == 0, print(train_loss)
                        # scaler.scale(train_loss).backward()
                        loss = train_loss
                        if self.training_dic["enable_l1"]:
                            loss += self.training_dic["l1_ratio"] * self.l1_regularization()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        # scaler.step(self.optimizer)
                        # scaler.update()
                        self.optimizer.step()
                        if "discrimination" in self.training_dic and self.training_dic["discrimination"]:
                            if self.training_dic["batch_size"] != cfg.dis_train.model_params["dis_target_size"]:
                                if dis_cur_cnt < dis_batch_cnt:
                                    dis_pred.append(pred)
                                    dis_data.append(data_batch)
                                    dis_target.append(target_batch)
                                    dis_cur_cnt += 1
                                else:
                                    dis_pred = torch.cat(dis_pred, dim=0)
                                    dis_data = torch.cat(dis_data, dim=0)
                                    dis_target = torch.cat(dis_target, dim=0)
                                    loss_dis = self.adversarial_list[model_idx].train_with_model(dis_pred, dis_data, dis_target)
                                    adv_loss += loss_dis
                                    dis_pred = []
                                    dis_data = []
                                    dis_target = []
                                    dis_cur_cnt = 0
                            else:
                                loss_dis = self.adversarial_list[model_idx].train_with_model(pred, data_batch, target_batch)
                                adv_loss += loss_dis
                        # assert torch.isnan(self.model.parameters()).sum() == 0, print(self.model.parameters())
                    epoch_training_loss += train_loss.item()
                    batch_num += 1

                    if step_cnt == self.training_dic["eval_steps"]:
                        step_cnt = 0

                        train_loss = epoch_training_loss / batch_num
                        result[model_idx]["train_loss"].append(train_loss)
                        if "discrimination" in self.training_dic and self.training_dic["discrimination"]:
                            result[model_idx]["adv_loss"].append(adv_loss/batch_num)

                        # add discriminator here
                        # if "discrimination" in self.training_dic and self.training_dic["discrimination"]:
                        #     result["adv_loss"].append(adv_loss / batch_num)
                        #     self.discriminator.eval()
                        #     for data_batch, target_batch, date_weight in self.train_loader:
                        #         data_batch = data_batch.to(self.device, non_blocking=True)
                        #         target_batch = target_batch.to(self.device, non_blocking=True)
                        #         pred = self.model.forward(target_batch)
                        #         loss_adv = adv_loss_fn(self.discriminator.forward(torch.cat([data_batch,pred.view(1, self.training_dic["batch_size"])], dim=0)), real_label)
                        #         loss_adv.backward()
                        #         self.optimizer.step()

                        # evaluation after each of the epochs
                        eval_loss = 0
                        mse_loss = 0
                        correlation_loss = 0
                        correlation_mask = 0

                        for name, param in self.model.named_parameters():
                            total_norm = 0
                            if param.grad is not None and "weight" in name:
                                param_norm = param.grad.data.norm(2)
                                total_norm += param_norm.item() ** 2
                                total_norm = total_norm ** (1. / 2)
                                writer.add_scalar(f"Model_{model_idx}/{name}_grad", total_norm, total_step)
                            # writer.add_histogram(f"{name}_grad", param.grad, total_step)

                        self.model.eval()
                        total_pred = [torch.tensor([]).to(self.device)]
                        total_target = [torch.tensor([]).to(self.device)]

                        with torch.no_grad():
                            for data in self.eval_loader_list[model_idx]:
                                if "extra_feature" in self.training_dic and self.training_dic["extra_feature"]:
                                    data_batch = torch.cat([data[0], data[-1]], dim=1).to(self.device, non_blocking=True, dtype=data[0].dtype)
                                else:
                                    data_batch = data[0].to(self.device, non_blocking=True)
                                if self.training_dic["denormalizer"]:
                                    batch_mean = torch.unsqueeze(torch.mean(data_batch, dim=1), 1)
                                    batch_std = torch.unsqueeze(torch.std(data_batch, dim=1), 1)
                                    data_batch = torch.cat([data_batch, batch_mean, batch_std.pow(0.5)], dim=1).to(self.device, non_blocking=True)
                                if "horizontal_norm" in self.training_dic and self.training_dic["horizontal_norm"]:
                                    data_batch = F.normalize(data_batch, p=2.0, dim=1)     
                                target_batch = data[1].to(self.device, non_blocking=True)
                                weight = torch.ones_like(target_batch)
                                weight += data[2].to(self.device, non_blocking=True)
                                pred = torch.squeeze(self.model.forward(data_batch))
                                if self.training_dic["ignore_negative_target"]:
                                    mask = (target_batch <= self.training_dic["ignore_threshold"]) | (pred <= self.training_dic["ignore_threshold"])
                                    pred = pred[mask]
                                    target_batch = target_batch[mask]
                                new_mse_loss = nn.MSELoss()(pred, target_batch).item()
                                mse_loss += new_mse_loss
                                if self.training_dic["weight_essential"]:
                                    # mask = target_batch >= mean + self.training_dic["weight_pos"] * std
                                    # weight = torch.where(mask, 2 * torch.ones_like(target_batch), weight)
                                    weight += F.sigmoid(8 * (pred - pred_mean - self.training_dic["mask_pos"] * pred_std)) * self.training_dic["weight_pow_under"]
                                if self.training_dic["loss_fn"] == "CCC" or self.training_dic["loss_fn"] == "Correlation":
                                    new_correlation_loss = self.loss_fn_eval(pred, target_batch, weight).item()
                                    correlation_loss += new_correlation_loss
                                    eval_loss += -self.correlation_ratio * new_correlation_loss + (1-self.correlation_ratio) * new_mse_loss
                                # compute both of the combined loss and the mse loss
                                else:
                                    eval_loss += new_mse_loss
                                    correlation_loss += self.correlation_loss(pred, target_batch).item()
                                if (self.training_dic["update_delta"] and not (epoch+1) % self.training_dic["update_period"]) or self.training_dic["eval_mask"]:
                                    if not total_pred:
                                        total_pred = [pred.detach()]
                                        total_target = [target_batch.detach()]
                                    else:
                                        total_pred.append(pred.detach())  
                                        total_target.append(target_batch.detach())
                        
                        self.model.train()

                        eval_loss /= len(self.eval_loader_list[model_idx])
                        mse_loss /= len(self.eval_loader_list[model_idx])
                        correlation_loss /= len(self.eval_loader_list[model_idx])
                        if (self.training_dic["update_delta"] and not (epoch+1) % self.training_dic["update_period"]) or self.training_dic["eval_mask"]:
                            total_pred = torch.cat(total_pred,dim=0)
                            total_target = torch.cat(total_target,dim=0)
                        if self.training_dic["eval_mask"]:
                            pred_mean = torch.mean(total_pred)
                            pred_std = torch.std(total_pred)
                            mask =  ((total_pred > (pred_mean + self.training_dic["mask_pos"] * pred_std))&(total_target < (mean + self.training_dic["mask_pos"] * std)))
                            # |((total_pred < (pred_mean + self.training_dic["mask_pos"] * pred_std))&(total_target > (mean + self.training_dic["mask_pos"] * std)))
                            total_pred_mask = total_pred[mask]
                            total_target_mask = total_target[mask]
                            if total_pred_mask.shape[0] == 0:
                                correlation_mask = 0
                            else:
                                correlation_mask = (total_pred_mask - pred_mean - self.training_dic["mask_pos"] * pred_std).pow(2) * (total_target_mask - mean - self.training_dic["mask_pos"] * std) / pred_std.pow(2)
                                correlation_mask = torch.nan_to_num(torch.sum(correlation_mask), nan=-1e5, neginf=-1e5).item()
                        result[model_idx]["good_points"].append(torch.sum((total_pred > pred_mean + self.training_dic["mask_pos"] * pred_std)&(total_target > mean + self.training_dic["mask_pos"] * std)).item())



                        # save the model weights if the evaluation loss is improved
                        if not result[model_idx]["eval_loss"] or (mse_loss - 5 * correlation_loss < best_eval_loss):
                            # best_correlation_loss = correlation_loss
                            # if eval_loss < best_eval_loss:
                            best_eval_loss = mse_loss - 5 * correlation_loss
                            torch.save(self.model.state_dict(), self.model_path_list[model_idx])
                            torch.save(self.optimizer.state_dict(), self.optimizer_path_list[model_idx])
                            # self.model.to(self.device)
                        if self.training_dic["check_mask"] and self.training_dic["eval_mask"]:
                            if not result[model_idx]["eval_correlation_mask"] or correlation_mask + 10 * correlation_loss> best_correlation_mask_loss:
                                best_correlation_mask_loss = correlation_mask + 10 * correlation_loss
                                torch.save(self.model.state_dict(), self.model_mask_path_list[model_idx])
                                torch.save(self.optimizer.state_dict(), self.optimizer_mask_path_list[model_idx])
                                # self.model.to(self.device)
                        if self.training_dic["check_most"] and (not result[model_idx]["good_points"] or result[model_idx]["good_points"][-1] > best_points_cnt):
                            best_points_cnt = result[model_idx]["good_points"][-1]
                            torch.save(self.model.state_dict(), self.model_most_path_list[model_idx])
                            torch.save(self.optimizer.state_dict(), self.optimizer_most_path_list[model_idx])
                            # self.model.to(self.device)

                        if result[model_idx]["eval_loss"] and eval_loss > min(result[model_idx]["eval_loss"]) and total_step >= self.training_dic["overfit_threshold"]:
                            overfit_cnt += 1
                        else:
                            overfit_cnt = 0

                        result[model_idx]["eval_loss"].append(eval_loss)

                        pbar.set_description("Model :%d|Epoch :%d|Train Loss: %.2e|Evaluation Loss: %.2e|MSE Loss: %.2e|Correlation: %.2e|LR: %.2e" % (model_idx+1, epoch+1, train_loss, eval_loss, mse_loss, correlation_loss, self.optimizer_list[model_idx].param_groups[0]["lr"]))

                        writer.add_scalar("Training_Stats/Model %d/Train Loss" % model_idx, train_loss, total_step)
                        writer.add_scalar("Training_Stats/Model %d/Evaluation Loss" % model_idx, eval_loss, total_step)
                        writer.add_scalar("Training_Stats/Model %d/MSE Loss" % model_idx, mse_loss, total_step)
                        writer.add_scalar("Training_Stats/Model %d/Correlation" % model_idx, correlation_loss, total_step)
                        writer.add_scalar("Training_Stats/Model %d/Learning rate" % model_idx, self.optimizer_list[model_idx].param_groups[0]["lr"], total_step)
                        total_step += 1



                        result[model_idx]["eval_mse_loss"].append(mse_loss)
                        result[model_idx]["eval_correlation_loss"].append(correlation_loss)
                        if self.training_dic["eval_mask"]:
                            result[model_idx]["eval_correlation_mask"].append(correlation_mask)


                        if self.training_dic["retrain"] and overfit_cnt == self.training_dic["overfit_patience"]:
                            overfit_cnt = 0
                            self.training_dic["retrain_lr"] = max(self.optimizer_list[model_idx].param_groups[0]["lr"]/2, self.training_dic["retrain_lr"])
                            if self.training_dic["check_mask"]:
                                self.model.load_state_dict(torch.load(self.model_mask_path_list[model_idx], map_location=self.device))
                                self.optimizer_list[model_idx].load_state_dict(torch.load(self.optimizer_mask_path_list[model_idx]))
                            else:
                                self.model.load_state_dict(torch.load(self.model_path_list[model_idx], map_location=self.device))
                                self.optimizer_list[model_idx].load_state_dict(torch.load(self.optimizer_path_list[model_idx]))
                            for param_group in self.optimizer_list[model_idx].param_groups:
                                param_group["lr"] = self.training_dic["retrain_lr"]
                            del self.scheduler
                            gc.collect()
                            torch.cuda.empty_cache()
                            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_list[model_idx], "min", factor=self.training_dic["scheduler_factor"], patience=self.training_dic["scheduler_patience"], threshold=self.training_dic["scheduler_threshold"])

                            self.training_dic["retrain_lr"] = max(self.training_dic["retrain_lr"]*self.training_dic["retrain_lr_factor"], self.training_dic["scheduler_threshold"])
                            retrain_cnt = 1
                        else:
                            if self.training_dic["loss_fn"] == "CCC" or self.training_dic["loss_fn"] == "Correlation":
                                self.scheduler.step(mse_loss - 5 * correlation_loss)
                            else:
                                self.scheduler.step(eval_loss)
                            retrain_cnt += 1

                        if self.training_dic["update_delta"] and self.training_dic["reg_loss"] == "Huber":
                            if not (epoch+1) % self.training_dic["update_period"]:
                                if total_pred.shape[0] > 0:
                                    self.training_dic["huber_delta"] = np.percentile(np.abs(total_pred.cpu().numpy()-total_target.cpu().numpy()), 90)
                                    self.reg_fn = nn.HuberLoss(delta=self.training_dic["huber_delta"])
                        result[model_idx]["huber_delta"].append(self.training_dic["huber_delta"])

                        df = pd.DataFrame(result[model_idx])
                        df.to_csv(self.log_path_list[model_idx], index=False, header = True)

        del self.train_loader_list
        gc.collect()
        torch.cuda.empty_cache()

        if "finetune" in self.training_dic and self.training_dic["finetune"]:
            self.fine_tune()

        if "discrimination" in self.training_dic and self.training_dic["discrimination"]:
            for model_idx in range(self.training_dic["num_buckets"]):                
                if "save_result_graph" in self.training_dic and self.training_dic["save_result_graph"]:
                    self.model_nd = create_model(self.cfg_list[model_idx].model)
                    self.model_nd.load_state_dict(torch.load(self.model_path_list[model_idx], map_location=self.device))
                    torch.save(self.model_nd.state_dict(), self.model_nd_path_list[model_idx])
                self.adversarial_list[model_idx].adv_against_model(self.model_path_list[model_idx], self.optimizer_path_list[model_idx], self.finetune_train_loader_list[model_idx], self.finetune_eval_loader_list[model_idx])
                if "save_result_graph" in self.training_dic and self.training_dic["save_result_graph"]:
                    self.model = create_model(self.cfg_list[model_idx].model)
                    self.model.load_state_dict(torch.load(self.model_path_list[model_idx], map_location=self.device)) 
                    self.model.eval()
                    self.model_nd.eval()
                    total_target = []
                    total_pred = []
                    total_pred_nd = []
                    with torch.no_grad():
                        for data in self.eval_loader_list[model_idx]:
                            if "extra_feature" in self.training_dic and self.training_dic["extra_feature"]:
                                data_batch = torch.cat([data[0], data[-1]], dim=1).to(self.device, non_blocking=True, dtype=data[0].dtype)
                            else:                    
                                data_batch = data[0].to(self.device, non_blocking=True)
                            target_batch = data[1].to(self.device, non_blocking=True)
                            pred = torch.squeeze(self.model.forward(data_batch))
                            pred_nd = torch.squeeze(self.model_nd.forward(data_batch))
                            total_target.append(target_batch.detach())
                            total_pred.append(pred.detach())
                            total_pred_nd.append(pred_nd.detach())
                    total_target = torch.cat(total_target, dim=0)
                    total_pred = torch.cat(total_pred, dim=0)
                    total_pred_nd = torch.cat(total_pred_nd, dim=0)
                    result_to_graph = {"target":total_target.cpu().tolist(), "pred":total_pred.cpu().tolist(), "pred_nd":total_pred_nd.cpu().tolist()}
                    df_graph = pd.DataFrame(result_to_graph)
                    df_graph.to_csv(Path(self.training_dic["model_path"]) / model_name / f'result_to_graph_{model_idx}.csv', index=False, header = True)
        
        for model_idx in range(self.training_dic["num_buckets"]):
            self.model_list[model_idx].load_state_dict(torch.load(self.model_path_list[model_idx], map_location=self.device))
            self.model_list[model_idx].eval()

        if "regressor_type" in self.training_dic and self.training_dic["regressor_type"] == "nn.linear":        
            self.regressor_training()
        else:
            pred_list = []
            target = []
            sample_weights = []
            for data in self.eval_loader:
                cur_pred = []
                if "extra_feature" in self.training_dic and self.training_dic["extra_feature"]:
                    data_batch = torch.cat([data[0], data[-1]], dim=1).to(self.device, non_blocking=True, dtype=data[0].dtype)
                else:                    
                    data_batch = data[0].to(self.device, non_blocking=True)
                target.append(data[1].to(non_blocking=True).detach())
                for model_idx in range(self.training_dic["num_buckets"]):
                    cur_pred.append(self.model_list[model_idx].forward(data_batch[:,self.feature_list[model_idx]]).to("cpu").detach())
                cur_pred = torch.cat(cur_pred, dim=-1)
                pred_list.append(cur_pred)
                weight = torch.ones(data_batch.shape[0])
                weight += data[2].to(non_blocking=True)
                if "weight_essential" in self.training_dic and self.training_dic["weight_essential"]:
                    weight += F.sigmoid(8 * (pred - pred_mean - self.training_dic["mask_pos"] * pred_std)) * self.training_dic["weight_pow_under"]
                sample_weights.append(weight)
            pred = torch.cat(pred_list, dim=0).numpy()
            target = torch.cat(target, dim=0).numpy()
            sample_weights = torch.cat(sample_weights, dim=0).numpy()
            self.linear_regressor.fit(pred, target, sample_weight=sample_weights)
            self.linear_coef = self.linear_regressor.coef_
            np.save(Path(self.training_dic["model_path"])/model_name/"linear_coefficient.npy", self.linear_coef)
            print(f"Linear Regression fit to {self.linear_coef}")
            # import ipdb; ipdb.set_trace()





        if ("finetune" in self.training_dic and self.training_dic["finetune"]) or ("discrimination" in self.training_dic and self.training_dic["discrimination"]):
            del self.finetune_train_loader_list
            del self.finetune_eval_loader_list
            del finetune_train_dataset_list
            del finetune_eval_dataset_list
        gc.collect()
        torch.cuda.empty_cache()

        pearson_ic, spearman_ic, mse_loss = self.evaluation(self.cfg_list)

        del self.eval_loader_list
        del self.train_dataset_list
        del self.eval_dataset_list

        gc.collect()
        torch.cuda.empty_cache()
        writer.close()

        # train_input_buffer.unlink()

        return pearson_ic, spearman_ic, mse_loss

    # def fine_tune(self):
    #     self.model.load_state_dict(torch.load(self.model_state_path))
    #     self.model.to(self.device)
    #     pbar_finetune = tqdm(range(self.training_dic["finetune_epoch"]), desc="finetune", ncols=160)

    #     layer_len = len(self.cfg.model.model_params["layers_hidden"]) - 1
    #     layer_threshold = layer_len - self.training_dic["finetune_layer"]

    #     params_to_update = []

    #     result_finetune = {"train_loss":[], "eval_loss":[], "eval_mse_loss":[], "eval_correlation_loss":[]}

    #     for name, params in self.model.named_parameters():
    #         if int(name.split(".")[1]) < layer_threshold:
    #             params.requires_grad = False
    #         else:
    #             params.requires_grad = True
    #             params_to_update.append(params)

    #     self.optimizer = torch.optim.AdamW(params_to_update, lr=self.training_dic["lr"]/self.training_dic["finetune_lr_ratio"], weight_decay=self.weight_decay)

    #     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, "min", factor=self.training_dic["scheduler_factor"], patience=self.training_dic["finetune_scheduler_patience"], threshold=self.training_dic["scheduler_threshold"])

    #     finetune_step = len(self.finetune_train_loader) // self.training_dic["finetune_step_num"]

    #     total_pred = [torch.tensor([]).to(self.device)]
    #     with torch.no_grad():
    #         for data in self.finetune_train_loader:
    #             if "extra_feature" in self.training_dic and self.training_dic["extra_feature"]:
    #                 data_batch = torch.cat([data[0], data[-1]], dim=1).to(self.device, non_blocking=True, dtype=data[0].dtype)
    #             else:                    
    #                 data_batch = data[0].to(self.device, non_blocking=True)
    #             pred = torch.squeeze(self.model.forward(data_batch))
    #             if not total_pred:
    #                 total_pred = [pred.detach()]
    #             else:
    #                 total_pred.append(pred.detach())
    #     total_pred = torch.cat(total_pred, dim=0)
    #     pred_mean = torch.mean(total_pred)
    #     pred_std = torch.std(total_pred)

    #     mse_loss = 0
    #     correlation_loss = 0

        
    #     with torch.no_grad():
    #         for data in self.finetune_train_loader:
    #             if "extra_feature" in self.training_dic and self.training_dic["extra_feature"]:
    #                 data_batch = torch.cat([data[0], data[-1]], dim=1).to(self.device, non_blocking=True, dtype=data[0].dtype)
    #             else:                    
    #                 data_batch = data[0].to(self.device, non_blocking=True)
    #             target_batch = data[1].to(self.device, non_blocking=True)
    #             pred = torch.squeeze(self.model.forward(data_batch))
    #             weight = torch.ones_like(target_batch)
    #             new_mse_loss = F.mse_loss(pred, target_batch).item()
    #             mse_loss += new_mse_loss
    #             if "finetune_weight_essential" in self.training_dic and self.training_dic["finetune_weight_essential"]:
    #                 weight += F.sigmoid(8 * (pred - pred_mean - self.training_dic["mask_pos"] * pred_std)) * self.training_dic["weight_pow_under"]
    #             if self.training_dic["loss_fn"] == "CCC" or self.training_dic["loss_fn"] == "Correlation":
    #                 new_correlation_loss = self.loss_fn_eval(pred, target_batch, weight).item()
    #                 correlation_loss += new_correlation_loss
    #             else:
    #                 correlation_loss += self.correlation_loss(pred, target_batch).item()
    #     mse_loss /= len(self.finetune_eval_loader)
    #     correlation_loss /= len(self.finetune_eval_loader)

    #     best_eval_loss = mse_loss - 5 * correlation_loss
    #     overfit_cnt = 0

    #     end_train = False

    #     for epoch in pbar_finetune:
    #         if end_train:
    #             break
    #         self.model.train()
    #         current_loss = 0
    #         step_cnt = 0
    #         for data in self.finetune_train_loader:
    #             step_cnt += 1
    #             if "extra_feature" in self.training_dic and self.training_dic["extra_feature"]:
    #                 data_batch = torch.cat([data[0], data[-1]], dim=1).to(self.device, non_blocking=True, dtype=data[0].dtype)
    #             else:                    
    #                 data_batch = data[0].to(self.device, non_blocking=True)
    #             target_batch = data[1].to(self.device, non_blocking=True)
    #             weight = torch.ones_like(target_batch) + data[2].to(self.device, non_blocking=True)
    #             pred = torch.squeeze(self.model.forward(data_batch))
    #             if "finetune_weight_essential" in self.training_dic and self.training_dic["finetune_weight_essential"]:
    #                 weight += torch.max(F.sigmoid(8 * (target_batch - self.mean -self.training_dic["weight_pos"] * self.std)), F.sigmoid(8 * (pred- pred_mean -self.training_dic["weight_pos"] * pred_std))) * self.training_dic["weight_pow_over"]
    #             if self.training_dic["loss_fn"] == "CCC" or self.training_dic["loss_fn"] == "Correlation":
    #                 correlation_loss = self.loss_fn(pred, target_batch, weight)
    #                 reg_loss = (self.reg_fn(pred, target_batch) * weight).mean()
    #                 loss = -self.correlation_ratio * correlation_loss + (1 - self.correlation_ratio) * reg_loss
    #             else:
    #                 reg_loss = (self.reg_fn(pred, target_batch) * weight).mean()
    #                 loss = reg_loss
    #             current_loss += loss.item()
    #             if self.training_dic["enable_l1"]:
    #                 loss += self.training_dic["l1_ratio"] * self.l1_regularization()
    #             loss.backward()
    #             torch.nn.utils.clip_grad_norm_(params_to_update, max_norm=1.0)
    #             self.optimizer.step()

    #             if not step_cnt % finetune_step:
    #                 eval_loss = 0
    #                 mse_loss = 0
    #                 correlation_loss = 0

    #                 train_loss = current_loss / step_cnt
    #                 step_cnt = 0

    #                 self.model.eval()

    #                 total_pred = [torch.tensor([]).to(self.device)]
    #                 # total_target = [torch.tensor([]).to(self.device)]

    #                 with torch.no_grad():
    #                     for data in self.finetune_eval_loader:
    #                         if "extra_feature" in self.training_dic and self.training_dic["extra_feature"]:
    #                             data_batch = torch.cat([data[0], data[-1]], dim=1).to(self.device, non_blocking=True, dtype=data[0].dtype)
    #                         else:                    
    #                             data_batch = data[0].to(self.device, non_blocking=True)
    #                         target_batch = data[1].to(self.device, non_blocking=True)
    #                         weight = torch.ones_like(target_batch) + data[2].to(self.device, non_blocking=True)
    #                         pred = torch.squeeze(self.model.forward(data_batch))
    #                         new_mse_loss = F.mse_loss(pred, target_batch).item()
    #                         mse_loss += new_mse_loss
    #                         if "finetune_weight_essential" in self.training_dic and self.training_dic["finetune_weight_essential"]:
    #                             weight += F.sigmoid(8 * (pred - pred_mean - self.training_dic["mask_pos"] * pred_std)) * self.training_dic["weight_pow_under"]
    #                         if self.training_dic["loss_fn"] == "CCC" or self.training_dic["loss_fn"] == "Correlation":
    #                             new_correlation_loss = self.loss_fn_eval(pred, target_batch, weight).item()
    #                             correlation_loss += new_correlation_loss
    #                             eval_loss += -self.correlation_ratio * new_correlation_loss + (1-self.correlation_ratio) * new_mse_loss
    #                         else:
    #                             eval_loss += new_mse_loss
    #                             correlation_loss += self.correlation_loss(pred, target_batch).item()
    #                         if not total_pred:
    #                             total_pred = [pred.detach()]
    #                             # total_target = [target_batch.detach()]
    #                         else:
    #                             total_pred.append(pred.detach())
    #                             # total_target.append(target_batch.detach())
                            
                        
    #                     eval_loss /= len(self.finetune_eval_loader)
    #                     mse_loss /= len(self.finetune_eval_loader)
    #                     correlation_loss /= len(self.finetune_eval_loader)

    #                     total_pred = torch.cat(total_pred, dim=0)
    #                     # total_target = torch.cat(total_target, dim=0)

    #                     pred_mean = torch.mean(total_pred)
    #                     pred_std = torch.std(total_pred)

    #                     if "finetune_earlystop" in self.training_dic and self.training_dic["finetune_earlystop"]:
    #                         if result_finetune["eval_loss"] and (mse_loss - 5 * correlation_loss > best_eval_loss):
    #                             overfit_cnt += 1
    #                         else:
    #                             overfit_cnt = 0
    #                         if overfit_cnt >= self.training_dic["finetune_patience"]:
    #                             end_train = True
    #                             break

    #                     if (mse_loss - 5 * correlation_loss < best_eval_loss):
    #                         best_eval_loss = mse_loss - 5 * correlation_loss
    #                         torch.save(self.model.state_dict(), self.model_state_path)
    #                         torch.save(self.optimizer.state_dict(), self.optimizer_path)
    #                         # self.model.to(self.device)
                        
                        
    #                     result_finetune["train_loss"].append(train_loss)
    #                     result_finetune["eval_loss"].append(eval_loss)
    #                     result_finetune["eval_mse_loss"].append(mse_loss)
    #                     result_finetune["eval_correlation_loss"].append(correlation_loss)

    #                     if self.training_dic["loss_fn"] == "CCC" or self.training_dic["loss_fn"] == "Correlation":
    #                         scheduler.step(mse_loss - 5 * correlation_loss)
    #                     else:
    #                         scheduler.step(mse_loss)

    #                     pbar_finetune.set_description("Epoch :%d|Train Loss: %.2e|Evaluation Loss: %.2e|MSE Loss: %.2e|Correlation: %.2e|LR: %.2e" % (epoch+1, train_loss, eval_loss, mse_loss, correlation_loss, self.optimizer.param_groups[0]["lr"]))
                    
    #                     df = pd.DataFrame(result_finetune)
    #                     df.to_csv(self.log_path_finetune, index=False, header = True)

    # def quick_evaluation(self):
    #     self.check_most = self.training_dic["check_most"]
    #     self.check_mask = self.training_dic["check_mask"]
    #     cfg = torch.load(self.dict_path)
    #     self.training_dic = cfg.training.model_params
    #     self.training_dic["check_most"] = self.check_most
    #     self.training_dic["check_mask"] = self.check_mask
    #     self.vertical_norm = self.training_dic["vertical_norm"]
    #     if not "denormalizer" in self.training_dic:
    #         self.training_dic["denormalizer"] = False

    #     pearson_ic, spearman_ic, mse_loss = self.evaluation(cfg)

    #     del self.eval_loader
    #     gc.collect()
    #     torch.cuda.empty_cache()

    #     return pearson_ic, spearman_ic, mse_loss


    def l1_regularization(self):
        l1_loss = sum(param.abs().sum() for param in self.model.parameters())
        return l1_loss


    def evaluation(self, cfg_list):
        # load the best model from the training process
        self.model_result_list = []
        for cfg in cfg_list:
            self.model_result_list.append(create_model(cfg.model))
        # if "check_mask" in cfg.training.model_params and cfg.training.model_params["check_mask"]:
        #     model_result.load_state_dict(torch.load(self.model_mask_path, map_location=self.device))
        # elif "check_most" in cfg.training.model_params and cfg.training.model_params["check_most"]:
        #     model_result.load_state_dict(torch.load(self.model_most_path, map_location=self.device))
        # else:
        for model_result, model_path in zip(self.model_result_list, self.model_path_list):
            model_result.load_state_dict(torch.load(model_path, map_location=self.device))
        for model_result in self.model_result_list:
            model_result.eval()
        self.regressor.load_state_dict(torch.load(self.regressor_path, map_location=self.device))
        self.regressor.eval()
        total_pred = None
        np_target = None
        with torch.no_grad():
            for data in self.eval_loader:
                if "extra_feature" in self.training_dic and self.training_dic["extra_feature"]:
                    data_batch = torch.cat([data[0], data[-1]], dim=1).to(self.device, non_blocking=True, dtype=data[0].dtype)
                else:                    
                    data_batch = data[0].to(self.device, non_blocking=True)
                if self.training_dic["denormalizer"]:
                    batch_mean = torch.unsqueeze(torch.mean(data_batch, dim=1), 1)
                    batch_std = torch.unsqueeze(torch.std(data_batch, dim=1), 1)
                    data_batch = torch.cat([data_batch, batch_mean, batch_std.pow(0.5)], dim=1).to(self.device, non_blocking=True)
                if "horizontal_norm" in self.training_dic and self.training_dic["horizontal_norm"]:
                    data_batch = F.normalize(data_batch, p=2.0, dim=1)
                target_batch = data[1].to(non_blocking=True)                
                pred = self.regressor_predict(data_batch)
                if not total_pred:
                    total_pred = [torch.squeeze(pred).to("cpu").detach().numpy()]
                    np_target = [target_batch.numpy()]
                else:
                    total_pred.append(torch.squeeze(pred).to("cpu").detach().numpy())
                    np_target.append(target_batch.numpy())
        total_pred = np.concatenate(total_pred, axis=0)
        np_target = np.concatenate(np_target, axis=0)

        # evaluate the model on the evaluation dataset. Return and print the correlation coefficients as well as mse loss
        # total_pred = model_result.forward(self.eval_dataset.x).detach().numpy()
        # total_pred = np.squeeze(total_pred)
        # assert np.isnan(total_pred).sum() == 0, print(total_pred)
        # np_target = self.eval_dataset.y.numpy()
        delta_candidate = np.percentile(np.abs(total_pred-np_target), 90)
        mse_loss = mean_squared_error(total_pred, np_target)
        pearson_ic = WeightedCorrNp(x=total_pred, y=np_target, w=np.ones(len(np_target)))('pearson')
        spearman_ic = WeightedCorrNp(x=total_pred, y=np_target, w=np.ones(len(np_target)))('spearman')
        print(f"peason_ic: {pearson_ic} | spearman_ic: {spearman_ic} | mse_loss: {mse_loss} | std of pred: {np.std(total_pred)} | mean of pred: {np.mean(total_pred)} | max of pred: {np.max(total_pred)} | min of pred: {np.min(total_pred)}")
        print(f"according to the evaluation, the suggested delta value is {delta_candidate}")
        return pearson_ic, spearman_ic, mse_loss
    
    def regressor_training(self):
        regressor_step = len(self.regressor_train_loader) // self.training_dic["step_num"]
        self.regressor.train()
        optimizer = torch.optim.Adam(self.regressor.parameters(), lr=self.training_dic["regressor_lr"])
        self.training_dic["retrain_lr"] = self.training_dic["regressor_lr"]
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, "min", factor=self.training_dic["scheduler_factor"], patience=1, threshold=self.training_dic["scheduler_threshold"])
        pbar_regressor = tqdm(range(self.training_dic["regressor_epoch"]))
        best_eval_loss = float("inf")
        pred_mean = 0
        pred_std = 0
        retrain_cnt = 0
        overfit_cnt = 0

        for epoch in pbar_regressor:
            cur_step = 0
            total_step = 0
            for data in self.regressor_train_loader:
                if "extra_feature" in self.training_dic and self.training_dic["extra_feature"]:
                    data_batch = torch.cat([data[0], data[-1]], dim=1).to(self.device, non_blocking=True, dtype=data[0].dtype)
                else:                    
                    data_batch = data[0].to(self.device, non_blocking=True)
                regressor_input = torch.cat([self.model_list[i](data_batch[:, self.feature_list[i]]) for i in range(self.training_dic["num_buckets"])], dim=1)
                target_batch = data[1].to(self.device, non_blocking=True)
                pred = self.regressor(regressor_input).squeeze()
                weight = torch.ones_like(target_batch)
                weight += data[2].to(self.device, non_blocking=True)
                if self.training_dic["weight_essential"]:
                    weight += torch.max(F.sigmoid(8 * (target_batch - self.mean - self.training_dic["weight_pos"] * self.std)), F.sigmoid(8 * (pred - pred_mean - self.training_dic["weight_pos"] * pred_std))) * self.training_dic["weight_pow_over"]
                loss = - self.training_dic["correlation_ratio"] * self.correlation_loss(pred, target_batch, weight) + (1 - self.training_dic["correlation_ratio"]) * (self.reg_fn(pred, target_batch) * weight).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                cur_step += 1
                if cur_step >= regressor_step:
                    cur_step = 0
                    total_step += 1
                    self.regressor.eval()
                    with torch.no_grad():
                        for params in self.regressor.parameters():
                            params.clamp_(min=0.0)
                        total_pred = []
                        eval_loss = 0
                        mse_loss = 0
                        correlation_loss = 0
                        for data in self.regressor_eval_loader:
                            if "extra_feature" in self.training_dic and self.training_dic["extra_feature"]:
                                data_batch = torch.cat([data[0], data[-1]], dim=1).to(self.device, non_blocking=True, dtype=data[0].dtype)
                            else:                    
                                data_batch = data[0].to(self.device, non_blocking=True)
                            regressor_input = torch.cat([self.model_list[i](data_batch[:, self.feature_list[i]]) for i in range(self.training_dic["num_buckets"])], dim=1)
                            target_batch = data[1].to(self.device, non_blocking=True)
                            pred = self.regressor(regressor_input).squeeze()
                            if not total_pred:
                                total_pred = [torch.squeeze(pred).to("cpu").detach()]
                            else:
                                total_pred.append(torch.squeeze(pred).to("cpu").detach())
                            weight = torch.ones_like(target_batch)
                            weight += data[2].to(self.device, non_blocking=True)
                            weight += F.sigmoid(8 * (pred - pred_mean - self.training_dic["mask_pos"] * pred_std)) * self.training_dic["weight_pow_under"]
                            new_correlation_loss = self.correlation_loss(pred, target_batch, weight)
                            new_mse_loss = (self.reg_fn(pred, target_batch) * weight).mean()
                            loss = - self.training_dic["correlation_ratio"] * new_correlation_loss + (1 - self.training_dic["correlation_ratio"]) * new_mse_loss
                            eval_loss += loss.item()
                            mse_loss += new_mse_loss.item()
                            correlation_loss += new_correlation_loss.item()

                    eval_loss /= len(self.regressor_eval_loader)
                    mse_loss /= len(self.regressor_eval_loader)
                    correlation_loss /= len(self.regressor_eval_loader)
                    pbar_regressor.set_description(f"Epoch {epoch+1} Step {total_step}|Regressor loss: {eval_loss:.4f} | MSE loss: {mse_loss:.4f} | Correlation loss: {correlation_loss:.4f}")

                        # pbar.set_description("Model :%d|Epoch :%d|Train Loss: %.2e|Evaluation Loss: %.2e|MSE Loss: %.2e|Correlation: %.2e|LR: %.2e" % (model_idx, epoch+1, train_loss, eval_loss, mse_loss, correlation_loss, self.optimizer_list[model_idx].param_groups[0]["lr"]))
                    if mse_loss - 5 * correlation_loss < best_eval_loss:
                        best_eval_loss = mse_loss - 5 * correlation_loss
                        torch.save(self.regressor.state_dict(), self.regressor_path)
                        torch.save(optimizer.state_dict(), self.regressor_optimizer_path)
                        overfit_cnt = 0
                    else:
                        overfit_cnt += 1
                    scheduler.step(mse_loss - 5 * correlation_loss)

                    if self.training_dic["retrain"] and overfit_cnt == self.training_dic["overfit_patience"] and epoch >= self.training_dic["overfit_threshold"]:
                        overfit_cnt = 0
                        self.training_dic["retrain_lr"] = max(optimizer.param_groups[0]["lr"]/2, self.training_dic["retrain_lr"])
                        self.regressor.load_state_dict(torch.load(self.regressor_path, map_location=self.device))
                        optimizer.load_state_dict(torch.load(self.regressor_optimizer_path))
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = self.training_dic["retrain_lr"]
                        del scheduler
                        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, "min", factor=self.training_dic["scheduler_factor"], patience=1, threshold=self.training_dic["scheduler_threshold"])
                        retrain_cnt = 1
                    else:
                        scheduler.step(mse_loss - 5 * correlation_loss)
                        retrain_cnt += 1

                    total_pred = torch.cat(total_pred, dim=0)
                    pred_mean = total_pred.mean()
                    pred_std = total_pred.std()
                    self.regressor.train()
    
    def regressor_predict(self, data_batch):
        self.regressor.eval()
        with torch.no_grad():
            if "regressor_type" in self.training_dic and self.training_dic["regressor_type"] == "nn.linear":
                regressor_input = torch.cat([self.model_result_list[i](data_batch[:, self.feature_list[i]]) for i in range(self.training_dic["num_buckets"])], dim=1)
                pred = self.regressor(regressor_input)
            else:
                pred = torch.zeros(data_batch.shape[0], device=self.device)
                for i in range(self.training_dic["num_buckets"]):
                    pred += self.linear_coef[i] * torch.squeeze(self.model_result_list[i](data_batch[:, self.feature_list[i]]))
        return pred


    def correlation_loss(self, pred, target, weight=None):
        # correlation_ratio = self.training_dic["correlation_ratio"]

        pred_centered = pred - pred.mean()
        target_centered = target - target.mean()
        if weight is None:
            weight = torch.ones_like(pred)
        corr_nume = torch.sum(pred_centered * target_centered * weight)
        corr_denom = torch.sqrt((torch.sum((pred_centered ** 2)*weight) + 1e-7) * (torch.sum((target_centered ** 2)*weight) + 1e-7))

        corr = corr_nume / corr_denom
        # return - correlation_ratio * corr + (1-correlation_ratio) * torch.nn.MSELoss()(pred, target)

        return corr

    def ccc(self, pred, target, weight=None):
        # compute the combined loss of correlation loss and mse loss with correlation_ratio in the training dict
        # correlation_ratio = self.training_dic["correlation_ratio"]
        if weight is None:
            weight = torch.ones_like(pred)
        sxy = torch.mean((pred - torch.mean(pred)) * weight * (target - torch.mean(target)) * weight)
        ccc_loss = 2 * sxy /(torch.var(pred * weight) + torch.var(target * weight) + ((pred *  weight).mean() - (target *  weight).mean()) ** 2 + 1e-7)
        # return - correlation_ratio * ccc_loss + (1-correlation_ratio) * torch.nn.MSELoss()(pred, target)
        return ccc_loss

    def spearman_ic(self, x, y):
        day_min = np.min(x, axis=1)
        day_max = np.max(x, axis=1)
        res = np.zeros_like(day_min).squeeze()
        days, stocks = x.shape
        for i in range(days):
            if day_min[i] == day_max[i]:
                res[i] = 0
            else:
                res[i] = WeightedCorrNp(x=x[i], y=y[i], w=np.ones(stocks))('spearman')
        return res.unsqueeze(dim=0)

    def calculate_weights(value_vector, indices):        
        mean = np.mean(np.nan_to_num(value_vector * indices))
        weights = value_vector - mean
        return weights - np.sum(np.abs(np.nan_to_num(weights)))

    def calculate_turnover(weights):
        total_tvr = 0
        n = 0
        days = len(weights)
        for j in range(days-1):
            diff_weights = np.abs(np.nan_to_num(weights[j+1]-weights[j]))
            daily_tvr = np.sum(diff_weights)
            if daily_tvr != 0:
                total_tvr += daily_tvr
                n += 1
        if n != 0:
            return total_tvr / n
        else:
            return float("inf")

    def feature_bucketing(self, cfg):
        features = np.where(self.indices[0])[0]
        alpha_indices = {}
        for pos, idx in enumerate(features):
            alpha_indices[idx] = pos
        weights = {}

        days, stocks = self.indices[1].shape

        pbar_weights = tqdm(range(self.indices[0].shape[0]), desc="calculating weights")
        with ThreadPoolExecutor(max_workers=75) as executor:
            for i in pbar_weights:
                weights[i] = [np.zeros(stocks) for _ in range(days)]
                if self.indices[0][i]:
                    for j in range(self.indices[1].shape[0]):
                        weights[i][j] = executor.submit(calculate_weights_cython, self.train_input[i][j,:], self.indices[1][j,:].view(np.uint8))
                        pbar_weights.set_description(f"calculating weights: feature {i}, day {j} finished")

        for key in weights:
            for j in range(days):
                if self.indices[0][key]:
                    weights[key][j] = weights[key][j].result()
            weights[key] = np.array(weights[key])
        turn_over = {}

        pbar_turnover = tqdm(range(self.indices[0].shape[0]), desc="calculating turnover")
        with ThreadPoolExecutor(max_workers=75) as executor:
            for i in pbar_turnover:
                if self.indices[0][i]:
                    turn_over[i] = executor.submit(calculate_turnover_cython, weights[i]) 
                    pbar_turnover.set_description(f"calculating turnover: feature {i} finished")

        for key in turn_over:
            turn_over[key] = turn_over[key].result()

        feature_indices = sorted(turn_over.keys(), key=lambda x: turn_over[x], reverse=True)

        
        feature_buckets = []
        
        bucket_size = math.ceil(np.sum(self.indices[0]) / self.training_dic["num_buckets"])

        current_bucket = []
        for idx in feature_indices:
            current_bucket.append(idx)
            if len(current_bucket) >= bucket_size:
                feature_buckets.append(current_bucket)
                current_bucket = []

        if current_bucket:
            feature_buckets.append(current_bucket)

        feature_list = []

        for bucket in feature_buckets:
            feature_list.append([alpha_indices[idx] for idx in bucket])
            
        feature_buckets_indices = [np.zeros_like(self.indices[0]).astype(bool) for _ in range(len(feature_list))]

        for i in range(len(feature_buckets)):
            for idx in feature_buckets[i]:
                feature_buckets_indices[i][idx] = True

        feature_list_indices = [np.concatenate([np.zeros(np.sum(self.indices[0])), np.ones(self.feature_matrix.shape[0])]).astype(bool) for i in range(len(feature_list))]  

        for i in range(len(feature_list)):
            for idx in feature_list[i]:
                feature_list_indices[i][idx] = True
      
        # import ipdb; ipdb.set_trace()

        return feature_buckets_indices, feature_list_indices

    def feature_selection(self, cfg):

        features = np.where(self.indices[0])[0]

        selection_res = [True] * np.sum(self.indices[0])

        # Mark the position of the features after dataloader

        alpha_indices = {}

        for pos, idx in enumerate(features):
            alpha_indices[idx] = pos

        corr_dict = defaultdict(list)
        
        to_delete = set()

        # Pick the features that doesn't correlated to the target or have a low information ratio

        horizon = cfg.training.model_params["horizon"]

        # self.widgets = [progressbar.FormatLabel(f"[{self.model_name[-10:]}] "), progressbar.FormatLabel(item_name), "|", progressbar.Percentage(), "|", progressbar.Bar(marker="█", left="[", right="]", fill="-"), "|", progressbar.SimpleProgress(), "|", progressbar.ETA()]

        my_widgets = [
            f"[{self.model_name}] ",  
            # 我们创建一个 TextColumn，引用一个名为 'status' 的自定义字段。
            # 我们用 {task.fields[status]} 来访问它。
            TextColumn(" [bold black]{task.fields[status]}[/bold black]"),
            # 进度百分比 (Rich 默认支持)
            TextColumn("[bold blue]{task.percentage:>3.0f}%"), 
            # 进度条本体
            BarColumn(), 
            # 预计剩余时间 (Rich 默认支持)
            TimeRemainingColumn(),   
        ]

        # for j in pbar_fs:
        #     corr_dict[j] = niocorr(self.train_input[j][:-horizon,:], self.train_output[:-horizon,:])
        #     # print(f"finished feature {j}")
        #     pbar_fs.set_description(f"feature {j} finished")

        # alpha_score= {}

        
        if "single_mode" not in cfg.training.model_params:
            cfg.training.model_params["single_mode"] = "pearson"

        
        if "single_threshold" not in cfg.training.model_params:
            cfg.training.model_params["single_threshold"] = 0.01

        MAX_STEPS = len(features)
        with Progress(*my_widgets) as progress:
            task_id = progress.add_task(
                "Processing Data", 
                total=MAX_STEPS, 
                status="[dim]Initializing...[/dim]" # 设置自定义字段 status 的初始值
            )
            with ProcessPoolExecutor(max_workers=50) as executor:
                for j in features:
                    if cfg.training.model_params["single_mode"] == "spearman":
                        corr_dict[j] = executor.submit(niocorr_spearman, self.train_input[j][:-horizon,:], self.train_output[:-horizon,:])
                    else:
                        corr_dict[j] = executor.submit(niocorr, self.train_input[j][:-horizon,:], self.train_output[:-horizon,:])
                    progress.update(
                        task_id, 
                        advance=1, 
                        status=f"Feature {j} is done" # 通过关键字参数更新自定义字段
                    )
        alpha_score= {}
        for i in corr_dict:
            corr_dict[i] = corr_dict[i].result()

        for i in corr_dict:
            mean = np.abs(np.mean(np.nan_to_num(corr_dict[i])))
            std = np.std(np.nan_to_num(corr_dict[i]))
            alpha_score[i] = mean / std
            if mean < cfg.training.model_params["single_threshold"] or alpha_score[i] < 0.3:
                to_delete.add(i)

        # Add more feature selection logics here

        # Pick features which perform worse on the previous test from pairs of features that too correlated

        if "pair_ic_selection" in cfg.training.model_params and cfg.training.model_params["pair_ic_selection"]:

            pair_ic = {}

            for i in range(len(features)-1):
                for j in range(i+1, len(features)):
                    if features[i] not in to_delete and features[j] not in to_delete:
                        pair_ic[(features[i], features[j])] = None

            pbar_pairic = tqdm(pair_ic, desc="Pairwise Correlation", ncols=160)

            with ProcessPoolExecutor(max_workers=50) as Executor:
                for i, j in pbar_pairic:
                    if cfg.training.model_params["single_mode"] == "spearman":
                        # pair_ic[(i,j)] = np.mean([WeightedCorrNp(x=train_input[features[i]][k][indices[1][k,:]], y=train_input[features[j]][k][indices[1][k,:]], w=np.ones(np.sum(indices[1][k,:])))("spearman") for k in range(indices[1].shape[0]-horizon)])
                        pair_ic[(i,j)] = Executor.submit(niocorr_spearman, self.train_input[i][:-horizon,:], self.train_input[j][:-horizon,:])
                    else:
                        pair_ic[(i,j)] = Executor.submit(niocorr, self.train_input[i][:-horizon,:], self.train_input[j][:-horizon,:])
                    pbar_pairic.set_description(f"Pair {i, j} is finished")
            
            for pair in pair_ic:
                pair_ic[pair] = np.mean(pair_ic[pair].result())

            # import ipdb; ipdb.set_trace()

            for i, j in pair_ic:
                if i not in to_delete and j not in to_delete and pair_ic[(i,j)] > 0.7:
                    if alpha_score[i] > alpha_score[j]:
                        to_delete.add(j)
                    else:
                        to_delete.add(i)


        # Eliminate the features that are picked and mark them in selection result to inform the inference module

        for i in to_delete:
            self.indices[0][i] = False
            selection_res[alpha_indices[i]] = False
        
        print(f"{len(to_delete)} out of {len(features)} alphas deleted due to low information")


        return selection_res

    def load(self, model_path, model_name, **kwargs):
        if model_path is None:
            assert self.model_path is not None
            model_path = self.model_path
        self.model_name = model_name
        self.folder_path = Path(model_path) / model_name
        return 1

    def predict(self, x, cfg, date=None, test_stage=True, lock=None, valid_stock=None, *args, **kwargs):
        if cfg.get("data_x", None):
            for op in cfg.get('data_x'):
                ProcessDataForTrainingIntermediate = create_op_process(cfg.data_x[op])
                ProcessDataForTrainingIntermediate.apply(x, test_stage=True)
        """        
        if cfg.model.type == "FastKAN":
            for i in range(len(cfg.model.model_params["layers_hidden"])-1):
                cfg.model.model_params["layers_hidden"][i] *= input_dim
        """

        # print(valid_stock.shape)
        # print(valid_stock.sum())
        # print(valid_stock)
        # return
        
        if cfg.training.model_params["check_mask"]:
            self.model_state_dict_path = self.model_state_dict_path.with_name("final_model_mask.pt")
        elif "check_most" in cfg.training.model_params and cfg.training.model_params["check_most"]:
            self.model_state_dict_path = self.model_state_dict_path.with_name("final_model_most.pt")
            
        self.dict_path_list = [self.folder_path / f"training_dict_{i}.pt" for i in range(self.training_dic["num_buckets"])]
        self.model_dict_path_list = [self.folder_path / f"final_model_{i}.pt" for i in range(self.training_dic["num_buckets"])]
        self.regressor_path = self.folder_path / f"final_regressor.pt"
        self.linear_regressor_path = self.folder_path / f"linear_coefficient.npy"


        cfg_list = [torch.load(self.dict_path_list[i]) for i in range(self.training_dic["num_buckets"])]
        # if not "denormalizer" in cfg.training.model_params:
        #     cfg.training.model_params["denormalizer"] = False

        # if "feature_select" in cfg.training.model_params and cfg.training.model_params["feature_select"]:
        #     # print(x.shape, len(cfg.training.model_params["features_delete"]))
        #     x = x[:,cfg.training.model_params["features_delete"]]

        # if "mode" not in cfg.model.model_params:
        #     cfg.model.model_params["mode"] = "default"
        # if "Cheby" in cfg.model.model_params:
        #     if cfg.model.model_params["Cheby"]:
        #         cfg.model.model_params["mode"] = "Cheby"
        #     del cfg.model.model_params["Cheby"]
        # if "KAF" in cfg.model.model_params:
        #     if cfg.model.model_params["KAF"]:
        #         cfg.model.model_params["mode"] = "KAF"
        #     del cfg.model.model_params["KAF"]
        # if "ResNet" in cfg.model.model_params:
        #     cfg.model.model_params["res_net"] = cfg.model.model_params["ResNet"]
        #     del cfg.model.model_params["ResNet"]
        # self.vertical_norm = cfg.training.model_params["vertical_norm"]
        # self.horizontal_norm = "horizontal_norm" in cfg.training.model_params and cfg.training.model_params["horizontal_norm"]

        # import ipdb; ipdb.set_trace()

        if "extra_feature" in cfg_list[0].training.model_params and cfg_list[0].training.model_params["extra_feature"]:
            # print(x.shape, self.feature_matrix.squeeze().transpose(1,0).numpy().shape, valid_stock.sum())
            self.feature_matrix = ((self.feature_matrix.squeeze().transpose(1,0))[valid_stock]).numpy()
            # print("here ", self.feature_matrix.shape)
            x = np.concatenate([x, self.feature_matrix], axis=1)
            x = np.nan_to_num(x)
        else:
            x = np.nan_to_num(x)


        if cfg.training.model_params["denormalizer"]:
            x = np.concatenate([x, np.mean(x, axis=1).reshape(-1, 1), np.power(np.std(x, axis=1), 0.5).reshape(-1, 1)], axis=1)

        if lock is not None:
            with lock:
                super().gpu_helper(*args, **kwargs)
        
        start_time = time.time()
        for cfg in cfg_list:
            cfg.model.model_params["device"] = self.device
        for cfg in cfg_list:
            cfg.model.model_params["infer"] = True
        self.feature_list = [cfg_list[i].training.model_params["feature_chosen"] for i in range(self.training_dic["num_buckets"])]
        
        self.model_result_list = [create_model(cfg.model) for cfg in cfg_list]
        for i in range(self.training_dic["num_buckets"]):
            self.model_result_list[i].load_state_dict(torch.load(self.model_dict_path_list[i], map_location=self.device))
            self.model_result_list[i].eval()
        
        self.regressor = Regressor(self.training_dic["num_buckets"], device=self.device)
        if "regressor_type" in self.training_dic and self.training_dic["regressor_type"] == "nn.linear":
            self.regressor.load_state_dict(torch.load(self.regressor_path, map_location=self.device))
            self.regressor.eval()
        else:
            self.linear_coef = np.load(self.linear_regressor_path)

        print(f"it takes {time.time()-start_time} to initiate")


        start_time = time.time()
        with torch.no_grad():
            pred = self.regressor_predict(torch.from_numpy(x).contiguous().to(torch.float32).to(self.device))

        ret = pred.squeeze().to("cpu").numpy()
        print(f"it takes {time.time()-start_time} to predict")

        if "regressor_type" in self.training_dic and self.training_dic["regressor_type"] == "nn.linear":
            del self.regressor
        del pred, self.model_result_list
        gc.collect()
        torch.cuda.empty_cache()

        return ret


        
    def create_raw_features(self, cfg, model_name, reader):

        self.training_dic = cfg.training.model_params

        tidx = int(model_name.split("_")[-1])

        buffer = cfg.train_basics.get("cache_reader_buffer", 0)

        block4 = reader("block4n")

        secind = reader("secoindustry")
        secind_num = secind.max() + 1

        days, stocks = block4.shape[0]-buffer, block4.shape[1]

        stock_block = np.zeros((4, days, stocks))
        
        # stock_block will add one-hot block information into features
        stock_block[0,:,:] = (block4[buffer:,:] == 0).astype(np.float32)
        stock_block[1,:,:] = (block4[buffer:,:] == 1).astype(np.float32)
        stock_block[2,:,:] = (block4[buffer:,:] == 2).astype(np.float32)
        stock_block[3,:,:] = (block4[buffer:,:] == 3).astype(np.float32)

        stock_block = torch.tensor(stock_block)

        stock_secind = np.zeros((secind_num+1, days, stocks))
        for i in range(secind_num):
            stock_secind[i, :, :] = (secind[buffer:, :] == i).astype(np.float32)
        stock_secind[-1, :, :] = (secind[buffer:, :] < 0).astype(np.float32)
        stock_secind = torch.tensor(stock_secind)


        # load CNE5 factor list (including style, industry and contry)
        self.styleFactorDataNames = [f"CNE5D_RISK.{f}" for f in STYLE_FACTORS["cne5"]]
        self.industryFactorDataNames = INDUSTRY_COV_ORDER["cne5"]
        self.factorList = self.styleFactorDataNames + self.industryFactorDataNames + ["COUNTRY"]

        # load CNE5 factor covariance matrix
        barra_cne5_factor_cov = reader("CNE5D_COV")
        factorCOV = barra_cne5_factor_cov /240 /10000

        barra_cne5_spec_cov = reader("CNE5S_RISK.SRISK")
        specCov = (np.power(barra_cne5_spec_cov, 2) /240 /10000)
        date = barra_cne5_factor_cov.shape[0]

        factorExp = np.zeros((len(self.factorList), date, stocks), dtype=np.float32)
        for i in range(len(self.styleFactorDataNames)):
            names = [f.strip() for f in self.styleFactorDataNames[i].split(",")]
            factorExp[i, :, :] = reader(names[0])

        for i in range(len(self.industryFactorDataNames)):
            indLabel = i + 1
            stockInd = reader("CNE5S_RISK.IND") == indLabel
            factorExp[len(self.styleFactorDataNames)+i, :, :] = stockInd

        factorExp[len(self.factorList) - 1, :, :] = 1

        factorExp = torch.from_numpy(factorExp[:,buffer:,:])


        features_dic = {}

        SH000001_close = reader("000001.SH.IntervalIndices.close")[:,tidx]
        SH000001_amount = reader("000001.SH.IntervalIndices.amount")[:,tidx]
        SZ399001_close = reader("399001.SZ.IntervalIndices.close")[:,tidx]
        SZ399001_amount = reader("399001.SZ.IntervalIndices.amount")[:,tidx]

        SH000300_close = reader("000300.SH.IntervalIndices.close")[:,tidx]
        SH000300_amount = reader("000300.SH.IntervalIndices.amount")[:,tidx]
        SH000905_close = reader("000905.SH.IntervalIndices.close")[:,tidx]
        SH000905_amount = reader("000905.SH.IntervalIndices.amount")[:,tidx]
        SH000906_close = reader("000906.SH.IntervalIndices.close")[:,tidx]
        SH000906_amount = reader("000906.SH.IntervalIndices.amount")[:,tidx]

        SH000001_leading_return = (SH000001_close[1:] - SH000001_close[:-1]) / (SH000001_close[:-1] + 1e-8)
        SZ399001_leading_return = (SZ399001_close[1:] - SZ399001_close[:-1]) / (SZ399001_close[:-1] + 1e-8)
        SH000300_leading_return = (SH000300_close[1:] - SH000300_close[:-1]) / (SH000300_close[:-1] + 1e-8)
        SH000905_leading_return = (SH000905_close[1:] - SH000905_close[:-1]) / (SH000905_close[:-1] + 1e-8)
        SH000906_leading_return = (SH000906_close[1:] - SH000906_close[:-1]) / (SH000906_close[:-1] + 1e-8)

        features_dic["feature_SH000001_leading_return"] = SH000001_leading_return[buffer-1:]
        features_dic["feature_SZ399001_leading_return"] = SZ399001_leading_return[buffer-1:]
        features_dic["feature_SH000300_leading_return"] = SH000300_leading_return[buffer-1:]
        features_dic["feature_SH000905_leading_return"] = SH000905_leading_return[buffer-1:]
        features_dic["feature_SH000906_leading_return"] = SH000906_leading_return[buffer-1:]

        features_dic["feature_SH000001_leading_return_mean_5"] = RollingStatistics(SH000001_leading_return).mean(5)[buffer-1:]
        features_dic["feature_SZ399001_leading_return_mean_5"] = RollingStatistics(SZ399001_leading_return).mean(5)[buffer-1:]
        features_dic["feature_SH000300_leading_return_mean_5"] = RollingStatistics(SH000300_leading_return).mean(5)[buffer-1:]
        features_dic["feature_SH000905_leading_return_mean_5"] = RollingStatistics(SH000905_leading_return).mean(5)[buffer-1:]
        features_dic["feature_SH000906_leading_return_mean_5"] = RollingStatistics(SH000906_leading_return).mean(5)[buffer-1:]

        features_dic["feature_SH000001_leading_return_mean_10"] = RollingStatistics(SH000001_leading_return).mean(10)[buffer-1:]
        features_dic["feature_SZ399001_leading_return_mean_10"] = RollingStatistics(SZ399001_leading_return).mean(10)[buffer-1:]
        features_dic["feature_SH000300_leading_return_mean_10"] = RollingStatistics(SH000300_leading_return).mean(10)[buffer-1:]
        features_dic["feature_SH000905_leading_return_mean_10"] = RollingStatistics(SH000905_leading_return).mean(10)[buffer-1:]
        features_dic["feature_SH000906_leading_return_mean_10"] = RollingStatistics(SH000906_leading_return).mean(10)[buffer-1:]

        features_dic["feature_SH000001_leading_return_mean_20"] = RollingStatistics(SH000001_leading_return).mean(20)[buffer-1:]
        features_dic["feature_SZ399001_leading_return_mean_20"] = RollingStatistics(SZ399001_leading_return).mean(20)[buffer-1:]
        features_dic["feature_SH000300_leading_return_mean_20"] = RollingStatistics(SH000300_leading_return).mean(20)[buffer-1:]
        features_dic["feature_SH000905_leading_return_mean_20"] = RollingStatistics(SH000905_leading_return).mean(20)[buffer-1:]
        features_dic["feature_SH000906_leading_return_mean_20"] = RollingStatistics(SH000906_leading_return).mean(20)[buffer-1:]
        
        features_dic["feature_SH000001_leading_return_mean_30"] = RollingStatistics(SH000001_leading_return).mean(30)[buffer-1:]
        features_dic["feature_SZ399001_leading_return_mean_30"] = RollingStatistics(SZ399001_leading_return).mean(30)[buffer-1:]
        features_dic["feature_SH000300_leading_return_mean_30"] = RollingStatistics(SH000300_leading_return).mean(30)[buffer-1:]
        features_dic["feature_SH000905_leading_return_mean_30"] = RollingStatistics(SH000905_leading_return).mean(30)[buffer-1:]
        features_dic["feature_SH000906_leading_return_mean_30"] = RollingStatistics(SH000906_leading_return).mean(30)[buffer-1:]
        
        features_dic["feature_SH000001_leading_return_mean_60"] = RollingStatistics(SH000001_leading_return).mean(60)[buffer-1:]
        features_dic["feature_SZ399001_leading_return_mean_60"] = RollingStatistics(SZ399001_leading_return).mean(60)[buffer-1:]
        features_dic["feature_SH000300_leading_return_mean_60"] = RollingStatistics(SH000300_leading_return).mean(60)[buffer-1:]
        features_dic["feature_SH000905_leading_return_mean_60"] = RollingStatistics(SH000905_leading_return).mean(60)[buffer-1:]
        features_dic["feature_SH000906_leading_return_mean_60"] = RollingStatistics(SH000906_leading_return).mean(60)[buffer-1:]

        features_dic["feature_SH000001_leading_return_std_5"] = RollingStatistics(SH000001_leading_return).std(5)[buffer-1:]
        features_dic["feature_SZ399001_leading_return_std_5"] = RollingStatistics(SZ399001_leading_return).std(5)[buffer-1:]
        features_dic["feature_SH000300_leading_return_std_5"] = RollingStatistics(SH000300_leading_return).std(5)[buffer-1:]
        features_dic["feature_SH000905_leading_return_std_5"] = RollingStatistics(SH000905_leading_return).std(5)[buffer-1:]
        features_dic["feature_SH000906_leading_return_std_5"] = RollingStatistics(SH000906_leading_return).std(5)[buffer-1:]

        features_dic["feature_SH000001_leading_return_std_10"] = RollingStatistics(SH000001_leading_return).std(10)[buffer-1:]
        features_dic["feature_SZ399001_leading_return_std_10"] = RollingStatistics(SZ399001_leading_return).std(10)[buffer-1:]
        features_dic["feature_SH000300_leading_return_std_10"] = RollingStatistics(SH000300_leading_return).std(10)[buffer-1:]
        features_dic["feature_SH000905_leading_return_std_10"] = RollingStatistics(SH000905_leading_return).std(10)[buffer-1:]
        features_dic["feature_SH000906_leading_return_std_10"] = RollingStatistics(SH000906_leading_return).std(10)[buffer-1:]
        
        features_dic["feature_SH000001_leading_return_std_20"] = RollingStatistics(SH000001_leading_return).std(20)[buffer-1:]
        features_dic["feature_SZ399001_leading_return_std_20"] = RollingStatistics(SZ399001_leading_return).std(20)[buffer-1:]
        features_dic["feature_SH000300_leading_return_std_20"] = RollingStatistics(SH000300_leading_return).std(20)[buffer-1:]
        features_dic["feature_SH000905_leading_return_std_20"] = RollingStatistics(SH000905_leading_return).std(20)[buffer-1:]
        features_dic["feature_SH000906_leading_return_std_20"] = RollingStatistics(SH000906_leading_return).std(20)[buffer-1:]        
        
        features_dic["feature_SH000001_leading_return_std_30"] = RollingStatistics(SH000001_leading_return).std(30)[buffer-1:]
        features_dic["feature_SZ399001_leading_return_std_30"] = RollingStatistics(SZ399001_leading_return).std(30)[buffer-1:]
        features_dic["feature_SH000300_leading_return_std_30"] = RollingStatistics(SH000300_leading_return).std(30)[buffer-1:]
        features_dic["feature_SH000905_leading_return_std_30"] = RollingStatistics(SH000905_leading_return).std(30)[buffer-1:]
        features_dic["feature_SH000906_leading_return_std_30"] = RollingStatistics(SH000906_leading_return).std(30)[buffer-1:]       
        
        features_dic["feature_SH000001_leading_return_std_60"] = RollingStatistics(SH000001_leading_return).std(60)[buffer-1:]
        features_dic["feature_SZ399001_leading_return_std_60"] = RollingStatistics(SZ399001_leading_return).std(60)[buffer-1:]
        features_dic["feature_SH000300_leading_return_std_60"] = RollingStatistics(SH000300_leading_return).std(60)[buffer-1:]
        features_dic["feature_SH000905_leading_return_std_60"] = RollingStatistics(SH000905_leading_return).std(60)[buffer-1:]
        features_dic["feature_SH000906_leading_return_std_60"] = RollingStatistics(SH000906_leading_return).std(60)[buffer-1:]

        features_dic["feature_SH000001_amount_mean_pct_5"] = (RollingStatistics(SH000001_amount).mean(5) / (SH000001_amount+1e-8))[buffer:]
        features_dic["feature_SZ399001_amount_mean_pct_5"] = (RollingStatistics(SZ399001_amount).mean(5) / (SZ399001_amount+1e-8))[buffer:]
        features_dic["feature_SH000300_amount_mean_pct_5"] = (RollingStatistics(SH000300_amount).mean(5) / (SH000300_amount+1e-8))[buffer:]
        features_dic["feature_SH000905_amount_mean_pct_5"] = (RollingStatistics(SH000905_amount).mean(5) / (SH000905_amount+1e-8))[buffer:]
        features_dic["feature_SH000906_amount_mean_pct_5"] = (RollingStatistics(SH000906_amount).mean(5) / (SH000906_amount+1e-8))[buffer:]

        features_dic["feature_SH000001_amount_mean_pct_10"] = (RollingStatistics(SH000001_amount).mean(10) / (SH000001_amount+1e-8))[buffer:]
        features_dic["feature_SZ399001_amount_mean_pct_10"] = (RollingStatistics(SZ399001_amount).mean(10) / (SZ399001_amount+1e-8))[buffer:]
        features_dic["feature_SH000300_amount_mean_pct_10"] = (RollingStatistics(SH000300_amount).mean(10) / (SH000300_amount+1e-8))[buffer:]
        features_dic["feature_SH000905_amount_mean_pct_10"] = (RollingStatistics(SH000905_amount).mean(10) / (SH000905_amount+1e-8))[buffer:]
        features_dic["feature_SH000906_amount_mean_pct_10"] = (RollingStatistics(SH000906_amount).mean(10) / (SH000906_amount+1e-8))[buffer:]

        features_dic["feature_SH000001_amount_mean_pct_20"] = (RollingStatistics(SH000001_amount).mean(20) / (SH000001_amount+1e-8))[buffer:]
        features_dic["feature_SZ399001_amount_mean_pct_20"] = (RollingStatistics(SZ399001_amount).mean(20) / (SZ399001_amount+1e-8))[buffer:]
        features_dic["feature_SH000300_amount_mean_pct_20"] = (RollingStatistics(SH000300_amount).mean(20) / (SH000300_amount+1e-8))[buffer:]
        features_dic["feature_SH000905_amount_mean_pct_20"] = (RollingStatistics(SH000905_amount).mean(20) / (SH000905_amount+1e-8))[buffer:]
        features_dic["feature_SH000906_amount_mean_pct_20"] = (RollingStatistics(SH000906_amount).mean(20) / (SH000906_amount+1e-8))[buffer:]

        features_dic["feature_SH000001_amount_mean_pct_30"] = (RollingStatistics(SH000001_amount).mean(30) / (SH000001_amount+1e-8))[buffer:]
        features_dic["feature_SZ399001_amount_mean_pct_30"] = (RollingStatistics(SZ399001_amount).mean(30) / (SZ399001_amount+1e-8))[buffer:]
        features_dic["feature_SH000300_amount_mean_pct_30"] = (RollingStatistics(SH000300_amount).mean(30) / (SH000300_amount+1e-8))[buffer:]
        features_dic["feature_SH000905_amount_mean_pct_30"] = (RollingStatistics(SH000905_amount).mean(30) / (SH000905_amount+1e-8))[buffer:]
        features_dic["feature_SH000906_amount_mean_pct_30"] = (RollingStatistics(SH000906_amount).mean(30) / (SH000906_amount+1e-8))[buffer:]

        features_dic["feature_SH000001_amount_mean_pct_60"] = (RollingStatistics(SH000001_amount).mean(60) / (SH000001_amount+1e-8))[buffer:]
        features_dic["feature_SZ399001_amount_mean_pct_60"] = (RollingStatistics(SZ399001_amount).mean(60) / (SZ399001_amount+1e-8))[buffer:]
        features_dic["feature_SH000300_amount_mean_pct_60"] = (RollingStatistics(SH000300_amount).mean(60) / (SH000300_amount+1e-8))[buffer:]
        features_dic["feature_SH000905_amount_mean_pct_60"] = (RollingStatistics(SH000905_amount).mean(60) / (SH000905_amount+1e-8))[buffer:]
        features_dic["feature_SH000906_amount_mean_pct_60"] = (RollingStatistics(SH000906_amount).mean(60) / (SH000906_amount+1e-8))[buffer:]

        features_dic["feature_SH000001_amount_std_pct_5"] = (RollingStatistics(SH000001_amount).std(5) / (SH000001_amount+1e-8))[buffer:]
        features_dic["feature_SZ399001_amount_std_pct_5"] = (RollingStatistics(SZ399001_amount).std(5) / (SZ399001_amount+1e-8))[buffer:]
        features_dic["feature_SH000300_amount_std_pct_5"] = (RollingStatistics(SH000300_amount).std(5) / (SH000300_amount+1e-8))[buffer:]
        features_dic["feature_SH000905_amount_std_pct_5"] = (RollingStatistics(SH000905_amount).std(5) / (SH000905_amount+1e-8))[buffer:]
        features_dic["feature_SH000906_amount_std_pct_5"] = (RollingStatistics(SH000906_amount).std(5) / (SH000906_amount+1e-8))[buffer:]

        features_dic["feature_SH000001_amount_std_pct_10"] = (RollingStatistics(SH000001_amount).std(10) / (SH000001_amount+1e-8))[buffer:]
        features_dic["feature_SZ399001_amount_std_pct_10"] = (RollingStatistics(SZ399001_amount).std(10) / (SZ399001_amount+1e-8))[buffer:]
        features_dic["feature_SH000300_amount_std_pct_10"] = (RollingStatistics(SH000300_amount).std(10) / (SH000300_amount+1e-8))[buffer:]
        features_dic["feature_SH000905_amount_std_pct_10"] = (RollingStatistics(SH000905_amount).std(10) / (SH000905_amount+1e-8))[buffer:]
        features_dic["feature_SH000906_amount_std_pct_10"] = (RollingStatistics(SH000906_amount).std(10) / (SH000906_amount+1e-8))[buffer:]

        features_dic["feature_SH000001_amount_std_pct_20"] = (RollingStatistics(SH000001_amount).std(20) / (SH000001_amount+1e-8))[buffer:]
        features_dic["feature_SZ399001_amount_std_pct_20"] = (RollingStatistics(SZ399001_amount).std(20) / (SZ399001_amount+1e-8))[buffer:]
        features_dic["feature_SH000300_amount_std_pct_20"] = (RollingStatistics(SH000300_amount).std(20) / (SH000300_amount+1e-8))[buffer:]
        features_dic["feature_SH000905_amount_std_pct_20"] = (RollingStatistics(SH000905_amount).std(20) / (SH000905_amount+1e-8))[buffer:]
        features_dic["feature_SH000906_amount_std_pct_20"] = (RollingStatistics(SH000906_amount).std(20) / (SH000906_amount+1e-8))[buffer:]

        features_dic["feature_SH000001_amount_std_pct_30"] = (RollingStatistics(SH000001_amount).std(30) / (SH000001_amount+1e-8))[buffer:]
        features_dic["feature_SZ399001_amount_std_pct_30"] = (RollingStatistics(SZ399001_amount).std(30) / (SZ399001_amount+1e-8))[buffer:]
        features_dic["feature_SH000300_amount_std_pct_30"] = (RollingStatistics(SH000300_amount).std(30) / (SH000300_amount+1e-8))[buffer:]
        features_dic["feature_SH000905_amount_std_pct_30"] = (RollingStatistics(SH000905_amount).std(30) / (SH000905_amount+1e-8))[buffer:]
        features_dic["feature_SH000906_amount_std_pct_30"] = (RollingStatistics(SH000906_amount).std(30) / (SH000906_amount+1e-8))[buffer:]

        features_dic["feature_SH000001_amount_std_pct_60"] = (RollingStatistics(SH000001_amount).std(60) / (SH000001_amount+1e-8))[buffer:]
        features_dic["feature_SZ399001_amount_std_pct_60"] = (RollingStatistics(SZ399001_amount).std(60) / (SZ399001_amount+1e-8))[buffer:]
        features_dic["feature_SH000300_amount_std_pct_60"] = (RollingStatistics(SH000300_amount).std(60) / (SH000300_amount+1e-8))[buffer:]
        features_dic["feature_SH000905_amount_std_pct_60"] = (RollingStatistics(SH000905_amount).std(60) / (SH000905_amount+1e-8))[buffer:]
        features_dic["feature_SH000906_amount_std_pct_60"] = (RollingStatistics(SH000906_amount).std(60) / (SH000906_amount+1e-8))[buffer:]

        sorted_keys = sorted(features_dic.keys())
        feature_matrix = np.column_stack([features_dic[k] for k in sorted_keys]).transpose(1, 0)
        self.feature_matrix = torch.tensor(feature_matrix).unsqueeze(2).expand(-1, -1, stocks)
        # for i in range(self.feature_matrix.shape[1]):
        #     if cfg.get("data_x", None):
        #         for op in cfg.get('data_x'):
        #             ProcessDataForTrainingIntermediate = create_op_process(cfg.data_x[op])
        #             ProcessDataForTrainingIntermediate.apply(self.feature_matrix[:,i,:].squeeze(), test_stage=True)
        # import ipdb; ipdb.set_trace()
        if "extra_feature" in cfg.training.model_params and cfg.training.model_params["extra_feature"]:
            self.feature_matrix = torch.cat([stock_block, factorExp], dim=0)
        else:
            self.feature_matrix = None

        return


# class Regressor(nn.Module):
#     def __init__(
#         self,
#         input_dim,
#         device = 'cpu',
#     ):
#         super().__init__()
#         self.input_dim = input_dim
#         self.raw_weights = nn.Parameter(torch.zeros(1, input_dim, device = device), requires_grad=True)
#         # self.regression_layer = nn.Linear(input_dim, 1, device=device)
#         self.device = device

#     def forward(self, input):
#         actual_weights = F.softmax(self.raw_weights, dim=-1)
#         return torch.sum(input * actual_weights, dim=-1, keepdim=True)

class Regressor(nn.Module):
    def __init__(
        self,
        input_dim,
        device = 'cpu',
    ):
        super().__init__()
        self.input_dim = input_dim
        self.regression_layer = nn.Linear(input_dim, 1, device=device, bias=False)
        self.device = device
        nn.init.constant_(self.regression_layer.weight, 1.0 / input_dim)

    def forward(self, input):
        return self.regression_layer(input)
