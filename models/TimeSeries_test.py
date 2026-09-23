import numpy as np
from pathlib import Path
from os.path import isfile, join, dirname
from os import listdir
from scipy.stats import rankdata
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import time
from copy import deepcopy
from collections import Counter, defaultdict
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import train_test_split
from multiprocessing.shared_memory import SharedMemory
from lion_pytorch import Lion
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from scipy.stats import skew, kurtosis
from numba import prange, njit
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import random
import math
import shutil


from prometheus.utils.utils import WeightedCorrNp, calc_fcst_weighted_ic, weighted_mse_wrap, weighted_mse_eval_wrap, filter_eligible_data, filter_eligible_data_v2, operations_by_group
from prometheus.utils.torch_utils import select_gpu_with_minimum_memory, early_stopping_func, WarmupLR, DecayingCosineWarmRestarts, SharedMemDataset, SharedMemSeqDataset, SharedMemSeqDatasetRandom
from prometheus.ops import create_op_process
from prometheus.modelpool.basemodel import BaseModel, create_model
from prometheus.utils.registry_factory import TRAINING_REGISTRY, MODEL_REGISTRY
from prometheus.utils.speedup_package import calculate_stock_mean_3d_parallel
from prometheus.utils.pearson import niocorr
from prometheus.modelpool.Adversarial import Adversarial
from prometheus.utils.data_processing import RollingStatistics
from prometheus.riskmodel.barra.factor_dict import *
from prometheus.utils.corr import calc_correlation, calc_rank_correlation
from prometheus.utils.muon import Muon, MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam



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



@TRAINING_REGISTRY.register('TimeSeries_test')
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

        Path.mkdir(Path(self.training_dic["model_path"]) / model_name, exist_ok=True, parents=True)

        self.model_state_path = Path(self.training_dic["model_path"]) / model_name / 'final_model.pt'
        self.model_nd_state_path = Path(self.training_dic["model_path"]) / model_name / 'final_model_nd.pt'
        self.optimizer_path = Path(self.training_dic["model_path"]) / model_name / 'final_optimizer.pt'
        self.dict_path = Path(self.training_dic["model_path"]) / model_name / 'training_dict.pt'
        self.log_path = Path(self.training_dic["model_path"]) / model_name / 'training_log.csv'
        if "finetune" in self.training_dic and self.training_dic["finetune"]:
            self.log_path_finetune = Path(self.training_dic["model_path"]) / model_name / 'finetune_log.csv'
        self.summary_path = Path("/dfs/data/tensorBoard") / cfg.train_basics["model_path_name"] / model_name
        if self.training_dic["eval_mask"]:
            self.model_mask_path = Path(self.training_dic["model_path"]) / model_name / 'final_model_mask.pt'
            self.optimizer_mask_path = Path(self.training_dic["model_path"]) / model_name / 'final_optimizer_mask.pt'
        self.model_most_path = Path(self.training_dic["model_path"]) / model_name / 'final_model_most.pt'
        self.optimizer_most_path = Path(self.training_dic["model_path"]) / model_name / 'final_optimizer_most.pt'

        if "feature_select" in self.training_dic and self.training_dic["feature_select"]:
            self.training_dic["features_delete"] = self.feature_selection(cfg)

        num_workers = max(4, min(16, torch.cuda.device_count() * 4))  # 根据GPU数量动态调整

        print("num_workers: ", num_workers)

        if Path(self.summary_path).exists():
            shutil.rmtree(self.summary_path)
        Path.mkdir(self.summary_path, exist_ok=True, parents=True)

        self.seq_len = self.training_dic["bptt"]
        
        if "large_batch_size" not in self.training_dic:
            self.training_dic["large_batch_size"] = self.training_dic["batch_size"]

        days, _ = self.indices[1].shape
        if "date_weight" in self.training_dic and self.training_dic["date_weight"]:
            date_weight = torch.arange(days)
            date_weight = torch.sigmoid(date_weight-(days-20*self.training_dic["date_weight_month"])) * self.training_dic["date_weight_pow"]
        else:
            date_weight = torch.zeros(days)

        if not "sample_mode" in self.training_dic:
            self.training_dic["sample_mode"] = "random"
        rows, cols = np.where(indices[1])
        if self.training_dic["sample_mode"] == "recent":
            train_days = math.ceil(days * (1 - self.training_dic["eval_size"]))
            train_indices = np.zeros(indices[1].shape).astype(bool)
            pos_holder = np.zeros(indices[1].shape).astype(bool)
            for i in range(train_days):
                train_indices[i] = indices[1][i]
                pos_holder[i] = indices[1][i]
                for i in range(self.seq_len+self.training_dic["horizon"]):
                    pos_holder[train_days + i] = indices[1][i]
            eval_indices = indices[1] & ~pos_holder
            self.train_dataset = SharedMemSeqDatasetRandom(model_name, self.train_input.shape, self.train_input.dtype, indices[0], train_indices, self.train_output, self.seq_len, np.zeros(days), False)
            self.eval_dataset = SharedMemSeqDatasetRandom(model_name, self.train_input.shape, self.train_input.dtype, indices[0], eval_indices, self.train_output, self.seq_len, np.zeros(days), False)
        elif self.training_dic["sample_mode"] == "recent_non_overlapping":
            train_days = math.ceil(days * (1 - self.training_dic["eval_size"]))
            train_indices = np.zeros(indices[1].shape).astype(bool)
            pos_holder = np.zeros(indices[1].shape).astype(bool)
            for i in range(train_days):
                train_indices[i] = indices[1][i]
                pos_holder[i] = indices[1][i]
                for i in range(self.seq_len+self.training_dic["horizon"]):
                    pos_holder[train_days + i] = indices[1][i]
            eval_indices = indices[1] & ~pos_holder
            train_idx = self.find_end_points(train_indices, horizon=self.training_dic["horizon"])
            train_endpoints = np.zeros(indices[1].shape).astype(bool)
            for i,j in train_idx:
                train_endpoints[i][j] = True
            # print(train_endpoints.sum())
            for j in range(indices[1].shape[1]):
                # print(train_endpoints[:,j].sum())
                for i in range(indices[1].shape[0]):
                    if train_endpoints[i][j]:
                        for k in range(i+1, i+self.seq_len//2):
                            train_endpoints[k][j] = False
                        i += self.seq_len
                # print(train_endpoints[:,j].sum())
            self.train_dataset = SharedMemSeqDatasetRandom(model_name, self.train_input.shape, self.train_input.dtype, indices[0], train_indices, self.train_output, self.seq_len, np.zeros(days), True, train_endpoints, self.feature_matrix)
            self.eval_dataset = SharedMemSeqDatasetRandom(model_name, self.train_input.shape, self.train_input.dtype, indices[0], eval_indices, self.train_output, self.seq_len, np.zeros(days), False, None, self.feature_matrix)
        elif self.training_dic["eval_mode"] == "period":
            if "eval_period" not in self.training_dic:
                self.eval_period = 10
            else:
                self.eval_period = self.training_dic["eval_period"]
            period_len = days // self.training_dic["eval_period"]
            eval_days = math.floor(period_len * self.training_dic["eval_size"])
            eval_indices = np.zeros(indices[1].shape).astype(bool)
            pos_holder = np.zeros(indices[1].shape).astype(bool)
            for i in range(self.eval_period):
                for j in range(period_len * i - eval_days, period_len * i):
                    eval_indices[j] = indices[1][j]
                if "avoid_horizon" in self.training_dic and self.training_dic["avoid_horizon"]:
                    for l in range(max(0, period_len*i-eval_days-self.training_dic["horizon"]), min(days, period_len*i+self.training_dic["horizon"])):
                        pos_holder[l] = indices[1][l]
                else:
                    for l in range(period_len * i - eval_days, period_len * i):
                        pos_holder[l] = indices[1][l]
            train_indices = indices[1] & ~pos_holder
            self.train_dataset = SharedMemSeqDataset(model_name, self.train_input.shape, self.train_input.dtype, indices[0], train_indices, self.train_output, self.seq_len)
            self.eval_dataset = SharedMemSeqDataset(model_name, self.train_input.shape, self.train_input.dtype, indices[0], eval_indices, self.train_output, self.seq_len)
        elif self.training_dic["sample_mode"] == "by_date":
            all_dates = np.unique(rows)
            train_dates, eval_dates = train_test_split(range(len(all_dates)), test_size=self.training_dic["eval_size"], random_state=0, shuffle=True)
            train_indices = np.zeros(indices[1].shape).astype(bool)
            row_filter = np.unique(rows)[train_dates]
            for i in range(indices[1].shape[0]):
                if i in row_filter:
                    train_indices[i] = indices[1][i]
            eval_indices = np.zeros(indices[1].shape).astype(bool)
            row_filter = np.unique(rows)[eval_dates]
            for i in range(indices[1].shape[0]):
                if i in row_filter:
                    eval_indices[i] = indices[1][i]
            self.train_dataset = SharedMemSeqDataset(model_name, self.train_input.shape, self.train_input.dtype, indices[0], train_indices, self.train_output, self.seq_len)
            self.eval_dataset = SharedMemSeqDataset(model_name, self.train_input.shape, self.train_input.dtype, indices[0], eval_indices, self.train_output, self.seq_len)
        elif self.training_dic["sample_mode"] == "non-overlapping":
            condition_eval = self.find_end_points(indices[1])
            _, eval_idx = train_test_split(range(len(condition_eval)), test_size=self.training_dic["eval_size"], random_state=0, shuffle=True)
            eval_endpoints = np.zeros(indices[1].shape).astype(bool)
            eval_indices = np.zeros(indices[1].shape).astype(bool)
            for i in eval_idx:
                eval_endpoints[condition_eval[i][0]][condition_eval[i][1]] = True
                for j in range(condition_eval[i][0]-self.seq_len+1, condition_eval[i][0]+1):
                    eval_indices[j][condition_eval[i][1]] = True
            train_idx = self.find_end_points(indices[1] & ~eval_endpoints)
            train_endpoints = np.zeros(indices[1].shape).astype(bool)
            train_indices = np.zeros(indices[1].shape).astype(bool)
            for i,j in train_idx:
                train_endpoints[i][j] = True
                for k in range(i-self.seq_len+1, i+1):
                    train_indices[k][j] = True
            self.train_dataset = SharedMemSeqDatasetRandom(model_name, self.train_input.shape, self.train_input.dtype, indices[0], train_indices, self.train_output, self.seq_len, True, train_endpoints)
            self.eval_dataset = SharedMemSeqDatasetRandom(model_name, self.train_input.shape, self.train_input.dtype, indices[0], eval_indices, self.train_output, self.seq_len, True, eval_endpoints)
        else:
            condition_eval = self.find_end_points(indices[1])
            train_idx, eval_idx = train_test_split(range(len(condition_eval)), test_size=self.training_dic["eval_size"], random_state=0, shuffle=True)
            train_endpoints = np.zeros(indices[1].shape).astype(bool)
            train_indices = np.zeros(indices[1].shape).astype(bool)
            for i in train_idx:
                train_endpoints[condition_eval[i][0]][condition_eval[i][1]] = True
                for j in range(condition_eval[i][0]-self.seq_len+1, condition_eval[i][0]+1):
                    train_indices[j][condition_eval[i][1]] = True
            eval_endpoints = np.zeros(indices[1].shape).astype(bool)
            eval_indices = np.zeros(indices[1].shape).astype(bool)
            for i in eval_idx:
                eval_endpoints[condition_eval[i][0]][condition_eval[i][1]] = True
                for j in range(condition_eval[i][0]-self.seq_len+1, condition_eval[i][0]+1):
                    eval_indices[j][condition_eval[i][1]] = True
            self.train_dataset = SharedMemSeqDatasetRandom(model_name, self.train_input.shape, self.train_input.dtype, indices[0], train_indices, self.train_output, self.seq_len, date_weight, True, train_endpoints)
            self.eval_dataset = SharedMemSeqDatasetRandom(model_name, self.train_input.shape, self.train_input.dtype, indices[0], eval_indices, self.train_output, self.seq_len, date_weight, True, eval_endpoints)

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.training_dic["large_batch_size"], shuffle=True, drop_last=True, prefetch_factor=4, pin_memory=True, num_workers=num_workers, persistent_workers=True)
        self.eval_loader = DataLoader(self.eval_dataset, batch_size=self.training_dic["large_batch_size"]*2, shuffle=False, drop_last=True, prefetch_factor=4, pin_memory=True, num_workers=num_workers, persistent_workers=True)


        if ("discrimination" in self.training_dic and self.training_dic["discrimination"]) or ("finetune" in self.training_dic and self.training_dic["finetune"]):
            # recent_indices = (np.arange(days) > (days - (20 * self.training_dic["finetune_month"]))).reshape(-1,1)
            # recent_indices = np.repeat(recent_indices, stocks, axis=1)

            finetune_indices = np.zeros(indices[1].shape).astype(bool)
            for i in range((days - (20 * self.training_dic["finetune_month"])), days):
                finetune_indices[i] = indices[1][i]

            condition_eval = self.find_end_points(finetune_indices)
            train_idx, eval_idx = train_test_split(range(len(condition_eval)), test_size=self.training_dic["finetune_eval_size"], random_state=0, shuffle=True)

            train_endpoints = np.zeros(indices[1].shape).astype(bool)
            train_indices = np.zeros(indices[1].shape).astype(bool)
            for i in train_idx:
                train_endpoints[condition_eval[i][0]][condition_eval[i][1]] = True
                for j in range(condition_eval[i][0]-self.seq_len+1, condition_eval[i][0]+1):
                    train_indices[j][condition_eval[i][1]] = True
            eval_endpoints = np.zeros(indices[1].shape).astype(bool)
            eval_indices = np.zeros(indices[1].shape).astype(bool)
            for i in eval_idx:
                eval_endpoints[condition_eval[i][0]][condition_eval[i][1]] = True
                for j in range(condition_eval[i][0]-self.seq_len+1, condition_eval[i][0]+1):
                    eval_indices[j][condition_eval[i][1]] = True

            finetune_train_dataset = SharedMemSeqDatasetRandom(model_name, self.train_input.shape, self.train_input.dtype, indices[0], train_indices, self.train_output, self.seq_len, np.zeros(days), True, train_endpoints, self.feature_matrix)
            finetune_eval_dataset = SharedMemSeqDatasetRandom(model_name, self.train_input.shape, self.train_input.dtype, indices[0], eval_indices, self.train_output, self.seq_len, np.zeros(days), True, eval_endpoints, self.feature_matrix)

            self.finetune_train_loader = DataLoader(finetune_train_dataset, batch_size=self.training_dic["batch_size"], shuffle=True, drop_last=True, prefetch_factor=4, pin_memory=True, num_workers=num_workers, persistent_workers=True)
            self.finetune_eval_loader = DataLoader(finetune_eval_dataset, batch_size=self.training_dic["batch_size"], shuffle=False, drop_last=True, prefetch_factor=4, pin_memory=True, num_workers=num_workers, persistent_workers=True)


        print(f"len of train_data: {len(self.train_dataset)}, len of eval_data: {len(self.eval_dataset)}, len of train_loader: {len(self.train_loader)}, len of eval_loader: {len(self.eval_loader)}")
        if ("discrimination" in self.training_dic and self.training_dic["discrimination"]) or ("finetune" in self.training_dic and self.training_dic["finetune"]):
            print(f"len of finetune_train_data: {len(finetune_train_dataset)}, len of finetune_eval_data: {len(finetune_eval_dataset)}, len of finetune_train_loader: {len(self.finetune_train_loader)}, len of finetune_eval_loader: {len(self.finetune_eval_loader)}")

        if not "step_num" in self.training_dic:
            self.training_dic["step_num"] = 30
        self.training_dic["eval_steps"] = len(self.train_dataset) // (self.training_dic["step_num"] * self.training_dic["large_batch_size"])

        self.input_dim = np.sum(indices[0])


        self.cfg = deepcopy(cfg)
        if "extra_feature" in self.training_dic and self.training_dic["extra_feature"]:
            self.input_dim += self.feature_matrix.shape[0]
        if cfg.model.type in ["KANGRU", "KANsformer"]:
            self.cfg.model.model_params["input_dim"] = self.input_dim
        if cfg.model.type == "KANGRU":
            hidden_dim = 1
            while hidden_dim < self.input_dim // 2:
                hidden_dim <<= 1
            self.cfg.model.model_params["hidden_dim"] = hidden_dim
        if cfg.model.type == "KANsformer":
            self.cfg.model.model_params["seq_len"] = self.seq_len
            embed_dim = 1
            while embed_dim < self.input_dim // 4:
                embed_dim <<= 1
            # embed_dim = self.input_dim // 4
            embed_dim //= 2
            if isinstance(self.cfg.model.model_params["embedding_dim"],list):
                self.cfg.model.model_params["embedding_dim"][-1] = embed_dim
            else:
                self.cfg.model.model_params["embedding_dim"] = embed_dim
            self.cfg.model.model_params["expand_dim"] = math.ceil(embed_dim * 1.5)
            self.cfg.model.model_params["head_dim"] = embed_dim // (self.cfg.model.model_params["num_heads"])

        if lock is not None:
            with lock:
                time.sleep(random.randint(1,30))
                super().gpu_helper(*args, **kwargs)
        self.cfg.model.model_params["device"] = self.device

        self.model = create_model(self.cfg.model).to(self.device)

        torch.save(self.cfg, self.dict_path)

        train_y = train_output[indices[1]]

        var = np.var(train_y)
        std = np.std(train_y)
        mean = np.mean(train_y)
        skewness = skew(train_y)
        kurt = kurtosis(train_y)

        print(f"target stats| Mean: {mean.item()}| Variance: {var.item()}| Skewness: {skewness}| Kurtosis: {kurt}")

        pbar = tqdm(range(self.training_dic["epochs"]), desc="Training", ncols=180)

        if "tau" not in self.training_dic:
            self.training_dic["tau"] = 0.5

        if self.training_dic["loss_fn"] == "CCC":
            loss_fn = loss_fn_eval = lambda x, y, w: self.ccc(x,y,w)
        elif self.training_dic["loss_fn"] == "Correlation":
            loss_fn = loss_fn_eval = lambda x, y, w: self.correlation_loss(x,y,w)
        elif self.training_dic["loss_fn"] == "RankCorrelation":
            loss_fn = loss_fn_eval = lambda x, y, w: calc_rank_correlation(x,y,self.training_dic["tau"],w)
        else:
            raise ValueError("loss_fn must be CCC, Correlation, or RankCorrelation.")
        if self.training_dic["reg_loss"] == "Huber":
            reg_fn = nn.HuberLoss(reduction="none", delta=self.training_dic["huber_delta"])
        else:
            reg_fn = nn.MSELoss(reduction="none")

        lr = self.training_dic["lr"]
        self.weight_decay = self.training_dic["weight_decay"]
        self.correlation_ratio = self.training_dic["correlation_ratio"]
        
        if "eps" not in self.training_dic:
            self.eps = 1e-4
        else:
            self.eps = self.training_dic["eps"]
            
        if "clip_norm" not in self.training_dic:
            self.clip_norm = 5
        else:
            self.clip_norm = self.training_dic["clip_norm"]

        if "threshold_ratio" not in self.training_dic:
            self.threshold_ratio = 5e2
        else:
            self.threshold_ratio = self.training_dic["threshold_ratio"]

        if self.training_dic['opt'] == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=self.weight_decay)
        elif self.training_dic['opt'] == "AdamW":
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=self.weight_decay)
        elif self.training_dic['opt'] == "RMSprop":
            self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=lr, weight_decay=self.weight_decay)
        elif self.training_dic["opt"] == "Lion":
            self.optimizer = Lion(self.model.parameters(), lr=lr/10, weight_decay=self.weight_decay)
        elif self.training_dic["opt"] == "Muon":
            muon_params = [p for p in self.model.parameters() if p.ndim >= 2]
            adamw_params = [p for p in self.model.parameters() if p.ndim < 2]
            self.optimizer = SingleDeviceMuonWithAuxAdam([dict(params=muon_params, use_muon=True, lr=lr, weight_decay=self.weight_decay), dict(params=adamw_params, use_muon=False, lr=lr/10, weight_decay=self.weight_decay)])
        elif self.training_dic['opt'] == "SGD":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=self.weight_decay)
        elif self.training_dic['opt'] == "LBFGS":
            self.optimizer = torch.optim.LBFGS(self.model.parameters(), lr=lr, history_size=10, tolerance_grad=1e-32, tolerance_change=1e-32)
        else:
            raise ValueError("opt must be Adam, AdamW, SGD, LBFGS, or RMSprop.")

        writer = SummaryWriter(self.summary_path / "run")

        result = {"train_loss": [], "eval_loss": [], "eval_mse_loss": [], "eval_correlation_loss": [], "eval_correlation_mask": [],  "good_points": []}

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, "min", factor=self.training_dic["scheduler_factor"], patience=self.training_dic["scheduler_patience"], threshold=self.training_dic["scheduler_threshold"])


        if "discrimination" in self.training_dic and self.training_dic["discrimination"]:
            result["adv_loss"] = []

            self.adversarial = Adversarial(self.cfg, self.device, self.input_dim)

        best_eval_loss = float("inf")
        if self.training_dic["eval_mask"]:
            best_correlation_mask_loss = -float("inf")
        retrain_cnt = 1
        overfit_cnt = 0
        pred_mean = 0
        pred_std = 0
        best_points_cnt = 0
        total_step = 0

        # print(len(self.optimizer.param_groups))

        # if len(self.optimizer.param_groups) > 1:
        #     lambda1 = lambda total_step: total_step * self.training_dic["lr"]/self.training_dic["overfit_threshold"]
        #     lambda2 = lambda total_step: total_step * self.training_dic["lr"]/self.training_dic["overfit_threshold"] / 10
        #     lambda_function = [lambda1, lambda2]
        # else:
        #     lambda1 = lambda total_step: total_step * self.training_dic["lr"]/self.training_dic["overfit_threshold"]
        #     lambda_function = [lambda1]

        # warmup = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda = lambda_function)

        for epoch in pbar:
            if "discrimination" in self.training_dic and self.training_dic["discrimination"] and self.training_dic["batch_size"] != cfg.dis_train.model_params["dis_target_size"]:
                assert cfg.dis_train.model_params["dis_target_size"] % self.training_dic["batch_size"] == 0, f"Target size in discrimination must be divisible by batch size"
                dis_batch_cnt = cfg.dis_train.model_params["dis_target_size"] // self.training_dic["large_batch_size"]
                dis_cur_cnt = 0
                dis_pred = []
                dis_data = []
                dis_target = []
            epoch_training_loss = 0
            batch_num = 0
            step_cnt = 0
            adv_loss = 0    
            self.model.train()
            for data in self.train_loader:
                if "extra_feature" in self.training_dic and self.training_dic["extra_feature"]:
                    data[-1] = data[-1].transpose(2,1)
                    # print(data[0].shape, data[-1].shape)
                    data_batch = torch.cat([data[0], data[-1]], dim=-1).to(self.device, non_blocking=True, dtype=data[0].dtype)
                else:
                    data_batch = data[0].to(self.device, non_blocking=True)
                target_batch = data[1]
                if "only_target" in self.training_dic and self.training_dic["only_target"]:
                    target_batch = target_batch[:,-1].to(self.device, non_blocking=True)
                    weight = 1 + data[2][:,-1].to(self.device, non_blocking=True)
                else:
                    target_batch = target_batch.to(self.device, non_blocking=True)
                    weight = 1 + data[2].to(self.device, non_blocking=True)
                with torch.no_grad():
                    pred = self.model.forward(data_batch)
                pred = pred.detach().requires_grad_(True)
                if "only_target" in self.training_dic and self.training_dic["only_target"]:
                    pred_used = pred[:,-1]
                else:
                    pred_used = pred
                if self.training_dic["weight_essential"]:
                    if "weight_pos_up" in self.training_dic:
                        upper_adjust_t = F.sigmoid(8 * (mean + self.training_dic["weight_pos_up"] * std - target_batch))
                        upper_adjust_p = F.sigmoid(8 * (pred_mean + self.training_dic["weight_pos_up"] * pred_std - pred))
                    else:
                        upper_adjust_t = upper_adjust_p = 1
                    weight += torch.max(F.sigmoid(8 * (target_batch - mean - self.training_dic["weight_pos"] * std)) * upper_adjust_t, F.sigmoid(8 * (pred_used - pred_mean - self.training_dic["weight_pos"] * pred_std)) * upper_adjust_p) * self.training_dic["weight_pow_over"]
                if self.training_dic["loss_fn"] in ["CCC", "Correlation", "RankCorrelation"]:
                    correlation_loss = loss_fn(pred_used, target_batch, weight)
                    reg_loss = reg_fn(pred_used, target_batch)
                    if self.training_dic["weight_essential"]:
                        reg_loss = (reg_loss * weight)
                    reg_loss = reg_loss.mean()
                    train_loss = - self.correlation_ratio * correlation_loss + (1 - self.correlation_ratio) * reg_loss
                else:
                    reg_loss = reg_fn(pred_used, target_batch)
                    if self.training_dic["weight_essential"]:
                        reg_loss = (reg_loss * weight)
                    reg_loss = torch.mean(reg_loss)
                    train_loss = reg_loss
                if self.training_dic["enable_l1"]:
                    l1_reg = self.training_dic["l1_ratio"] * self.l1_regularization()
                    train_loss += l1_reg
                self.optimizer.zero_grad()
                train_loss.backward()
                intermediate_grad = pred.grad.detach()
                step_cnt += 1
                for i in range(self.training_dic["large_batch_size"] // self.training_dic["batch_size"]):
                    grad_chunk = intermediate_grad[self.training_dic["batch_size"]*i: self.training_dic["batch_size"]*(i+1)]
                    y_chunk = torch.squeeze(self.model.forward(data_batch[self.training_dic["batch_size"]*i: self.training_dic["batch_size"]*(i+1)]))
                    y_chunk.backward(gradient=grad_chunk)
                # import ipdb; ipdb.set_trace()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_norm)
                self.optimizer.step()
                pred.requires_grad_(False)
                if "discrimination" in self.training_dic and self.training_dic["discrimination"]:
                    if self.training_dic["large_batch_size"] != cfg.dis_train.model_params["dis_target_size"]:
                        if "only_target" not in self.training_dic or not self.training_dic["only_target"]:
                            pred = pred[:,-1]
                            target_batch = target_batch[:,-1]
                        else:
                            pred = pred[:,-1]
                        if dis_cur_cnt < dis_batch_cnt:
                            dis_pred.append(pred)
                            dis_data.append(data_batch[:,-1,:])
                            dis_target.append(target_batch)
                            dis_cur_cnt += 1
                        else:
                            dis_pred = torch.cat(dis_pred, dim=0)
                            dis_data = torch.cat(dis_data, dim=0)
                            dis_target = torch.cat(dis_target, dim=0)
                            # print(dis_pred.shape, dis_data.shape, dis_target.shape)
                            loss_dis = self.adversarial.train_with_model(dis_pred, dis_data, dis_target)
                            adv_loss += loss_dis
                            # del dis_pred, dis_data, dis_target
                            # gc.collect()
                            # torch.cuda.empty_cache()
                            dis_pred = []
                            dis_data = []
                            dis_target = []
                            dis_cur_cnt = 0
                    else:
                        if "only_target" not in self.training_dic or not self.training_dic["only_target"]:
                            pred = pred[:,-1]
                            target_batch = target_batch[:,-1]
                        else:
                            pred = pred[:,-1]
                        # print(pred.shape)
                        loss_dis = self.adversarial.train_with_model(pred, data_batch[:,-1,:], target_batch)
                        adv_loss += loss_dis
                epoch_training_loss += train_loss.item()
                batch_num += 1

                if step_cnt == self.training_dic["eval_steps"]:
                    
                    step_cnt = 0
                    total_step += 1

                    result["train_loss"].append(epoch_training_loss / batch_num)

                    if "discrimination" in self.training_dic and self.training_dic["discrimination"]:
                        result["adv_loss"].append(adv_loss/batch_num)

                    eval_loss = 0
                    mse_loss = 0
                    correlation_loss = 0
                    correlation_mask = 0

                    self.model.eval()
                    total_pred = []
                    total_target = []

                    for name, param in self.model.named_parameters():
                        total_norm = 0
                        if param.grad is not None and "weight" in name:
                            param_norm = param.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                            total_norm = total_norm ** (1. / 2)
                            writer.add_scalar(f"Model/{name}_grad", total_norm, total_step)
                        # writer.add_histogram(f"{name}_grad", param.grad, total_step)

                    with torch.no_grad():
                        start_time = time.time()
                        for data in self.eval_loader:
                            if "extra_feature" in self.training_dic and self.training_dic["extra_feature"]:
                                data[-1] = data[-1].transpose(2,1)
                                data_batch = torch.cat([data[0], data[-1]], dim=-1).to(self.device, non_blocking=True, dtype=data[0].dtype)
                            else:
                                data_batch = data[0].to(self.device, non_blocking=True)
                            target_batch = data[1][:,-1].to(self.device, non_blocking=True)
                            weight = 1 + data[2][:,-1].to(self.device, non_blocking=True)
                            pred = self.model.forward(data_batch, infer=True)[:,-1]
                            new_mse_loss = F.mse_loss(pred, target_batch).item()
                            mse_loss += new_mse_loss
                            if self.training_dic["weight_essential"]:
                                if "mask_pos_up" in self.training_dic:
                                    upper_adjust_p = F.sigmoid(8 * (pred_mean + self.training_dic["mask_pos_up"] * pred_std - pred))
                                else:
                                    upper_adjust_p = 1
                                weight += F.sigmoid(8 * (pred - pred_mean - self.training_dic["mask_pos"] * pred_std)) * upper_adjust_p * self.training_dic["weight_pow_under"]
                            if self.training_dic["loss_fn"] in ["CCC", "Correlation", "RankCorrelation"]:
                                new_correlation_loss = loss_fn_eval(pred, target_batch, weight).item()
                                correlation_loss += new_correlation_loss
                                eval_loss += -self.correlation_ratio * new_correlation_loss + (1-self.correlation_ratio) * new_mse_loss
                            else:
                                eval_loss += new_mse_loss
                                correlation_loss += self.correlation_loss(pred, target_batch).item()
                            total_pred.append(pred.squeeze())
                            total_target.append(target_batch.squeeze())
                    eval_loss /= len(self.eval_loader)
                    mse_loss /= len(self.eval_loader)
                    correlation_loss /= len(self.eval_loader)

                    total_pred = torch.cat(total_pred, dim=0)
                    total_target = torch.cat(total_target, dim=0)

                    pred_mean = torch.mean(total_pred)
                    pred_std = torch.std(total_pred)

                    if self.training_dic["eval_mask"]:
                        mask = ((total_pred > (pred_mean + self.training_dic["mask_pos"] * pred_std)) & (total_target < (mean + self.training_dic["mask_pos"] * std)))
                        total_pred_mask = total_pred[mask]
                        total_target_mask = total_target[mask]

                        if total_pred_mask.shape[0] == 0:
                            correlation_mask = 0
                        else:
                            correlation_mask = (total_pred_mask - pred_mean - self.training_dic["mask_pos"] * pred_std).pow(2) * (total_target_mask - mean - self.training_dic["mask_pos"] * std) / pred_std.pow(2)
                            correlation_mask = torch.nan_to_num(torch.sum(correlation_mask), nan=-1e5, neginf=-1e5).item()
                    result["good_points"].append(torch.sum((total_pred > pred_mean + self.training_dic["mask_pos"] * pred_std)&(total_target > mean + self.training_dic["mask_pos"] * std)).item())
                    
                    if not result["eval_loss"] or (mse_loss - 10 * correlation_loss < best_eval_loss):
                        # best_correlation_loss = correlation_loss
                        # if eval_loss < best_eval_loss:
                        best_eval_loss = mse_loss - 10 * correlation_loss
                        torch.save(self.model.to("cpu").state_dict(), self.model_state_path)
                        torch.save(self.optimizer.state_dict(), self.optimizer_path)
                        self.model.to(self.device)
                    
                    if result["eval_loss"] and mse_loss - 10 * correlation_loss > min(result["eval_loss"]) -self.eps and total_step >= self.training_dic["overfit_threshold"]:
                        overfit_cnt += 1
                    else:
                        overfit_cnt = 0

                    result["eval_loss"].append(mse_loss - 10 * correlation_loss)

                    pbar.set_description("Epoch :%d|Train Loss: %.2e|Evaluation Loss: %.2e|MSE Loss: %.2e|Correlation: %.2e|LR: %.2e|Retrain_LR: %.2e|%.2e" % (epoch+1, result["train_loss"][-1], eval_loss, mse_loss, correlation_loss, self.optimizer.param_groups[0]["lr"], self.training_dic["retrain_lr"], time.time()-start_time))

                    writer.add_scalar("Training_Stats/Train Loss", train_loss, total_step)
                    writer.add_scalar("Training_Stats/Evaluation Loss", eval_loss, total_step)
                    writer.add_scalar("Training_Stats/MSE Loss", mse_loss, total_step)
                    writer.add_scalar("Training_Stats/Correlation", correlation_loss, total_step)
                    writer.add_scalar("Training_Stats/Learning rate", self.optimizer.param_groups[0]["lr"], total_step)

                    result["eval_mse_loss"].append(mse_loss)
                    result["eval_correlation_loss"].append(correlation_loss)
                    if self.training_dic["eval_mask"]:
                        result["eval_correlation_mask"].append(correlation_mask)


                    if self.training_dic["retrain"] and ((overfit_cnt >= self.training_dic["overfit_patience"]) or ((self.optimizer.param_groups[0]["lr"] <= self.training_dic["retrain_lr"]/self.threshold_ratio))):
                        if self.optimizer.param_groups[0]["lr"] <= self.training_dic["retrain_lr"]/self.threshold_ratio:
                            self.training_dic["retrain_lr"] *= self.training_dic["retrain_lr_factor"]
                        overfit_cnt = 0
                        skip_reset = False
                        if self.optimizer.param_groups[0]["lr"]/2 >= self.training_dic["retrain_lr"]:
                            self.training_dic["retrain_lr"] = self.optimizer.param_groups[0]["lr"]/2
                            skip_reset = True
                        self.model.load_state_dict(torch.load(self.model_state_path, map_location=self.device))
                        self.optimizer.load_state_dict(torch.load(self.optimizer_path))
                        for i in range(len(self.optimizer.param_groups)):
                            self.optimizer.param_groups[i]["lr"] = self.training_dic["retrain_lr"]
                            if i > 0:
                                self.optimizer.param_groups[i]["lr"] /= 10
                        del scheduler
                        gc.collect()
                        torch.cuda.empty_cache()
                        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, "min", factor=self.training_dic["scheduler_factor"], patience=self.training_dic["scheduler_patience"], threshold=self.training_dic["scheduler_threshold"])
                        if not skip_reset:
                            self.training_dic["retrain_lr"] = max(self.training_dic["retrain_lr"]*self.training_dic["retrain_lr_factor"], self.training_dic["scheduler_threshold"])
                            retrain_cnt = 1
                        else:
                            retrain_cnt = 0
                    else:
                        if self.training_dic["loss_fn"] in ["CCC", "Correlation", "RankCorrelation"]:
                            scheduler.step(mse_loss - 10 * correlation_loss)
                        else:
                            scheduler.step(eval_loss)

                    df = pd.DataFrame(result)
                    df.to_csv(self.log_path, index=False, header = True)

        del self.train_loader
        del self.train_dataset
        del self.model

        gc.collect()
        torch.cuda.empty_cache()

        if "discrimination" in self.training_dic and self.training_dic["discrimination"]:
            self.adversarial.adv_against_model(self.model_state_path, self.optimizer_path, self.finetune_train_loader, self.finetune_eval_loader)
            del self.finetune_train_loader
            del self.finetune_eval_loader
            del finetune_train_dataset
            del finetune_eval_dataset


        pearson_ic, spearman_ic, mse_loss = self.evaluation(self.cfg)

        del self.eval_loader
        del self.eval_dataset

        train_input_buffer.close()

        gc.collect()
        torch.cuda.empty_cache()

        return pearson_ic, spearman_ic, mse_loss


    def find_end_points_horizon_only(self, mask, horizon=1):
        mask_cumsum = mask.cumsum(axis=0)
        mask_upper = mask_cumsum[horizon-1:,:]
        mask_lower = np.pad(mask_cumsum[:-horizon,:], ((1,0),(0,0)), "constant", constant_values=0)
        cum_sum = mask_upper - mask_lower
        condition_mask = list(np.where(cum_sum == horizon))
        condition_mask[0] += horizon - 1
        return np.transpose(condition_mask)

    def find_end_points(self, mask, horizon=1):
        mask_cumsum = mask.cumsum(axis=0)
        mask_upper = mask_cumsum[self.seq_len-1:,:]
        mask_lower = np.pad(mask_cumsum[:-self.seq_len,:], ((1,0),(0,0)), "constant", constant_values=0)
        cum_sum = mask_upper - mask_lower
        condition_mask = list(np.where(cum_sum == self.seq_len))
        condition_mask[0] += self.seq_len - 1
        print(condition_mask[0].shape)
        return np.transpose(condition_mask)

    def fine_tune(self):
        self.model.load_state_dict(torch.load(self.model_state_path))
        self.model.to(self.device)
        pbar_finetune = tqdm(range(self.training_dic["finetune_epoch"]), desc="finetune", ncols=160)

        layer_len = len(self.cfg.model.model_params["layers_hidden"]) - 1
        layer_threshold = layer_len - self.training_dic["finetune_layer"]

        params_to_update = []

        result_finetune = {"train_loss":[], "eval_loss":[], "eval_mse_loss":[], "eval_correlation_loss":[]}

        for name, params in self.model.named_parameters():
            if int(name.split(".")[1]) < layer_threshold:
                params.requires_grad = False
            else:
                params.requires_grad = True
                params_to_update.append(params)

        self.optimizer = torch.optim.AdamW(params_to_update, lr=self.training_dic["lr"]/self.training_dic["finetune_lr_ratio"], weight_decay=self.weight_decay)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, "min", factor=self.training_dic["scheduler_factor"], patience=self.training_dic["finetune_scheduler_patience"], threshold=self.training_dic["scheduler_threshold"])

        finetune_step = len(self.finetune_train_loader) // self.training_dic["finetune_step_num"]

        total_pred = [torch.tensor([]).to(self.device)]
        with torch.no_grad():
            for data_batch, target_batch, weight in self.finetune_train_loader:
                data_batch = data_batch.to(self.device, non_blocking=True)
                pred = torch.squeeze(self.model.forward(data_batch))
                if not total_pred:
                    total_pred = [pred.detach()]
                else:
                    total_pred.append(pred.detach())
        total_pred = torch.cat(total_pred, dim=0)
        pred_mean = torch.mean(total_pred)
        pred_std = torch.std(total_pred)

        mse_loss = 0
        correlation_loss = 0

        
        with torch.no_grad():
            for data_batch, target_batch, weight in self.finetune_train_loader:
                data_batch = data_batch.to(self.device, non_blocking=True)
                target_batch = target_batch[:,-1].to(self.device, non_blocking=True)
                pred = self.model.forward(data_batch)[:,-1]
                weight = torch.ones_like(target_batch)
                new_mse_loss = F.mse_loss(pred, target_batch).item()
                mse_loss += new_mse_loss
                if "finetune_weight_essential" in self.training_dic and self.training_dic["finetune_weight_essential"]:
                    if "mask_pos_up" in self.training_dic:
                        upper_adjust_p = F.sigmoid(8 * (pred_mean + self.training_dic["mask_pos_up"] * pred_std - pred))
                    else:
                        upper_adjust_p = 1
                    weight += F.sigmoid(8 * (pred - pred_mean - self.training_dic["mask_pos"] * pred_std)) * upper_adjust_p * self.training_dic["weight_pow_under"]
                if self.training_dic["loss_fn"] == "CCC" or self.training_dic["loss_fn"] == "Correlation":
                    new_correlation_loss = self.loss_fn_eval(pred, target_batch, weight).item()
                    correlation_loss += new_correlation_loss
                else:
                    correlation_loss += self.correlation_loss(pred, target_batch).item()
        mse_loss /= len(self.finetune_eval_loader)
        correlation_loss /= len(self.finetune_eval_loader)

        best_eval_loss = mse_loss - 5 * correlation_loss
        overfit_cnt = 0

        end_train = False

        for epoch in pbar_finetune:
            if end_train:
                break
            self.model.train()
            current_loss = 0
            step_cnt = 0
            for data_batch, target_batch, date_weight in self.finetune_train_loader:
                step_cnt += 1
                data_batch = data_batch.to(self.device, non_blocking=True)
                target_batch = target_batch[:,-1].to(self.device, non_blocking=True)
                weight = torch.ones_like(target_batch) + date_weight[:,-1].to(self.device, non_blocking=True)
                pred = self.model.forward(data_batch)[:,-1]
                if "finetune_weight_essential" in self.training_dic and self.training_dic["finetune_weight_essential"]:
                    if "weight_pos_up" in self.training_dic:
                        upper_adjust_t = F.sigmoid(8 * (mean + self.training_dic["weight_pos_up"] * std - target_batch))
                        upper_adjust_p = F.sigmoid(8 * (pred_mean + self.training_dic["weight_pos_up"] * pred_std - pred))
                    else:
                        upper_adjust_t = upper_adjust_p = 1
                    weight += torch.max(F.sigmoid(8 * (target_batch - self.mean -self.training_dic["weight_pos"] * self.std)) * upper_adjust_t, F.sigmoid(8 * (pred- pred_mean -self.training_dic["weight_pos"] * pred_std)) * upper_adjust_p) * self.training_dic["weight_pow_over"]
                if self.training_dic["loss_fn"] == "CCC" or self.training_dic["loss_fn"] == "Correlation":
                    correlation_loss = self.loss_fn(pred, target_batch, weight)
                    reg_loss = (self.reg_fn(pred, target_batch) * weight).mean()
                    loss = -self.correlation_ratio * correlation_loss + (1 - self.correlation_ratio) * reg_loss
                else:
                    reg_loss = (self.reg_fn(pred, target_batch) * weight).mean()
                    loss = reg_loss
                current_loss += loss.item()
                if self.training_dic["enable_l1"]:
                    loss += self.training_dic["l1_ratio"] * self.l1_regularization()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params_to_update, max_norm=self.clip_norm)
                self.optimizer.step()

                if not step_cnt % finetune_step:
                    eval_loss = 0
                    mse_loss = 0
                    correlation_loss = 0

                    train_loss = current_loss / step_cnt
                    step_cnt = 0

                    self.model.eval()

                    total_pred = [torch.tensor([]).to(self.device)]
                    # total_target = [torch.tensor([]).to(self.device)]

                    with torch.no_grad():
                        for data_batch, target_batch, date_weight in self.finetune_eval_loader:
                            data_batch = data_batch.to(self.device, non_blocking=True)
                            target_batch = target_batch[:,-1].to(self.device, non_blocking=True)
                            weight = torch.ones_like(target_batch) + date_weight[:,-1].to(self.device, non_blocking=True)
                            pred = self.model.forward(data_batch)[:,-1]
                            new_mse_loss = F.mse_loss(pred, target_batch).item()
                            mse_loss += new_mse_loss
                            if "finetune_weight_essential" in self.training_dic and self.training_dic["finetune_weight_essential"]:
                                if "mask_pos_up" in self.training_dic:
                                    upper_adjust_p = F.sigmoid(8 * (pred_mean + self.training_dic["mask_pos_up"] * pred_std - pred))
                                else:
                                    upper_adjust_p = 1
                                weight += F.sigmoid(8 * (pred - pred_mean - self.training_dic["mask_pos"] * pred_std)) * upper_adjust_p * self.training_dic["weight_pow_under"]
                            if self.training_dic["loss_fn"] == "CCC" or self.training_dic["loss_fn"] == "Correlation":
                                new_correlation_loss = self.loss_fn_eval(pred, target_batch, weight).item()
                                correlation_loss += new_correlation_loss
                                eval_loss += -self.correlation_ratio * new_correlation_loss + (1-self.correlation_ratio) * new_mse_loss
                            else:
                                eval_loss += new_mse_loss
                                correlation_loss += self.correlation_loss(pred, target_batch).item()
                            if not total_pred:
                                total_pred = [pred.detach()]
                                # total_target = [target_batch.detach()]
                            else:
                                total_pred.append(pred.detach())
                                # total_target.append(target_batch.detach())
                            
                        
                        eval_loss /= len(self.finetune_eval_loader)
                        mse_loss /= len(self.finetune_eval_loader)
                        correlation_loss /= len(self.finetune_eval_loader)

                        total_pred = torch.cat(total_pred, dim=0)
                        # total_target = torch.cat(total_target, dim=0)

                        pred_mean = torch.mean(total_pred)
                        pred_std = torch.std(total_pred)

                        if "finetune_earlystop" in self.training_dic and self.training_dic["finetune_earlystop"]:
                            if result_finetune["eval_loss"] and (mse_loss - 5 * correlation_loss > best_eval_loss):
                                overfit_cnt += 1
                            else:
                                overfit_cnt = 0
                            if overfit_cnt >= self.training_dic["finetune_patience"]:
                                end_train = True
                                break

                        if (mse_loss - 5 * correlation_loss < best_eval_loss):
                            best_eval_loss = mse_loss - 5 * correlation_loss
                            torch.save(self.model.to("cpu").state_dict(), self.model_state_path)
                            torch.save(self.optimizer.state_dict(), self.optimizer_path)
                            self.model.to(self.device)
                        
                        
                        result_finetune["train_loss"].append(train_loss)
                        result_finetune["eval_loss"].append(eval_loss)
                        result_finetune["eval_mse_loss"].append(mse_loss)
                        result_finetune["eval_correlation_loss"].append(correlation_loss)

                        if self.training_dic["loss_fn"] == "CCC" or self.training_dic["loss_fn"] == "Correlation":
                            scheduler.step(mse_loss - 5 * correlation_loss)
                        else:
                            scheduler.step(mse_loss)

                        pbar_finetune.set_description("Epoch :%d|Train Loss: %.2e|Evaluation Loss: %.2e|MSE Loss: %.2e|Correlation: %.2e|LR: %.2e" % (epoch+1, train_loss, eval_loss, mse_loss, correlation_loss, self.optimizer.param_groups[0]["lr"]))
                    
                        df = pd.DataFrame(result_finetune)
                        df.to_csv(self.log_path_finetune, index=False, header = True)

    def evaluation(self, cfg):
        # load the best model from the training process
        model_result = create_model(cfg.model)
        model_result.load_state_dict(torch.load(self.model_state_path))
        model_result = model_result.to(self.device)
        model_result.eval()
        total_pred = None
        np_target = None
        with torch.no_grad():
            for data in self.eval_loader:
                if "extra_feature" in self.training_dic and self.training_dic["extra_feature"]:
                    data[-1] = data[-1].transpose(2,1)
                    data_batch = torch.cat([data[0], data[-1]], dim=-1).to(self.device, non_blocking=True, dtype=data[0].dtype)
                else:
                    data_batch = data[0].to(self.device, non_blocking=True)
                target_batch = data[1]
                if self.training_dic["denormalizer"]:
                    batch_mean = torch.unsqueeze(torch.mean(data_batch, dim=1), 1)
                    batch_std = torch.unsqueeze(torch.std(data_batch, dim=1), 1)
                    data_batch = torch.cat([data_batch, batch_mean, batch_std.pow(0.5)], dim=1).to(self.device, non_blocking=True)
                if "horizontal_norm" in self.training_dic and self.training_dic["horizontal_norm"]:
                    data_batch = F.normalize(data_batch, p=2.0, dim=1)
                target_batch = target_batch[:,-1].to(non_blocking=True)
                pred = model_result.forward(data_batch)[:,-1]
                if not total_pred:
                    total_pred = [pred.flatten().to("cpu").detach().numpy()]
                    np_target = [target_batch.flatten().numpy()]
                else:
                    total_pred.append(pred.flatten().to("cpu").detach().numpy())
                    np_target.append(target_batch.flatten().numpy())
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

    def l1_regularization(self):
        l1_loss = sum(param.abs().sum() for param in self.model.parameters())
        return l1_loss

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

    def feature_selection(self, cfg):
        
        features = np.where(self.indices[0])[0]

        selection_res = [True] * np.sum(self.indices[0])

        # Mark the position of the features after dataloader

        alpha_indices = {}

        for pos, idx in enumerate(features):
            alpha_indices[idx] = pos

        spearman_dict = defaultdict(list)
        
        to_delete = set()

        # Pick the features that doesn't correlated to the target or have a low information ratio

        horizon = cfg.training.model_params["horizon"]


        pbar_fs = tqdm(features, desc='feature_selection', ncols=160)

        # for j in pbar_fs:
        #     spearman_dict[j] = niocorr(self.train_input[j][:-horizon,:], self.train_output[:-horizon,:])
        #     # print(f"finished feature {j}")
        #     pbar_fs.set_description(f"feature {j} finished")

        # alpha_score= {}

        with ProcessPoolExecutor() as executor:
                for j in pbar_fs:
                    spearman_dict[j] = executor.submit(niocorr, self.train_input[j][:-horizon,:], self.train_output[:-horizon,:])
                    pbar_fs.set_description(f"feature {j} finished")

        alpha_score= {}
        for i in spearman_dict:
            spearman_dict[i] = spearman_dict[i].result()

        for i in spearman_dict:
            mean = np.abs(np.mean(np.nan_to_num(spearman_dict[i])))
            std = np.std(np.nan_to_num(spearman_dict[i]))
            alpha_score[i] = mean / std
            if mean < 0.01 or alpha_score[i] < 0.3:
                to_delete.add(i)

        # Add more feature selection logics here

        # Eliminate the features that are picked and mark them in selection result to inform the inference module

        for i in to_delete:
            self.indices[0][i] = False
            selection_res[alpha_indices[i]] = False
        
        print(f"{len(to_delete)} alphas deleted due to low information")

        return selection_res
    
    def load(self, model_path, model_name, **kwargs):
        if model_path is None:
            assert self.model_path is not None
            model_path = self.model_path
        self.model_name = model_name
        self.dict_path = Path(model_path) / model_name / "training_dict.pt"
        self.model_state_dict_path = Path(model_path) / model_name / "final_model.pt"
        return 1

    def predict(self, x, cfg, date=None, test_stage=True, lock=None, valid_stock=None, *args, **kwargs):
        
        start_time = time.time()
        ProcessDataForTrainingIntermediate = {}
        if cfg.get("data_x", None):
            for op in cfg.get("data_x"):
                ProcessDataForTrainingIntermediate[op] = create_op_process(cfg.data_x[op])            
            for i in range(x.shape[1]):
                for op in ProcessDataForTrainingIntermediate:
                    ProcessDataForTrainingIntermediate[op].apply(x[:,i,:].squeeze(), test_stage=True)
        print(f"data processing takes {time.time()-start_time}")

        """
        if cfg.model.type == "FastKAN":
            for i in range(len(cfg.model.model_params["layers_hidden"])-1):
                cfg.model.model_params["layers_hidden"][i] *= input_dim
        """

        cfg = torch.load(self.dict_path)
        if not "denormalizer" in cfg.training.model_params:
            cfg.training.model_params["denormalizer"] = False

        if "feature_select" in cfg.training.model_params and cfg.training.model_params["feature_select"]:
            # print(x.shape, len(cfg.training.model_params["features_delete"]))
            x = x[:,:,cfg.training.model_params["features_delete"]]

        if "mode" not in cfg.model.model_params:
            cfg.model.model_params["mode"] = "default"
        self.vertical_norm = cfg.training.model_params["vertical_norm"]
        self.horizontal_norm = "horizontal_norm" in cfg.training.model_params and cfg.training.model_params["horizontal_norm"]
        # cfg.model.model_params["device"] = self.device
        

        if "extra_feature" in cfg.training.model_params and cfg.training.model_params["extra_feature"]:
            self.feature_matrix = ((self.feature_matrix.squeeze().transpose(2,0))[valid_stock]).numpy()
            x = np.concatenate([x, self.feature_matrix], axis=-1)

        x = np.nan_to_num(x)
        
        # print(f"the shape of x is {x.shape}")
        if cfg.training.model_params["denormalizer"]:
            x = np.concatenate([x, np.mean(x, axis=1).reshape(-1, 1), np.power(np.std(x, axis=1), 0.5).reshape(-1, 1)], axis=1)

        if lock is not None:
            with lock:
                time.sleep(10)
                super().gpu_helper(*args, **kwargs)

        start_time = time.time()

        self.batch_size = cfg.training.model_params["batch_size"] * 4

        stocks = x.shape[0]
        steps = math.ceil(stocks / self.batch_size)

        cfg.model.model_params["device"] = self.device
        cfg.model.model_params["infer"] = True
        model = create_model(cfg.model)
        model.load_state_dict(torch.load(self.model_state_dict_path, map_location=self.device))
        model.eval()
        print(f"it takes {time.time()-start_time} to initiate")

        start_time = time.time()
        with torch.no_grad():
            pred = []
            for i in range(steps):
                # print(x[i*self.batch_size:(i+1)*self.batch_size,:,:].shape)
                data_input = torch.from_numpy(x[i*self.batch_size:(i+1)*self.batch_size,:,:]).contiguous().to(torch.float32).to(self.device)
                pred.append((model.forward(data_input)).to("cpu").detach())
        pred = torch.cat(pred, dim=0)
            # if self.vertical_norm:
            #     if cfg.model.type == "KANsformer":
            #         pred = model.forward_infer(F.normalize(torch.from_numpy(x).contiguous(), p=2.0, dim=0).to(torch.float32).to(self.device))
            #     else:
            #         pred = model.forward(F.normalize(torch.from_numpy(x).contiguous(), p=2.0, dim=0).to(torch.float32).to(self.device))
            # elif self.horizontal_norm:
            #     if cfg.model.type == "KANsformer":
            #         pred = model.forward_infer(F.normalize(torch.from_numpy(x).contiguous(), p=2.0, dim=1).to(torch.float32).to(self.device))
            #     else:
            #         pred = model.forward(F.normalize(torch.from_numpy(x).contiguous(), p=2.0, dim=1).to(torch.float32).to(self.device))
            # else:
            #     data_input = torch.from_numpy(x).contiguous().to(torch.float32).to(self.device)
            #     pred = model.forward(data_input)

        print(f"it takes {time.time()-start_time} to predict")
        result = pred[:,-1].numpy()

        del model, pred, data_input
        gc.collect()
        torch.cuda.empty_cache()

        return result

     
    def create_raw_features(self, cfg, model_name, reader):

        self.training_dic = deepcopy(cfg).training.model_params

        tidx = int(model_name.split("_")[-1])

        buffer = cfg.train_basics.get("cache_reader_buffer", 0)

        block4 = reader("block4n")

        # secind = reader("secoindustry")
        # secind_num = secind.max() + 1

        days, stocks = block4.shape[0]-buffer, block4.shape[1]

        stock_block = np.zeros((4, days, stocks))
        
        # stock_block will add one-hot block information into features
        stock_block[0,:,:] = (block4[buffer-1:-1,:] == 0).astype(np.float32)
        stock_block[1,:,:] = (block4[buffer-1:-1,:] == 1).astype(np.float32)
        stock_block[2,:,:] = (block4[buffer-1:-1,:] == 2).astype(np.float32)
        stock_block[3,:,:] = (block4[buffer-1:-1,:] == 3).astype(np.float32)

        stock_block = torch.tensor(stock_block)

        # stock_secind = np.zeros((secind_num+1, days, stocks))
        # for i in range(secind_num):
        #     stock_secind[i, :, :] = (secind[buffer:, :] == i).astype(np.float32)
        # stock_secind[-1, :, :] = (secind[buffer:, :] < 0).astype(np.float32)
        # stock_secind = torch.tensor(stock_secind)

        # load CNE5 factor list (including style, industry and contry)
        self.styleFactorDataNames = [f"CNE5D_RISK.{f}" for f in STYLE_FACTORS["cne5"]]
        self.industryFactorDataNames = INDUSTRY_COV_ORDER["cne5"]
        self.factorList = self.styleFactorDataNames + self.industryFactorDataNames + ["COUNTRY"]

        # load CNE5 factor covariance matrix
        barra_cne5_factor_cov = reader("CNE5D_COV")
        # factorCOV = barra_cne5_factor_cov /240 /10000

        # barra_cne5_spec_cov = reader("CNE5S_RISK.SRISK")
        # specCov = (np.power(barra_cne5_spec_cov, 2) /240 /10000)
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

        factorExp = torch.from_numpy(factorExp[:,buffer-1:-1,:])

        features_dic = {}

        # SH000001_close = reader("000001.SH.IntervalIndices.close")[:,tidx]
        # SH000001_amount = reader("000001.SH.IntervalIndices.amount")[:,tidx]
        # SZ399001_close = reader("399001.SZ.IntervalIndices.close")[:,tidx]
        # SZ399001_amount = reader("399001.SZ.IntervalIndices.amount")[:,tidx]

        # SH000300_close = reader("000300.SH.IntervalIndices.close")[:,tidx]
        # SH000300_amount = reader("000300.SH.IntervalIndices.amount")[:,tidx]
        # SH000905_close = reader("000905.SH.IntervalIndices.close")[:,tidx]
        # SH000905_amount = reader("000905.SH.IntervalIndices.amount")[:,tidx]
        # SH000906_close = reader("000906.SH.IntervalIndices.close")[:,tidx]
        # SH000906_amount = reader("000906.SH.IntervalIndices.amount")[:,tidx]

        # SH000001_leading_return = (SH000001_close[1:] - SH000001_close[:-1]) / (SH000001_close[:-1] + 1e-8)
        # SZ399001_leading_return = (SZ399001_close[1:] - SZ399001_close[:-1]) / (SZ399001_close[:-1] + 1e-8)
        # SH000300_leading_return = (SH000300_close[1:] - SH000300_close[:-1]) / (SH000300_close[:-1] + 1e-8)
        # SH000905_leading_return = (SH000905_close[1:] - SH000905_close[:-1]) / (SH000905_close[:-1] + 1e-8)
        # SH000906_leading_return = (SH000906_close[1:] - SH000906_close[:-1]) / (SH000906_close[:-1] + 1e-8)

        # features_dic["feature_SH000001_leading_return"] = SH000001_leading_return[buffer-1:]
        # features_dic["feature_SZ399001_leading_return"] = SZ399001_leading_return[buffer-1:]
        # features_dic["feature_SH000300_leading_return"] = SH000300_leading_return[buffer-1:]
        # features_dic["feature_SH000905_leading_return"] = SH000905_leading_return[buffer-1:]
        # features_dic["feature_SH000906_leading_return"] = SH000906_leading_return[buffer-1:]

        # features_dic["feature_SH000001_leading_return_mean_5"] = RollingStatistics(SH000001_leading_return).mean(5)[buffer-1:]
        # features_dic["feature_SZ399001_leading_return_mean_5"] = RollingStatistics(SZ399001_leading_return).mean(5)[buffer-1:]
        # features_dic["feature_SH000300_leading_return_mean_5"] = RollingStatistics(SH000300_leading_return).mean(5)[buffer-1:]
        # features_dic["feature_SH000905_leading_return_mean_5"] = RollingStatistics(SH000905_leading_return).mean(5)[buffer-1:]
        # features_dic["feature_SH000906_leading_return_mean_5"] = RollingStatistics(SH000906_leading_return).mean(5)[buffer-1:]

        # features_dic["feature_SH000001_leading_return_mean_10"] = RollingStatistics(SH000001_leading_return).mean(10)[buffer-1:]
        # features_dic["feature_SZ399001_leading_return_mean_10"] = RollingStatistics(SZ399001_leading_return).mean(10)[buffer-1:]
        # features_dic["feature_SH000300_leading_return_mean_10"] = RollingStatistics(SH000300_leading_return).mean(10)[buffer-1:]
        # features_dic["feature_SH000905_leading_return_mean_10"] = RollingStatistics(SH000905_leading_return).mean(10)[buffer-1:]
        # features_dic["feature_SH000906_leading_return_mean_10"] = RollingStatistics(SH000906_leading_return).mean(10)[buffer-1:]

        # features_dic["feature_SH000001_leading_return_mean_20"] = RollingStatistics(SH000001_leading_return).mean(20)[buffer-1:]
        # features_dic["feature_SZ399001_leading_return_mean_20"] = RollingStatistics(SZ399001_leading_return).mean(20)[buffer-1:]
        # features_dic["feature_SH000300_leading_return_mean_20"] = RollingStatistics(SH000300_leading_return).mean(20)[buffer-1:]
        # features_dic["feature_SH000905_leading_return_mean_20"] = RollingStatistics(SH000905_leading_return).mean(20)[buffer-1:]
        # features_dic["feature_SH000906_leading_return_mean_20"] = RollingStatistics(SH000906_leading_return).mean(20)[buffer-1:]
        
        # features_dic["feature_SH000001_leading_return_mean_30"] = RollingStatistics(SH000001_leading_return).mean(30)[buffer-1:]
        # features_dic["feature_SZ399001_leading_return_mean_30"] = RollingStatistics(SZ399001_leading_return).mean(30)[buffer-1:]
        # features_dic["feature_SH000300_leading_return_mean_30"] = RollingStatistics(SH000300_leading_return).mean(30)[buffer-1:]
        # features_dic["feature_SH000905_leading_return_mean_30"] = RollingStatistics(SH000905_leading_return).mean(30)[buffer-1:]
        # features_dic["feature_SH000906_leading_return_mean_30"] = RollingStatistics(SH000906_leading_return).mean(30)[buffer-1:]
        
        # features_dic["feature_SH000001_leading_return_mean_60"] = RollingStatistics(SH000001_leading_return).mean(60)[buffer-1:]
        # features_dic["feature_SZ399001_leading_return_mean_60"] = RollingStatistics(SZ399001_leading_return).mean(60)[buffer-1:]
        # features_dic["feature_SH000300_leading_return_mean_60"] = RollingStatistics(SH000300_leading_return).mean(60)[buffer-1:]
        # features_dic["feature_SH000905_leading_return_mean_60"] = RollingStatistics(SH000905_leading_return).mean(60)[buffer-1:]
        # features_dic["feature_SH000906_leading_return_mean_60"] = RollingStatistics(SH000906_leading_return).mean(60)[buffer-1:]

        # features_dic["feature_SH000001_leading_return_std_5"] = RollingStatistics(SH000001_leading_return).std(5)[buffer-1:]
        # features_dic["feature_SZ399001_leading_return_std_5"] = RollingStatistics(SZ399001_leading_return).std(5)[buffer-1:]
        # features_dic["feature_SH000300_leading_return_std_5"] = RollingStatistics(SH000300_leading_return).std(5)[buffer-1:]
        # features_dic["feature_SH000905_leading_return_std_5"] = RollingStatistics(SH000905_leading_return).std(5)[buffer-1:]
        # features_dic["feature_SH000906_leading_return_std_5"] = RollingStatistics(SH000906_leading_return).std(5)[buffer-1:]

        # features_dic["feature_SH000001_leading_return_std_10"] = RollingStatistics(SH000001_leading_return).std(10)[buffer-1:]
        # features_dic["feature_SZ399001_leading_return_std_10"] = RollingStatistics(SZ399001_leading_return).std(10)[buffer-1:]
        # features_dic["feature_SH000300_leading_return_std_10"] = RollingStatistics(SH000300_leading_return).std(10)[buffer-1:]
        # features_dic["feature_SH000905_leading_return_std_10"] = RollingStatistics(SH000905_leading_return).std(10)[buffer-1:]
        # features_dic["feature_SH000906_leading_return_std_10"] = RollingStatistics(SH000906_leading_return).std(10)[buffer-1:]
        
        # features_dic["feature_SH000001_leading_return_std_20"] = RollingStatistics(SH000001_leading_return).std(20)[buffer-1:]
        # features_dic["feature_SZ399001_leading_return_std_20"] = RollingStatistics(SZ399001_leading_return).std(20)[buffer-1:]
        # features_dic["feature_SH000300_leading_return_std_20"] = RollingStatistics(SH000300_leading_return).std(20)[buffer-1:]
        # features_dic["feature_SH000905_leading_return_std_20"] = RollingStatistics(SH000905_leading_return).std(20)[buffer-1:]
        # features_dic["feature_SH000906_leading_return_std_20"] = RollingStatistics(SH000906_leading_return).std(20)[buffer-1:]        
        
        # features_dic["feature_SH000001_leading_return_std_30"] = RollingStatistics(SH000001_leading_return).std(30)[buffer-1:]
        # features_dic["feature_SZ399001_leading_return_std_30"] = RollingStatistics(SZ399001_leading_return).std(30)[buffer-1:]
        # features_dic["feature_SH000300_leading_return_std_30"] = RollingStatistics(SH000300_leading_return).std(30)[buffer-1:]
        # features_dic["feature_SH000905_leading_return_std_30"] = RollingStatistics(SH000905_leading_return).std(30)[buffer-1:]
        # features_dic["feature_SH000906_leading_return_std_30"] = RollingStatistics(SH000906_leading_return).std(30)[buffer-1:]       
        
        # features_dic["feature_SH000001_leading_return_std_60"] = RollingStatistics(SH000001_leading_return).std(60)[buffer-1:]
        # features_dic["feature_SZ399001_leading_return_std_60"] = RollingStatistics(SZ399001_leading_return).std(60)[buffer-1:]
        # features_dic["feature_SH000300_leading_return_std_60"] = RollingStatistics(SH000300_leading_return).std(60)[buffer-1:]
        # features_dic["feature_SH000905_leading_return_std_60"] = RollingStatistics(SH000905_leading_return).std(60)[buffer-1:]
        # features_dic["feature_SH000906_leading_return_std_60"] = RollingStatistics(SH000906_leading_return).std(60)[buffer-1:]

        # features_dic["feature_SH000001_amount_mean_pct_5"] = (RollingStatistics(SH000001_amount).mean(5) / (SH000001_amount+1e-8))[buffer:]
        # features_dic["feature_SZ399001_amount_mean_pct_5"] = (RollingStatistics(SZ399001_amount).mean(5) / (SZ399001_amount+1e-8))[buffer:]
        # features_dic["feature_SH000300_amount_mean_pct_5"] = (RollingStatistics(SH000300_amount).mean(5) / (SH000300_amount+1e-8))[buffer:]
        # features_dic["feature_SH000905_amount_mean_pct_5"] = (RollingStatistics(SH000905_amount).mean(5) / (SH000905_amount+1e-8))[buffer:]
        # features_dic["feature_SH000906_amount_mean_pct_5"] = (RollingStatistics(SH000906_amount).mean(5) / (SH000906_amount+1e-8))[buffer:]

        # features_dic["feature_SH000001_amount_mean_pct_10"] = (RollingStatistics(SH000001_amount).mean(10) / (SH000001_amount+1e-8))[buffer:]
        # features_dic["feature_SZ399001_amount_mean_pct_10"] = (RollingStatistics(SZ399001_amount).mean(10) / (SZ399001_amount+1e-8))[buffer:]
        # features_dic["feature_SH000300_amount_mean_pct_10"] = (RollingStatistics(SH000300_amount).mean(10) / (SH000300_amount+1e-8))[buffer:]
        # features_dic["feature_SH000905_amount_mean_pct_10"] = (RollingStatistics(SH000905_amount).mean(10) / (SH000905_amount+1e-8))[buffer:]
        # features_dic["feature_SH000906_amount_mean_pct_10"] = (RollingStatistics(SH000906_amount).mean(10) / (SH000906_amount+1e-8))[buffer:]

        # features_dic["feature_SH000001_amount_mean_pct_20"] = (RollingStatistics(SH000001_amount).mean(20) / (SH000001_amount+1e-8))[buffer:]
        # features_dic["feature_SZ399001_amount_mean_pct_20"] = (RollingStatistics(SZ399001_amount).mean(20) / (SZ399001_amount+1e-8))[buffer:]
        # features_dic["feature_SH000300_amount_mean_pct_20"] = (RollingStatistics(SH000300_amount).mean(20) / (SH000300_amount+1e-8))[buffer:]
        # features_dic["feature_SH000905_amount_mean_pct_20"] = (RollingStatistics(SH000905_amount).mean(20) / (SH000905_amount+1e-8))[buffer:]
        # features_dic["feature_SH000906_amount_mean_pct_20"] = (RollingStatistics(SH000906_amount).mean(20) / (SH000906_amount+1e-8))[buffer:]

        # features_dic["feature_SH000001_amount_mean_pct_30"] = (RollingStatistics(SH000001_amount).mean(30) / (SH000001_amount+1e-8))[buffer:]
        # features_dic["feature_SZ399001_amount_mean_pct_30"] = (RollingStatistics(SZ399001_amount).mean(30) / (SZ399001_amount+1e-8))[buffer:]
        # features_dic["feature_SH000300_amount_mean_pct_30"] = (RollingStatistics(SH000300_amount).mean(30) / (SH000300_amount+1e-8))[buffer:]
        # features_dic["feature_SH000905_amount_mean_pct_30"] = (RollingStatistics(SH000905_amount).mean(30) / (SH000905_amount+1e-8))[buffer:]
        # features_dic["feature_SH000906_amount_mean_pct_30"] = (RollingStatistics(SH000906_amount).mean(30) / (SH000906_amount+1e-8))[buffer:]

        # features_dic["feature_SH000001_amount_mean_pct_60"] = (RollingStatistics(SH000001_amount).mean(60) / (SH000001_amount+1e-8))[buffer:]
        # features_dic["feature_SZ399001_amount_mean_pct_60"] = (RollingStatistics(SZ399001_amount).mean(60) / (SZ399001_amount+1e-8))[buffer:]
        # features_dic["feature_SH000300_amount_mean_pct_60"] = (RollingStatistics(SH000300_amount).mean(60) / (SH000300_amount+1e-8))[buffer:]
        # features_dic["feature_SH000905_amount_mean_pct_60"] = (RollingStatistics(SH000905_amount).mean(60) / (SH000905_amount+1e-8))[buffer:]
        # features_dic["feature_SH000906_amount_mean_pct_60"] = (RollingStatistics(SH000906_amount).mean(60) / (SH000906_amount+1e-8))[buffer:]

        # features_dic["feature_SH000001_amount_std_pct_5"] = (RollingStatistics(SH000001_amount).std(5) / (SH000001_amount+1e-8))[buffer:]
        # features_dic["feature_SZ399001_amount_std_pct_5"] = (RollingStatistics(SZ399001_amount).std(5) / (SZ399001_amount+1e-8))[buffer:]
        # features_dic["feature_SH000300_amount_std_pct_5"] = (RollingStatistics(SH000300_amount).std(5) / (SH000300_amount+1e-8))[buffer:]
        # features_dic["feature_SH000905_amount_std_pct_5"] = (RollingStatistics(SH000905_amount).std(5) / (SH000905_amount+1e-8))[buffer:]
        # features_dic["feature_SH000906_amount_std_pct_5"] = (RollingStatistics(SH000906_amount).std(5) / (SH000906_amount+1e-8))[buffer:]

        # features_dic["feature_SH000001_amount_std_pct_10"] = (RollingStatistics(SH000001_amount).std(10) / (SH000001_amount+1e-8))[buffer:]
        # features_dic["feature_SZ399001_amount_std_pct_10"] = (RollingStatistics(SZ399001_amount).std(10) / (SZ399001_amount+1e-8))[buffer:]
        # features_dic["feature_SH000300_amount_std_pct_10"] = (RollingStatistics(SH000300_amount).std(10) / (SH000300_amount+1e-8))[buffer:]
        # features_dic["feature_SH000905_amount_std_pct_10"] = (RollingStatistics(SH000905_amount).std(10) / (SH000905_amount+1e-8))[buffer:]
        # features_dic["feature_SH000906_amount_std_pct_10"] = (RollingStatistics(SH000906_amount).std(10) / (SH000906_amount+1e-8))[buffer:]

        # features_dic["feature_SH000001_amount_std_pct_20"] = (RollingStatistics(SH000001_amount).std(20) / (SH000001_amount+1e-8))[buffer:]
        # features_dic["feature_SZ399001_amount_std_pct_20"] = (RollingStatistics(SZ399001_amount).std(20) / (SZ399001_amount+1e-8))[buffer:]
        # features_dic["feature_SH000300_amount_std_pct_20"] = (RollingStatistics(SH000300_amount).std(20) / (SH000300_amount+1e-8))[buffer:]
        # features_dic["feature_SH000905_amount_std_pct_20"] = (RollingStatistics(SH000905_amount).std(20) / (SH000905_amount+1e-8))[buffer:]
        # features_dic["feature_SH000906_amount_std_pct_20"] = (RollingStatistics(SH000906_amount).std(20) / (SH000906_amount+1e-8))[buffer:]

        # features_dic["feature_SH000001_amount_std_pct_30"] = (RollingStatistics(SH000001_amount).std(30) / (SH000001_amount+1e-8))[buffer:]
        # features_dic["feature_SZ399001_amount_std_pct_30"] = (RollingStatistics(SZ399001_amount).std(30) / (SZ399001_amount+1e-8))[buffer:]
        # features_dic["feature_SH000300_amount_std_pct_30"] = (RollingStatistics(SH000300_amount).std(30) / (SH000300_amount+1e-8))[buffer:]
        # features_dic["feature_SH000905_amount_std_pct_30"] = (RollingStatistics(SH000905_amount).std(30) / (SH000905_amount+1e-8))[buffer:]
        # features_dic["feature_SH000906_amount_std_pct_30"] = (RollingStatistics(SH000906_amount).std(30) / (SH000906_amount+1e-8))[buffer:]

        # features_dic["feature_SH000001_amount_std_pct_60"] = (RollingStatistics(SH000001_amount).std(60) / (SH000001_amount+1e-8))[buffer:]
        # features_dic["feature_SZ399001_amount_std_pct_60"] = (RollingStatistics(SZ399001_amount).std(60) / (SZ399001_amount+1e-8))[buffer:]
        # features_dic["feature_SH000300_amount_std_pct_60"] = (RollingStatistics(SH000300_amount).std(60) / (SH000300_amount+1e-8))[buffer:]
        # features_dic["feature_SH000905_amount_std_pct_60"] = (RollingStatistics(SH000905_amount).std(60) / (SH000905_amount+1e-8))[buffer:]
        # features_dic["feature_SH000906_amount_std_pct_60"] = (RollingStatistics(SH000906_amount).std(60) / (SH000906_amount+1e-8))[buffer:]

        # sorted_keys = sorted(features_dic.keys())
        # feature_matrix = np.column_stack([features_dic[k] for k in sorted_keys]).transpose(1, 0)
        # self.feature_matrix = torch.tensor(feature_matrix).unsqueeze(2).expand(-1, -1, stocks)
        # for i in range(self.feature_matrix.shape[1]):
        #     if cfg.get("data_x", None):
        #         for op in cfg.get('data_x'):
        #             ProcessDataForTrainingIntermediate = create_op_process(cfg.data_x[op])
        #             ProcessDataForTrainingIntermediate.apply(self.feature_matrix[:,i,:].squeeze(), test_stage=True)
        # import ipdb; ipdb.set_trace()
        if "extra_feature" in cfg.training.model_params and cfg.training.model_params["extra_feature"]:
            self.feature_matrix = torch.cat([factorExp, stock_block], dim=0)
        else:
            self.feature_matrix = None

        return

