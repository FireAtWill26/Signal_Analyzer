from contextlib import redirect_stderr
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
import os
import math
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



@TRAINING_REGISTRY.register('Refined_test')
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

        self.training_dic = cfg.training.model_params
        Path.mkdir(Path(self.training_dic["model_path"]) / model_name, exist_ok=True, parents=True)

        self.model_state_path = Path(self.training_dic["model_path"]) / model_name / 'final_model.pt'
        self.optimizer_path = Path(self.training_dic["model_path"]) / model_name / 'final_optimizer.pt'
        self.dict_path = Path(self.training_dic["model_path"]) / model_name / 'training_dict.pt'
        self.log_path = Path(self.training_dic["model_path"]) / model_name / 'training_log.csv'
        if "finetune" in self.training_dic and self.training_dic["finetune"]:
            self.log_path_finetune = Path(self.training_dic["model_path"]) / model_name / 'finetune_log.csv'
        self.summary_path = Path("/dfs/data/tensorBoard") / model_name
        if self.training_dic["eval_mask"]:
            self.model_mask_path = Path(self.training_dic["model_path"]) / model_name / 'final_model_mask.pt'
            self.optimizer_mask_path = Path(self.training_dic["model_path"]) / model_name / 'final_optimizer_mask.pt'
        self.model_most_path = Path(self.training_dic["model_path"]) / model_name / 'final_model_most.pt'
        self.optimizer_most_path = Path(self.training_dic["model_path"]) / model_name / 'final_optimizer_most.pt'

        if "feature_select" in self.training_dic and self.training_dic["feature_select"]:
            self.training_dic["features_delete"] = self.feature_selection(cfg)
        
        num_workers = max(4, min(16, torch.cuda.device_count() * 4))  # 根据GPU数量动态调整
        
        Path.mkdir(self.summary_path, exist_ok=True, parents=True)

        writer = SummaryWriter(self.summary_path / "run")

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

        self.train_dataset = SharedMemDataset(model_name, self.train_input.shape, self.train_input.dtype, self.indices[0], train_indices, self.train_output,date_weight)
        self.eval_dataset = SharedMemDataset(model_name, self.train_input.shape, self.train_input.dtype, self.indices[0], eval_indices, self.train_output,date_weight)

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

            finetune_train_dataset = SharedMemDataset(model_name, self.train_input.shape, self.train_input.dtype, self.indices[0], finetune_train_indices, self.train_output,np.ones(days))
            finetune_eval_dataset = SharedMemDataset(model_name, self.train_input.shape, self.train_input.dtype, self.indices[0], finetune_eval_indices, self.train_output,np.ones(days))

            self.finetune_train_loader = DataLoader(finetune_train_dataset, batch_size=self.training_dic["batch_size"], shuffle=True, drop_last=True, prefetch_factor=4, pin_memory=True, num_workers=num_workers, persistent_workers=True)
            self.finetune_eval_loader = DataLoader(finetune_eval_dataset, batch_size=self.training_dic["batch_size"], shuffle=False, drop_last=True, prefetch_factor=4, pin_memory=True, num_workers=num_workers, persistent_workers=True)

        print(f"len of train_data: {len(self.train_dataset)}, len of eval_data: {len(self.eval_dataset)}, len of train_loader: {len(self.train_loader)}, len of eval_loader: {len(self.eval_loader)}")

        if lock is not None:
            with lock:
                time.sleep(30)
                super().gpu_helper(*args, **kwargs)

        if not "denormalizer" in self.training_dic:
            self.training_dic["denormalizer"] = False
        if not "step_num" in self.training_dic:
            self.training_dic["step_num"] = 30
        self.training_dic["eval_steps"] = np.sum(indices[1]) // (self.training_dic["step_num"] * self.training_dic["batch_size"])

        self.input_dim = np.sum(indices[0])
        if self.training_dic["denormalizer"]:
            self.input_dim += 2
        self.cfg = deepcopy(cfg)
        if "FastKAN" in self.cfg.model.type:
            for i in range(len(self.cfg.model.model_params["layers_hidden"])-self.training_dic["plain_layer"]):
                self.cfg.model.model_params["layers_hidden"][i] *= self.input_dim
        elif self.cfg.model.type == "KANUNet":
            for i in range(self.training_dic["mult_layer"]):
                self.cfg.model.model_params["layers_hidden"][i] *= self.input_dim
        self.model = create_model(self.cfg.model).to(self.device)
        self.tanh_output = nn.Tanh()

        result = {"train_loss": [], "eval_loss": [], "eval_mse_loss": [], "eval_correlation_loss": [], "eval_correlation_mask": [],  "good_points": [], "huber_delta": []}

        if "discrimination" in self.training_dic and self.training_dic["discrimination"]:
            
            result["adv_loss"] = []

            adv_loss_fn = nn.BCEWithLogitsLoss()

            real_label = torch.ones(self.training_dic["batch_size"], 1).to(self.device)
            fake_label = torch.zeros(self.training_dic["batch_size"], 1).to(self.device)

            if "FastKAN" in self.cfg.discriminator.type:
                for i in range(len(cfg.discriminator.model_params["layers_hidden"])-3):
                    self.cfg.discriminator.model_params["layers_hidden"][i] *= self.input_dim+1
            self.discriminator = create_model(self.cfg.discriminator).to(self.device)
            self.discriminator_optimizer = torch.optim.AdamW(self.discriminator.parameters(), lr=self.training_dic["lr"], weight_decay=self.training_dic["weight_decay"])

        # if only_eval set to true in config, call the quick_evaluation function which will not train the model but only do the evaluation.

        if self.training_dic["only_eval"]:
            return self.quick_evaluation()

        torch.save(self.cfg, self.dict_path)
        
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

        pbar = tqdm(range(self.training_dic['epochs']), desc='Training', ncols=160)

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
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=self.weight_decay)
        elif self.training_dic['opt'] == "AdamW":
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=self.weight_decay)
        elif self.training_dic['opt'] == "RMSprop":
            self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=lr, weight_decay=self.weight_decay)
        elif self.training_dic["opt"] == "Lion":
            self.optimizer = Lion(self.model.parameters(), lr=lr/10, weight_decay=self.weight_decay)
        # elif self.training_dic["opt"] == "Muon":
        #     muon_params = [p for p in self.model.parameters() if p.ndim >= 2]
        #     adamw_params = [p for p in self.model.parameters() if p.ndim < 2]
        #     self.optimizer = MuonWithAuxAdam(dict(params=muon_params, use_muon=True, lr=lr, weight_decay=self.weight_decay), dict(params=adamw_params, use_muon=False, lr=lr/10, weight_decay=self.weight_decay))
        elif self.training_dic['opt'] == "SGD":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=self.weight_decay)
        elif self.training_dic['opt'] == "LBFGS":
            self.optimizer = torch.optim.LBFGS(self.model.parameters(), lr=lr, history_size=10, tolerance_grad=1e-32, tolerance_change=1e-32)
        else:
            raise ValueError("opt must be Adam, AdamW, SGD, LBFGS, or RMSprop.")

        writer = SummaryWriter(self.summary_path / "run")
        
        if self.training_dic["opt"] == "LBFGS":
            def closure():
                self.optimizer.zero_grad()
                pred = self.model.forward(data_batch)
                train_loss = self.loss_fn(pred, target_batch)
                train_loss.backward()
                return train_loss

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, "min", factor=self.training_dic["scheduler_factor"], patience=self.training_dic["scheduler_patience"], threshold=self.training_dic["scheduler_threshold"])


        best_eval_loss = float("inf")
        best_correlation_loss = -float("inf")
        if self.training_dic["eval_mask"]:
            best_correlation_mask_loss = -float("inf")
        retrain_cnt = 1
        overfit_cnt = 0
        pred_mean = 0
        pred_std = 0
        best_points_cnt = 0
        for epoch in pbar:
            
            self.model.train()
            epoch_training_loss = 0
            batch_num = 0
            step_cnt = 0
            adv_loss = 0
            for data_batch, target_batch, date_weight in self.train_loader:
                step_cnt += 1
                # 提前批量移动数据到设备
                if self.training_dic["denormalizer"]:
                    data_batch = data_batch.to(self.device)
                    batch_mean = torch.unsqueeze(torch.mean(data_batch, dim=1), 1)
                    batch_std = torch.unsqueeze(torch.std(data_batch, dim=1), 1)
                    data_batch = torch.cat([data_batch, batch_mean, batch_std.pow(0.5)], dim=1).to(self.device, non_blocking=True)
                else:
                    data_batch = data_batch.to(self.device, non_blocking=True)  
                target_batch = target_batch.to(self.device, non_blocking=True)
                weight = torch.ones_like(target_batch)
                weight += date_weight.to(self.device, non_blocking=True)
                if "horizontal_norm" in self.training_dic and self.training_dic["horizontal_norm"]:
                    data_batch = F.normalize(data_batch, p=2.0, dim=1)
                if self.training_dic["opt"] == "LBFGS":
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    train_loss = self.optimizer.step(closure)
                else:
                    self.optimizer.zero_grad()
                    if "discrimination" in self.training_dic and self.training_dic["discrimination"]:
                        self.discriminator_optimizer.zero_grad()
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
                        loss_dis = adv_loss_fn(self.discriminator.forward(torch.cat([data_batch.detach(),target_batch.detach().view(-1,1)], dim=1)), real_label) + adv_loss_fn(self.discriminator.forward(torch.cat([data_batch.detach(),pred.detach().view(-1,1)], dim=1)), fake_label)
                        loss_dis /= 2
                        loss_dis.backward()
                        self.discriminator_optimizer.step()
                        adv_loss += loss_dis.item()
                    # assert torch.isnan(self.model.parameters()).sum() == 0, print(self.model.parameters())
                epoch_training_loss += train_loss.item()
                batch_num += 1

                if step_cnt == self.training_dic["eval_steps"]:
                    step_cnt = 0

                    train_loss = epoch_training_loss / batch_num
                    result["train_loss"].append(train_loss)
                    if "discrimination" in self.training_dic and self.training_dic["discrimination"]:
                        result["adv_loss"].append(adv_loss/batch_num)

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

                    self.model.eval()
                    total_pred = [torch.tensor([]).to(self.device)]
                    total_target = [torch.tensor([]).to(self.device)]

                    with torch.no_grad():
                        for data_batch, target_batch, date_weight in self.eval_loader:
                            if self.training_dic["denormalizer"]:
                                batch_mean = torch.unsqueeze(torch.mean(data_batch, dim=1), 1)
                                batch_std = torch.unsqueeze(torch.std(data_batch, dim=1), 1)
                                data_batch = torch.cat([data_batch, batch_mean, batch_std.pow(0.5)], dim=1).to(self.device, non_blocking=True)
                            else:
                                data_batch = data_batch.to(self.device, non_blocking=True)
                            if "horizontal_norm" in self.training_dic and self.training_dic["horizontal_norm"]:
                                data_batch = F.normalize(data_batch, p=2.0, dim=1)     
                            target_batch = target_batch.to(self.device, non_blocking=True)
                            weight = torch.ones_like(target_batch)
                            weight += date_weight.to(self.device, non_blocking=True)
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

                    eval_loss /= len(self.eval_loader)
                    mse_loss /= len(self.eval_loader)
                    correlation_loss /= len(self.eval_loader)
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
                    result["good_points"].append(torch.sum((total_pred > pred_mean + self.training_dic["mask_pos"] * pred_std)&(total_target > mean + self.training_dic["mask_pos"] * std)).item())



                    # save the model weights if the evaluation loss is improved
                    if not result["eval_loss"] or (mse_loss - 5 * correlation_loss < best_eval_loss):
                        # best_correlation_loss = correlation_loss
                        # if eval_loss < best_eval_loss:
                        best_eval_loss = mse_loss - 5 * correlation_loss
                        torch.save(self.model.to("cpu").state_dict(), self.model_state_path)
                        torch.save(self.optimizer.state_dict(), self.optimizer_path)
                        self.model.to(self.device)
                    if self.training_dic["check_mask"] and self.training_dic["eval_mask"]:
                        if not result["eval_correlation_mask"] or correlation_mask + 10 * correlation_loss> best_correlation_mask_loss:
                            best_correlation_mask_loss = correlation_mask + 10 * correlation_loss
                            torch.save(self.model.to("cpu").state_dict(), self.model_mask_path)
                            torch.save(self.optimizer.state_dict(), self.optimizer_mask_path)
                            self.model.to(self.device)
                    if self.training_dic["check_most"] and (not result["good_points"] or result["good_points"][-1] > best_points_cnt):
                        best_points_cnt = result["good_points"][-1]
                        torch.save(self.model.to("cpu").state_dict(), self.model_most_path)
                        torch.save(self.optimizer.state_dict(), self.optimizer_most_path)
                        self.model.to(self.device)

                    if result["eval_loss"] and eval_loss > min(result["eval_loss"]) and epoch >= self.training_dic["overfit_threshold"]:
                        overfit_cnt += 1
                    else:
                        overfit_cnt = 0

                    result["eval_loss"].append(eval_loss)

                    pbar.set_description("Epoch :%d|Train Loss: %.2e|Evaluation Loss: %.2e|MSE Loss: %.2e|Correlation: %.2e|LR: %.2e" % (epoch+1, train_loss, eval_loss, mse_loss, correlation_loss, self.optimizer.param_groups[0]["lr"]))

                    result["eval_mse_loss"].append(mse_loss)
                    result["eval_correlation_loss"].append(correlation_loss)
                    if self.training_dic["eval_mask"]:
                        result["eval_correlation_mask"].append(correlation_mask)


                    if self.training_dic["retrain"] and overfit_cnt == self.training_dic["overfit_patience"]:
                        overfit_cnt = 0
                        self.training_dic["retrain_lr"] = max(self.optimizer.param_groups[0]["lr"]/2, self.training_dic["retrain_lr"])
                        if self.training_dic["check_mask"]:
                            self.model.load_state_dict(torch.load(self.model_mask_path, map_location=self.device))
                            self.optimizer.load_state_dict(torch.load(self.optimizer_mask_path))
                        else:
                            self.model.load_state_dict(torch.load(self.model_state_path, map_location=self.device))
                            self.optimizer.load_state_dict(torch.load(self.optimizer_path))
                        for param_group in self.optimizer.param_groups:
                            param_group["lr"] = self.training_dic["retrain_lr"]
                        del scheduler
                        gc.collect()
                        torch.cuda.empty_cache()
                        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, "min", factor=self.training_dic["scheduler_factor"], threshold=self.training_dic["scheduler_threshold"])
                        self.training_dic["retrain_lr"] = max(self.training_dic["retrain_lr"]*self.training_dic["retrain_lr_factor"], self.training_dic["scheduler_threshold"])
                        retrain_cnt = 1
                    else:
                        if self.training_dic["loss_fn"] == "CCC" or self.training_dic["loss_fn"] == "Correlation":
                            scheduler.step(mse_loss - 5 * correlation_loss)
                        else:
                            scheduler.step(eval_loss)
                        retrain_cnt += 1

                    if self.training_dic["update_delta"] and self.training_dic["reg_loss"] == "Huber":
                        if not (epoch+1) % self.training_dic["update_period"]:
                            if total_pred.shape[0] > 0:
                                self.training_dic["huber_delta"] = np.percentile(np.abs(total_pred.cpu().numpy()-total_target.cpu().numpy()), 90)
                                self.reg_fn = nn.HuberLoss(delta=self.training_dic["huber_delta"])
                    result["huber_delta"].append(self.training_dic["huber_delta"])

                    df = pd.DataFrame(result)
                    df.to_csv(self.log_path, index=False, header = True)

        if "finetune" in self.training_dic and self.training_dic["finetune"]:
            self.fine_tune()

        if "discrimination" in self.training_dic and self.training_dic["discrimination"]:
            self.model.load_state_dict(torch.load(self.model_state_path, map_location=self.device))
            layer_len = len(self.cfg.model.model_params["layers_hidden"]) - 1
            if "dis_finetune_layer" not in self.training_dic:
                self.training_dic["dis_finetune_layer"] = 0
            layer_threshold = layer_len - self.training_dic["dis_finetune_layer"]

            params_to_update = []

            for name, params in self.model.named_parameters():
                if int(name.split(".")[1]) < layer_threshold:
                    params.requires_grad = False
                else:
                    params.requires_grad = True
                    params_to_update.append(params)

            self.optimizer = torch.optim.AdamW(params_to_update, lr=self.training_dic["lr"]/1e2, weight_decay=self.weight_decay)
            self.discriminator_optimizer = torch.optim.AdamW(self.discriminator.parameters(), lr=self.training_dic["lr"]/self.training_dic["dis_lr_ratio"], weight_decay=self.training_dic["weight_decay"])

            dis_step = len(self.finetune_train_loader) // self.training_dic["dis_step_num"]

            pbar_dis = tqdm(range(self.training_dic["dis_epoch"]), desc="discrimination", ncols=160)
            best_eval = float("inf")
            for epoch in pbar_dis:
                self.discriminator.train()
                total_adv_loss = 0
                batch_num = 0
                step_cnt = 0
                for data_batch, target_batch, _ in self.finetune_train_loader:
                    self.discriminator_optimizer.zero_grad()
                    data_batch = data_batch.to(self.device, non_blocking=True)
                    target_batch = target_batch.to(self.device, non_blocking=True)
                    pred = torch.squeeze(self.model.forward(data_batch))
                    adv_loss = adv_loss_fn(self.discriminator.forward(torch.cat([data_batch.detach(),target_batch.detach().view(-1,1)], dim=1)), real_label) + adv_loss_fn(self.discriminator.forward(torch.cat([data_batch.detach(),pred.detach().view(-1,1)], dim=1)), fake_label)
                    adv_loss /= 2
                    adv_loss.backward()
                    total_adv_loss += adv_loss.item()
                    batch_num += 1
                    step_cnt += 1
                    self.discriminator_optimizer.step()
                    if step_cnt == dis_step:
                        step_cnt = 0
                        self.discriminator.eval()
                        self.model.train()
                        # for data_batch, target_batch, _ in self.finetune_train_loader:
                        #     self.optimizer.zero_grad()
                        #     data_batch = data_batch.to(self.device, non_blocking=True)
                        #     pred = torch.squeeze(self.model.forward(data_batch))
                        #     dis_loss = adv_loss_fn(self.discriminator.forward(torch.cat([data_batch,pred.view(-1,1)], dim=1)), real_label)
                        #     dis_loss.backward()
                        #     self.optimizer.step()
                        self.optimizer.zero_grad()
                        pred = torch.squeeze(self.model.forward(data_batch))
                        dis_loss = adv_loss_fn(self.discriminator.forward(torch.cat([data_batch,pred.view(-1,1)], dim=1)), real_label)
                        dis_loss.backward()
                        self.optimizer.step()
                        self.model.eval()
                        mse_loss = 0
                        correlation_loss = 0
                        eval_batch_num = 0
                        for data_batch, target_batch, _ in self.finetune_eval_loader:
                            data_batch = data_batch.to(self.device, non_blocking=True)
                            target_batch = target_batch.to(self.device, non_blocking=True)
                            pred = torch.squeeze(self.model.forward(data_batch))
                            mse_loss += torch.mean(F.mse_loss(pred, target_batch)).item()
                            correlation_loss += self.correlation_loss(pred, target_batch, torch.ones_like(target_batch)).item()
                            eval_batch_num += 1
                        mse_loss /= eval_batch_num
                        self.discriminator.train()
                        correlation_loss /= eval_batch_num
                        pbar_dis.set_description("Epoch :%d|Adv Loss: %.2e|MSE Loss: %.2e|Correlation Loss: %.2e" % (epoch+1, total_adv_loss/batch_num, mse_loss, correlation_loss))
                        if best_eval > mse_loss - 5 * correlation_loss:
                            best_eval = mse_loss - 5 * correlation_loss
                            torch.save(self.model.to("cpu").state_dict(), self.model_state_path)
                            torch.save(self.optimizer.state_dict(), self.optimizer_path)
                            self.model.to(self.device)

        del self.train_loader
        if ("finetune" in self.training_dic and self.training_dic["finetune"]) or ("discrimination" in self.training_dic and self.training_dic["discrimination"]):
            del self.finetune_train_loader
            del self.finetune_eval_loader
        gc.collect()
        torch.cuda.empty_cache()

        pearson_ic, spearman_ic, mse_loss = self.evaluation(self.cfg)

        del self.eval_loader
        gc.collect()
        torch.cuda.empty_cache()

        return pearson_ic, spearman_ic, mse_loss

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
                target_batch = target_batch.to(self.device, non_blocking=True)
                pred = torch.squeeze(self.model.forward(data_batch))
                weight = torch.ones_like(target_batch)
                new_mse_loss = F.mse_loss(pred, target_batch).item()
                mse_loss += new_mse_loss
                if "finetune_weight_essential" in self.training_dic and self.training_dic["finetune_weight_essential"]:
                    weight += F.sigmoid(8 * (pred - pred_mean - self.training_dic["mask_pos"] * pred_std)) * self.training_dic["weight_pow_under"]
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
                target_batch = target_batch.to(self.device, non_blocking=True)
                weight = torch.ones_like(target_batch) + date_weight.to(self.device, non_blocking=True)
                pred = torch.squeeze(self.model.forward(data_batch))
                if "finetune_weight_essential" in self.training_dic and self.training_dic["finetune_weight_essential"]:
                    weight += torch.max(F.sigmoid(8 * (target_batch - self.mean -self.training_dic["weight_pos"] * self.std)), F.sigmoid(8 * (pred- pred_mean -self.training_dic["weight_pos"] * pred_std))) * self.training_dic["weight_pow_over"]
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
                torch.nn.utils.clip_grad_norm_(params_to_update, max_norm=1.0)
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
                            target_batch = target_batch.to(self.device, non_blocking=True)
                            weight = torch.ones_like(target_batch) + date_weight.to(self.device, non_blocking=True)
                            pred = torch.squeeze(self.model.forward(data_batch))
                            new_mse_loss = F.mse_loss(pred, target_batch).item()
                            mse_loss += new_mse_loss
                            if "finetune_weight_essential" in self.training_dic and self.training_dic["finetune_weight_essential"]:
                                weight += F.sigmoid(8 * (pred - pred_mean - self.training_dic["mask_pos"] * pred_std)) * self.training_dic["weight_pow_under"]
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

    def quick_evaluation(self):
        self.check_most = self.training_dic["check_most"]
        self.check_mask = self.training_dic["check_mask"]
        cfg = torch.load(self.dict_path)
        self.training_dic = cfg.training.model_params
        self.training_dic["check_most"] = self.check_most
        self.training_dic["check_mask"] = self.check_mask
        self.vertical_norm = self.training_dic["vertical_norm"]
        if not "denormalizer" in self.training_dic:
            self.training_dic["denormalizer"] = False

        pearson_ic, spearman_ic, mse_loss = self.evaluation(cfg)

        del self.eval_loader
        gc.collect()
        torch.cuda.empty_cache()

        return pearson_ic, spearman_ic, mse_loss


    def l1_regularization(self):
        l1_loss = sum(param.abs().sum() for param in self.model.parameters())
        return l1_loss


    def evaluation(self, cfg):
        # load the best model from the training process
        model_result = create_model(cfg.model)
        if "check_mask" in cfg.training.model_params and cfg.training.model_params["check_mask"]:
            model_result.load_state_dict(torch.load(self.model_mask_path))
        elif "check_most" in cfg.training.model_params and cfg.training.model_params["check_most"]:
            model_result.load_state_dict(torch.load(self.model_most_path))
        else:
            model_result.load_state_dict(torch.load(self.model_state_path))
        model_result = model_result.to(self.device)
        model_result.eval()
        total_pred = None
        np_target = None
        for data_batch, target_batch, date_weight in self.eval_loader:
            if self.training_dic["denormalizer"]:
                batch_mean = torch.unsqueeze(torch.mean(data_batch, dim=1), 1)
                batch_std = torch.unsqueeze(torch.std(data_batch, dim=1), 1)
                data_batch = torch.cat([data_batch, batch_mean, batch_std.pow(0.5)], dim=1).to(self.device, non_blocking=True)
            else:
                data_batch = data_batch.to(self.device, non_blocking=True)
            if "horizontal_norm" in self.training_dic and self.training_dic["horizontal_norm"]:
                data_batch = F.normalize(data_batch, p=2.0, dim=1)
            target_batch = target_batch.to(non_blocking=True)
            pred = model_result.forward(data_batch)
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
        self.dict_path = Path(model_path) / model_name / "training_dict.pt"
        self.model_state_dict_path = Path(model_path) / model_name / "final_model.pt"
        return 1

    def predict(self, x, cfg, date=None, test_stage=True, lock=None, *args, **kwargs):
        with lock:
            super().gpu_helper(*args, **kwargs)
        if cfg.get("data_x", None):
            for op in cfg.get('data_x'):
                ProcessDataForTrainingIntermediate = create_op_process(cfg.data_x[op])
                ProcessDataForTrainingIntermediate.apply(x, test_stage=True)
        """        
        if cfg.model.type == "FastKAN":
            for i in range(len(cfg.model.model_params["layers_hidden"])-1):
                cfg.model.model_params["layers_hidden"][i] *= input_dim
        """
        if cfg.training.model_params["check_mask"]:
            self.model_state_dict_path = self.model_state_dict_path.with_name("final_model_mask.pt")
        elif "check_most" in cfg.training.model_params and cfg.training.model_params["check_most"]:
            self.model_state_dict_path = self.model_state_dict_path.with_name("final_model_most.pt")

        cfg = torch.load(self.dict_path)
        if not "denormalizer" in cfg.training.model_params:
            cfg.training.model_params["denormalizer"] = False

        if "feature_select" in cfg.training.model_params and cfg.training.model_params["feature_select"]:
            # print(x.shape, len(cfg.training.model_params["features_delete"]))
            x = x[:,cfg.training.model_params["features_delete"]]

        if "mode" not in cfg.model.model_params:
            cfg.model.model_params["mode"] = "default"
        if "Cheby" in cfg.model.model_params:
            if cfg.model.model_params["Cheby"]:
                cfg.model.model_params["mode"] = "Cheby"
            del cfg.model.model_params["Cheby"]
        if "KAF" in cfg.model.model_params:
            if cfg.model.model_params["KAF"]:
                cfg.model.model_params["mode"] = "KAF"
            del cfg.model.model_params["KAF"]
        if "ResNet" in cfg.model.model_params:
            cfg.model.model_params["res_net"] = cfg.model.model_params["ResNet"]
            del cfg.model.model_params["ResNet"]
        self.vertical_norm = cfg.training.model_params["vertical_norm"]
        self.horizontal_norm = "horizontal_norm" in cfg.training.model_params and cfg.training.model_params["horizontal_norm"]
        model = create_model(cfg.model)
        model.load_state_dict(torch.load(self.model_state_dict_path))
        model = model.to(self.device)
        model.eval()
        x = np.nan_to_num(x)

        if cfg.training.model_params["denormalizer"]:
            x = np.concatenate([x, np.mean(x, axis=1).reshape(-1, 1), np.power(np.std(x, axis=1), 0.5).reshape(-1, 1)], axis=1)

        with torch.no_grad():
            if self.vertical_norm:
                pred = model.forward(F.normalize(torch.from_numpy(x).contiguous(), p=2.0, dim=0).to(torch.float32).to(self.device))
            elif self.horizontal_norm:
                pred = model.forward(F.normalize(torch.from_numpy(x).contiguous(), p=2.0, dim=1).to(torch.float32).to(self.device))
            else:
                pred = model.forward(torch.from_numpy(x).contiguous().to(torch.float32).to(self.device))

        return pred.squeeze().to("cpu").numpy()

