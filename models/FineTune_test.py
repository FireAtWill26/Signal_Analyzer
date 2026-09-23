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
from prometheus.utils.muon import Muon, MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam

@TRAINING_REGISTRY.register('FineTune_test')
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
        self.cfg = deepcopy(cfg)

        self.training_dic = cfg.training.model_params
        Path.mkdir(Path(self.training_dic["model_path"]) / model_name, exist_ok=True, parents=True)

        self.model_state_path = Path(self.training_dic["model_path"]) / model_name / 'final_model.pt'
        self.embedding_state_path = Path(self.training_dic["model_path"]) / model_name / 'final_embedding.pt'
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
            for i in range(train_days):
                train_indices[i] = indices[1][i]
            eval_indices = indices[1] & ~train_indices
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

        self.input_dim = np.sum(indices[0])
        for i in range(len(self.cfg.embedding.model_params["layers"])):
            self.cfg.embedding.model_params["layers"] *= self.input_dim
        self.cfg.embedding.model_params["layers"].append(self.cfg.model.model_params["layers_hidden"][0])

        if lock is not None:
            with lock:
                time.sleep(30)
                super().gpu_helper(*args, **kwargs)

        self.embedding = create_model(self.cfg.embedding).to(self.device)
        self.model = create_model(self.cfg.model).to(self.device)

        torch.save(self.cfg, self.dict_path)
        
        train_y = self.train_output[indices[1]]

        var = np.var(train_y)
        std = np.std(train_y)
        mean = np.mean(train_y)
        skewness = skew(train_y)
        kurt = kurtosis(train_y)

        self.mean = mean
        self.std = std

        print(f"target stats| Mean: {mean.item()}| Variance: {var.item()}| Skewness: {skewness}| Kurtosis: {kurt}")

        if model_name not in self.training_dic["prev_pit"]:
            self.train()
        else:
            self.prev_model_state_path = Path(self.training_dic["model_path"]) / self.training_dic["prev_pit"][model_name] / 'final_model.pt'
            self.fine_tune()

        del self.train_loader
        gc.collect()
        torch.cuda.empty_cache()

        pearson_ic, spearman_ic, mse_loss = self.evaluation(self.cfg)

        del self.eval_loader
        gc.collect()
        torch.cuda.empty_cache()

        return pearson_ic, spearman_ic, mse_loss

    def train(self):
        
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
        self.lr = self.training_dic['lr']
        self.weight_decay = self.training_dic["weight_decay"]

        self.training_parameters = [p for p in self.embedding.parameters()] + [p for p in self.model.parameters()]

        if self.training_dic['opt'] == "Adam":
            self.optimizer = torch.optim.Adam(self.training_parameters, lr=self.lr, weight_decay=self.weight_decay)
        elif self.training_dic['opt'] == "AdamW":
            self.optimizer = torch.optim.AdamW(self.training_parameters, lr=self.lr, weight_decay=self.weight_decay)
        elif self.training_dic['opt'] == "RMSprop":
            self.optimizer = torch.optim.RMSprop(self.training_parameters, lr=self.lr, weight_decay=self.weight_decay)   
        elif self.training_dic["opt"] == "Lion":
            self.optimizer = Lion(self.training_parameters, lr=self.lr/10, weight_decay=self.weight_decay)
        # elif self.training_dic["opt"] == "Muon":
        #     muon_params = [p for p in self.model.parameters() if p.ndim >= 2]
        #     adamw_params = [p for p in self.model.parameters() if p.ndim < 2]
        #     self.optimizer = MuonWithAuxAdam(dict(params=muon_params, use_muon=True, lr=lr, weight_decay=self.weight_decay), dict(params=adamw_params, use_muon=False, lr=lr/10, weight_decay=self.weight_decay))
        elif self.training_dic['opt'] == "SGD":
            self.optimizer = torch.optim.SGD(self.training_parameters, lr=self.lr, weight_decay=self.weight_decay)
        elif self.training_dic['opt'] == "LBFGS":
            self.optimizer = torch.optim.LBFGS(self.training_parameters, lr=self.lr, history_size=10, tolerance_grad=1e-32, tolerance_change=1e-32)
        else:
            raise ValueError("opt must be Adam, AdamW, SGD, LBFGS, or RMSprop.")
        
        self.result = {"train_loss": [], "eval_loss": [], "eval_mse_loss": [], "eval_correlation_loss": [], "good_points": []}

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, "min", factor=self.training_dic["scheduler_factor"], patience=self.training_dic["scheduler_patience"], threshold=self.training_dic["scheduler_threshold"])

        self.best_eval_loss = float("inf")
        self.retrain_cnt = 1
        self.overfit_cnt = 0
        self.pred_mean = 0
        self.pred_std = 0

        for epoch in pbar:
            self.embedding.train()
            self.model.train()
            epoch_training_loss = 0
            batch_num = 0
            step_cnt = 0
            for data_batch, target_batch, date_weight in self.train_loader:
                step_cnt += 1
                data_batch = data_batch.to(self.device, non_blocking=True)
                target_batch = target_batch.to(self.device, non_blocking=True)
                weight = torch.ones_like(target_batch)
                weight += date_weight.to(self.device, non_blocking=True)
                self.optimizer.zero_grad()
                pred = self.embedding.forward(data_batch)
                pred = torch.squeeze(self.model.forward(pred))
                if self.training_dic["weight_essential"]:
                    weight += torch.max(F.sigmoid(8 * (target_batch - self.mean - self.training_dic["weight_pos"] * self.std)), F.sigmoid(8 * (pred - self.pred_mean - self.training_dic["weight_pos"] * self.pred_std))) * self.training_dic["weight_pow_over"]
                if self.training_dic["loss_fn"] == "CCC" or self.training_dic["loss_fn"] == "Correlation":
                    correlation_loss = self.loss_fn(pred, target_batch, weight)
                    reg_loss = self.reg_fn(pred, target_batch)
                    if self.training_dic["weight_essential"]:
                        reg_loss = reg_loss * weight
                    reg_loss = reg_loss.mean()
                    train_loss = -self.correlation_ratio * correlation_loss + (1 - self.correlation_ratio) * reg_loss
                else:
                    reg_loss = self.reg_fn(pred, target_batch)
                    if self.training_dic["weight_essential"]:
                        reg_loss = reg_loss * weight
                    train_loss = reg_loss
                loss = train_loss
                if self.training_dic["enable_l1"]:
                    loss += self.training_dic["l1_ratio"] * self.l1_regularization()
                loss.backward()
                nn.utils.clip_grad_norm_(self.training_parameters, max_norm=1.0)
                self.optimizer.step()
            epoch_training_loss += train_loss.item()
            batch_num += 1

            if step_cnt == self.training_dic["eval_steps"]:
                step_cnt = 0
                train_loss = epoch_training_loss / batch_num
                self.result["train_loss"].append(train_loss)
                self.validation(epoch, self.eval_loader)
                pbar.set_description("Epoch: %d|Train Loss:%.2e|Eval Loss:%.2e|MSE Loss:%.2e|Correlation Loss:%.2e|LR: %.2e" % (epoch, self.result["train_loss"][-1], self.result["eval_loss"], self.result["eval_mse_loss"], self.result["eval_correlation_loss"], self.embedding_optimizer.param_groups[0]["lr"]))


    def fine_tune(self):

        days = self.indices[1].shape[0]
        train_days, eval_days = train_test_split(range((days - (20 * self.training_dic["finetune_month"])), days), test_size=self.training_dic["eval_size"], random_state=0, shuffle=True)

        finetune_train_indices = np.zeros(indices[1].shape).astype(bool)
        for i in train_days:
            finetune_train_indices[i] = indices[1][i]
        finetune_eval_indices = np.zeros(indices[1].shape).astype(bool)
        for i in eval_days:
            finetune_eval_indices[i] = indices[1][i]

        num_workers = max(4, min(16, torch.cuda.device_count() * 4))

        finetune_train_dataset = SharedMemDataset(self.model_name, self.train_input.shape, self.train_input.dtype, self.indices[0], finetune_train_indices, self.train_output, np.ones(days))
        finetune_eval_dataset = SharedMemDataset(self.model_name, self.train_input.shape, self.train_input.dtype, self.indices[0], finetune_eval_indices, self.train_output, np.ones(days))

        self.train_loader = DataLoader(finetune_train_dataset, batch_size=self.training_dic["batch_size"], shuffle=True, drop_last=True, prefetch_factor=4, pin_memory=True, num_workers=num_workers, persistent_workers=True)
        self.eval_loader = DataLoader(finetune_eval_dataset, batch_size=self.training_dic["batch_size"], shuffle=False, drop_last=True, prefetch_factor=4, pin_memory=True, num_workers=num_workers, persistent_workers=True)

        self.model.load_state_dict(torch.load(self.prev_model_state_path))
        if self.training_dic['opt'] == "Adam":
            self.embedding_optimizer = torch.optim.Adam(self.embedding.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.training_dic['opt'] == "AdamW":
            self.embedding_optimizer = torch.optim.AdamW(self.embedding.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.training_dic['opt'] == "Lion":
            self.embedding_optimizer = Lion(self.embedding.parameters(), lr=self.lr/10, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"optimizer {self.training_dic['opt']} not supported")

        pbar = tqdm(range(self.training_dic["epoch"]+self.training_dic["fine_tune_epoch"]), desc="Embedding Train")

        self.retrain_cnt = 1
        self.overfit_cnt = 0
        self.pred_mean = 0
        self.pred_std = 0
        for epoch in pbar:
            if epoch < self.training_dic["epoch"]:
                self.embedding.train()
                self.model.eval()
                epoch_training_loss = 0
                batch_num = 0
                step_cnt = 0
                for data_batch, target_batch, date_weight in self.train_loader:
                    step_cnt += 1
                    data_batch = data_batch.to(self.device, non_blocking=True)
                    target_batch = target_batch.to(self.device, non_blocking=True)
                    weight = torch.ones_like(target_batch) + date_weight.to(self.device, non_blocking=True)
                    self.embedding_optimizer.zero_grad()
                    pred = torch.squeeze(self.model.forward(self.embedding.forward(data_batch)))
                    if self.training_dic["weight_essential"]:
                        weight += torch.max(F.sigmoid(8 * (target_batch - self.mean - self.training_dic["weight_pos"] * self.std)), F.sigmoid(8 * (pred - self.pred_mean - self.training_dic["weight_pos"] * self.pred_std))) * self.training_dic["weight_pow_over"]
                    if self.training_dic["loss_fn"] == "CCC" or self.training_dic["loss_fn"] == "Correlation":
                        correlation_loss = self.loss_fn(pred, target_batch, weight)
                        reg_loss = self.reg_fn(pred, target_batch)
                        reg_loss = torch.mean(reg_loss * weight)
                        train_loss = -self.correlation_ratio * correlation_loss + (1 - self.correlation_ratio) * reg_loss
                    else:
                        reg_loss = self.reg_fn(pred, target_batch)
                        reg_loss = torch.mean(reg_loss * weight)
                        train_loss = reg_loss
                        loss = train_loss
                        if self.training_dic["enable_l1"]:
                            loss += self.training_dic["l1_ratio"] * self.l1_regularization()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.embedding.parameters(), max_norm=1.0)
                        self.embedding_optimizer.step()
                    epoch_training_loss += train_loss.item()
                    batch_num += 1
                    if step_cnt == self.training_dic["eval_steps"]:
                        step_cnt = 0
                        self.result["train_loss"].append(epoch_training_loss / batch_num)
                        self.validation(epoch, self.eval_loader)
                        pbar.set_description("Epoch: %d|Train Loss:%.2e|Eval Loss:%.2e|MSE Loss:%.2e|Correlation Loss:%.2e|LR: %.2e" % (epoch, self.result["train_loss"][-1], self.result["eval_loss"], self.result["eval_mse_loss"], self.result["eval_correlation_loss"], self.embedding_optimizer.param_groups[0]["lr"]))
            else:
                self.training_dic["retrain_lr"] = self.training_dic["finetune_lr"]
                self.embedding.train()
                self.model.train()
            break

    def validation(self, epoch, eval_loader):
        mse_loss = 0
        eval_loss = 0
        correlation_loss = 0
        total_pred = []
        total_target = []
        self.embedding.eval()
        self.model.eval()
        with torch.no_grad():
            for data_batch, target_batch, date_weight in eval_loader:
                data_batch = data_batch.to(self.device, non_blocking=True)
                target_batch = target_batch.to(self.device, non_blocking=True)
                weight = torch.ones_like(target_batch)
                weight += date_weight.to(self.device, non_blocking=True)
                pred = self.embedding.forward(data_batch)
                pred = torch.squeeze(self.model.forward(pred))
                if not total_pred:
                    total_pred = [pred.detach()]
                    total_target = [target_batch.detach()]
                else:
                    total_pred.append(pred.detach())
                    total_target.append(target_batch.detach())
                new_mse_loss = F.mse_loss(pred, target_batch)
                mse_loss += new_mse_loss.item()
                if self.training_dic["weight_essential"]:
                    weight += F.sigmoid(8 * (pred - self.pred_mean - self.training_dic["mask_pos"] * self.pred_std)) * self.training_dic["weight_pow_under"]
                if self.training_dic["loss_fn"] == "CCC" or self.training_dic["loss_fn"] == "Correlation":
                    new_correlation_loss = self.loss_fn_eval(pred, target_batch, weight).item()
                    correlation_loss += new_correlation_loss
                    eval_loss += -self.correlation_ratio * new_correlation_loss + (1 - self.correlation_ratio) * new_mse_loss.item()
                else:
                    eval_loss += new_mse_loss.item()
                    correlation_loss += self.correlation_loss(pred, target_batch).item()
        eval_loss /= len(eval_loader)
        mse_loss /= len(eval_loader)
        correlation_loss /= len(eval_loader)
        self.result["mse_loss"].append(mse_loss)
        self.result["correlation_loss"].append(correlation_loss)
        self.result["good_points"].append(torch.sum((total_pred > self.pred_mean + self.training_dic["mask_pos"] * self.pred_std) & (total_target > self.mean + self.training_dic["mask_pos"] * self.std)).item())

        if mse_loss - 5 * correlation_loss < self.best_eval_loss:
            self.best_eval_loss = mse_loss - 5 * correlation_loss
            torch.save(self.embedding.to("cpu").state_dict(), self.embedding_state_path)
            torch.save(self.model.to("cpu").state_dict(), self.model_state_path)
            torch.save(self.optimizer.state_dict(), self.optimizer_path)
            self.embedding.to(self.device)
            self.model.to(self.device)

        if self.result["eval_loss"] and eval_loss > min(self.result["eval_loss"]) and epoch > self.training_dic["overfit_threshold"]:
            self.overfit_cnt += 1
        else:
            self.overfit_cnt = 0

        self.result["eval_loss"].append(eval_loss)

        if self.training_dic["retrain"] and self.overfit_cnt == self.training_dic["overfit_patience"]:
            self.overfit_cnt = 0
            self.training_dic["retrain_lr"] = max(self.optimizer.param_groups[0]["lr"]/2, self.training_dic["retrain_lr"])
            if self.training_dic["check_mask"]:
                self.model.load_state_dict(torch.load(self.model_mask_path, map_location=self.device))
                self.optimizer.load_state_dict(torch.load(self.optimizer_mask_path))
            else:
                self.model.load_state_dict(torch.load(self.model_state_path, map_location=self.device))
                self.optimizer.load_state_dict(torch.load(self.optimizer_path))
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.training_dic["retrain_lr"]
            del self.scheduler
            gc.collect()
            torch.cuda.empty_cache()
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, "min", factor=self.training_dic["scheduler_factor"], threshold=self.training_dic["scheduler_threshold"])
            self.training_dic["retrain_lr"] = max(self.training_dic["retrain_lr"]*self.training_dic["retrain_lr_factor"], self.training_dic["scheduler_threshold"])
            self.retrain_cnt = 1
        else:
            self.retrain_cnt += 1                
        if self.training_dic["loss_fn"] == "CCC" or self.training_dic["loss_fn"] == "Correlation":
            self.scheduler.step(mse_loss - 5 * correlation_loss)
        else:
            self.scheduler.step(eval_loss)

        df = pd.DataFrame(self.result)
        df.to_csv(self.log_path, index=False, head=True)

    def evaluation(self, cfg):

        model_result = create_model(cfg.model)
        embedding_result = create_model(cfg.embedding)

        model_result.load_state_dict(torch.load(self.model_state_path))
        embedding_result.load_state_dict(torch.load(self.embedding_state_path))

        model_result.to(self.device)
        embedding_result.to(self.device)

        model_result.eval()
        embedding_result.eval()

        total_pred = None
        total_target = None

        for data_batch, target_batch, _ in self.eval_loader:
            data_batch = data_batch.to(self.device, non_blocking=True)
            target_batch = target_batch.to(non_blocking=True)
            pred = embedding_result.forward(data_batch)
            pred = torch.squeeze(model_result.forward(pred)).to("cpu").detach().numpy()
            if not total_pred:
                total_pred = [pred]
                total_target = [target_batch.numpy()]
            else:
                total_pred.append(pred)
                total_target.append(target_batch.numpy())
        total_pred = np.concatenate(total_pred, axis=0)
        total_target = np.concatenate(total_target, axis=0)
        mse_loss = mean_squared_error(total_pred, np_target)
        pearson_ic = WeightedCorrNp(x=total_pred, y=np_target, w=np.ones(len(np_target)))('pearson')
        spearman_ic = WeightedCorrNp(x=total_pred, y=np_target, w=np.ones(len(np_target)))('spearman')
        print(f"peason_ic: {pearson_ic} | spearman_ic: {spearman_ic} | mse_loss: {mse_loss} | std of pred: {np.std(total_pred)} | mean of pred: {np.mean(total_pred)} | max of pred: {np.max(total_pred)} | min of pred: {np.min(total_pred)}")
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
        self.model_state_path = Path(model_path) / model_name / "final_model.pt"
        self.embedding_state_path = Path(model_path) / model_name / "final_embedding.pt"

        return 1

    def predict(self, x, cfg, date=None, test_stage=True, lock=None, *args, **kwargs):
        with lock:
            super().gpu_helper(*args, **kwargs)
        if cfg.get("data_x", None):
            for op in cfg.get('data_x'):
                ProcessDataForTrainingIntermediate = create_op_process(cfg.data_x[op])
                ProcessDataForTrainingIntermediate.apply(x, test_stage=True)

        cfg = torch.load(self.dict_path)

        if "feature_select" in cfg.training.model_params and cfg.training.model_params["feature_select"]:
            # print(x.shape, len(cfg.training.model_params["features_delete"]))
            x = x[:,cfg.training.model_params["features_delete"]]

        model = create_model(cfg.model)
        model.load_state_dict(torch.load(self.model_state_path))
        embedding = create_embedding(cfg.embedding)
        embedding.load_state_dict(torch.load(self.embedding_state_path))
        model = model.to(self.device)
        model.eval()
        x = np.nan_to_num(x)

        with torch.no_grad():
            if self.vertical_norm:
                pred = embedding.forward(F.normalize(torch.from_numpy(x).contiguous(), p=2.0, dim=0).to(torch.float32).to(self.device))
            elif self.horizontal_norm:
                pred = embedding.forward(F.normalize(torch.from_numpy(x).contiguous(), p=2.0, dim=1).to(torch.float32).to(self.device))
            else:
                pred = embedding.forward(torch.from_numpy(x).contiguous().to(torch.float32).to(self.device))
            pred = model.forward(pred)

        return pred.squeeze().to("cpu").numpy()

