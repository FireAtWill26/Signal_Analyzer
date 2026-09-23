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
from prometheus.utils.torch_utils import select_gpu_with_minimum_memory, early_stopping_func, WarmupLR, DecayingCosineWarmRestarts, SharedMemDataset, SharedMemSeqDataset, SharedMemSeqDataset, SharedMemSeqDatasetRandom
from prometheus.ops import create_op_process
from prometheus.modelpool.basemodel import BaseModel, create_model
from prometheus.utils.registry_factory import TRAINING_REGISTRY, MODEL_REGISTRY
from prometheus.utils.speedup_package import calculate_stock_mean_3d_parallel
from prometheus.utils.pearson import niocorr
from prometheus.utils.spearman import niocorr_spearman
from prometheus.modelpool.Adversarial import Adversarial
from prometheus.utils.data_processing import RollingStatistics
from prometheus.riskmodel.barra.factor_dict import *
from prometheus.utils.utils import calc_correlation, ccc


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



@TRAINING_REGISTRY.register('Boosting_test')
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

        self.model_state_path_list = [Path(self.training_dic["model_path"]) / model_name / f'final_model_{i}.pt' for i in range(self.training_dic["num_models"])]
        # self.model_nd_state_path = Path(self.training_dic["model_path"]) / model_name / 'final_model_nd.pt'
        self.optimizer_path_list = [Path(self.training_dic["model_path"]) / model_name / f'final_optimizer_{i}.pt' for i in range(self.training_dic["num_models"])]
        self.regressor_state_path = Path(self.training_dic["model_path"]) / model_name/ "regressor.pt"
        self.regressor_opt_path = Path(self.training_dic["model_path"]) / model_name / "regressor_opt.pt"
        self.dict_path = Path(self.training_dic["model_path"]) / model_name / 'training_dict.pt'
        self.log_path_list = [Path(self.training_dic["model_path"]) / model_name / f'training_log_{i}.csv' for i in range(self.training_dic["num_models"])]
        self.summary_path = Path("/dfs/data/tensorBoard") / cfg.train_basics["model_path_name"] / model_name

        if "feature_select" in self.training_dic and self.training_dic["feature_select"]:
            self.training_dic["features_delete"] = self.feature_selection(cfg)
        
        num_workers = max(4, min(16, torch.cuda.device_count() * 4))  # 根据GPU数量动态调整
        
        Path.mkdir(self.summary_path, exist_ok=True, parents=True)

        
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

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.training_dic["batch_size"], shuffle=True, drop_last=True, prefetch_factor=4, pin_memory=True, num_workers=num_workers, persistent_workers=True)
        self.eval_loader = DataLoader(self.eval_dataset, batch_size=self.training_dic["batch_size"], shuffle=False, drop_last=True, prefetch_factor=4, pin_memory=True, num_workers=num_workers, persistent_workers=True)

        print(f"len of train_data: {len(self.train_dataset)}, len of eval_data: {len(self.eval_dataset)}, len of train_loader: {len(self.train_loader)}, len of eval_loader: {len(self.eval_loader)}")

        if lock is not None:
            with lock:
                time.sleep(30)
                super().gpu_helper(*args, **kwargs)

        self.cfg = deepcopy(cfg)
        
        self.input_dim = np.sum(indices[0])
        if "extra_feature" in self.training_dic and self.training_dic["extra_feature"]:
            self.input_dim += self.feature_matrix.shape[0]
        if "feature_insert_pos" in self.cfg.model.model_params and self.cfg.model.model_params["feature_insert_pos"] is not None:
            if "feature_insert" in self.cfg.model.model_params and self.cfg.model.model_params["feature_insert"] is not None:
                self.input_dim -= self.cfg.model.model_params["feature_insert"].sum()
        if self.training_dic["denormalizer"]:
            self.input_dim += 2
        if "FastKAN" in self.cfg.model.type:
            for i in range(len(self.cfg.model.model_params["layers_hidden"])-self.training_dic["plain_layer"]):
                self.cfg.model.model_params["layers_hidden"][i] *= self.input_dim
        elif self.cfg.model.type == "KANUNet":
            for i in range(self.training_dic["mult_layer"]):
                self.cfg.model.model_params["layers_hidden"][i] *= self.input_dim
        self.cfg.model.model_params["device"] = self.device
        self.models = {i:create_model(self.cfg.model) for i in range(self.training_dic["num_models"])}
        self.tanh_output = nn.Tanh()
        self.regressor = Regressor(self.training_dic["num_models"], device=self.device)

        writer = SummaryWriter(self.summary_path / "run")
        writer.add_graph(self.model, torch.ones(self.input_dim, device=self.device))

        result = {}

        for i in range(self.training_dic["num_models"]):
            result[i] = {"train_loss": [], "eval_loss": [], "eval_mse_loss": [], "eval_correlation_loss": [], "good_points": [], "LR": []}

        
        torch.save(self.cfg, self.dict_path)
        
        train_y = train_output[indices[1]]

        var = np.var(train_y)
        std = np.std(train_y)
        mean = np.mean(train_y)
        skewness = skew(train_y)
        kurt = kurtosis(train_y)

        self.mean = mean
        self.std = std

        print(f"target stats| Mean: {mean.item()}| Variance: {var.item()}| Skewness: {skewness}| Kurtosis: {kurt}")

        pbar = tqdm(range(self.training_dic['num_models']), desc='Training', ncols=160)

        if self.training_dic["loss_fn"] == "CCC":
            self.loss_fn = self.loss_fn_eval = lambda x, y, w: (x,y,w)
        elif self.training_dic["loss_fn"] == "Correlation":
            self.loss_fn = self.loss_fn_eval = lambda x, y, w: (x,y,w)
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
            self.optimizer_list = {i:torch.optim.Adam(self.models[i].parameters(), lr=lr, weight_decay=self.weight_decay) for i in range(self.training_dic["num_models"])}
        elif self.training_dic['opt'] == "AdamW":
            self.optimizer_list = {i:torch.optim.AdamW(self.models[i].parameters(), lr=lr, weight_decay=self.weight_decay) for i in range(self.training_dic["num_models"])}
        elif self.training_dic['opt'] == "RMSprop":
            self.optimizer_list = {i:torch.optim.RMSprop(self.models[i].parameters(), lr=lr, weight_decay=self.weight_decay) for i in range(self.training_dic["num_models"])}
        elif self.training_dic["opt"] == "Lion":
            self.optimizer_list = {i:Lion(self.models[i].parameters(), lr=lr/10, weight_decay=self.weight_decay) for i in range(self.training_dic["num_models"])}
        # elif self.training_dic["opt"] == "Muon":
        #     muon_params = [p for p in self.model.parameters() if p.ndim >= 2]
        #     adamw_params = [p for p in self.model.parameters() if p.ndim < 2]
        #     self.optimizer = MuonWithAuxAdam(dict(params=muon_params, use_muon=True, lr=lr, weight_decay=self.weight_decay), dict(params=adamw_params, use_muon=False, lr=lr/10, weight_decay=self.weight_decay))
        elif self.training_dic['opt'] == "SGD":
            self.optimizer_list = {i:torch.optim.SGD(self.models[i].parameters(), lr=lr, weight_decay=self.weight_decay) for i in range(self.training_dic["num_models"])}
        elif self.training_dic['opt'] == "LBFGS":
            self.optimizer_list = {i:torch.optim.LBFGS(self.models[i].parameters(), lr=lr, history_size=10, tolerance_grad=1e-32, tolerance_change=1e-32) for i in range(self.training_dic["num_models"])}
        else:
            raise ValueError("opt must be Adam, AdamW, SGD, LBFGS, or RMSprop.")

        scheduler_list = {i:torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_list[i], "min", factor=self.training_dic["scheduler_factor"], patience=self.training_dic["scheduler_patience"], threshold=self.training_dic["scheduler_threshold"]) for i in range(self.training_dic["num_models"])}

        for model_idx in pbar:
            
            total_step = 0
            best_eval_loss = float("inf")
            best_correlation_loss = -float("inf")
            retrain_cnt = 1
            overfit_cnt = 0
            pred_mean = 0
            pred_std = 0
            best_points_cnt = 0

            self.model = self.models[model_idx]
            self.optimizer = self.optimizer_list[model_idx]
            self.scheduler = scheduler_list[model_idx]

            for epoch in self.training_dic["epochs"]:
                self.model.train()
                epoch_training_loss = 0
                batch_num = 0
                step_cnt = 0
                
                for data in self.train_loader:
                    step_cnt += 1
                    if "extra_feature" in self.training_dic and self.training_dic["extra_feature"]:
                        data_batch = torch.cat([data[0], data[-1]], dim=-1).to(self.device, non_blocking=True, dtype=data[0].dtype)
                    else:
                        data_batch = data[0].to(self.device, non_blocking=True)
                    if model_idx == 0:
                        diff = torch.zeros_like(data[1], device=self.device)
                    else:
                        diff = torch.zeros_like(data[1], device=self.device)
                        with torch.no_grad():
                            for i in range(model_idx-1):
                                diff += torch.squeeze(self.models[i].forward(data_batch))
                    target_batch = data[1].to(self.device, non_blocking=True) - diff
                    weight = torch.ones_like(target_batch)
                    weight += data[2].to(self.device, non_blocking=True)
                    self.optimizer.zero_grad()
                    pred = torch.squeeze(self.model.forward(data_batch), dim=-1)
                    if self.training_dic["weight_essential"]:
                        weight += torch.max(F.sigmoid(8 * (target_batch - mean - self.training_dic["weight_pos"] * std)), F.sigmoid(8 * (pred - pred_mean - self.training_dic["weight_pos"] * pred_std))) * self.training_dic["weight_pow_over"]
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
                        reg_loss = torch.mean(reg_loss)
                        train_loss = reg_loss
                    if self.training_dic["enable_l1"]:
                        train_loss += self.training_dic["l1_ratio"] * self.l1_regularization()
                    train_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    epoch_training_loss += train_loss.item()
                    batch_num += 1

                    if step_cnt == self.training_dic["eval_steps"]:
                        step_cnt = 0

                        train_loss = epoch_training_loss / batch_num
                        result[model_idx]["train_loss"].append(train_loss)

                        eval_loss = 0
                        mse_loss = 0
                        correlation_loss = 0

                        total_step += 1

                        for name, param in self.model.named_parameters():
                            total_norm = 0
                            if param.grad is not None and "spline_linear" in name:
                                param_norm = param.grad.data.norm(2)
                                total_norm += param_norm.item() ** 2
                                total_norm = total_norm ** (1./2)
                                writer.add_scalar(f"Gradients\Model_{model_idx}\{name}_grad", total_norm, total_step)

                        self.model.eval()
                        total_pred = [torch.tensor([], device=self.device)]
                        total_target = [torch.tensor([], device=self.device)]

                        with torch.no_grad():
                            for data in self.eval_loader:
                                if "extra_feature" in self.training_dic and self.training_dic["extra_feature"]:
                                    data_batch = torch.cat([data[0], data[-1]], dim=-1).to(self.device, non_blocking=True, dtype=data[0].dtype)
                                else:
                                    data_batch = data[0].to(self.device, non_blocking=True)
                                if model_idx == 0:
                                    diff = torch.zeros_like(data[1], device=self.device)
                                else:
                                    diff = torch.zeros_like(data[1], device=self.device)
                                    with torch.no_grad():
                                        for i in range(model_idx):
                                            diff += torch.squeeze(self.models[i].forward(data_batch))
                                target_batch = data[1].to(self.device, non_blocking=True) - diff
                                weight = torch.ones_like(target_batch)
                                weight += data[2].to(self.device, non_blocking=True)
                                pred = torch.squeeze(self.model.forward(data_batch), dim=-1)
                                new_mse_loss = nn.MSELoss()(pred, target_batch).item()
                                mse_loss += new_mse_loss
                                if self.training_dic["weight_essential"]:
                                    weight+= F.sigmoid(8 * (pred- pred_mean - self.training_dic["mask_pos"] * pred_std)) * self.training_dic["weight_pow_under"]
                                if self.training_dic["loss_fn"] == "CCC" or self.training_dic["loss_fn"] == "Correlation":
                                    new_correlation_loss = self.loss_fn_eval(pred, target_batch, weight).item()
                                    correlation_loss += new_correlation_loss
                                    eval_loss += self.correlation_ratio * new_correlation_loss + (1-self.correlation_ratio) * new_mse_loss
                                else:
                                    eval_loss += new_mse_loss
                                    correlation_loss += self.correlation_loss(pred, target_batch).item()
                                if not total_pred:
                                    total_pred = [pred[:,-1].detach()]
                                    total_target = [target_batch[:,-1].detach()]
                                else:
                                    total_pred.append(pred[:,-1].detach())
                                    total_target.append(target_batch[:,-1].detach())

                        self.model.train()

                        eval_loss /= len(self.eval_loader)
                        mse_loss /= len(self.eval_loader)
                        correlation_loss /= len(self.eval_loader)
                        total_pred = torch.cat(total_pred, dim=0)
                        total_target = torch.cat(total_target, dim=0)
                        pred_mean = torch.mean(total_pred)
                        pred_std = torch.std(total_pred)
                        result[model_idx]["good_points"].append(torch.sum((total_pred > pred_mean + self.training_dic["mask_pos"] * pred_std) & (total_target > mean + self.training_dic["mask_pos"] * std)).item())

                        if not result[model_idx]["eval_loss"] or (mse_loss - 5 * correlation_loss < best_eval_loss):
                            best_eval_loss = mse_loss - 5 * correlation_loss
                            torch.save(self.model.state_dict(), self.model_state_path_list[model_idx])
                            torch.save(self.optimizer.state_dict(), self.optimizer_path_list[model_idx])

                        if result[model_idx]["eval_loss"] and eval_loss > min(result[model_idx]["eval_loss"]) and total_step >= self.training_dic["overfit_threshold"]:
                            overfit_cnt += 1
                        else:
                            overfit_cnt = 0

                        result[model_idx]["eval_loss"].append(eval_loss)

                        pbar.set_description("Model :%d|Epoch :%d|Train Loss: %.2e|Evaluation Loss: %.2e|MSE Loss: %.2e|Correlation: %.2e|LR: %.2e" % (model_idx+1, epoch+1, train_loss, eval_loss, mse_loss, correlation_loss, self.optimizer_list[model_idx].param_groups[0]["lr"]))

                        result[model_idx]["eval_mse_loss"].append(mse_loss)
                        result[model_idx]["eval_correlation_loss"].append(correlation_loss)
                        result[model_idx]["LR"].append(self.optimizer.param_groups[0]["lr"])

                        if self.training_dic["retrain"] and overfit_cnt >= self.training_dic["overfit_patience"]:
                            overfit_cnt = 0
                            self.training_dic["retrain_lr"] = max(self.optimizer.param_groups[0]["lr"]/2, self.training_dic["retrain_lr"])
                            self.model.load_state_dict(torch.load(self.model_state_path_list[model_idx], map_location=self.device))
                            self.optimizer.load_state_dict(torch.load(self.optimizer_path_list[model_idx]))
                            for param_group in self.optimizer.param_groups:
                                param_group["lr"] = self.training_dic["retrain_lr"]
                            del self.scheduler
                            gc.collect()
                            torch.cuda.empty_cache()
                            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, "min", factor=self.training_dic["scheduler_factor"], threshold=self.training_dic["scheduler_threshold"])
                            self.training_dic["retrain_lr"] = max(self.training_dic["retrain_lr"]*self.training_dic["retrain_lr_factor"], self.training_dic["scheduler_threshold"])
                        else:
                            if self.training_dic["loss_fn"] == "CCC" or self.training_dic["loss_fn"] == "Correlation":
                                self.scheduler.step(mse_loss - 5 * correlation_loss)
                            else:
                                self.scheduler.step(eval_loss)
                        
                        df = pd.DataFrame(result[model_idx])
                        df.to_csv(self.log_path_list[model_idx], index=False)


            self.model.load_state_dict(torch.load(self.model_state_path_list[model_idx], map_location=self.device))
            self.models[model_idx] = self.model

    def regressor_train(self, num_models):
        for data in self.eval_loader:
            if "extra_feature" in self.training_dic and self.training_dic["extra_feature"]:
                data_batch = torch.cat([data[0], data[-1]], dim=1).to(self.device, non_blocking=True, dtype=data[0].dtype)
            else:                    
                data_batch = data[0].to(self.device, non_blocking=True)
            target_batch = data[1].to(self.device, non_blocking=True)
            regressor_input = []
            for i in range(num_models):
                self.models[i].eval()
                regressor_input.append(self.models[i].forward(data_batch).view(1, self.training_dic["bptt"], -1))
            if num_models != len(self.models):
                for i in range(len(self.models) - num_models):
                    regressor_input.append(torch.zeros_like(regressor_input[0]))
            regressor_input = torch.cat(regressor_input, dim=0).transpose(2,0)
            pred = self.regressor.forward(regressor_input)
            loss = self.reg_fn(pred, target_batch)

    def regressor_predict(self, num_models, data_batch):
        regressor_input = []
        for i in range(num_models):
            self.models[i].eval()
            regressor_input.append(self.models[i].forward(data_batch).view(1, self.training_dic["bptt"], -1))
        if num_models != len(self.models):
            for i in range(len(self.models) - num_models):
                regressor_input.append(torch.zeros_like(regressor_input[0]))
        regressor_input = torch.cat(regressor_input, dim=0).transpose(2,0)
        return self.regressor.forward(regressor_input)

    def l1_regularization(self):
        l1_loss = sum(param.abs().sum() for param in self.model.parameters())
        return l1_loss

    def find_end_points(self, mask, horizon=1):
        mask_cumsum = mask.cumsum(axis=0)
        mask_upper = mask_cumsum[self.seq_len-1:,:]
        mask_lower = np.pad(mask_cumsum[:-self.seq_len,:], ((1,0),(0,0)), "constant", constant_values=0)
        cum_sum = mask_upper - mask_lower
        condition_mask = list(np.where(cum_sum == self.seq_len))
        condition_mask[0] += self.seq_len - 1
        print(condition_mask[0].shape)
        return np.transpose(condition_mask)

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

        
    def create_raw_features(self, cfg, model_name, reader):

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

class Regressor(nn.Module):
    def __init__(
        input_dim,
        device = 'cpu',
    ):
        super().__init__()
        self.input_dim = input_dim
        self.regression_layer = nn.Linear(input_dim, 1, device=device)
        self.device = device

    def forward(self, input):
        return self.regression_layer(input)