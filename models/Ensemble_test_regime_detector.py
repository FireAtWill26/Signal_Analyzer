from contextlib import redirect_stderr
import numpy as np
from pathlib import Path
from os.path import isfile, join, dirname
from os import listdir
from scipy.stats import rankdata
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
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
from collections import Counter, defaultdict, deque
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import train_test_split
from multiprocessing.shared_memory import SharedMemory
from lion_pytorch import Lion
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from scipy.stats import skew, kurtosis
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn, Task
from rich.console import Console

from prometheus.utils.utils import WeightedCorrNp, calc_fcst_weighted_ic, weighted_mse_wrap, weighted_mse_eval_wrap, filter_eligible_data, filter_eligible_data_v2, operations_by_group
from prometheus.utils.torch_utils import select_gpu_with_minimum_memory, early_stopping_func, WarmupLR, DecayingCosineWarmRestarts, SharedMemDataset, SharedMemSeqDataset, SharedMemSeqDatasetDaily
from prometheus.ops import create_op_process
from prometheus.modelpool.basemodel import BaseModel, create_model
from prometheus.utils.registry_factory import TRAINING_REGISTRY, MODEL_REGISTRY
from prometheus.utils.speedup_package import calculate_stock_mean_3d_parallel
from prometheus.utils.pearson import niocorr
from prometheus.utils.spearman import niocorr_spearman
from prometheus.modelpool.Adversarial import Adversarial
from prometheus.modelpool.fastkan import *
from prometheus.utils.data_processing import RollingStatistics
from prometheus.riskmodel.barra.factor_dict import *
from prometheus.utils.weight_calculation import calculate_weights_cython, calculate_turnover_cython
from prometheus.utils.corr import *
from prometheus.utils.muon import Muon, MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam
from prometheus.utils.date_info_extraction import *
from prometheus.utils.util_funcs import *
from prometheus.utils.cache_reader import CacheReader
from prometheus.label_engineering.base_labels import *

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



@TRAINING_REGISTRY.register('Ensemble_test_multiple_labels')
class training_test(BaseModel):
    def __init__(self, *args, **kwargs):
        # super().__init__(*args, **kwargs)
        pass

    def fit(self, cfg, indices, train_output, sample_stocks_date, sample_stocks_seccode, model_name, lock=None, *args, **kwargs):

        self.model_path = self.training_dic["model_path"]
        
        super().fit_helper(cfg, indices, train_output, sample_stocks_date=sample_stocks_date, model_name=model_name, *args, **kwargs)

        train_input_buffer = SharedMemory(model_name)
        self.train_input = np.ndarray((indices[0].shape[0], indices[1].shape[0], indices[1].shape[1]), dtype=np.float32, buffer=train_input_buffer.buf)
        self.train_output = train_output
        self.indices = indices
        self.model_name = model_name
        self.pid = os.getpid()

        if self.training_dic.get("custom_labels", False):
            fit_config, extension_config, projection_config = build_pca_config(cfg.label.model_params)
            self.train_output = pca_transform_target(self.train_output, fit_returns="dsrt", reader=self.r, model_name=model_name, fit_start_date=self.start_date, fit_end_date=self.current_date, target_horizon=self.training_dic.get("horizon", 1), fit_config=fit_config, extension_config=extension_config, projection_config=projection_config)

        Path.mkdir(Path(self.training_dic["model_path"]) / model_name, exist_ok=True, parents=True)

        model_path_name = (self.training_dic["model_path"]).split("/")[-1]

        pit_tidx = model_name.split("_")
        pit_tidx = pit_tidx[1] + "_" + pit_tidx[2]
        tidx = int(model_name.split("_")[-1])

        if "ah_buckets" not in self.training_dic:
            self.training_dic["ah_buckets"] = 0
        
        if "ensemble_tidx" in self.training_dic:
            if tidx not in self.training_dic["ensemble_tidx"]:
                self.training_dic["num_buckets"] = 1
                self.training_dic["ah_buckets"] = 0
                self.training_dic["training_ratio"] = 0.99
                
        if isinstance(self.training_dic["num_buckets"], int):
            self.num_buckets = self.training_dic["num_buckets"]
            self.buckets_info = deepcopy(self.training_dic["num_buckets"])
        else:
            self.num_buckets = sum(self.training_dic["num_buckets"])
            self.buckets_info = deepcopy(self.training_dic["num_buckets"])
        
        if cfg.model.type == "Trident":
            self.actual_buckets = self.num_buckets
            self.num_buckets = 1

        if self.num_buckets == 1:
            self.training_dic["ah_buckets"] = 0
            self.training_dic["training_ratio"] = 0.99

        if "saved_alpha_path" in self.training_dic:
            self.saved_alphas_loc = Path(self.training_dic["saved_alpha_path"]) / ("saved_train_alphas_" + pit_tidx + "_processed.npy")
        else:
            self.saved_alphas_loc = Path("/dfs/data/ksim/automation/20240116/readcache_new/data") / model_path_name / ("saved_train_alphas_" + pit_tidx + "_processed.npy")

        self.num_labels = len(self.training_dic.get("blocks_sep", [["SH", "SZ", "gem", "star"]]))

        self.num_buckets_loc_list = [(Path(self.training_dic["model_path"]) / model_name / f'num_buckets_{i}.npy') for i in range(self.num_labels)]
        self.feature_buckets_loc_list = [(Path(self.training_dic["model_path"]) / model_name / f'feature_bucket_{i}.npy') for i in range(self.num_labels)]
        self.feature_list_loc_list = [(Path(self.training_dic["model_path"]) / model_name / f'feature_list_{i}.npy') for i in range(self.num_labels)]
        self.num_bags_loc_list = [(Path(self.training_dic["model_path"]) / model_name / f'num_bags_{i}.npy') for i in range(self.num_labels)]
        self.statistic_loc_list = [(Path(self.training_dic["model_path"]) / model_name / f'statistic_{i}.npy') for i in range(self.num_labels)]
        self.model_stats_loc_list = [(Path(self.training_dic["model_path"]) / model_name / f'model_stats_{i}.npy') for i in range(self.num_labels)]
        self.model_info_loc = Path(self.training_dic["model_path"]) / model_name / f'model_info.txt'
        self.stat_loc = Path(self.training_dic["model_path"]) / model_name / f'stat.npy'
        self.finish_flag_loc = Path(self.training_dic["model_path"]) / model_name / f'finished.npy'
        self.used_alphas_loc = Path(self.training_dic["model_path"]) / model_name / f'used_alphas.npy'
        
        begin_tidx = self.training_dic.get("tidx_list", [0, int(tidx)])[0]
        if tidx == begin_tidx:
            self.alpha_indices_loc = Path(self.training_dic["model_path"]) / model_name / f'alpha_indices.npy'
        else:
            base_model_name = "_".join(model_name.split("_")[:-1]) + "_" + str(begin_tidx)
            self.alpha_indices_loc = Path(self.training_dic["model_path"]) / base_model_name / f'alpha_indices.npy'

        self.model_path_list = [[Path(self.training_dic["model_path"]) / model_name / f'final_model_{j}_{i}.pt' for i in range(self.num_buckets)] for j in range(self.num_labels)]
        self.model_nd_path_list = [[Path(self.training_dic["model_path"]) / model_name / f'final_model_nd_{j}_{i}.pt' for i in range(self.num_buckets)] for j in range(self.num_labels)]
        self.optimizer_path_list = [[Path(self.training_dic["model_path"]) / model_name / f'final_optimizer_{j}_{i}.pt' for i in range(self.num_buckets)] for j in range(self.num_labels)]
        self.optimizer_nd_path_list = [[Path(self.training_dic["model_path"]) / model_name / f'final_optimizer_nd_{j}_{i}.pt' for i in range(self.num_buckets)] for j in range(self.num_labels)]
        self.dict_path_list = [[Path(self.training_dic["model_path"]) / model_name / f'training_dict_{j}_{i}.pt' for i in range(self.num_buckets)] for j in range(self.num_labels)]
        self.log_path_list = [[Path(self.training_dic["model_path"]) / model_name / f'training_log_{j}_{i}.csv' for i in range(self.num_buckets)] for j in range(self.num_labels)]
        if "finetune" in self.training_dic and self.training_dic["finetune"]:
            self.log_path_finetune = Path(self.training_dic["model_path"]) / model_name / 'finetune_log.csv'
        self.summary_path = Path("/dfs/data/tensorBoard") / cfg.train_basics["model_path_name"] / model_name
        self.regressor_path_list = [(Path(self.training_dic["model_path"]) / model_name / f'final_regressor_{i}.pt') for i in range(self.num_labels)]
        self.regressor_optimizer_path_list = [(Path(self.training_dic["model_path"]) / model_name / f'final_regressor_optimizer_{i}.pt') for i in range(self.num_labels)]

        if "feature_select" in self.training_dic and self.training_dic["feature_select"]:
            self.training_dic["features_delete"] = self.feature_selection(cfg)
        
        num_workers = max(4, min(16, torch.cuda.device_count() * 4))  # 根据GPU数量动态调整
        num_workers //= cfg.train_basics["no_of_process"]
        print(f"the number of workers is {num_workers}")

        self.label_handling()

        self.feature_buckets_list = []
        self.feature_select_list = []
        
        self.tidx_incr_learning = self.training_dic.get("tidx_incr_learning", False)
        if self.tidx_incr_learning and tidx == begin_tidx:
            np.save(self.alpha_indices_loc, indices[0])
        if self.tidx_incr_learning and tidx != begin_tidx:
            tidx_list = self.training_dic.get("tidx_list", [0, int(tidx)])
            tidx_idx = -1
            if self.training_dic.get("base_model", "default") == "begin_tidx":
                tidx_idx = 0
            else:
                for i in range(len(tidx_list)):
                    if tidx_list[i] == int(tidx):
                        tidx_idx = i - 1
                        break
            base_model_name = "_".join(model_name.split("_")[:-1])+ "_" + str(tidx_list[tidx_idx])
            base_pit_tidx = model_name.split("_")
            base_pit_tidx = base_pit_tidx[1] + "_" + str(tidx_list[tidx_idx])
            if "saved_alpha_path" in self.training_dic:
                self.saved_alphas_base_loc = Path(self.training_dic["saved_alpha_path"]) / ("saved_train_alphas_" + base_pit_tidx + "_processed.npy")
            else:
                self.saved_alphas_base_loc = Path("/dfs/data/ksim/automation/20240116/readcache_new/data") / model_path_name / ("saved_train_alphas_" + base_pit_tidx + "_processed.npy")
            saved_alphas = np.load(self.saved_alphas_base_loc)
            np.save(self.saved_alphas_loc, saved_alphas)
            # self.saved_alphas = np.array([alpha[:6] for alpha in saved_alphas])
            self.model_base_path_list = [[Path(self.training_dic["model_path"]) / base_model_name / f'final_model_{j}_{i}.pt' for i in range(self.num_buckets)] for j in range(self.num_labels)]
            self.optimizer_base_path_list = [[Path(self.training_dic["model_path"]) / base_model_name / f'final_optimizer_nd_{j}_{i}.pt' for i in range(self.num_buckets)] for j in range(self.num_labels)]
            self.indices[0] = np.load(self.alpha_indices_loc)
            if isinstance(self.training_dic["num_buckets"], int) and self.training_dic["num_buckets"]==1:
                self.feature_buckets_base_list = [[self.indices[0]]]
                self.feature_list_base_list = [[np.concatenate([np.ones(np.sum(self.indices[0])), np.ones(self.feature_matrix.shape[0])]).astype(bool) for j in range(self.num_labels)]]
            else:
                self.feature_buckets_base_list = [(np.load(Path(self.training_dic["model_path"]) / base_model_name / f'feature_bucket_{i}.npy')) for i in range(self.num_labels)]
                self.feature_list_base_list = [(np.load(Path(self.training_dic["model_path"]) / base_model_name / f'feature_list_{i}.npy')) for i in range(self.num_labels)]
            for i in range(self.num_labels):
                self.feature_buckets_list.append(self.feature_buckets_base_list[i])
                self.feature_select_list.append(self.feature_list_base_list[i])
                np.save(self.feature_buckets_loc_list[i], self.feature_buckets_list[i])
                np.save(self.feature_list_loc_list[i], self.feature_select_list[i])
            clean_input_and_sample_mask(self.train_input, self.indices[0], self.indices[1])
            self.training_dic["epochs"] = 3
        else:
            for label_idx in range(self.num_labels):
                feature_buckets, feature_list = self.feature_bucketing(cfg, label_idx)
                self.feature_buckets_list.append(deepcopy(feature_buckets))
                self.feature_select_list.append(deepcopy(feature_list))

        # import ipdb; ipdb.set_trace()

        self.cfg_list = [[deepcopy(cfg) for _ in range(self.num_buckets)] for j in range(self.num_labels)]

        if self.training_dic["ah_buckets"] >= 0:
            for j in range(self.num_labels):
                for i in range(self.training_dic["ah_buckets"]):
                    self.cfg_list[j][-1-i].training.model_params["avoid_horizon"] = True
        else:
            for j in range(self.num_labels):
                for i in range(-self.training_dic["ah_buckets"]):
                    self.cfg_list[j][i].training.model_params["avoid_horizon"] = True

        days, stocks = self.indices[1].shape
        feature_num = self.indices[0].sum()

        if self.training_dic.get("validation_by_return", False) or "ret" in self.training_dic.get("loss_fn") or self.training_dic.get("cross_section", False):
            if self.training_dic["eval_mode"] not in ["by_date", "recent", "period"]:
                self.training_dic["eval_mode"] = "by_date"
            self.training_dic["eval_size"] = min(self.training_dic["eval_size"], 0.2)
            for i in range(days):
                if self.indices[1][i].sum() < 500:
                    self.indices[1][i] = False

        if not "eval_mode" in self.training_dic:
            self.training_dic["eval_mode"] = "random"
        if "training_ratio" not in self.training_dic:
            self.training_dic["training_ratio"] = 0.8
        training_days = math.ceil(days*self.training_dic["training_ratio"])
        rows, cols = np.where(indices[1][:training_days,:])
        
        if self.training_dic["eval_mode"] == "recent":
            train_days = math.ceil(training_days * (1 - self.training_dic["eval_size"]))
            train_indice = np.zeros(indices[1].shape).astype(bool)
            pos_holder = np.zeros(indices[1].shape).astype(bool)
            for i in range(train_days):
                train_indice[i] = indices[1][i]
                pos_holder[i] = indices[1][i]
                if self.training_dic.get("avoid_horizon", False):
                    for i in range(self.training_dic["horizon"]):
                        pos_holder[train_days + i] = indices[1][i]
            eval_indice = indices[1] & ~pos_holder
        elif self.training_dic["eval_mode"] == "period":
            if "eval_period" not in self.training_dic:
                self.eval_period = 10
            else:
                self.eval_period = self.training_dic["eval_period"]
            period_len = training_days // self.training_dic["eval_period"]
            eval_days = math.floor(period_len * self.training_dic["eval_size"])
            eval_indice = np.zeros(indices[1].shape).astype(bool)
            pos_holder = np.zeros(indices[1].shape).astype(bool)
            for i in range(self.eval_period):
                for j in range(period_len * i - eval_days, period_len * i):
                    eval_indice[j] = indices[1][j]
                if self.training_dic.get("avoid_horizon", False):
                    for l in range(max(0, period_len*i-eval_days-self.training_dic["horizon"]//2), min(training_days, period_len*i+self.training_dic["horizon"]//2)):
                        pos_holder[l] = indices[1][l]
                else:
                    for l in range(period_len * i - eval_days, period_len * i):
                        pos_holder[l] = indices[1][l]            
            train_indice = indices[1] & ~pos_holder
        elif self.training_dic["eval_mode"] == "by_date":
            all_dates = np.unique(rows)
            _, eval_dates = train_test_split(range(len(all_dates)), test_size=self.training_dic["eval_size"], random_state=0, shuffle=True)
            eval_indice = np.zeros(indices[1].shape).astype(bool)
            pos_holder = np.zeros(indices[1].shape).astype(bool)
            row_filter = all_dates[eval_dates]
            for i in range(indices[1].shape[0]):
                if i in row_filter:
                    eval_indice[i] = indices[1][i]
                    if self.training_dic.get("avoid_horizon", False):
                        for j in range(max(0, i-self.training_dic["horizon"]), min(days, i+self.training_dic["horizon"])):
                            pos_holder[j] = indices[1][j]
                    else:
                        pos_holder[i] = indices[1][i]
            train_indice = indices[1] & ~pos_holder
        else:
            _, eval_idx = train_test_split(range(len(rows)), test_size=self.training_dic["eval_size"], random_state=0, shuffle=True)
            pos_holder = np.zeros(indices[1].shape).astype(bool)
            eval_indice = np.zeros(indices[1].shape).astype(bool)
            for i in eval_idx:
                eval_indice[rows[i]][cols[i]] = True
                if self.training_dic.get("avoid_horizon", False):
                    for j in range(max(0, rows[i]-self.training_dic["horizon"]//2), min(days, rows[i]+self.training_dic["horizon"]//2)):
                        pos_holder[j][cols[i]] = True
                else:
                    pos_holder[rows[i]][cols[i]] = True
            train_indice = indices[1] & ~pos_holder

        self.output_dim = 1
        if len(train_output.shape) > 2:
            self.output_dim = train_output.shape[0]
        if "date_weight" in self.training_dic and self.training_dic["date_weight"]:
            date_weight = torch.arange(days)
            # if self.output_dim > 1:
            #     date_weight = date_weight.unsqueeze(0).expand(self.output_dim, days)
            date_weight = torch.sigmoid(date_weight-(days-20*self.training_dic["date_weight_month"])) * self.training_dic["date_weight_pow"]
        else:
            date_weight = torch.zeros(days)
            # if self.output_dim > 1:
            #     date_weight = date_weight.unsqueeze(0).expand(self.output_dim, days)


        train_extra_info = {"features": self.feature_matrix}
        if "ret" in self.training_dic.get("loss_fn") or self.training_dic.get("validation_by_return", False):
            train_extra_info["ret_ratio"] = self.ret_ratio
        else:
            train_extra_info["ret_ratio"] = torch.zeros(1, days, stocks)
        if self.training_dic.get("down_weight_limited", False):
            train_extra_info["limited"] = self.limited
        else:
            train_extra_info["limited"] = torch.zeros(1, days, stocks)
        if self.training_dic.get("reduce_low_cap", False):
            train_extra_info["reduction_cap"] = self.reduction_cap

        if "ret" in self.training_dic.get("loss_fn") or self.training_dic.get("cross_section", False):
            self.train_dataset_list = [SharedMemSeqDatasetDaily(model_name, self.train_input.shape, self.train_input.dtype, self.indices[0], train_indice & self.stocks_choice[i], self.train_output, 1, feature_matrix=train_extra_info, time_as_batch=True) for i in range(self.num_labels)]
            self.train_loader_list = [DataLoader(self.train_dataset_list[i], batch_size=1, shuffle=True, drop_last=True, prefetch_factor=4, pin_memory=True, num_workers=num_workers, persistent_workers=True, collate_fn=lambda x:x[0]) for i in range(self.num_labels)]
        else:
            self.train_dataset_list = [SharedMemDataset(model_name, self.train_input.shape, self.train_input.dtype, self.indices[0], train_indice & self.stocks_choice[i], self.train_output, date_weight,feature_matrix=train_extra_info) for i in range(self.num_labels)]
            self.train_loader_list = [DataLoader(self.train_dataset_list[i], batch_size=self.training_dic["batch_size"], shuffle=True, drop_last=True, prefetch_factor=4, pin_memory=True, num_workers=num_workers, persistent_workers=True) for i in range(self.num_labels)]

        if self.training_dic.get("validation_by_return", False) or self.training_dic.get("cross_section", False):
            self.eval_dataset_list = [SharedMemSeqDatasetDaily(model_name, self.train_input.shape, self.train_input.dtype, self.indices[0], eval_indice & self.stocks_choice[i], self.train_output, 1, feature_matrix=train_extra_info, time_as_batch=True) for i in range(self.num_labels)]
            self.eval_loader_list = [DataLoader(self.eval_dataset_list[i], batch_size=1, shuffle=False, drop_last=False, prefetch_factor=4, pin_memory=True, num_workers=num_workers, persistent_workers=True, collate_fn=lambda x:x[0]) for i in range(self.num_labels)]
        else:
            self.eval_dataset_list = [SharedMemDataset(model_name, self.train_input.shape, self.train_input.dtype, self.indices[0], eval_indice & self.stocks_choice[i], self.train_output, date_weight,feature_matrix=train_extra_info) for i in range(self.num_labels)]
            self.eval_loader_list = [DataLoader(self.eval_dataset_list[i], batch_size=self.training_dic["batch_size"]*8, shuffle=False, drop_last=True, prefetch_factor=4, pin_memory=True, num_workers=num_workers, persistent_workers=True) for i in range(self.num_labels)]

        self.batch_date = self.training_dic.get("batch_date", 1)
        if (self.training_dic.get("cross_section", False) or "ret" in self.training_dic.get("loss_fn")) and self.batch_date != 1:
            self.train_loader_list = [ExteriorLoader(self.batch_date, self.train_loader_list[i]) for i in range(self.num_labels)]

        if ("finetune" in self.training_dic and self.training_dic["finetune"]) or ("discrimination" in self.training_dic and self.training_dic["discrimination"]):
            # recent_indices = (np.arange(days) > (days - (20 * self.training_dic["finetune_month"]))).reshape(-1,1)
            # recent_indices = np.repeat(recent_indices, stocks, axis=1)

            train_days, eval_days = train_test_split(range((training_days - (20 * self.training_dic["finetune_month"])), training_days), test_size=self.training_dic["eval_size"], random_state=0, shuffle=True)

            finetune_train_indices = np.zeros(indices[1].shape).astype(bool)
            for i in train_days:
                finetune_train_indices[i] = indices[1][i]
            finetune_eval_indices = np.zeros(indices[1].shape).astype(bool)
            for i in eval_days:
                finetune_eval_indices[i] = indices[1][i]

            finetune_train_dataset_list = [SharedMemDataset(model_name, self.train_input.shape, self.train_input.dtype, self.indices[0], finetune_train_indices & self.stocks_choice[i], self.train_output,np.ones(days),feature_matrix=train_extra_info) for i in range(self.num_labels)]
            finetune_eval_dataset_list = [SharedMemDataset(model_name, self.train_input.shape, self.train_input.dtype, self.indices[0], finetune_eval_indices & self.stocks_choice[i], self.train_output,np.ones(days),feature_matrix=train_extra_info) for i in range(self.num_labels)]

            self.finetune_train_loader_list = [DataLoader(finetune_train_dataset_list[i], batch_size=self.training_dic["batch_size"], shuffle=True, drop_last=True, prefetch_factor=4, pin_memory=True, num_workers=num_workers, persistent_workers=True) for i in range(self.num_labels)]
            self.finetune_eval_loader_list = [DataLoader(finetune_eval_dataset_list[i], batch_size=self.training_dic["batch_size"], shuffle=False, drop_last=True, prefetch_factor=4, pin_memory=True, num_workers=num_workers, persistent_workers=True) for i in range(self.num_labels)]

        rows_reg, col_reg = np.where(indices[1][training_days:,:])

        if self.num_buckets != 1:
            train_idx, eval_idx = train_test_split(range(len(rows_reg)), test_size=self.training_dic["eval_size"], random_state=0, shuffle=True)

            regressor_train_indices = np.zeros(indices[1].shape).astype(bool)
            for i in train_idx:
                regressor_train_indices[training_days+rows_reg[i]][col_reg[i]] = True
            regressor_eval_indices = np.zeros(indices[1].shape).astype(bool)
            for i in train_idx:
                regressor_eval_indices[training_days+rows_reg[i]][col_reg[i]] = True
            regressor_total_indices = regressor_train_indices | regressor_eval_indices

            regressor_train_dataset_list = [SharedMemDataset(model_name, self.train_input.shape, self.train_input.dtype, self.indices[0], regressor_train_indices & self.stocks_choice[i], self.train_output,None,feature_matrix=train_extra_info) for i in range(self.num_labels)]
            regressor_eval_dataset_list = [SharedMemDataset(model_name, self.train_input.shape, self.train_input.dtype, self.indices[0], regressor_eval_indices & self.stocks_choice[i], self.train_output,None,feature_matrix=train_extra_info) for i in range(self.num_labels)]
            regressor_total_dataset_list = [SharedMemDataset(model_name, self.train_input.shape, self.train_input.dtype, self.indices[0], regressor_total_indices & self.stocks_choice[i], self.train_output,None,feature_matrix=train_extra_info) for i in range(self.num_labels)]

            self.regressor_train_loader_list = [DataLoader(regressor_train_dataset_list[i], batch_size=self.training_dic["batch_size"]*8, shuffle=True, drop_last=True, prefetch_factor=4, pin_memory=True, num_workers=num_workers, persistent_workers=True) for i in range(self.num_labels)]
            self.regressor_eval_loader_list = [DataLoader(regressor_eval_dataset_list[i], batch_size=self.training_dic["batch_size"]*8, shuffle=False, drop_last=True, prefetch_factor=4, pin_memory=True, num_workers=num_workers, persistent_workers=True) for i in range(self.num_labels)]
            self.regressor_total_loader_list = [DataLoader(regressor_total_dataset_list[i], batch_size=self.training_dic["batch_size"]*8, shuffle=False, drop_last=True, prefetch_factor=4, pin_memory=True, num_workers=num_workers, persistent_workers=True) for i in range(self.num_labels)]

        for i in range(self.num_labels):
            print(f"len of train_data: {len(self.train_dataset_list[i])}, len of eval_data: {len(self.eval_dataset_list[i])}, len of train_loader: {len(self.train_loader_list[i])}, len of eval_loader: {len(self.eval_loader_list[i])}")

        if lock is not None:
            with lock:
                time.sleep(30)
                super().gpu_helper(*args, **kwargs)

        if not "denormalizer" in self.training_dic:
            self.training_dic["denormalizer"] = False
        if not "step_num" in self.training_dic:
            self.training_dic["step_num"] = 30
        for i in range(self.num_buckets):
            for j in range(self.num_labels):
                self.cfg_list[j][i].training.model_params["eval_steps"] = len(self.train_loader_list[j]) // self.training_dic["step_num"]
        
        # self.training_dic = self.cfg_list[0].training.model_params
        # self.training_dic["lr"] *= self.num_buckets
        # self.training_dic["retrain_lr"] *= self.num_buckets
        
        if cfg.model.type == "Trident":
            for label_idx in range(self.num_labels):
                self.cfg_list[label_idx][0].model.model_params["trident_heads"] = deepcopy(self.feature_select_list[label_idx])
                self.feature_select_list[label_idx][0] = (self.feature_select_list[label_idx][0] * 0 + 1).astype(bool)
                self.feature_buckets_list[label_idx][0] = self.indices[0]

        self.input_dim = np.sum(self.indices[0]) + self.feature_matrix.shape[0]
        self.input_dim_list = [[np.sum(self.feature_buckets_list[j][i]) for i in range(self.num_buckets)] for j in range(self.num_labels)]
        self.original_dim = deepcopy(self.input_dim_list)
        if "extra_feature" in self.training_dic and self.training_dic["extra_feature"]:
            self.input_dim_list = [[input_dim+self.feature_matrix.shape[0] for input_dim in self.input_dim_list[k]] for k in range(self.num_labels)]
        # self.modified_dim = deepcopy(self.input_dim_list)

        # import ipdb; ipdb.set_trace()

        # add the feature information corresponding to deeper layer injection to the config
        if "feature_insert_pos" in self.cfg_list[0][0].model.model_params and self.cfg_list[0][0].model.model_params["feature_insert_pos"] is not None:
            feature_insert_pos = self.cfg_list[0][0].model.model_params["feature_insert_pos"]
            if isinstance(feature_insert_pos, int):
                for i in range(self.num_buckets):
                    for k in range(self.num_labels):
                        self.cfg_list[k][i].model.model_params["feature_insert"] = np.zeros(self.input_dim_list[k][i], dtype=bool)
                        if "insert_type" in self.training_dic and self.training_dic["insert_type"] == "all":
                            for j in range(self.feature_matrix.shape[0]):
                                self.cfg_list[k][i].model.model_params["feature_insert"][-1-j] = True
                        else:
                            for j in range(self.insert_num):
                                self.cfg_list[k][i].model.model_params["feature_insert"][-1-j] = True
            elif isinstance(feature_insert_pos, list):
                if "insert_features" not in self.training_dic:
                    self.training_dic["insert_features"] = [["alpha.", "alpha_"]]
                insert_features = self.find_by_featurename(self.training_dic["insert_features"])
                for i in range(self.num_buckets):
                    for k in range(self.num_labels):
                        feature_insert = np.zeros(self.input_dim_list[k][i], dtype=bool)
                        if "insert_type" in self.training_dic and self.training_dic["insert_type"] == "all":
                            for j in range(self.feature_matrix.shape[0]):
                                feature_insert[-1-j] = True
                        else:
                            for j in range(4):
                                feature_insert[-1-j] = True
                        self.cfg_list[k][i].model.model_params["feature_insert"] = [insert_feature[self.feature_select_list[k][i]] for insert_feature in insert_features] + [feature_insert]
                        to_delete = []
                        for j in range(len(self.cfg_list[k][i].model.model_params["feature_insert"])):
                            if self.cfg_list[k][i].model.model_params["feature_insert"][j].sum() == 0 or self.cfg_list[k][i].model.model_params["feature_insert"][j].sum() == self.feature_select_list[k][i].sum() - self.feature_matrix.shape[0] + 1:
                                to_delete.append(j)
                        # import ipdb; ipdb.set_trace()
                        if to_delete:
                            self.cfg_list[k][i].model.model_params["feature_insert"] = [self.cfg_list[k][i].model.model_params["feature_insert"][j] for j in range(len(self.cfg_list[k][i].model.model_params["feature_insert"])) if j not in to_delete]
                            self.cfg_list[k][i].model.model_params["feature_insert_pos"] = [self.cfg_list[k][i].model.model_params["feature_insert_pos"][j] for j in range(len(self.cfg_list[k][i].model.model_params["feature_insert_pos"])) if j not in to_delete]
                        self.cfg_list[k][i].model.model_params["feature_insert"] = np.array(self.cfg_list[k][i].model.model_params["feature_insert"])
            else:
                raise ValueError("feature_insert_pos must be int or list")
            # for i in range(self.num_buckets):
            #     for k in range(self.num_labels):
            #         if "feature_insert" in self.cfg_list[k][i].model.model_params and self.cfg_list[k][i].model.model_params["feature_insert"].shape[-1] == self.input_dim_list[k][i]:
            #             self.input_dim_list[k][i] -= self.cfg_list[k][i].model.model_params["feature_insert"].sum()
        
        # add the features information corresponding to time choice to the config
        if self.training_dic.get("time_choice_block", False):
            for i in range(self.num_labels):
                for j in range(self.num_buckets):
                    if self.cfg_list[i][j].model.model_params["time_choice_method"] == "barra_estimator":
                        self.cfg_list[i][j].model.model_params["barra_dim"] = self.barra_dim
                    self.cfg_list[i][j].model.model_params["time_choice_feature"] = np.zeros(self.input_dim_list[i][j], dtype=bool)
                    for k in range(self.time_choice_feature_num):
                        self.cfg_list[i][j].model.model_params["time_choice_feature"][self.original_dim[i][j]+k] = True
        else:
            for i in range(self.num_labels):
                for j in range(self.num_buckets):
                    self.cfg_list[i][j].model.model_params["time_choice_pos"] = None
        for i in range(self.num_labels):
            for j in range(self.num_buckets):
                if "feature_insert" in self.cfg_list[i][j].model.model_params and self.cfg_list[i][j].model.model_params["feature_insert"] is not None and self.cfg_list[i][j].model.model_params["feature_insert"].shape[-1] == self.input_dim_list[i][j]:
                    self.input_dim_list[i][j] -= self.cfg_list[i][j].model.model_params["feature_insert"].sum()
                if self.training_dic.get("time_choice_block", False):
                    self.input_dim_list[i][j] -= self.cfg_list[i][j].model.model_params["time_choice_feature"].sum()
        
        if self.training_dic["denormalizer"]:
            self.input_dim += 2
        for i in range(self.num_labels):
            for j in range(self.num_buckets):
                self.cfg_list[i][j].model.model_params["layers_hidden"][-1] = self.output_dim
        if "FastKAN" in cfg.model.type:
            for i in range(len(cfg.model.model_params["layers_hidden"])-self.training_dic["plain_layer"]):
                for j in range(self.num_buckets):
                    for k in range(self.num_labels):
                        self.cfg_list[k][j].model.model_params["layers_hidden"][i] = math.ceil(self.input_dim_list[k][j] * self.cfg_list[k][j].model.model_params["layers_hidden"][i])
        # elif self.cfg.model.type == "KANUNet":
        #     for i in range(self.training_dic["mult_layer"]):
        #         self.cfg.model.model_params["layers_hidden"][i] *= self.input_dim
        for i in range(len(self.cfg_list)):
            for j in range(len(self.cfg_list[i])):
                self.cfg_list[i][j].model.model_params["device"] = self.device
                self.cfg_list[i][j].training.model_params["feature_chosen"] = self.feature_select_list[i][j]
        self.model_list = [[create_model(self.cfg_list[j][i].model) for i in range(self.num_buckets)] for j in range(self.num_labels)]

        if self.training_dic["regressor_type"] == "nn.linear":
            self.regressor_list = [Regressor(self.num_buckets, device=self.device) for _ in range(self.num_labels)]
        elif self.training_dic["regressor_type"] == "fastkan":
            self.regressor_list = [KANRegressor(self.num_buckets, device=self.device) for _ in range(self.num_labels)]

        if "regressor_alpha" not in self.training_dic:
            self.training_dic["regressor_alpha"] = 1
        
        if "positive" not in self.training_dic:
            self.training_dic["positive"] = True

        self.rank_ratio = self.training_dic.get("rank_ratio", 0)
        
        if "reg_intercept" not in self.training_dic:
            self.reg_intercept = False
        else:
            self.reg_intercept = self.training_dic["reg_intercept"]

        if "regression_method" in self.training_dic and self.training_dic["regression_method"] == "Lasso":
            self.linear_regressor_list = [Lasso(alpha=self.training_dic["regressor_alpha"], positive=self.training_dic["positive"], fit_intercept=self.reg_intercept) for _ in range(self.num_labels)]
        elif "regression_method" in self.training_dic and self.training_dic["regression_method"] == "Ridge":
            self.linear_regressor_list = [Ridge(alpha=self.training_dic["regressor_alpha"], positive=self.training_dic["positive"], fit_intercept=self.reg_intercept) for _ in range(self.num_labels)]

        else:
            self.linear_regressor_list = [LinearRegression(positive=self.training_dic["positive"], fit_intercept=self.reg_intercept) for _ in range(self.num_labels)]

        if "reg_min" not in self.training_dic:
            self.training_dic["reg_min"] = 0.1
        if "reg_max" not in self.training_dic:
            self.training_dic["reg_max"] = 1
        # self.linear_regressor = RidgeCV(np.arange(0.01, 1.01, 0.1), fit_intercept=False)

        if Path(self.summary_path).exists():
            shutil.rmtree(self.summary_path)

        Path.mkdir(self.summary_path, exist_ok=True, parents=True)

        writer = SummaryWriter(self.summary_path / "run")
        # writer.add_graph(self.model_list[0], torch.ones(self.input_dim_list[0], device=self.cfg_list[0].model.model_params["device"]))

        result = {}
        for i in range(self.num_buckets):
            for j in range(self.num_labels):
                result[(j,i)] = {"train_loss": [], "eval_loss": [], "eval_mse_loss": [], "eval_correlation_loss": [], "eval_correlation_mask": [],  "good_points": [], "huber_delta": []}
                if self.training_dic.get("validation_by_return", False) or self.training_dic.get("cross_section", False):
                    if self.training_dic.get("validation_by_return", False):
                        result[(j,i)]["return"] = []
                    result[(j,i)].pop("good_points")
                    result[(j,i)].pop("huber_delta")
                    result[(j,i)].pop("eval_correlation_mask")

        if "discrimination" in self.training_dic and self.training_dic["discrimination"]:
            for i in range(self.num_buckets):
                for j in range(self.num_labels):
                    result[(j,i)]["adv_loss"] = []

            self.adversarial_list = [[Adversarial(self.cfg_list[j][i], self.device, self.input_dim_list[j][i]) for i in range(self.num_buckets)] for j in range(self.num_labels)]

        # if only_eval set to true in config, call the quick_evaluation function which will not train the model but only do the evaluation.
        if self.training_dic.get("breakpoint_continue", False) and Path(self.finish_flag_loc).exists() and Path(self.used_alphas_loc).exists():
            used_alphas = np.load(self.used_alphas_loc)
            saved_alphas = np.load(self.saved_alphas_loc)
            if len(used_alphas) == len(saved_alphas) and np.all(used_alphas==saved_alphas):
                self.training_dic["regression_only"] = True

        for i in range(self.num_buckets):
            for j in range(self.num_labels):
                torch.save(self.cfg_list[j][i], self.dict_path_list[j][i])

        self.vertical_norm = self.training_dic["vertical_norm"]

        tau = self.training_dic.get("tau", 0.5)

        if self.training_dic["loss_fn"] == "CCC":
            self.loss_fn = self.loss_fn_eval = lambda x, y, w: self.ccc(x,y,w)
        elif self.training_dic["loss_fn"] == "Correlation" or "ret" in self.training_dic["loss_fn"]:
            self.loss_fn = self.loss_fn_eval = lambda x, y, w: self.correlation_loss(x,y,w)
        elif self.training_dic["loss_fn"] == "RankCorrelation":
            self.loss_fn = self.loss_fn_eval = lambda x, y, w: calc_rank_correlation(x,y,tau,w)
        else:
            self.loss_fn = self.loss_fn_eval = lambda x, y, w: 0
        if self.training_dic["reg_loss"] == "Huber":
            self.reg_fn = nn.HuberLoss(reduction="none", delta=self.training_dic["huber_delta"])
        else:
            self.reg_fn = nn.MSELoss(reduction="none")

        self.correlation_ratio = self.training_dic["correlation_ratio"]
        lr = self.training_dic['lr']
        self.weight_decay = self.training_dic["weight_decay"]

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

        if "regression_only" not in self.training_dic or not self.training_dic["regression_only"]:

            for i in range(self.num_buckets):
                for j in range(self.num_labels):
                    ratio = (self.indices[0].sum()/self.feature_buckets_list[j][i].sum()) ** (0.5)
                    self.cfg_list[j][i].training.model_params["lr"] *= ratio
                    self.cfg_list[j][i].training.model_params["retrain_lr"] *= ratio
                    if self.tidx_incr_learning and tidx != begin_tidx:
                        incr_lr_ratio = self.training_dic.get("incr_lr_ratio", 50)
                        incr_rt_lr_ratio = self.training_dic.get("incr_rt_lr_ratio", incr_lr_ratio/5)
                        self.cfg_list[j][i].training.model_params["lr"] /= incr_lr_ratio
                        self.cfg_list[j][i].training.model_params["retrain_lr"] /= incr_rt_lr_ratio

            parameter_groups = {}

            model_info = []

            for i in range(self.num_buckets):
                for j in range(self.num_labels):
                    if "stem_component" in self.cfg_list[j][i].model.model_params and self.cfg_list[j][i].model.model_params["stem_component"]:

                        parameter_groups[(j,i)] = []
                        parameter_groups[(j,i)].append([p for n, p in self.model_list[j][i].named_parameters() if "stem_component" not in n])
                        parameter_groups[(j,i)].append([p for n, p in self.model_list[j][i].named_parameters() if "stem_component" in n])
                        if not parameter_groups[(j,i)][-1]:
                            parameter_groups[(j,i)].pop()
                    elif "separate_lr" in self.training_dic and self.training_dic["separate_lr"] == "spline":
                        parameter_groups[(j,i)] = []
                        parameter_groups[(j,i)].append([p for n, p in self.model_list[j][i].named_parameters() if "spline_linear" in n])
                        parameter_groups[(j,i)].append([p for n, p in self.model_list[j][i].named_parameters() if "spline_linear" not in n])
                        if not parameter_groups[(j,i)][-1]:
                            parameter_groups[(j,i)].pop()
                    elif "separate_lr" in self.training_dic and self.training_dic["separate_lr"] == "embedding":
                        parameter_groups[(j,i)] = []
                        parameter_groups[(j,i)].append([p for n, p in self.model_list[j][i].named_parameters() if ("layers.0" not in n and "layers.1" not in n)])
                        parameter_groups[(j,i)].append([p for n, p in self.model_list[j][i].named_parameters() if ("layers.0" in n or "layers.1" in n)])
                        if not parameter_groups[(j,i)][-1]:
                            parameter_groups[(j,i)].pop()
                    else:
                        parameter_groups[(j,i)] = [[p for n, p in self.model_list[j][i].named_parameters()]]

            if self.training_dic.get("tidx_incr_learning", False) and tidx != begin_tidx:
                for i in range(self.num_labels):
                    for j in range(self.num_buckets):
                        for name, params in self.model_list[i][j].named_parameters():
                            if len(name.split(".")) > 1 and name.split(".")[1].isdigit() and int(name.split(".")[1]) < self.training_dic.get("static_layers", 3):
                                params.requires_grad = False

            if self.training_dic['opt'] == "Adam":
                self.optimizer_list = [[torch.optim.Adam([{
                    "params": parameter_groups[(j,i)][k],
                    "lr": self.cfg_list[j][i].training.model_params["lr"],
                    "weight_decay": self.weight_decay
                } for k in range(len(parameter_groups[(j,i)]))]) for i in range(self.num_buckets)] for j in range(self.num_labels)]
            elif self.training_dic['opt'] == "AdamW":
                self.optimizer_list = [[torch.optim.AdamW([{
                    "params": parameter_groups[(j,i)][k],
                    "lr": self.cfg_list[j][i].training.model_params["lr"],
                    "weight_decay": self.weight_decay
                } for k in range(len(parameter_groups[(j,i)]))]) for i in range(self.num_buckets)] for j in range(self.num_labels)]
            elif self.training_dic['opt'] == "RMSprop":
                self.optimizer_list = [[torch.optim.RMSprop([{
                    "params": parameter_groups[(j,i)][k],
                    "lr": self.cfg_list[j][i].training.model_params["lr"],
                    "weight_decay": self.weight_decay
                } for k in range(len(parameter_groups[(j,i)]))]) for i in range(self.num_buckets)] for j in range(self.num_labels)]
            elif self.training_dic["opt"] == "Lion":
                self.optimizer_list = [[Lion([{
                    "params": parameter_groups[(j,i)][k],
                    "lr": self.cfg_list[j][i].training.model_params["lr"]/10,
                    "weight_decay": self.weight_decay
                } for k in range(len(parameter_groups[(j,i)]))]) for i in range(self.num_buckets)] for j in range(self.num_labels)]
            elif self.training_dic["opt"] == "Muon":
                muon_params_list = [[[p for p in self.model_list[j][i].parameters() if p.ndim >= 2] for i in range(self.num_buckets)] for j in range(self.num_labels)]
                adamw_params_list = [[[p for p in self.model_list[j][i].parameters() if p.ndim < 2] for i in range(self.num_buckets)] for j in range(self.num_labels)]
                self.optimizer_list = [[SingleDeviceMuonWithAuxAdam([dict(params=muon_params_list[j][i], use_muon=True, lr=self.cfg_list[j][i].training.model_params["lr"], weight_decay=self.weight_decay), dict(params=adamw_params_list[j][i], use_muon=False, lr=self.cfg_list[j][i].training.model_params["lr"]/10, weight_decay=self.weight_decay)]) for i in range(self.num_buckets)] for j in range(self.num_labels)]
            elif self.training_dic['opt'] == "SGD":
                self.optimizer_list = [[torch.optim.SGD([{
                    "params": parameter_groups[(j,i)][k],
                    "lr": self.cfg_list[j][i].training.model_params["lr"],
                    "weight_decay": self.weight_decay
                } for k in range(len(parameter_groups[(j,i)]))]) for i in range(self.num_buckets)] for j in range(self.num_labels)]
            # elif self.training_dic['opt'] == "LBFGS":
            #     self.optimizer_list = [torch.optim.LBFGS(self.model_list[i].parameters(), lr=lr, history_size=10, tolerance_grad=1e-32, tolerance_change=1e-32) for i in range(self.num_buckets)]
            else:
                raise ValueError("opt must be Adam, AdamW, SGD, LBFGS, or RMSprop.")

            if "ret" not in self.training_dic.get("loss_fn") and not self.training_dic.get("cross_section", False):
                self.batch_date = 1
            
            # if self.training_dic["opt"] == "LBFGS":
            #     def closure():
            #         self.optimizer.zero_grad()
            #         pred = self.model.forward(data_batch)
            #         train_loss = self.loss_fn(pred, target_batch)
            #         train_loss.backward()
            #         return train_loss

            scheduler_list = [[torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_list[j][i], "min", factor=self.training_dic["scheduler_factor"], patience=self.training_dic["scheduler_patience"], threshold=self.training_dic["scheduler_threshold"]) for i in range(self.num_buckets)] for j in range(self.num_labels)]

            self.retrain_lr_list = [[self.cfg_list[j][i].training.model_params["retrain_lr"] for i in range(self.num_buckets)] for j in range(self.num_labels)]

            self.retrain_lr = self.training_dic["retrain_lr"]

            self.mean = []
            self.std = []

            for label_idx in range(self.num_labels):

                train_y = train_output[..., indices[1] & self.stocks_choice[label_idx]]

                var = np.var(train_y, axis=-1)
                std = np.std(train_y, axis=-1)
                mean = np.mean(train_y, axis=-1)
                skewness = skew(train_y, axis=-1)
                kurt = kurtosis(train_y, axis=-1)

                np.save(self.statistic_loc_list[label_idx], [mean, std])

            
                print(f"target stats| Mean: {mean}| Variance: {var}| Skewness: {skewness}| Kurtosis: {kurt}")

                if len(train_y.shape) > 1:
                    mean = torch.from_numpy(mean).to(self.device)
                    std = torch.from_numpy(std).to(self.device)

                self.mean.append(mean)
                self.std.append(std)

                for model_idx in range(self.num_buckets):
                    pbar = tqdm(range(self.training_dic["epochs"]), desc='Training', ncols=200)

                    self.model = self.model_list[label_idx][model_idx]
                    self.optimizer = self.optimizer_list[label_idx][model_idx]
                    if self.tidx_incr_learning and tidx != begin_tidx:
                        self.model.load_state_dict(torch.load(self.model_base_path_list[label_idx][model_idx]))
                        # self.optimizer.load_state_dict(torch.load(self.optimizer_base_path_list[label_idx][model_idx]))
                    if len(self.optimizer.param_groups) > 1:
                        if "separate_lr" in self.training_dic and self.training_dic["separate_lr"] in ["spline", "embedding"]:
                            self.optimizer.param_groups[1]["lr"] /= 10
                        elif "stem_component" in self.cfg_list[label_idx][model_idx].model.model_params["stem_component"] and self.cfg_list[label_idx][model_idx].model.model_params["stem_component"]:

                            self.optimizer.param_groups[1]["lr"] *= 5

                    self.scheduler = scheduler_list[label_idx][model_idx]

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

                    if self.training_dic.get("validation_by_return", False):
                        highest_ret = -float("inf")

                    self.training_dic["retrain_lr"] = self.retrain_lr

                    if "ret" not in self.training_dic.get("loss_fn"):                    
                        loss_type = "Train Loss"
                    else:
                        loss_type = "Train Ret"

                    for epoch in pbar:
                        
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
                            if "ret" in self.training_dic.get("loss_fn") or self.training_dic.get("cross_section", False):
                                dis_pred = torch.tensor(dis_pred, device=self.device)
                                dis_data = torch.tensor(dis_data, device=self.device)
                                dis_target = torch.tensor(dis_target, device=self.device)


                        for data in self.train_loader_list[label_idx]:

                            # import ipdb; ipdb.set_trace()
                            step_cnt += 1
                            # 提前批量移动数据到设备
                            if self.batch_date == 1:
                                extra_dict = data[-1]
                                if self.training_dic.get("reduce_low_cap", False):
                                    if self.output_dim > 1:
                                        reduction_cap = extra_dict["reduction_cap"]
                                    else:
                                        reduction_cap = extra_dict["reduction_cap"].squeeze()
                                extra_feat = extra_dict["features"].squeeze()
                                # import ipdb; ipdb.set_trace()
                                # print(extra_feat.shape)
                                if "extra_feature" in self.training_dic and self.training_dic["extra_feature"]:
                                    data_batch = torch.cat([data[0].squeeze(), extra_feat], dim=1).to(self.device, non_blocking=True, dtype=data[0].dtype)
                                else:
                                    data_batch = data[0].squeeze().to(self.device, non_blocking=True)
                                if "ret" in self.training_dic.get("loss_fn"):
                                    if self.output_dim > 1:
                                        ret_ratio = extra_dict["ret_ratio"].to(self.device, non_blocking=True, dtype=data[1].dtype)
                                    else:
                                        ret_ratio = extra_dict["ret_ratio"].squeeze().to(self.device, non_blocking=True, dtype=data[1].dtype)
                                limited = 1
                                if self.training_dic.get("down_weight_limited", False):
                                    if self.output_dim > 1:
                                        limited = extra_dict["limited"].to(self.device, non_blocking=True, dtype=data[1].dtype)
                                    else:
                                        limited = extra_dict["limited"].squeeze().to(self.device, non_blocking=True, dtype=data[1].dtype)
                                if self.training_dic["denormalizer"]:
                                    batch_mean = torch.unsqueeze(torch.mean(data_batch, dim=1), 1)
                                    batch_std = torch.unsqueeze(torch.std(data_batch, dim=1), 1)
                                    data_batch = torch.cat([data_batch, batch_mean, batch_std.pow(0.5)], dim=1).to(self.device, non_blocking=True)
                                if self.training_dic.get("reduce_low_cap", False):
                                    if self.training_dic.get("reduce_method", "addition") == "addition":
                                        target_batch = (data[1].squeeze() + reduction_cap).to(self.device, non_blocking=True, dtype=data[1].dtype)
                                    elif self.training_dic.get("reduce_method", "addition") == "multiplication":
                                        target_batch = (data[1].squeeze() * reduction_cap).to(self.device, non_blocking=True, dtype=data[1].dtype)
                                else:
                                    target_batch = data[1].squeeze().to(self.device, non_blocking=True)
                                if self.training_dic.get("residual_training", False):
                                    for prev_model in range(model_idx):
                                        target_batch -= torch.squeeze(self.model_list[label_idx][prev_model].forward(data_batch[:,self.feature_select_list[label_idx][prev_model]]))
                                if "horizontal_norm" in self.training_dic and self.training_dic["horizontal_norm"]:
                                    data_batch = F.normalize(data_batch, p=2.0, dim=0)

                                pred = torch.squeeze(self.model.forward(data_batch[:,self.feature_select_list[label_idx][model_idx]]))
                                if "ret" in self.training_dic.get("loss_fn") or self.training_dic.get("cross_section", False):
                                    weight = torch.ones_like(target_batch)
                                else:
                                    date_weight = data[2].to(self.device, non_blocking=True)
                                    if self.output_dim > 1:
                                        date_weight = date_weight.unsqueeze(1)
                                    weight = torch.ones_like(target_batch) + date_weight
                                if "ret" in self.training_dic.get("loss_fn") or self.training_dic.get("cross_section", False):
                                    pred = pred.detach().requires_grad_(True)
                                if self.training_dic["ignore_negative_target"]:
                                    mask = (target_batch <= self.training_dic["ignore_threshold"]) | (pred <= self.training_dic["ignore_threshold"])
                                    pred = pred[mask]
                                    target_batch = target_batch[mask]
                                if self.training_dic["weight_essential"]:
                                    if "weight_pos_up" in self.training_dic:
                                        upper_adjust_t = F.sigmoid(8 * (mean + self.training_dic["weight_pos_up"] * std - target_batch))
                                        upper_adjust_p = F.sigmoid(8 * (pred_mean + self.training_dic["weight_pos_up"] * pred_std - pred))
                                    else:
                                        upper_adjust_t = upper_adjust_p = 1
                                    weight += torch.max(F.sigmoid(8 * (target_batch - mean - self.training_dic["weight_pos"] * std)) * upper_adjust_t, F.sigmoid(8 * (pred - pred_mean - self.training_dic["weight_pos"] * pred_std)) * upper_adjust_p) * self.training_dic["weight_pow_over"] * limited
                                if self.training_dic["loss_fn"] == "CCC" or self.training_dic["loss_fn"] == "Correlation":
                                    correlation_loss = self.loss_fn(pred, target_batch, weight)
                                    reg_loss = self.reg_fn(pred, target_batch)
                                    if self.training_dic["weight_essential"]:
                                        reg_loss = (reg_loss * weight)
                                    reg_loss = reg_loss.mean()
                                    train_loss = -self.correlation_ratio * correlation_loss + (1 - self.correlation_ratio) * reg_loss
                                elif "ret" in self.training_dic.get("loss_fn"):
                                    r_loss = -ret_loss(pred, ret_ratio, torch.ones_like(pred, device=self.device))
                                    reg_loss = self.reg_fn(pred, target_batch)
                                    if self.training_dic["weight_essential"]:
                                        reg_loss = (reg_loss * weight)
                                    reg_loss = reg_loss.mean()
                                    train_loss = self.correlation_ratio * r_loss + (1 - self.correlation_ratio) * reg_loss
                                else:
                                    reg_loss = self.reg_fn(pred, target_batch)
                                    if self.training_dic["weight_essential"]:
                                        reg_loss = (reg_loss * weight)
                                    reg_loss = torch.mean(reg_loss)
                                    train_loss = reg_loss
                                if self.training_dic.get("pairwise_rank", False):
                                    train_loss += weighted_pairwise_rank_loss(pred, target_batch) * self.training_dic.get("pairwise_ratio", 0.1)
                                # assert torch.isnan(train_loss).sum() == 0, print(train_loss)
                                # scaler.scale(train_loss).backward()
                                total_pred = pred
                                total_data = data_batch
                                total_target = target_batch
                            else:
                                total_pred = []
                                total_data = []
                                total_target = []
                                train_loss = 0
                                for chunk in data:
                                    extra_dict = chunk[-1]
                                    extra_feat = extra_dict["features"].squeeze()
                                    if self.training_dic.get("reduce_low_cap", False):
                                        if self.output_dim > 1:
                                            reduction_cap = extra_dict["reduction_cap"]
                                        else:
                                            reduction_cap = extra_dict["reduction_cap"].squeeze()
                                    # import ipdb; ipdb.set_trace()
                                    # print(extra_feat.shape)
                                    if "extra_feature" in self.training_dic and self.training_dic["extra_feature"]:
                                        data_batch = torch.cat([chunk[0].squeeze(), extra_feat], dim=1).to(self.device, non_blocking=True, dtype=chunk[0].dtype)
                                    else:
                                        data_batch = chunk[0].squeeze().to(self.device, non_blocking=True)
                                    if "ret" in self.training_dic.get("loss_fn"):
                                        if self.output_dim > 1:
                                            ret_ratio = extra_dict["ret_ratio"].to(self.device, non_blocking=True, dtype=chunk[1].dtype)
                                        else:
                                            ret_ratio = extra_dict["ret_ratio"].squeeze().to(self.device, non_blocking=True, dtype=chunk[1].dtype)
                                    limited = 1
                                    if self.training_dic.get("down_weight_limited", False):
                                        if self.output_dim > 1:
                                            limited = extra_dict["limited"].to(self.device, non_blocking=True, dtype=chunk[1].dtype)
                                        else:
                                            limited = extra_dict["limited"].squeeze().to(self.device, non_blocking=True, dtype=chunk[1].dtype)
                                    if self.training_dic.get("reduce_low_cap", False):
                                        if self.training_dic.get("reduce_method", "addition") == "addition":
                                            target_batch = (chunk[1].squeeze() + reduction_cap).to(self.device, non_blocking=True, dtype=data[1].dtype)
                                        elif self.training_dic.get("reduce_method", "addition") == "multiplication":
                                            target_batch = (chunk[1].squeeze() * reduction_cap).to(self.device, non_blocking=True, dtype=data[1].dtype)
                                    else:
                                        target_batch = chunk[1].squeeze().to(self.device, non_blocking=True)
                                    if self.training_dic.get("residual_training", False):
                                        for prev_model in range(model_idx):
                                            target_batch -= torch.squeeze(self.model_list[label_idx][prev_model].forward(data_batch[:,self.feature_select_list[label_idx][prev_model]]))
                                    weight = 1
                                    pred = torch.squeeze(self.model.forward(data_batch[:,self.feature_select_list[label_idx][model_idx]]))
                                    pred = pred.detach().requires_grad_(True)
                                    total_pred.append(pred)
                                    total_data.append(data_batch)
                                    total_target.append(target_batch)
                                    if self.training_dic["weight_essential"]:
                                        if "weight_pos_up" in self.training_dic:
                                            upper_adjust_t = F.sigmoid(8 * (mean + self.training_dic["weight_pos_up"] * std - target_batch))
                                            upper_adjust_p = F.sigmoid(8 * (pred_mean + self.training_dic["weight_pos_up"] * pred_std - pred))
                                        else:
                                            upper_adjust_t = upper_adjust_p = 1
                                        weight += torch.max(F.sigmoid(8 * (target_batch - mean - self.training_dic["weight_pos"] * std)) * upper_adjust_t, F.sigmoid(8 * (pred - pred_mean - self.training_dic["weight_pos"] * pred_std)) * upper_adjust_p) * self.training_dic["weight_pow_over"] * limited
                                    if self.training_dic["loss_fn"] == "CCC" or self.training_dic["loss_fn"] == "Correlation":
                                        correlation_loss = self.loss_fn(pred, target_batch, weight)
                                        reg_loss = self.reg_fn(pred, target_batch)
                                        if self.training_dic["weight_essential"]:
                                            reg_loss = (reg_loss * weight)
                                        reg_loss = reg_loss.mean()
                                        train_loss += -self.correlation_ratio * correlation_loss + (1 - self.correlation_ratio) * reg_loss
                                    elif "ret" in self.training_dic.get("loss_fn"):
                                        r_loss = -ret_loss(pred, ret_ratio, torch.ones_like(pred, device=self.device))
                                        reg_loss = self.reg_fn(pred, target_batch)
                                        if self.training_dic["weight_essential"]:
                                            reg_loss = (reg_loss * weight)
                                        reg_loss = reg_loss.mean()
                                        train_loss += self.correlation_ratio * r_loss + (1 - self.correlation_ratio) * reg_loss
                                    else:
                                        reg_loss = self.reg_fn(pred, target_batch)
                                        if self.training_dic["weight_essential"]:
                                            reg_loss = (reg_loss * weight)
                                        reg_loss = torch.mean(reg_loss)
                                        train_loss += reg_loss
                                    if self.training_dic.get("pairwise_rank", False):
                                        train_loss += weighted_pairwise_rank_loss(pred, target_batch) * self.training_dic.get("pairwise_ratio", 0.1)
                                    # assert torch.isnan(train_loss).sum() == 0, print(train_loss)
                                    # scaler.scale(train_loss).backward()
                                total_data = torch.cat(total_data, dim=0)
                                total_target = torch.cat(total_target, dim=0)
                            loss = train_loss
                            if self.training_dic.get("time_choice_block", False) and not self.training_dic.get("cross_section", False) and self.model.time_choice_method == "moe_output":
                                router_weights = self.model.last_router_weights
                                weights_mean = router_weights.mean()
                                num_of_experts = self.model.num_experts
                                balance_loss = num_of_experts * (weights_mean - 1 / num_of_experts).mean()
                                loss += balance_loss * self.training_dic.get("balance_ratio", 0.01) + router_entropy_loss(router_weights) * self.training_dic.get("balance_ratio", 0.01) / 10
                            if self.training_dic["enable_l1"]:
                                loss += self.training_dic["l1_ratio"] * self.l1_regularization()
                            self.optimizer.zero_grad()
                            loss.backward()
                            if "ret" in self.training_dic.get("loss_fn") or self.training_dic.get("cross_section", False):
                                if self.batch_date != 1:
                                    intermediate_grad = torch.cat([pred.grad.detach().squeeze()/pred.shape[0]*self.training_dic["batch_size"] for pred in total_pred], dim=0)
                                    length = sum([pred.shape[0] for pred in total_pred])
                                else:
                                    intermediate_grad = pred.grad.detach().squeeze() / pred.shape[0] * self.training_dic["batch_size"]
                                    length = total_pred.shape[0]
                                for i in range(math.ceil(length / self.training_dic["batch_size"])):
                                    grad_chunk = intermediate_grad[self.training_dic["batch_size"]*i: self.training_dic["batch_size"]*(i+1)].squeeze()
                                    # print(data_batch[self.training_dic["batch_size"]*i: self.training_dic["batch_size"]*(i+1)].shape)
                                    y_chunk = torch.squeeze(self.model.forward(total_data[self.training_dic["batch_size"]*i: self.training_dic["batch_size"]*(i+1),self.feature_select_list[label_idx][model_idx]]))
                                    y_chunk.backward(gradient=grad_chunk)
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_norm)
                                self.optimizer.step()
                                if self.batch_date != 1:
                                    for pred in total_pred:
                                        pred.requires_grad_(False)
                                    total_pred = torch.cat(total_pred, dim=0)
                                else:
                                    pred.requires_grad_(False)
                            else:
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_norm)
                                self.optimizer.step()
                            if "discrimination" in self.training_dic and self.training_dic["discrimination"]:
                                if "ret" not in self.training_dic.get("loss_fn") and not self.training_dic.get("cross_section", False):
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
                                            loss_dis = self.adversarial_list[label_idx][model_idx].train_with_model(dis_pred, dis_data[:,self.feature_select_list[label_idx][model_idx]], dis_target)
                                            # del dis_pred, dis_data, dis_target
                                            # gc.collect()
                                            # torch.cuda.empty_cache()
                                            adv_loss += loss_dis
                                            dis_pred = []
                                            dis_data = []
                                            dis_target = []
                                            dis_cur_cnt = 0
                                    else:
                                        loss_dis = self.adversarial_list[label_idx][model_idx].train_with_model(pred, data_batch[:,self.feature_select_list[label_idx][model_idx]], target_batch)
                                        adv_loss += loss_dis
                                else:
                                    unfinished_len = total_pred.shape[0]
                                    pos = 0
                                    len_diff = cfg.dis_train.model_params["dis_target_size"] - dis_pred.shape[0]
                                    while unfinished_len > 0:
                                        unfinished_len -= len_diff
                                        dis_pred = torch.cat([dis_pred, total_pred[pos:len_diff+pos]], dim=0)
                                        dis_data = torch.cat([dis_data, total_data[pos:len_diff+pos,self.feature_select_list[label_idx][model_idx]]], dim=0)
                                        dis_target = torch.cat([dis_target, total_target[pos:len_diff+pos]], dim=0)
                                        pos=len_diff+pos
                                        if cfg.dis_train.model_params["dis_target_size"] == dis_pred.shape[0]:
                                            loss_dis = self.adversarial_list[label_idx][model_idx].train_with_model(dis_pred, dis_data,dis_target)
                                            len_diff = cfg.dis_train.model_params["dis_target_size"]
                                            dis_pred = torch.tensor([], device=self.device)
                                            dis_data = torch.tensor([], device=self.device)
                                            dis_target = torch.tensor([], device=self.device)
                            # assert torch.isnan(self.model.parameters()).sum() == 0, print(self.model.parameters())
                            epoch_training_loss += train_loss.item()
                            batch_num += 1

                            if step_cnt == self.cfg_list[label_idx][model_idx].training.model_params["eval_steps"]:
                                step_cnt = 0

                                train_loss = epoch_training_loss / batch_num
                                result[(label_idx,model_idx)]["train_loss"].append(train_loss)
                                if "discrimination" in self.training_dic and self.training_dic["discrimination"]:
                                    result[(label_idx,model_idx)]["adv_loss"].append(adv_loss/batch_num)

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
                                        writer.add_scalar(f"Model_{label_idx}_{model_idx}/{name}_grad", total_norm, total_step)
                                    # writer.add_histogram(f"{name}_grad", param.grad, total_step)

                                self.model.eval()
                                total_pred = [torch.tensor([]).to(self.device)]
                                total_target = [torch.tensor([]).to(self.device)]

                                if self.training_dic.get("validation_by_return", False) or self.training_dic.get("cross_section", False):
                                    with torch.no_grad():
                                        if self.training_dic.get("validation_by_return"):
                                            total_ret = 0
                                        for data in self.eval_loader_list[label_idx]:
                                            # import ipdb; ipdb.set_trace()
                                            extra_dict = data[-1]
                                            extra_feat = extra_dict["features"].squeeze()
                                            if self.training_dic.get("reduce_low_cap", False):
                                                if self.output_dim > 1:
                                                    reduction_cap = extra_dict["reduction_cap"].to(dtype=data[0].dtype)
                                                else:
                                                    reduction_cap = extra_dict["reduction_cap"].squeeze().to(dtype=data[0].dtype)
                                            limited = 1
                                            if self.training_dic.get("down_weight_limited", False):
                                                if self.output_dim > 1:
                                                    limited = extra_dict["limited"].to(self.device, non_blocking=True, dtype=data[0].dtype) * 2 - 1
                                                else:
                                                    limited = extra_dict["limited"].squeeze().to(self.device, non_blocking=True, dtype=data[0].dtype) * 2 - 1
                                            if "extra_feature" in self.training_dic and self.training_dic["extra_feature"]:
                                                # print(data[0].shape, extra_feat.shape)
                                                data_batch = torch.cat([data[0].squeeze(), extra_feat], dim=1).to(self.device, non_blocking=True, dtype=data[0].dtype)
                                            else:
                                                data_batch = data[0].squeeze().to(self.device, non_blocking=True)
                                            # import ipdb; ipdb.set_trace()
                                            target_batch = data[1].squeeze().to(self.device, non_blocking=True)
                                            if self.training_dic.get("residual_training", False):
                                                for prev_model in range(model_idx):
                                                    target_batch -= torch.squeeze(self.model_list[label_idx][prev_model].forward(data_batch[:,self.feature_select_list[label_idx][prev_model]]))
                                            pred = torch.squeeze(self.model.forward(data_batch[:,self.feature_select_list[label_idx][model_idx]]))
                                            if not total_pred:
                                                total_pred = [pred.detach()]
                                            else:
                                                total_pred.append(pred.detach()) 
                                            if self.training_dic.get("validation_by_return"):
                                                if self.output_dim > 1:
                                                    ret_ratio = extra_dict["ret_ratio"].to(self.device, non_blocking=True, dtype=data[1].dtype)
                                                else:
                                                    ret_ratio = extra_dict["ret_ratio"].squeeze().to(self.device, non_blocking=True, dtype=data[1].dtype)
                                                mse_loss += F.mse_loss(pred, target_batch).item()
                                                weight = torch.ones(target_batch.shape, device=target_batch.device)
                                                correlation_loss += self.loss_fn_eval(pred, target_batch,weight).item()
                                                pred -= pred.mean(dim=0)
                                                abs_sum = torch.sum(torch.abs(pred), dim=0)
                                                # print(abs_sum)
                                                total_ret += ((((pred * ret_ratio).sum(dim=0)) / abs_sum).sum()).item()
                                            else:
                                                weight = torch.ones_like(target_batch)
                                                if self.training_dic["ignore_negative_target"]:
                                                    mask = (target_batch <= self.training_dic["ignore_threshold"]) | (pred <= self.training_dic["ignore_threshold"])
                                                    pred = pred[mask]
                                                    target_batch = target_batch[mask]
                                                new_mse_loss = F.mse_loss(pred, target_batch).item()
                                                mse_loss += new_mse_loss
                                                if self.training_dic["weight_essential"]:
                                                    if "mask_pos_up" in self.training_dic:
                                                        upper_adjust_p = F.sigmoid(8 * (pred_mean + self.training_dic["mask_pos_up"] * pred_std - pred))
                                                    else:
                                                        upper_adjust_p = 1
                                                    # mask = target_batch >= mean + self.training_dic["weight_pos"] * std
                                                    # weight = torch.where(mask, 2 * torch.ones_like(target_batch), weight)
                                                    weight += F.sigmoid(8 * (pred - pred_mean - self.training_dic["mask_pos"] * pred_std)) * upper_adjust_p * self.training_dic["weight_pow_under"] * limited
                                                if self.training_dic["loss_fn"] == "CCC" or self.training_dic["loss_fn"] == "Correlation":
                                                    new_correlation_loss = self.loss_fn_eval(pred, target_batch, weight).item()
                                                    correlation_loss += new_correlation_loss
                                                    eval_loss += -self.correlation_ratio * new_correlation_loss + (1-self.correlation_ratio) * new_mse_loss
                                                # compute both of the combined loss and the mse loss
                                                else:
                                                    eval_loss += new_mse_loss
                                                    correlation_loss += self.correlation_loss(pred, target_batch).item()
                                    total_pred = torch.cat(total_pred, dim=0)
                                    pred_mean = total_pred.mean(dim=0)
                                    pred_std = total_pred.std(dim=0)
                                    self.model.train()
                                    if self.training_dic.get("validation_by_return"):
                                        eval_loss = (-self.correlation_ratio * correlation_loss + (1-self.correlation_ratio) * mse_loss) / len(self.eval_loader_list[label_idx])
                                    else:
                                        eval_loss /= len(self.eval_loader_list[label_idx])
                                    mse_loss /= len(self.eval_loader_list[label_idx])
                                    correlation_loss /= len(self.eval_loader_list[label_idx])
 
                                    if self.training_dic.get("validation_by_return"):
                                    # save the model weights if the evaluation loss is improved
                                        if not result[(label_idx,model_idx)]["return"] or total_ret + 5 * correlation_loss > highest_ret:
                                            # best_correlation_loss = correlation_loss
                                            # if eval_loss < best_eval_loss:
                                            highest_ret = total_ret + 5 * correlation_loss
                                            torch.save(self.model.state_dict(), self.model_path_list[label_idx][model_idx])
                                            torch.save(self.optimizer.state_dict(), self.optimizer_path_list[label_idx][model_idx])
                                            torch.save(self.optimizer.state_dict(), self.optimizer_nd_path_list[label_idx][model_idx])
                                            current_time_stamp = [epoch, total_step]

                                        if result[(label_idx,model_idx)]["return"] and total_ret < max(result[(label_idx,model_idx)]["return"]) + self.eps and total_step >= self.training_dic["overfit_threshold"]:
                                            overfit_cnt += 1
                                        else:
                                            overfit_cnt = 0


                                        pbar.set_description("Label :%d|Model :%d|Epoch :%d|%s: %.2e|Total Ret: %.2e|MSE Loss: %.2e|Correlation: %.2e|LR: %.2e|Retrain_LR: %.2e" % (label_idx+1, model_idx+1, epoch+1, loss_type, train_loss, total_ret, mse_loss, correlation_loss, self.optimizer.param_groups[0]["lr"], self.retrain_lr_list[label_idx][model_idx]))


                                        writer.add_scalar("Training_Stats_Model_%d/Train Loss" % model_idx, train_loss, total_step)
                                        writer.add_scalar("Training_Stats_Model_%d/Total Ret" % model_idx, total_ret, total_step)
                                        writer.add_scalar("Training_Stats_Model_%d/MSE Loss" % model_idx, mse_loss, total_step)
                                        writer.add_scalar("Training_Stats_Model_%d/Correlation" % model_idx, correlation_loss, total_step)
                                        writer.add_scalar("Training_Stats_Model_%d/Learning rate" % model_idx, self.optimizer.param_groups[0]["lr"], total_step)
                                        total_step += 1

                                        result[(label_idx,model_idx)]["eval_loss"].append(mse_loss - 5 * correlation_loss)
                                        result[(label_idx,model_idx)]["eval_mse_loss"].append(mse_loss)
                                        result[(label_idx,model_idx)]["eval_correlation_loss"].append(correlation_loss)
                                        result[(label_idx,model_idx)]["return"].append(total_ret)

                                    else:
                                        if not result[(label_idx,model_idx)]["eval_loss"] or (mse_loss - 5 * correlation_loss < best_eval_loss):
                                            # best_correlation_loss = correlation_loss
                                            # if eval_loss < best_eval_loss:
                                            best_eval_loss = mse_loss - 5 * correlation_loss
                                            torch.save(self.model.state_dict(), self.model_path_list[label_idx][model_idx])
                                            torch.save(self.optimizer.state_dict(), self.optimizer_path_list[label_idx][model_idx])
                                            torch.save(self.optimizer.state_dict(), self.optimizer_nd_path_list[label_idx][model_idx])
                                            # self.model.to(self.device)
                                            current_time_stamp = [epoch, total_step]

                                        if result[(label_idx,model_idx)]["eval_loss"] and mse_loss - 5 * correlation_loss > min(result[(label_idx,model_idx)]["eval_loss"]) - self.eps and total_step >= self.training_dic["overfit_threshold"]:
                                            overfit_cnt += 1
                                        else:
                                            overfit_cnt = 0

                                        result[(label_idx,model_idx)]["eval_loss"].append(mse_loss - 5 * correlation_loss)

                                        pbar.set_description("Label :%d|Model :%d|Epoch :%d|%s: %.2e|Evaluation Loss: %.2e|MSE Loss: %.2e|Correlation: %.2e|LR: %.2e|Retrain_LR: %.2e" % (label_idx+1, model_idx+1, epoch+1, loss_type, train_loss, eval_loss, mse_loss, correlation_loss, self.optimizer.param_groups[0]["lr"], self.retrain_lr_list[label_idx][model_idx]))

                                        writer.add_scalar("Training_Stats_Model_%d/Train Loss" % model_idx, train_loss, total_step)
                                        writer.add_scalar("Training_Stats_Model_%d/Evaluation Loss" % model_idx, eval_loss, total_step)
                                        writer.add_scalar("Training_Stats_Model_%d/MSE Loss" % model_idx, mse_loss, total_step)
                                        writer.add_scalar("Training_Stats_Model_%d/Correlation" % model_idx, correlation_loss, total_step)
                                        writer.add_scalar("Training_Stats_Model_%d/Learning rate" % model_idx, self.optimizer.param_groups[0]["lr"], total_step)
                                        total_step += 1



                                        result[(label_idx,model_idx)]["eval_mse_loss"].append(mse_loss)
                                        result[(label_idx,model_idx)]["eval_correlation_loss"].append(correlation_loss)

                                    if (self.training_dic["retrain"] and overfit_cnt >= self.training_dic["overfit_patience"]) or (self.optimizer.param_groups[0]["lr"] <= self.retrain_lr_list[label_idx][model_idx]/self.threshold_ratio):
                                        if self.optimizer.param_groups[0]["lr"] <= self.retrain_lr_list[label_idx][model_idx]/self.threshold_ratio:
                                            self.retrain_lr_list[label_idx][model_idx] *= self.training_dic["retrain_lr_factor"]
                                        skip_reset = False
                                        overfit_cnt = 0
                                        if self.optimizer.param_groups[0]["lr"]/2 >= self.retrain_lr_list[label_idx][model_idx]:
                                            self.retrain_lr_list[label_idx][model_idx] = self.optimizer.param_groups[0]["lr"]/2
                                            skip_reset = True
                                        if "jumpback" in self.training_dic and self.training_dic["jump_back"]:
                                            self.model.load_state_dict(torch.load(self.model_path_list[label_idx][model_idx], map_location=self.device))
                                            # self.optimizer.load_state_dict(torch.load(self.optimizer_path_list[model_idx]))
                                            self.optimizer_list[label_idx][model_idx].load_state_dict(torch.load(self.optimizer_path_list[label_idx][model_idx]))
                                        for i in range(len(self.optimizer_list[label_idx][model_idx].param_groups)):
                                            self.optimizer_list[label_idx][model_idx].param_groups[i]["lr"] = self.retrain_lr_list[label_idx][model_idx]
                                            if i > 0:
                                                if "separate_lr" in self.training_dic and self.training_dic["separate_lr"] in ["spline", "embedding"]:
                                                    self.optimizer_list[label_idx][model_idx].param_groups[i]["lr"] /= 10
                                                elif "stem_component" in self.cfg_list[label_idx][model_idx].model.model_params["stem_component"] and self.cfg_list[label_idx][model_idx].model.model_params["stem_component"]:
                                                    self.optimizer_list[label_idx][model_idx].param_groups[i]["lr"] *= 5
                                        del self.scheduler
                                        gc.collect()
                                        torch.cuda.empty_cache()
                                        retrain_cnt += 1
                                        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_list[label_idx][model_idx], "min", factor=self.training_dic["scheduler_factor"], patience=self.training_dic["scheduler_patience"], threshold=self.training_dic["scheduler_threshold"])
                                        if not skip_reset:
                                            self.retrain_lr_list[label_idx][model_idx] = max(self.retrain_lr_list[label_idx][model_idx]*self.training_dic["retrain_lr_factor"], self.training_dic["scheduler_threshold"])
                                    else:
                                        if self.training_dic.get("validation_by_return", False):
                                            self.scheduler.step(-(total_ret + 5 * correlation_loss))
                                        elif self.training_dic["loss_fn"] == "CCC" or self.training_dic["loss_fn"] == "Correlation":
                                            self.scheduler.step(mse_loss - 5 * correlation_loss)
                                        else:
                                            self.scheduler.step(eval_loss)
                                    df = pd.DataFrame(result[(label_idx,model_idx)])
                                    df.to_csv(self.log_path_list[label_idx][model_idx], index=False, header = True)
                                    if self.optimizer_list[label_idx][model_idx].param_groups[0]["lr"] == self.training_dic["scheduler_threshold"] and self.retrain_lr_list[label_idx][model_idx] == self.training_dic["scheduler_threshold"]:
                                        break
                                else:
                                    with torch.no_grad():
                                        for data in self.eval_loader_list[label_idx]:
                                            extra_dict = data[-1]
                                            if self.training_dic.get("reduce_low_cap", False):
                                                if self.output_dim > 1:
                                                    reduction_cap = extra_dict["reduction_cap"].to(dtype=data[0].dtype)
                                                else:
                                                    reduction_cap = extra_dict["reduction_cap"].squeeze().to(dtype=data[0].dtype)
                                            extra_feat = extra_dict["features"].squeeze()
                                            limited = 1
                                            if self.training_dic.get("down_weight_limited", False):
                                                if self.output_dim > 1:
                                                    limited = extra_dict["limited"].to(self.device, non_blocking=True, dtype=data[0].dtype) * 2 - 1
                                                else:
                                                    limited = extra_dict["limited"].squeeze().to(self.device, non_blocking=True, dtype=data[0].dtype) * 2 - 1
                                            if "extra_feature" in self.training_dic and self.training_dic["extra_feature"]:
                                                data_batch = torch.cat([data[0], extra_feat], dim=1).to(self.device, non_blocking=True, dtype=data[0].dtype)
                                            else:
                                                data_batch = data[0].to(self.device, non_blocking=True)
                                            if self.training_dic["denormalizer"]:
                                                batch_mean = torch.unsqueeze(torch.mean(data_batch, dim=1), 1)
                                                batch_std = torch.unsqueeze(torch.std(data_batch, dim=1), 1)
                                                data_batch = torch.cat([data_batch, batch_mean, batch_std.pow(0.5)], dim=1).to(self.device, non_blocking=True)
                                            if "horizontal_norm" in self.training_dic and self.training_dic["horizontal_norm"]:
                                                data_batch = F.normalize(data_batch, p=2.0, dim=1)
                                            if self.training_dic.get("reduce_low_cap", False):
                                                if self.training_dic.get("reduce_method", "addition") == "addition":
                                                    target_batch = (data[1] + reduction_cap).to(self.device, non_blocking=True, dtype=data[1].dtype)
                                                elif self.training_dic.get("reduce_method", "addition") == "multiplication":
                                                    target_batch = (data[1] * reduction_cap).to(self.device, non_blocking=True, dtype=data[1].dtype)
                                            else:
                                                target_batch = data[1].to(self.device, non_blocking=True)
                                                
                                            if self.training_dic.get("residual_training", False):
                                                for prev_model in range(model_idx):
                                                    target_batch -= torch.squeeze(self.model_list[label_idx][prev_model].forward(data_batch[:,self.feature_select_list[label_idx][prev_model]]))

                                            weight = 1 + data[2].to(self.device, non_blocking=True)
                                            pred = torch.squeeze(self.model.forward(data_batch[:,self.feature_select_list[label_idx][model_idx]]))
                                            if self.training_dic["ignore_negative_target"]:
                                                mask = (target_batch <= self.training_dic["ignore_threshold"]) | (pred <= self.training_dic["ignore_threshold"])
                                                pred = pred[mask]
                                                target_batch = target_batch[mask]
                                            new_mse_loss = F.mse_loss(pred, target_batch).item()
                                            mse_loss += new_mse_loss
                                            if self.training_dic["weight_essential"]:
                                                if "mask_pos_up" in self.training_dic:
                                                    upper_adjust_p = F.sigmoid(8 * (pred_mean + self.training_dic["mask_pos_up"] * pred_std - pred))
                                                else:
                                                    upper_adjust_p = 1
                                                # mask = target_batch >= mean + self.training_dic["weight_pos"] * std
                                                # weight = torch.where(mask, 2 * torch.ones_like(target_batch), weight)
                                                weight += F.sigmoid(8 * (pred - pred_mean - self.training_dic["mask_pos"] * pred_std)) * upper_adjust_p * self.training_dic["weight_pow_under"] * limited
                                            if self.training_dic["loss_fn"] == "CCC" or self.training_dic["loss_fn"] == "Correlation":
                                                new_correlation_loss = (1 - self.rank_ratio) * self.loss_fn_eval(pred, target_batch, weight).item() + self.rank_ratio * calc_rank_correlation(pred, target_batch, tau, weight).item()
                                                correlation_loss += new_correlation_loss
                                                eval_loss += -self.correlation_ratio * new_correlation_loss + (1-self.correlation_ratio) * new_mse_loss
                                            # compute both of the combined loss and the mse loss
                                            else:
                                                eval_loss += new_mse_loss
                                                correlation_loss += self.correlation_loss(pred, target_batch).item()
                                            if (self.training_dic["update_delta"] and not (epoch+1) % self.training_dic["update_period"]) or self.training_dic["eval_mask"]:
                                                if not total_pred:
                                                    total_pred = [pred]
                                                    total_target = [target_batch]
                                                else:
                                                    total_pred.append(pred)  
                                                    total_target.append(target_batch)
                                    
                                    self.model.train()

                                    eval_loss /= len(self.eval_loader_list[label_idx])
                                    mse_loss /= len(self.eval_loader_list[label_idx])
                                    correlation_loss /= len(self.eval_loader_list[label_idx])
                                    if (self.training_dic["update_delta"] and not (epoch+1) % self.training_dic["update_period"]) or self.training_dic["eval_mask"]:
                                        total_pred = torch.cat(total_pred,dim=0)
                                        total_target = torch.cat(total_target,dim=0)
                                    if self.training_dic["eval_mask"]:
                                        pred_mean = torch.mean(total_pred,dim=0)
                                        pred_std = torch.std(total_pred,dim=0)
                                        mask =  ((total_pred > (pred_mean + self.training_dic["mask_pos"] * pred_std))&(total_target < (mean + self.training_dic["mask_pos"] * std)))
                                        # |((total_pred < (pred_mean + self.training_dic["mask_pos"] * pred_std))&(total_target > (mean + self.training_dic["mask_pos"] * std)))
                                        total_pred_mask = total_pred[mask]
                                        total_target_mask = total_target[mask]
                                        if total_pred_mask.shape[0] == 0:
                                            correlation_mask = 0
                                        else:
                                            correlation_mask = (total_pred_mask - pred_mean - self.training_dic["mask_pos"] * pred_std).pow(2) * (total_target_mask - mean - self.training_dic["mask_pos"] * std) / pred_std.pow(2)
                                            correlation_mask = torch.nan_to_num(torch.sum(correlation_mask), nan=-1e5, neginf=-1e5).item()
                                    if self.output_dim > 1:
                                        result[(label_idx,model_idx)]["good_points"].append(0)
                                    else:
                                        result[(label_idx,model_idx)]["good_points"].append(torch.sum((total_pred > pred_mean + self.training_dic["mask_pos"] * pred_std)&(total_target > mean + self.training_dic["mask_pos"] * std)).item())


                                    # save the model weights if the evaluation loss is improved
                                    if not result[(label_idx,model_idx)]["eval_loss"] or (mse_loss - 5 * correlation_loss < best_eval_loss):
                                        # best_correlation_loss = correlation_loss
                                        # if eval_loss < best_eval_loss:
                                        best_eval_loss = mse_loss - 5 * correlation_loss
                                        torch.save(self.model.state_dict(), self.model_path_list[label_idx][model_idx])
                                        torch.save(self.optimizer.state_dict(), self.optimizer_path_list[label_idx][model_idx])
                                        torch.save(self.optimizer.state_dict(), self.optimizer_nd_path_list[label_idx][model_idx])
                                        # self.model.to(self.device)
                                        current_time_stamp = [epoch, total_step]
                                    if self.training_dic["check_mask"] and self.training_dic["eval_mask"]:
                                        if not result[(label_idx,model_idx)]["eval_correlation_mask"] or correlation_mask + 10 * correlation_loss> best_correlation_mask_loss:
                                            best_correlation_mask_loss = correlation_mask + 10 * correlation_loss
                                            torch.save(self.model.state_dict(), self.model_mask_path_list[model_idx])
                                            torch.save(self.optimizer.state_dict(), self.optimizer_mask_path_list[model_idx])
                                            torch.save(self.optimizer.state_dict(), self.optimizer_nd_path_list[label_idx][model_idx])
                                            # self.model.to(self.device)
                                            current_time_stamp = [epoch, total_step]
                                    if self.training_dic["check_most"] and (not result[(label_idx,model_idx)]["good_points"] or result[(label_idx,model_idx)]["good_points"][-1] > best_points_cnt):
                                        best_points_cnt = result[(label_idx,model_idx)]["good_points"][-1]
                                        torch.save(self.model.state_dict(), self.model_most_path_list[(label_idx,model_idx)])
                                        torch.save(self.optimizer.state_dict(), self.optimizer_most_path_list[(label_idx,model_idx)])
                                        torch.save(self.optimizer.state_dict(), self.optimizer_nd_path_list[label_idx][model_idx])
                                        current_time_stamp = [epoch, total_step]
                                        # self.model.to(self.device)

                                    # if result[model_idx]["eval_loss"] and eval_loss > min(result[model_idx]["eval_loss"]) and total_step >= self.training_dic["overfit_threshold"]:
                                    #     overfit_cnt += 1
                                    # else:
                                    #     overfit_cnt = 0

                                    # result[model_idx]["eval_loss"].append(eval_loss)
                                    
                                    if result[(label_idx,model_idx)]["eval_loss"] and mse_loss - 5 * correlation_loss > min(result[(label_idx,model_idx)]["eval_loss"]) - self.eps and total_step >= self.training_dic["overfit_threshold"]:
                                        overfit_cnt += 1
                                    else:
                                        overfit_cnt = 0

                                    result[(label_idx,model_idx)]["eval_loss"].append(mse_loss - 5 * correlation_loss)

                                    pbar.set_description("Label :%d|Model :%d|Epoch :%d|%s: %.2e|Evaluation Loss: %.2e|MSE Loss: %.2e|Correlation: %.2e|LR: %.2e|Retrain_LR: %.2e" % (label_idx+1, model_idx+1, epoch+1, loss_type, train_loss, eval_loss, mse_loss, correlation_loss, self.optimizer.param_groups[0]["lr"], self.retrain_lr_list[label_idx][model_idx]))

                                    writer.add_scalar("Training_Stats_Model_%d/Train Loss" % model_idx, train_loss, total_step)
                                    writer.add_scalar("Training_Stats_Model_%d/Evaluation Loss" % model_idx, eval_loss, total_step)
                                    writer.add_scalar("Training_Stats_Model_%d/MSE Loss" % model_idx, mse_loss, total_step)
                                    writer.add_scalar("Training_Stats_Model_%d/Correlation" % model_idx, correlation_loss, total_step)
                                    writer.add_scalar("Training_Stats_Model_%d/Learning rate" % model_idx, self.optimizer.param_groups[0]["lr"], total_step)
                                    total_step += 1



                                    result[(label_idx,model_idx)]["eval_mse_loss"].append(mse_loss)
                                    result[(label_idx,model_idx)]["eval_correlation_loss"].append(correlation_loss)
                                    if self.training_dic["eval_mask"]:
                                        result[(label_idx,model_idx)]["eval_correlation_mask"].append(correlation_mask)


                                    if (self.training_dic["retrain"] and overfit_cnt >= self.training_dic["overfit_patience"]) or (self.optimizer.param_groups[0]["lr"] <= self.retrain_lr_list[label_idx][model_idx]/self.threshold_ratio):
                                        if self.optimizer.param_groups[0]["lr"] <= self.retrain_lr_list[label_idx][model_idx]/self.threshold_ratio:
                                            self.retrain_lr_list[label_idx][model_idx] *= self.training_dic["retrain_lr_factor"]
                                        skip_reset = False
                                        overfit_cnt = 0
                                        if self.optimizer.param_groups[0]["lr"]/2 >= self.retrain_lr_list[label_idx][model_idx]:
                                            self.retrain_lr_list[label_idx][model_idx] = self.optimizer.param_groups[0]["lr"]/2
                                            skip_reset = True
                                        if "jumpback" in self.training_dic and self.training_dic["jump_back"]:
                                            self.model.load_state_dict(torch.load(self.model_path_list[label_idx][model_idx], map_location=self.device))
                                            # self.optimizer.load_state_dict(torch.load(self.optimizer_path_list[model_idx]))
                                            self.optimizer_list[label_idx][model_idx].load_state_dict(torch.load(self.optimizer_path_list[label_idx][model_idx]))
                                        # for i in range(len(self.optimizer.param_groups)):
                                        #     self.optimizer.param_groups[i]["lr"] = self.retrain_lr_list[model_idx]
                                        #     if i > 0:
                                        #         self.optimizer.param_groups[i]["lr"] *= 5
                                        # del self.scheduler
                                        for i in range(len(self.optimizer_list[label_idx][model_idx].param_groups)):
                                            self.optimizer_list[label_idx][model_idx].param_groups[i]["lr"] = self.retrain_lr_list[label_idx][model_idx]
                                            if i > 0:
                                                if "separate_lr" in self.training_dic and self.training_dic["separate_lr"] in ["spline", "embedding"]:
                                                    self.optimizer_list[label_idx][model_idx].param_groups[i]["lr"] /= 10
                                                elif "stem_component" in self.cfg_list[label_idx][model_idx].model.model_params["stem_component"] and self.cfg_list[label_idx][model_idx].model.model_params["stem_component"]:
                                                    self.optimizer_list[label_idx][model_idx].param_groups[i]["lr"] *= 5
                                        del self.scheduler
                                        gc.collect()
                                        torch.cuda.empty_cache()
                                        retrain_cnt += 1
                                        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, "min", factor=self.training_dic["scheduler_factor"], patience=self.training_dic["scheduler_patience"], threshold=self.training_dic["scheduler_threshold"])
                                        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_list[label_idx][model_idx], "min", factor=self.training_dic["scheduler_factor"], patience=self.training_dic["scheduler_patience"], threshold=self.training_dic["scheduler_threshold"])
                                        if not skip_reset:
                                            self.retrain_lr_list[label_idx][model_idx] = max(self.retrain_lr_list[label_idx][model_idx]*self.training_dic["retrain_lr_factor"], self.training_dic["scheduler_threshold"])
                                    else:
                                        if self.training_dic["loss_fn"] == "CCC" or self.training_dic["loss_fn"] == "Correlation":
                                            self.scheduler.step(mse_loss - 5 * correlation_loss)
                                        else:
                                            self.scheduler.step(eval_loss)

                                    if self.training_dic["update_delta"] and self.training_dic["reg_loss"] == "Huber":
                                        if not (epoch+1) % self.training_dic["update_period"]:
                                            if total_pred.shape[0] > 0:
                                                self.training_dic["huber_delta"] = np.percentile(np.abs(total_pred.cpu().numpy()-total_target.cpu().numpy()), 90)
                                                self.reg_fn = nn.HuberLoss(delta=self.training_dic["huber_delta"])
                                    result[(label_idx,model_idx)]["huber_delta"].append(self.training_dic["huber_delta"])

                                    df = pd.DataFrame(result[(label_idx,model_idx)])
                                    df.to_csv(self.log_path_list[label_idx][model_idx], index=False, header = True)
                                    if self.optimizer_list[label_idx][model_idx].param_groups[0]["lr"] == self.training_dic["scheduler_threshold"] and self.retrain_lr_list[label_idx][model_idx] == self.training_dic["scheduler_threshold"]:
                                        break

                    info = f"Label {label_idx+1} Model {model_idx+1} is saved on epoch {current_time_stamp[0]+1} step {current_time_stamp[1]+1}"
                    print(info)

                    model_info.append(info)

                    self.model_list[label_idx][model_idx].load_state_dict(torch.load(self.model_path_list[label_idx][model_idx], map_location=self.device))
                    self.model_list[label_idx][model_idx].eval()
                    torch.cuda.empty_cache()

                for model_idx in range(self.num_buckets):
                    self.model_list[label_idx][model_idx].to("cpu")

            with open(self.model_info_loc, "w") as f:
                for info in model_info:
                    f.write(info+"\n")

            del self.train_dataset_list
            del self.train_loader_list

            if self.training_dic.get("residual_training", False):
                for label_idx in range(self.num_labels):
                    for model_idx in range(self.num_buckets):
                        self.model_list[label_idx][model_idx].to(self.device)
            
            gc.collect()

            if "finetune" in self.training_dic and self.training_dic["finetune"]:
                self.fine_tune()

            if "discrimination" in self.training_dic and self.training_dic["discrimination"]:
                for label_idx in range(self.num_labels):
                    for model_idx in range(self.num_buckets):                
                        if "save_result_graph" in self.training_dic and self.training_dic["save_result_graph"]:
                            self.model_nd = create_model(self.cfg_list[label_idx][model_idx].model)
                            self.model_nd.load_state_dict(torch.load(self.model_path_list[label_idx][model_idx], map_location=self.device))
                            torch.save(self.model_nd.state_dict(), self.model_nd_path_list[label_idx][model_idx])
                        if self.training_dic.get("residual_training", False):
                            if model_idx > 0:
                                self.model_list[label_idx][model_idx-1].load_state_dict(torch.load(self.model_path_list[label_idx][model_idx-1], map_location=self.device))
                            self.adversarial_list[label_idx][model_idx].adv_against_model(self.model_path_list[label_idx][model_idx], self.optimizer_path_list[label_idx][model_idx], self.finetune_train_loader_list[label_idx], self.finetune_eval_loader_list[label_idx], self.model_list[label_idx][:model_idx], self.feature_select_list[label_idx][:model_idx+1])
                        else:
                            self.adversarial_list[label_idx][model_idx].adv_against_model(self.model_path_list[label_idx][model_idx], self.optimizer_path_list[label_idx][model_idx], self.finetune_train_loader_list[label_idx], self.finetune_eval_loader_list[label_idx], None, self.feature_select_list[label_idx][:model_idx+1])

                        if "save_result_graph" in self.training_dic and self.training_dic["save_result_graph"]:
                            self.model = create_model(self.cfg_list[label_idx][model_idx].model)
                            self.model.load_state_dict(torch.load(self.model_path_list[label_idx][model_idx], map_location=self.device)) 
                            self.model.eval()
                            self.model_nd.eval()
                            total_target = []
                            total_pred = []
                            total_pred_nd = []
                            with torch.no_grad():
                                for data in self.eval_loader_list[label_idx]:
                                    extra_dict = data[-1]
                                    if self.training_dic.get("reduce_low_cap", False):
                                        reduction_cap = extra_dict["reduction_cap"].squeeze().to(dtype=data[0].dtype)
                                    extra_feat = extra_dict["features"].squeeze()
                                    if "extra_feature" in self.training_dic and self.training_dic["extra_feature"]:
                                        data_batch = torch.cat([data[0].squeeze(), extra_feat], dim=1).to(self.device, non_blocking=True, dtype=data[0].dtype)
                                    else:                    
                                        data_batch = data[0].squeeze().to(self.device, non_blocking=True)
                                    target_batch = data[1].squeeze().to(self.device, non_blocking=True)
                                    pred = torch.squeeze(self.model.forward(data_batch[:,self.feature_select_list[label_idx][model_idx]]))
                                    pred_nd = torch.squeeze(self.model_nd.forward(data_batch[:,self.feature_select_list[label_idx][model_idx]]))
                                    total_target.append(target_batch.detach())
                                    total_pred.append(pred.detach())
                                    total_pred_nd.append(pred_nd.detach())
                            total_target = torch.cat(total_target, dim=0)
                            total_pred = torch.cat(total_pred, dim=0)
                            total_pred_nd = torch.cat(total_pred_nd, dim=0)
                            result_to_graph = {"target":total_target.cpu().tolist(), "pred":total_pred.cpu().tolist(), "pred_nd":total_pred_nd.cpu().tolist()}
                            df_graph = pd.DataFrame(result_to_graph)
                            df_graph.to_csv(Path(self.training_dic["model_path"]) / model_name / f'result_to_graph_{label_idx}_{model_idx}.csv', index=False, header = True)
                del self.finetune_train_loader_list, finetune_train_dataset_list, self.finetune_eval_loader_list, finetune_eval_dataset_list

                gc.collect()
                torch.cuda.empty_cache()
        
        for label_idx in range(self.num_labels):
            for model_idx in range(self.num_buckets):
                self.model_list[label_idx][model_idx].to(self.device)
                self.model_list[label_idx][model_idx].load_state_dict(torch.load(self.model_path_list[label_idx][model_idx], map_location=self.device))
                self.model_list[label_idx][model_idx].eval()

        if "regressor_type" in self.training_dic and self.training_dic["regressor_type"] in ["nn.linear", "fastkan"]:        
            self.regressor_training()
            del self.regressor_eval_loader_list, self.regressor_train_loader_list, regressor_eval_dataset_list, regressor_train_dataset_list
            gc.collect()
            torch.cuda.empty_cache()
        else:
            if self.num_buckets == 1:
                self.linear_coef_list = [np.array([1]) for _ in range(self.num_labels)]
                np.save(Path(self.training_dic["model_path"])/model_name/"linear_coefficient.npy", self.linear_coef_list)
                print(f"Since we only have one bucket, the coefficient is {self.linear_coef_list}")
            elif "equal_weights" in self.training_dic and self.training_dic["equal_weights"]:
                del self.regressor_eval_loader_list, self.regressor_train_loader_list, regressor_eval_dataset_list, regressor_train_dataset_list
                self.linear_coef_list = [np.ones(self.num_buckets) / self.num_buckets for _ in range(self.num_labels)]
                np.save(Path(self.training_dic["model_path"])/model_name/"linear_coefficient.npy", self.linear_coef_list)
                print(f"Equal weights gives us {self.linear_coef_list}")
            else:
                del self.regressor_eval_loader_list, self.regressor_train_loader_list, regressor_eval_dataset_list, regressor_train_dataset_list
                self.linear_coef_list = []
                for label_idx in range(self.num_labels):
                    if "separate_regression" in self.training_dic and self.training_dic["separate_regression"] and isinstance(self.buckets_info, list):
                        prev_num = 0
                        self.linear_coef = []
                        for nbuckets in self.buckets_info:
                            pred_list = []
                            target = []
                            for data in self.regressor_total_loader_list[label_idx]:
                                cur_pred = []
                                extra_dict = data[-1]
                                if self.training_dic.get("reduce_low_cap", False):
                                    reduction_cap = extra_dict["reduction_cap"].squeeze().to(dtype=data[0].dtype)
                                extra_feat = extra_dict["features"].squeeze()
                                if "extra_feature" in self.training_dic and self.training_dic["extra_feature"]:
                                    data_batch = torch.cat([data[0], extra_feat], dim=1).to(self.device, non_blocking=True, dtype=data[0].dtype)
                                else:
                                    data_batch = data[0].to(self.device, non_blocking=True)
                                prev_pred = torch.zeros_like(data[1].to(non_blocking=True))
                                for model_idx in range(prev_num):
                                    prev_pred += (self.model_list[label_idx][model_idx].forward(data_batch[:,self.feature_select_list[label_idx][model_idx]]).to("cpu") * self.linear_coef[model_idx]).squeeze()
                                if self.training_dic.get("reduce_low_cap", False):
                                    if self.training_dic.get("reduce_method", "addition") == "addition":
                                        target.append(((data[1]+reduction_cap).to(non_blocking=True) - prev_pred).detach())
                                    elif self.training_dic.get("reduce_method", "addition") == "multiplication":
                                        target.append(((data[1]*reduction_cap).to(non_blocking=True) - prev_pred).detach())
                                else:
                                    target.append((data[1].to(non_blocking=True) - prev_pred).detach())
                                for model_idx in range(prev_num, prev_num+nbuckets):
                                    cur_pred.append(self.model_list[label_idx][model_idx].forward(data_batch[:,self.feature_select_list[label_idx][model_idx]]).to("cpu").detach())
                                cur_pred = torch.cat(cur_pred, dim=-1)
                                pred_list.append(cur_pred)
                            pred = torch.cat(pred_list, dim=0)
                            target = torch.cat(target, dim=0).numpy()
                            if "reg_weight_essential" in self.training_dic and self.training_dic["reg_weight_essential"]:
                                pred_mean = pred.mean(dim=0)
                                pred_std = pred.std(dim=0)
                                sample_weights = torch.zeros_like(pred)
                                if "mask_pos_up" in self.training_dic:
                                    upper_adjust_p = F.sigmoid(8 * (pred_mean + self.training_dic["mask_pos_up"] * pred_std - pred))
                                else:
                                    upper_adjust_p = 1
                                sample_weights += F.sigmoid(8 * (pred - pred_mean - self.training_dic["mask_pos"] * pred_std)) * upper_adjust_p * self.training_dic["weight_pow_under"]
                                sample_weights = sample_weights.sum(dim=1).numpy() + np.ones_like(target)
                            else:
                                sample_weights = np.ones_like(target)
                            pred = pred.numpy()
                            self.linear_regressor_list[label_idx].fit(pred, target, sample_weight=sample_weights)
                            cur_coef = np.clip(self.linear_regressor_list[label_idx].coef_, a_min=self.training_dic["reg_min"], a_max=self.training_dic["reg_max"])
                            self.linear_coef += cur_coef.tolist()
                            prev_num += nbuckets
                    else:
                        pred_list = []
                        target = []
                        for data in self.regressor_total_loader_list[label_idx]:
                            cur_pred = []
                            extra_dict = data[-1]
                            if self.training_dic.get("reduce_low_cap", False):
                                reduction_cap = extra_dict["reduction_cap"].squeeze().to(dtype=data[0].dtype)
                            extra_feat = extra_dict["features"].squeeze()
                            if "extra_feature" in self.training_dic and self.training_dic["extra_feature"]:
                                data_batch = torch.cat([data[0], extra_feat], dim=1).to(self.device, non_blocking=True, dtype=data[0].dtype)
                            else:
                                data_batch = data[0].to(self.device, non_blocking=True)
                            if self.training_dic.get("reduce_low_cap", False):
                                if self.training_dic.get("reduce_method", "addition") == "addition":
                                    target.append((data[1]+reduction_cap).to(non_blocking=True).detach())
                                elif self.training_dic.get("reduce_method", "addition") == "multiplication":
                                    target.append((data[1]*reduction_cap).to(non_blocking=True).detach())
                            else:
                                target.append(data[1].to(non_blocking=True).detach())
                            for model_idx in range(self.num_buckets):
                                cur_pred.append(self.model_list[label_idx][model_idx].forward(data_batch[:,self.feature_select_list[label_idx][model_idx]]).to("cpu").detach())
                            cur_pred = torch.cat(cur_pred, dim=-1)
                            pred_list.append(cur_pred)
                        pred = torch.cat(pred_list, dim=0)
                        target = torch.cat(target, dim=0).numpy()
                        if "reg_weight_essential" in self.training_dic and self.training_dic["reg_weight_essential"]:
                            pred_mean = pred.mean(dim=0)
                            pred_std = pred.std(dim=0)
                            sample_weights = torch.zeros_like(pred)
                            if "mask_pos_up" in self.training_dic:
                                upper_adjust_p = F.sigmoid(8 * (pred_mean + self.training_dic["mask_pos_up"] * pred_std - pred))
                            else:
                                upper_adjust_p = 1
                            sample_weights += F.sigmoid(8 * (pred - pred_mean - self.training_dic["mask_pos"] * pred_std)) * upper_adjust_p * self.training_dic["weight_pow_under"]
                            sample_weights = sample_weights.sum(dim=1).numpy() + np.ones_like(target)
                        else:
                            sample_weights = np.ones_like(target)
                        pred = pred.numpy()
                        self.linear_regressor_list[label_idx].fit(pred, target, sample_weight=sample_weights)
                        self.linear_coef = np.clip(self.linear_regressor_list[label_idx].coef_, a_min=self.training_dic["reg_min"], a_max=self.training_dic["reg_max"])
                    if "base_coef" in self.training_dic and (isinstance(self.training_dic["base_coef"], int) or len(self.training_dic["base_coef"])==len(self.linear_coef)):
                        self.linear_coef = np.array(self.linear_coef)
                        self.linear_coef += self.training_dic["base_coef"]
                    self.linear_coef_list.append(deepcopy(self.linear_coef))
                np.save(Path(self.training_dic["model_path"])/model_name/"linear_coefficient.npy", self.linear_coef_list)
                np.save(Path(self.training_dic["model_path"])/model_name/"linear_coefficient_bk.npy", self.linear_coef_list)
                print(f"Linear Regression fit to {self.linear_coef_list}")
                    # import ipdb; ipdb.set_trace()

        del self.model_list
        if self.num_buckets != 1:
            del regressor_total_dataset_list
            del self.regressor_total_loader_list

        gc.collect()
        torch.cuda.empty_cache()


        pearson_ic, spearman_ic, mse_loss = self.evaluation(self.cfg_list)

        del self.eval_loader_list
        del self.eval_dataset_list
        del self.indices
        del self.train_input

        gc.collect()
        torch.cuda.empty_cache()
        writer.close()
        
        train_input_buffer.close()

        # train_input_buffer.unlink()
        print(f"{self.model_name} is done")

        saved_alphas = np.load(self.saved_alphas_loc)
        np.save(self.used_alphas_loc, saved_alphas)
        np.save(self.finish_flag_loc, [])

        return pearson_ic, spearman_ic, mse_loss

    def l1_regularization(self):
        l1_loss = sum(param.abs().sum() for param in self.model.parameters())
        return l1_loss

    def evaluation(self, cfg_list):
        # load the best model from the training process
        self.model_result_list = []
        for cfgs in cfg_list:
            self.model_result_list.append([])
            for cfg in cfgs:
                self.model_result_list[-1].append(create_model(cfg.model))
        # if "check_mask" in cfg.training.model_params and cfg.training.model_params["check_mask"]:
        #     model_result.load_state_dict(torch.load(self.model_mask_path, map_location=self.device))
        # elif "check_most" in cfg.training.model_params and cfg.training.model_params["check_most"]:
        #     model_result.load_state_dict(torch.load(self.model_most_path, map_location=self.device))
        # else:
        for i in range(self.num_labels):
            for model_result, model_path in zip(self.model_result_list[i], self.model_path_list[i]):
                model_result.load_state_dict(torch.load(model_path, map_location=self.device))
            for model_result in self.model_result_list[i]:
                model_result.eval()
            if "regressor_type" in self.training_dic and self.training_dic["regressor_type"] in ["nn.linear", "fastkan"]:
                self.regressor_list[i].load_state_dict(torch.load(self.regressor_path_list[i], map_location=self.device))
                self.regressor_list[i].eval()
        total_pred = None
        np_target = None
        with torch.no_grad():
            for label_idx in range(self.num_labels):
                cur_pred = None
                for data in self.eval_loader_list[label_idx]:
                    extra_dict = data[-1]
                    if self.training_dic.get("validation_by_return", False):
                        ret_ratio = extra_dict["ret_ratio"].squeeze().to(self.device, non_blocking=True, dtype=data[0].dtype)
                    if self.training_dic.get("reduce_low_cap", False):
                        reduction_cap = extra_dict["reduction_cap"].squeeze().to(dtype=data[0].dtype)
                    extra_feat = extra_dict["features"].squeeze()
                    if "extra_feature" in self.training_dic and self.training_dic["extra_feature"]:
                        data_batch = torch.cat([data[0].squeeze(), extra_feat], dim=1).to(self.device, non_blocking=True, dtype=data[0].dtype)
                    else:                    
                        data_batch = data[0].squeeze().to(self.device, non_blocking=True)
                    if self.training_dic["denormalizer"]:
                        batch_mean = torch.unsqueeze(torch.mean(data_batch, dim=1), 1)
                        batch_std = torch.unsqueeze(torch.std(data_batch, dim=1), 1)
                        data_batch = torch.cat([data_batch, batch_mean, batch_std.pow(0.5)], dim=1).to(self.device, non_blocking=True)
                    if "horizontal_norm" in self.training_dic and self.training_dic["horizontal_norm"]:
                        data_batch = F.normalize(data_batch, p=2.0, dim=1)
                    target_batch = data[1].squeeze().to(non_blocking=True).numpy()         
                    pred = self.regressor_predict(data_batch, label_idx)
                    pred = torch.squeeze(pred).to("cpu").detach().numpy()
                    if not total_pred:
                        total_pred = [pred]
                        np_target = [target_batch]
                    else:
                        total_pred.append(pred)
                        np_target.append(target_batch)
                    if cur_pred is None:
                        cur_pred = pred
                    else:
                        cur_pred = np.concatenate((cur_pred, pred), axis=0)
                np.save(self.model_stats_loc_list[label_idx], [np.mean(cur_pred), np.std(cur_pred)])
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
        std = np.std(total_pred)
        mean = np.mean(total_pred)
        np.save(self.stat_loc, [mean, std])
        print(f"peason_ic: {pearson_ic} | spearman_ic: {spearman_ic} | mse_loss: {mse_loss} | std of pred: {std} | mean of pred: {mean} | max of pred: {np.max(total_pred)} | min of pred: {np.min(total_pred)}")
        print(f"according to the evaluation, the suggested delta value is {delta_candidate}")
        return pearson_ic, spearman_ic, mse_loss
    
    def regressor_training(self):
        for label_idx in range(self.num_labels):
            regressor_step = len(self.regressor_train_loader_list[label_idx]) // self.training_dic["step_num"]
            self.regressor_list[label_idx].train()
            optimizer = torch.optim.AdamW(self.regressor_list[label_idx].parameters(), lr=self.training_dic["regressor_lr"], weight_decay=self.training_dic["weight_decay"])
            self.training_dic["retrain_lr"] = self.training_dic["regressor_lr"]
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=self.training_dic["scheduler_factor"], patience=1, threshold=self.training_dic["scheduler_threshold"])
            pbar_regressor = tqdm(range(self.training_dic["regressor_epoch"]))
            best_eval_loss = float("inf")
            pred_mean = 0
            pred_std = 0
            retrain_cnt = 0
            overfit_cnt = 0

            for epoch in pbar_regressor:
                cur_step = 0
                total_step = 0
                for data in self.regressor_train_loader_list[label_idx]:
                    extra_dict = data[-1]
                    if self.training_dic.get("reduce_low_cap", False):
                        reduction_cap = extra_dict["reduction_cap"].to(dtype=data[0].dtype)
                    extra_feat = extra_dict["features"].squeeze()
                    if "extra_feature" in self.training_dic and self.training_dic["extra_feature"]:
                        data_batch = torch.cat([data[0], extra_feat], dim=1).to(self.device, non_blocking=True, dtype=data[0].dtype)
                    else:
                        data_batch = data[0].to(self.device, non_blocking=True)
                    regressor_input = torch.cat([self.model_list[label_idx][i](data_batch[:, self.feature_select_list[label_idx][i]]) for i in range(self.num_buckets)], dim=1)
                    target_batch = data[1].to(self.device, non_blocking=True)
                    pred = self.regressor_list[label_idx](regressor_input).squeeze()
                    weight = torch.ones_like(target_batch)
                    # if "ret" not in self.training_dic.get("loss_fn"):
                    weight += data[2].to(self.device, non_blocking=True)
                    if "reg_weight_essential" in self.training_dic and self.training_dic["reg_weight_essential"]:
                        if "weight_pos_up" in self.training_dic:
                            upper_adjust_t = F.sigmoid(8 * (self.mean[label_idx] + self.training_dic["weight_pos_up"] * self.std[label_idx] - target_batch))
                            upper_adjust_p = F.sigmoid(8 * (pred_mean + self.training_dic["weight_pos_up"] * pred_std - pred))
                        else:
                            upper_adjust_t = upper_adjust_p = 1
                        weight += torch.max(F.sigmoid(8 * (target_batch - self.mean[label_idx] - self.training_dic["weight_pos"] * self.std[label_idx])) * upper_adjust_t, F.sigmoid(8 * (pred - pred_mean - self.training_dic["weight_pos"] * pred_std)) * upper_adjust_p) * self.training_dic["weight_pow_over"]
                    loss = - self.training_dic["correlation_ratio"] * self.correlation_loss(pred, target_batch, weight) + (1 - self.training_dic["correlation_ratio"]) * (self.reg_fn(pred, target_batch) * weight).mean()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    cur_step += 1
                    if cur_step >= regressor_step:
                        cur_step = 0
                        total_step += 1
                        self.regressor_list[label_idx].eval()
                        with torch.no_grad():
                            if self.training_dic["regressor_type"] == "nn.linear":
                                for params in self.regressor_list[label_idx].parameters():
                                    params.clamp_(min=0.0)
                            total_pred = []
                            eval_loss = 0
                            mse_loss = 0
                            correlation_loss = 0
                            for data in self.regressor_eval_loader_list[label_idx]:
                                extra_dict = data[-1]
                                if self.training_dic.get("reduce_low_cap", False):
                                    reduction_cap = extra_dict["reduction_cap"].to(dtype=data[0].dtype)
                                extra_feat = extra_dict["features"].squeeze()
                                if "extra_feature" in self.training_dic and self.training_dic["extra_feature"]:
                                    data_batch = torch.cat([data[0], extra_feat], dim=1).to(self.device, non_blocking=True, dtype=data[0].dtype)
                                else:                    
                                    data_batch = data[0].to(self.device, non_blocking=True)
                                regressor_input = torch.cat([self.model_list[label_idx][i](data_batch[:, self.feature_select_list[label_idx][i]]) for i in range(self.num_buckets)], dim=1)
                                target_batch = data[1].to(self.device, non_blocking=True)
                                pred = self.regressor_list[label_idx](regressor_input).squeeze()
                                if not total_pred:
                                    total_pred = [torch.squeeze(pred).to("cpu").detach()]
                                else:
                                    total_pred.append(torch.squeeze(pred).to("cpu").detach())
                                weight = torch.ones_like(target_batch)
                                weight += data[2].to(self.device, non_blocking=True)
                                if "reg_weight_essential" in self.training_dic and self.training_dic["reg_weight_essential"]:
                                    if "mask_pos_up" in self.training_dic:
                                        upper_adjust_p = F.sigmoid(8 * (pred_mean + self.training_dic["mask_pos_up"] * pred_std - pred))
                                    else:
                                        upper_adjust_p = 1
                                    weight += F.sigmoid(8 * (pred - pred_mean - self.training_dic["mask_pos"] * pred_std)) * upper_adjust_p * self.training_dic["weight_pow_under"]
                                new_correlation_loss = self.correlation_loss(pred, target_batch, weight)
                                new_mse_loss = (self.reg_fn(pred, target_batch) * weight).mean()
                                loss = - self.training_dic["correlation_ratio"] * new_correlation_loss + (1 - self.training_dic["correlation_ratio"]) * new_mse_loss
                                eval_loss += loss.item()
                                mse_loss += new_mse_loss.item()
                                correlation_loss += new_correlation_loss.item()

                        eval_loss /= len(self.regressor_eval_loader_list[label_idx])
                        mse_loss /= len(self.regressor_eval_loader_list[label_idx])
                        correlation_loss /= len(self.regressor_eval_loader_list[label_idx])
                        pbar_regressor.set_description(f"Epoch {epoch+1} Step {total_step}|Regressor loss: {eval_loss:.4f} | MSE loss: {mse_loss:.4f} | Correlation loss: {correlation_loss:.4f}")

                            # pbar.set_description("Model :%d|Epoch :%d|Train Loss: %.2e|Evaluation Loss: %.2e|MSE Loss: %.2e|Correlation: %.2e|LR: %.2e" % (model_idx, epoch+1, train_loss, eval_loss, mse_loss, correlation_loss, self.optimizer_list[model_idx].param_groups[0]["lr"]))
                        if mse_loss - 5 * correlation_loss < best_eval_loss:
                            best_eval_loss = mse_loss - 5 * correlation_loss
                            torch.save(self.regressor_list[label_idx].state_dict(), self.regressor_path_list[label_idx])
                            torch.save(optimizer.state_dict(), self.regressor_optimizer_path_list[label_idx])
                            overfit_cnt = 0
                        else:
                            overfit_cnt += 1
                        scheduler.step(mse_loss - 5 * correlation_loss)

                        if self.training_dic["retrain"] and overfit_cnt == self.training_dic["overfit_patience"] and epoch >= self.training_dic["overfit_threshold"]:
                            overfit_cnt = 0
                            self.training_dic["retrain_lr"] = max(optimizer.param_groups[0]["lr"]/2, self.training_dic["retrain_lr"])
                            self.regressor.load_state_dict(torch.load(self.regressor_path_list[label_idx], map_location=self.device))
                            optimizer.load_state_dict(torch.load(self.regressor_optimizer_path_list[label_idx]))
                            for param_group in optimizer.param_groups:
                                param_group["lr"] = self.training_dic["retrain_lr"]
                                if "stem_component" in param_group["param_names"]:
                                    param_group["lr"] *= 5
                            del scheduler
                            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=self.training_dic["scheduler_factor"], patience=1, threshold=self.training_dic["scheduler_threshold"])
                            retrain_cnt = 1
                        else:
                            scheduler.step(mse_loss - 5 * correlation_loss)
                            retrain_cnt += 1

                        total_pred = torch.cat(total_pred, dim=0)
                        pred_mean = total_pred.mean()
                        pred_std = total_pred.std()
                        self.regressor.train()
    
    def regressor_predict(self, data_batch, label_idx):
        if "regressor_type" in self.training_dic and self.training_dic["regressor_type"] in ["nn.linear", "fastkan"]:
            self.regressor_list[label_idx].eval()
        with torch.no_grad():
            if "regressor_type" in self.training_dic and self.training_dic["regressor_type"] in ["nn.linear", "fastkan"]:
                regressor_input = torch.cat([self.model_result_list[label_idx][i](data_batch[:, self.feature_select_list[label_idx][i]]) for i in range(self.num_buckets)], dim=1)
                pred = self.regressor_list[label_idx](regressor_input)
            else:
                pred = torch.zeros(data_batch.shape[0], device=self.device)
                for i in range(self.num_buckets):
                    pred += self.linear_coef_list[label_idx][i] * torch.squeeze(self.model_result_list[label_idx][i](data_batch[:, self.feature_select_list[label_idx][i]]))
        return pred


    def correlation_loss(self, pred, target, weight=None):
        # correlation_ratio = self.training_dic["correlation_ratio"]

        pred_centered = pred - pred.mean(dim=0, keepdim=True)
        target_centered = target - target.mean(dim=0, keepdim=True)
        if weight is None:
            weight = torch.ones_like(pred)
        corr_nume = torch.sum(pred_centered * target_centered * weight, dim=0, keepdim=True)
        corr_denom = torch.sqrt((torch.sum((pred_centered ** 2)*weight, dim=0, keepdim=True) + 1e-7) * (torch.sum((target_centered ** 2)*weight, dim=0, keepdim=True) + 1e-7))

        corr = corr_nume / corr_denom
        # return - correlation_ratio * corr + (1-correlation_ratio) * torch.nn.MSELoss()(pred, target)

        return corr.sum()

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

    def find_by_featurename(self, names_list):
        feature_indices = [np.zeros(self.input_dim, dtype=bool) for _ in names_list]
        for j in range(len(names_list)):
            for i in range(len(self.saved_alphas)):
                if self.saved_alphas[i] in names_list[j]:
                    feature_indices[j][i] = True
        return feature_indices

    def weight_calculation_merged(self, feature_num, valid_indices):
        days, stocks = self.indices[1].shape
        weights = np.empty((days, stocks), dtype=np.float32)
        for i in range(days):
            calculate_weights_cython(self.train_input[feature_num][i,:], valid_indices[i,:], weights[i])
        return calculate_turnover_cython(weights)

    def feature_bucketing(self, cfg, label_idx):
        

        days, stocks = self.indices[1].shape
        features_num = self.indices[0].shape[0]
        valid_feature_num = self.indices[0].sum()

        # import ipdb; ipdb.set_trace()
        
        features = np.where(self.indices[0])[0]
        alpha_indices = {}
        for pos, idx in enumerate(features):
            alpha_indices[idx] = pos
        
        saved_alphas = np.load(self.saved_alphas_loc).tolist()
        self.saved_alphas = np.array([alpha[:6] for alpha in saved_alphas])

        if isinstance(self.training_dic["num_buckets"], int) and (self.training_dic["num_buckets"]==1 or self.training_dic.get("boosting", False)):
            feature_buckets_indices = [self.indices[0] for _ in range(self.training_dic["num_buckets"])]
            feature_list_indices = [np.concatenate([np.ones(np.sum(self.indices[0])), np.ones(self.feature_matrix.shape[0])]).astype(bool) for _ in range(self.training_dic["num_buckets"])]
            return feature_buckets_indices, feature_list_indices

        # if Path(self.feature_buckets_loc_list[label_idx]).exists() and Path(self.num_buckets_loc_list[label_idx]).exists() and (np.array(self.buckets_info).squeeze().shape == np.array(np.load(self.num_buckets_loc_list[label_idx])).squeeze().shape) and (np.array(self.buckets_info).squeeze() == np.load(self.num_buckets_loc_list[label_idx]).squeeze()).all():
        #     feature_buckets_indices = np.load(self.feature_buckets_loc_list[label_idx])
        #     saved_features = feature_buckets_indices[0].shape[0]
        #     # import ipdb; ipdb.set_trace()
        #     if saved_features == features_num and len(feature_buckets_indices) == self.num_buckets:
        #         if isinstance(self.training_dic["num_buckets"], list) or (Path(self.num_bags_loc_list[label_idx]).exists() and (np.load(self.num_bags_loc_list[label_idx]) == np.array(self.training_dic["num_bags"])).all()):
        #             feature_list_indices = np.load(self.feature_list_loc_list[label_idx])
        #             if valid_feature_num + self.feature_matrix.shape[0] == feature_list_indices[0].shape[0]:
        #                 return feature_buckets_indices, feature_list_indices

        np.save(self.num_buckets_loc_list[label_idx] ,self.buckets_info)

        alpha_pos_name = {}
        for pos, name in enumerate(saved_alphas):
            alpha_pos_name[pos] = name

        # weights = np.empty((features_num, days, stocks), dtype=np.float32)

        futures = []

        # valid_indices = (self.indices[1] & self.stocks_choice[label_idx]).view(np.uint8)

        # print("I'm here")
        # present_time = time.time
        
        # with ThreadPoolExecutor(max_workers=25) as executor:
        #     for i in range(features_num):
        #         if self.indices[0][i]:
        #             for j in range(self.indices[1].shape[0]):
        #                 future = executor.submit(calculate_weights_cython, self.train_input[i][j,:], valid_indices[j,:], weights[i][j])
        #                 futures.append((future, i, j))

        # pbar_weights = tqdm(as_completed([f[0] for f in futures]), total=len(futures), desc="Calculating")

        # print(time.time-present_time)

        # i = 0
        # for completed_future in pbar_weights:
        #     i += 1
        #     # 可以从 completed_future 提取返回值或捕捉异常
        #     try:
        #         completed_future.result() # 如果 Cython 函数报错，这里会抛出
        #     except Exception as e:
        #         print(f"任务发生异常: {e}")

        #     pbar_weights.set_description(f"calculating weights: feature {i} finished")
        #             # executor.submit(calculate_weights_cython, self.train_input[i], self.indices[1].view(np.uint8), weights[i])
        #             # pbar_weights.set_description(f"calculating weights: feature {i} finished")

        # # for key in range(features):
        # #     for j in range(days):
        # #         if self.indices[0][key]:
        # #             weights[key][j] = weights[key][j].result()
        # #     weights[key] = np.array(weights[key])

        valid_indices = (self.indices[1] & self.stocks_choice[label_idx]).view(np.uint8)

        turn_over = {}

        futures = []

        pbar_turnover = tqdm(range(self.indices[0].shape[0]), desc="calculating turnover")
        with ThreadPoolExecutor(max_workers=50) as executor:
            for i in range(self.indices[0].shape[0]):
                if self.indices[0][i]:
                    future = turn_over[i] = executor.submit(self.weight_calculation_merged, i, valid_indices) 
                    futures.append((future, i))
                
        pbar_turnover = tqdm(as_completed([f[0] for f in futures]), total=len(futures), desc="Calculating")
        i = 0
        for completed_future in pbar_turnover:
            i += 1

            # 可以从 completed_future 提取返回值或捕捉异常
            try:
                completed_future.result() # 如果 Cython 函数报错，这里会抛出
            except Exception as e:
                print(f"任务发生异常: {e}")

            pbar_turnover.set_description(f"calculating turnover: feature {i} finished")

        del valid_indices

        for key in turn_over:
            turn_over[key] = turn_over[key].result()

        # in the case of separate bucketing, num_buckets should be a list whose content are two integers, the second is the number of buckets for d1 alphas, while the first is for the rest
        if isinstance(self.training_dic["num_buckets"], list):
            if len(self.training_dic["num_buckets"]) == 2:
                bucket1_alphas = self.training_dic.get("bucket1_alphas", ["kalpha", "valpha"])
                bucket2_alphas = self.training_dic.get("bucket2_alphas", ["kalpha", "alpha.", "alpha_"])
                turn_over_0 = {key: turn_over[key] for key in turn_over if alpha_pos_name[alpha_indices[key]][:6] in bucket1_alphas}
                turn_over_1 = {key: turn_over[key] for key in turn_over if alpha_pos_name[alpha_indices[key]][:6] in bucket2_alphas}

                feature_indices_0 = sorted(turn_over_0.keys(), key=lambda x: turn_over_0[x], reverse=True)
                feature_indices_1 = sorted(turn_over_1.keys(), key=lambda x: turn_over_1[x], reverse=True)

                feature_buckets = []

                bucket_size_0 = math.ceil(len(feature_indices_0) / self.training_dic["num_buckets"][0])
                bucket_size_1 = math.ceil(len(feature_indices_1) / self.training_dic["num_buckets"][1])
                
                current_bucket_0 = []
                for idx in feature_indices_0:
                    current_bucket_0.append(idx)
                    if len(current_bucket_0) >= bucket_size_0:
                        feature_buckets.append(current_bucket_0)
                        current_bucket_0 = []
                
                if current_bucket_0:
                    feature_buckets.append(current_bucket_0)
                
                current_bucket_1 = []
                for idx in feature_indices_1:
                    current_bucket_1.append(idx)
                    if len(current_bucket_1) >= bucket_size_1:
                        feature_buckets.append(current_bucket_1)
                        current_bucket_1 = []
                
                if current_bucket_1:
                    feature_buckets.append(current_bucket_1)

                feature_list = []
                for bucket in feature_buckets:
                    feature_list.append([alpha_indices[idx] for idx in bucket]) 
                feature_buckets_indices = [np.zeros_like(self.indices[0]).astype(bool) for _ in range(len(feature_list))]

                for i in range(len(feature_buckets)):
                    for idx in feature_buckets[i]:
                        feature_buckets_indices[i][idx] = True
                
                del feature_buckets

                feature_list_indices = [np.concatenate([np.zeros(np.sum(self.indices[0])), np.ones(self.feature_matrix.shape[0])]).astype(bool) for i in range(len(feature_list))]

                for i in range(len(feature_list)):
                    for idx in feature_list[i]:
                        feature_list_indices[i][idx] = True

                del feature_list
                # import ipdb; ipdb.set_trace()

                np.save(self.feature_buckets_loc_list[label_idx], feature_buckets_indices)
                np.save(self.feature_list_loc_list[label_idx], feature_list_indices)
                
                return feature_buckets_indices, feature_list_indices

            if len(self.training_dic["num_buckets"]) == 3:
                
                bucket1_alphas = self.training_dic.get("bucket1_alphas", ["kalpha"])
                bucket2_alphas = self.training_dic.get("bucket2_alphas", ["valpha"])
                bucket3_alphas = self.training_dic.get("bucket3_alphas", ["alpha.", "alpha_"])

                turn_over_0 = {key: turn_over[key] for key in turn_over if alpha_pos_name[alpha_indices[key]][:6] in bucket1_alphas}
                turn_over_1 = {key: turn_over[key] for key in turn_over if alpha_pos_name[alpha_indices[key]][:6] in bucket2_alphas}
                turn_over_2 = {key: turn_over[key] for key in turn_over if alpha_pos_name[alpha_indices[key]][:6] in bucket3_alphas}

                feature_indices_0 = sorted(turn_over_0.keys(), key=lambda x: turn_over_0[x], reverse=True)
                feature_indices_1 = sorted(turn_over_1.keys(), key=lambda x: turn_over_1[x], reverse=True)
                feature_indices_2 = sorted(turn_over_2.keys(), key=lambda x: turn_over_2[x], reverse=True)

                feature_buckets = []

                bucket_size_0 = math.ceil(len(feature_indices_0) / self.training_dic["num_buckets"][0])
                bucket_size_1 = math.ceil(len(feature_indices_1) / self.training_dic["num_buckets"][1])
                bucket_size_2 = math.ceil(len(feature_indices_2) / self.training_dic["num_buckets"][2])
                
                current_bucket_0 = []
                for idx in feature_indices_0:
                    current_bucket_0.append(idx)
                    if len(current_bucket_0) >= bucket_size_0:
                        feature_buckets.append(current_bucket_0)
                        current_bucket_0 = []
                
                if current_bucket_0:
                    feature_buckets.append(current_bucket_0)
                
                current_bucket_1 = []
                for idx in feature_indices_1:
                    current_bucket_1.append(idx)
                    if len(current_bucket_1) >= bucket_size_1:
                        feature_buckets.append(current_bucket_1)
                        current_bucket_1 = []
                
                if current_bucket_1:
                    feature_buckets.append(current_bucket_1)

                current_bucket_2 = []
                for idx in feature_indices_2:
                    current_bucket_2.append(idx)
                    if len(current_bucket_2) >= bucket_size_2:
                        feature_buckets.append(current_bucket_2)
                        current_bucket_2 = []
                
                if current_bucket_2:
                    feature_buckets.append(current_bucket_2)

                feature_list = []
                for bucket in feature_buckets:
                    feature_list.append([alpha_indices[idx] for idx in bucket]) 
                feature_buckets_indices = [np.zeros_like(self.indices[0]).astype(bool) for _ in range(len(feature_list))]

                for i in range(len(feature_buckets)):
                    for idx in feature_buckets[i]:
                        feature_buckets_indices[i][idx] = True
                
                del feature_buckets

                feature_list_indices = [np.concatenate([np.zeros(np.sum(self.indices[0])), np.ones(self.feature_matrix.shape[0])]).astype(bool) for i in range(len(feature_list))]

                for i in range(len(feature_list)):
                    for idx in feature_list[i]:
                        feature_list_indices[i][idx] = True

                del feature_list
                # import ipdb; ipdb.set_trace()

                np.save(self.feature_buckets_loc_list[label_idx], feature_buckets_indices)
                np.save(self.feature_list_loc_list[label_idx], feature_list_indices)
                
                return feature_buckets_indices, feature_list_indices
            
            else:
                raise ValueError("num_buckets must be a list with two or three integers")

        elif isinstance(self.training_dic["num_buckets"], int):

            feature_indices = sorted(turn_over.keys(), key=lambda x: turn_over[x], reverse=True)

            feature_bags = []

            num_bags = self.training_dic.get("num_bags", [1] * self.training_dic["num_buckets"])
            if len(num_bags) != self.training_dic["num_buckets"]:
                print("number of bags must be a list whose length equal to the number of buckets, it will be reset to the default list")
                num_bags = [1] * self.training_dic["num_buckets"]
            
            np.save(self.num_bags_loc_list[label_idx], num_bags)

            bag_size = math.ceil(np.sum(self.indices[0]) / sum(num_bags))

            bucket_sizes = [bag_size * num_bag for num_bag in num_bags]

            batch_idx = 0

            current_bag = []

            for idx in feature_indices:
                current_bag.append(idx)
                if len(current_bag) >= bag_size:
                    feature_bags.append(current_bag)
                    current_bag = []
            
            if current_bag:
                feature_bags.append(current_bag)

            feature_buckets = [[]]

            # print([len(feature_bag) for feature_bag in feature_bags])
            # print(len(feature_buckets))

            num = 0
            pos = 0

            if self.training_dic.get("bucketing_method", "default") == "zigzag":
                for i in range(self.training_dic["num_buckets"]):
                    for j in range(num_bags[i]):
                        feature_buckets[-1] += feature_bags[i+j*self.training_dic["num_buckets"]]
                    feature_buckets.append([])
            elif self.training_dic.get("bucketing_method", "default") == "lowhigh":
                feature_bags = deque(feature_bags)
                indicator = 1
                while feature_bags:
                    if indicator > 0:
                        feature_buckets[-1] += feature_bags.popleft()
                    else:
                        feature_buckets[-1] += feature_bags.pop()
                    num += 1
                    indicator *= -1
                    if num >= num_bags[pos]:
                        pos += 1
                        num = 0
                        feature_buckets.append([])
            else:
                for i in range(len(feature_bags)):
                    feature_buckets[-1] += feature_bags[i]
                    num += 1
                    if num >= num_bags[pos]:
                        num = 0
                        pos += 1
                        feature_buckets.append([])
            
            feature_buckets.pop()

            print([len(feature_bucket) for feature_bucket in feature_buckets])

            # current_bucket = []
            # for idx in feature_indices:
            #     current_bucket.append(idx)
            #     if len(current_bucket) >= bucket_sizes[batch_idx]:
            #         feature_buckets.append(current_bucket)
            #         current_bucket = []
            #         batch_idx += 1

            # if current_bucket:
            #     feature_buckets.append(current_bucket)

            feature_list = []

            for bucket in feature_buckets:
                feature_list.append([alpha_indices[idx] for idx in bucket])
                
            feature_buckets_indices = [np.zeros_like(self.indices[0]).astype(bool) for _ in range(len(feature_list))]

            for i in range(len(feature_buckets)):
                for idx in feature_buckets[i]:
                    feature_buckets_indices[i][idx] = True
            
            del feature_buckets

            feature_list_indices = [np.concatenate([np.zeros(np.sum(self.indices[0])), np.ones(self.feature_matrix.shape[0])]).astype(bool) for i in range(len(feature_list))]

            for i in range(len(feature_list)):
                for idx in feature_list[i]:
                    feature_list_indices[i][idx] = True

            del feature_list
            # import ipdb; ipdb.set_trace()

            np.save(self.feature_buckets_loc_list[label_idx], feature_buckets_indices)
            np.save(self.feature_list_loc_list[label_idx], feature_list_indices)
            
            return feature_buckets_indices, feature_list_indices
        
        else:
            raise ValueError("num_buckets must be a list with two integers or an int")

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


    def predict(self, x, cfg, valid_alphas=None, non_valid_alphas=None, date=None, test_stage=True, lock=None, valid_stock=None, *args, **kwargs):
        x = x[:, valid_alphas]
        super().predict_helper(x, cfg)
        """
        if cfg.model.type == "FastKAN":
            for i in range(len(cfg.model.model_params["layers_hidden"])-1):
                cfg.model.model_params["layers_hidden"][i] *= input_dim
        """

        # print(valid_stock.shape)
        # print(valid_stock.sum())
        # print(valid_stock)
        # return
        
        # if cfg.training.model_params["check_mask"]:
        #     self.model_state_dict_path = self.model_state_dict_path.with_name("final_model_mask.pt")
        # elif "check_most" in cfg.training.model_params and cfg.training.model_params["check_most"]:
        #     self.model_state_dict_path = self.model_state_dict_path.with_name("final_model_most.pt")
        pit_tidx = self.model_name.split("_")
        pit_tidx = pit_tidx[1] + "_" + pit_tidx[2]
        
        model_path_name = (self.training_dic["model_path"]).split("/")[-1]
        
        if "saved_alpha_path" in self.training_dic:
            self.saved_alphas_loc = Path(self.training_dic["saved_alpha_path"]) / ("saved_train_alphas_" + pit_tidx + "_processed.npy")
        else:
            self.saved_alphas_loc = Path("/dfs/data/ksim/automation/20240116/readcache_new/data") / model_path_name / ("saved_train_alphas_" + pit_tidx + "_processed.npy")

        if isinstance(self.training_dic["num_buckets"], int):
            self.num_buckets = self.training_dic["num_buckets"]
        else:
            self.num_buckets = sum(self.training_dic["num_buckets"])

        if cfg.model.type == "Trident":
            self.num_buckets = 1
        
        self.num_labels = len(self.training_dic.get("blocks_sep", [["SH", "SZ", "gem", "star"]]))

        self.dict_path_list = [[self.folder_path / f"training_dict_{j}_{i}.pt" for i in range(self.num_buckets)] for j in range(self.num_labels)]
        self.model_dict_path_list = [[self.folder_path / f"final_model_{j}_{i}.pt" for i in range(self.num_buckets)] for j in range(self.num_labels)]
        self.regressor_path_list = [(self.folder_path / f"final_regressor_{i}.pt") for i in range(self.num_labels)]
        self.statistic_loc_list = [(self.folder_path / f"statistic_{i}.npy") for i in range(self.num_labels)]        
        self.model_stats_loc_list = [(self.folder_path / f"model_stats_{i}.npy") for i in range(self.num_labels)]
        self.linear_regressor_path = self.folder_path / f"linear_coefficient.npy"
        self.stat_loc = self.folder_path / f"stat.npy"

        if "ensemble_tidx" in self.training_dic and self.training_dic["ensemble_tidx"] is not None:
            tidx = int(self.model_name.split("_")[-1])
            if tidx not in self.training_dic["ensemble_tidx"]:
                self.training_dic["num_buckets"] = 1
                self.num_buckets = 1
                self.training_dic["ah_buckets"] = 0

        cfg_list = [[torch.load(self.dict_path_list[j][i]) for i in range(self.num_buckets)] for j in range(self.num_labels)]
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

        if "extra_feature" in cfg_list[0][0].training.model_params and cfg_list[0][0].training.model_params["extra_feature"]:
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
        for cfgs in cfg_list:
            for cfg in cfgs:
                cfg.model.model_params["device"] = self.device
        for cfgs in cfg_list:
            for cfg in cfgs:
                cfg.model.model_params["infer"] = True
        self.feature_select_list = [[cfg_list[j][i].training.model_params["feature_chosen"] for i in range(self.num_buckets)] for j in range(self.num_labels)]
        
        self.model_result_list = [[create_model(cfg.model) for cfg in cfg_list[i]] for i in range(self.num_labels)]
        for i in range(self.num_buckets):
            for j in range(self.num_labels):
                self.model_result_list[j][i].load_state_dict(torch.load(self.model_dict_path_list[j][i], map_location=self.device))
                self.model_result_list[j][i].eval()
        
        if "regressor_type" in self.training_dic and self.training_dic["regressor_type"] == "nn.linear":
            self.regressor_list = [Regressor(self.num_buckets, device=self.device) for _ in range(self.num_labels)]
        elif "regressor_type" in self.training_dic and self.training_dic["regressor_type"] == "fastkan":
            self.regressor_list = [KANRegressor(self.num_buckets, device=self.device) for _ in range(self.num_labels)]
        if "regressor_type" in self.training_dic and self.training_dic["regressor_type"] in ["nn.linear", "fastkan"]:
            for i in range(self.num_labels):
                self.regressor_list[i].load_state_dict(torch.load(self.regressor_path_list[i], map_location=self.device))
                self.regressor_list[i].eval()
        else:
            self.linear_coef_list = np.load(self.linear_regressor_path)

        print(f"it takes {time.time()-start_time} to initiate")


        self.label_handling()
        for i in range(self.num_labels):
            self.stocks_choice[i] = (self.stocks_choice[i].squeeze())[valid_stock]

        start_time = time.time()

        result = torch.zeros(x.shape[0], dtype=torch.float32, device=self.device)

        if not self.training_dic.get("label_ensemble", False):
            if self.training_dic.get("stats_calibration", False):
                target_stats_list = [np.load(self.statistic_loc_list[i]) for i in range(self.num_labels)]                
                pred_stats_list = [np.load(self.model_stats_loc_list[i]) for i in range(self.num_labels)]
            with torch.no_grad():
                for label_idx in range(self.num_labels):
                    pred = self.regressor_predict(torch.from_numpy(x[self.stocks_choice[label_idx]]).contiguous().to(torch.float32).to(self.device), label_idx)
                    if self.training_dic.get("stats_calibration", False):  
                        mean_diff = pred_stats_list[label_idx][0] - target_stats_list[label_idx][0]
                        std_ratio = target_stats_list[label_idx][1] / (pred_stats_list[label_idx][1] + 1e-8)
                        pred = (pred - mean_diff) * std_ratio
                    result[self.stocks_choice[label_idx]] = pred.squeeze()
        else:
            label_ensemble_ratio = np.array(self.training_dic.get("label_ensemble_ratio", [self.stocks_choice[i].sum() for i in range(self.num_labels)])).astype(np.float32)
            label_ensemble_ratio /= label_ensemble_ratio.sum()
            with torch.no_grad():
                for label_idx in range(self.num_labels):
                    pred = self.regressor_predict(torch.from_numpy(x).contiguous().to(torch.float32).to(self.device), label_idx)
                    result += pred.squeeze() * label_ensemble_ratio[label_idx]

        ret = result.to("cpu").numpy()

        sub_universe_selected = self.training_dic.get("stock_select", None)

        if sub_universe_selected is not None:
            stock_selection = np.zeros(x.shape[0], dtype=np.uint8)
            for sub_universe in self.sub_universe_dict:
                if sub_universe in sub_universe_selected:
                    stock_selection += self.sub_universe_dict[sub_universe].squeeze()[valid_stock]
            stock_selection = stock_selection.astype(bool)
            ret[~stock_selection] = np.nan
        
        if self.training_dic.get("validity_for_infer", False) and Path(self.stat_loc).exists():
            ratio = self.training_dic.get("validity_bound", 0.5)
            stat = np.load(self.stat_loc)
            mean, std = stat[0], stat[1]
            ret[abs(ret - mean) < ratio * std] = np.nan


        print(f"it takes {time.time()-start_time} to predict")

        if "regressor_type" in self.training_dic and self.training_dic["regressor_type"] in ["nn.linear", "fastkan"]:
            del self.regressor_list
        del pred, self.model_result_list
        gc.collect()
        torch.cuda.empty_cache()

        return ret

    def label_handling(self):
        self.stocks_choice = []
        days, stocks = self.SH_indices.shape[0], self.SH_indices.shape[1]
        block_collection = deepcopy(self.training_dic.get("blocks_sep", [["SH", "SZ", "gem", "star"]]))
        if ["star"] in block_collection:
            if self.indices[1][self.star_indices].sum() < 1e5:
                temp_col = []
                for block in block_collection:
                    if block != ["star"]:
                        temp_col.append(block)
                        if "gem" in block and "star" not in block:
                            block.append("star")
                block_collection = temp_col
        for blocks in block_collection:
            stocks_choice = np.zeros((days, stocks), dtype=bool)
            if "SH" in blocks:
                stocks_choice += self.SH_indices
            if "SZ" in blocks:
                stocks_choice += self.SZ_indices
            if "gem" in blocks:
                stocks_choice += self.gem_indices
            if "star" in blocks:
                stocks_choice += self.star_indices
            if "ZZ2000" in blocks:
                stocks_choice += self.ZZ2000_indices
            if "ZZ1000" in blocks:
                stocks_choice += self.ZZ1000_indices
            if "ZZ500" in blocks:
                stocks_choice += self.ZZ500_indices
            if "HS300" in blocks:
                stocks_choice += self.HS300_indices
            self.stocks_choice.append(stocks_choice.astype(bool))
        
        # std = self.train_output[:, self.stocks_choice[0][-1]].std(axis=1)
        # mean = self.train_output[:, self.stocks_choice[0][-1]].mean(axis=1)
        # for stocks_choice in self.stocks_choice:
        #     self.train_output[:, stocks_choice[-1]] -= mean
        #     self.train_output[:, stocks_choice[-1]] /= std
        
    def create_raw_features(self, cfg, model_name, reader):

        self.training_dic = deepcopy(cfg).training.model_params

        tidx = int(model_name.split("_")[-1])

        buffer = cfg.train_basics.get("cache_reader_buffer", 0)

        dates = reader(mode="dates")

        self.start_date = dates[buffer]
        self.current_date = dates[-1]

        if len(dates) != buffer:
            start_date = dates[buffer]
            self.r = r = CacheReader(cfg.env.root_directory)
            nstocks = r.uids.shape[0]
            start_idx = np.where(r.dates == start_date)[0][0]
            last_date = dates[-1]
            date_idx = np.where(r.dates == last_date)[0][0] + 1
            last_date = r.dates[date_idx]
            horizon = self.training_dic.get("horizon", 1)
            padding = np.zeros((horizon, nstocks), dtype=np.float32)
            if self.training_dic.get("validation_by_return", False):
                if self.training_dic.get("return_type", "raw") == "raw" or self.training_dic.get("loss_fn") == "ret_raw":
                    close = r.get_raw_data("IntervalFull.close", start_date, last_date)[:,:,tidx]
                    close = np.concatenate((close, padding), axis=0)
                    # close = np.concatenate((close, np.zeros((1, close.shape[1]), dtype=np.float32)), axis=0)
                    adj_fct = r.get_raw_data("adjfactor", start_date, last_date)
                    adj_fct = np.concatenate((adj_fct, padding), axis=0)
                    for d in range(len(close)-horizon):
                        for i in range(horizon):
                            close[d,:] = close[d,:] * adj_fct[d+i,:]
                    base_price = close[:-horizon,:]
                    self.raw_ret = np.nan_to_num(close[horizon:,:] / (base_price + 1e-8) - 1, 0)
                    self.ret_ratio = self.raw_ret - self.raw_ret.mean(axis=-1, keepdims=True)
                elif self.training_dic.get("return_type", "raw") == "dsrt" or self.training_dic.get("loss_fn") == "ret_dsrt":
                    dsrt_return = r.get_raw_data(f"CNE5Ret.DSRT.d{horizon}", r.dates[start_idx+horizon], last_date)[:,:,tidx]
                    dsrt_return = np.concatenate((dsrt_return, padding), axis=0)
                    dsrt_return -= np.nanmean(dsrt_return, axis=-1, keepdims=True)
                    self.ret_ratio = np.nan_to_num(dsrt_return, 0)

                self.ret_ratio = torch.tensor(self.ret_ratio).unsqueeze(0)
        
        feature_list = []

        # grab log_cap from cache and use it to get the reduction for low cap stocks when needed
        log_cap = torch.from_numpy(np.log(reader("cap")[buffer-1: -1,:].astype(np.float32))).unsqueeze(0)

        if self.training_dic.get("reduce_low_cap", False):
            reduce_ratio = self.training_dic.get("reduce_ratio", 1)
            if self.training_dic.get("reduce_method", "addition") == "addition":
                self.reduction_cap = (reduce_ratio * (F.sigmoid(log_cap - 13) - 1))
            elif self.training_dic.get("reduce_method", "addition") == "multiplication":
                self.reduction_cap = ((F.sigmoid(log_cap - 13))+reduce_ratio) / (reduce_ratio + 1)

            # self.feature_matrix = reduction

        log_cap -= 14

        # grab block information from cache
        block4 = reader("block4n")

        # secind = reader("secoindustry")
        # secind_num = secind.max() + 1

        days, stocks = block4.shape[0]-buffer, block4.shape[1]

        self.SH_indices = block4[buffer-1: -1,:] == 0
        self.SZ_indices = block4[buffer-1: -1,:] == 1
        self.gem_indices = block4[buffer-1: -1,:] == 2
        self.star_indices = block4[buffer-1: -1,:] == 3

        self.ZZ2000_indices = reader("ZZ2000")[buffer-1: -1,:]
        self.ZZ1000_indices = reader("ZZ1000_")[buffer-1: -1,:]
        self.ZZ500_indices = reader("ZZ500_")[buffer-1: -1,:]
        self.HS300_indices = reader("HS300")[buffer-1: -1,:]
        self.residual_indices = 1 - self.ZZ2000_indices - self.ZZ1000_indices - self.ZZ500_indices - self.HS300_indices

        self.sub_universe_dict = {"SH":self.SH_indices, "SZ":self.SZ_indices, "gem":self.gem_indices, "star":self.star_indices, "ZZ2000":self.ZZ2000_indices, "ZZ1000":self.ZZ1000_indices, "ZZ500":self.ZZ500_indices, "HS300":self.HS300_indices, "residual":self.residual_indices}

        stock_block = np.zeros((len(self.sub_universe_dict.keys()), days, stocks))
        
        # stock_block will add one-hot block information into features
        pos = 0
        for sub_universe in self.sub_universe_dict:
            stock_block[pos,:,:] = self.sub_universe_dict[sub_universe].astype(np.float32)
            pos += 1
        # stock_block[0,:,:] = (self.SH_indices).astype(np.float32)
        # stock_block[1,:,:] = (self.SZ_indices).astype(np.float32)
        # stock_block[2,:,:] = (self.gem_indices).astype(np.float32)
        # stock_block[3,:,:] = (self.star_indices).astype(np.float32)

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

        barra_indices = deepcopy(factorExp)

        factorExp = torch.from_numpy(factorExp[:,buffer-1: -1,:])

        barra_invert_indices = -deepcopy(barra_indices[:10])

        barra_indices = np.concatenate([barra_invert_indices, barra_indices], axis=0)

        for i in range(20):
            barra_indices[i] = np.tanh(barra_indices[i])

        #retrieve date information from cache and use these info to create seasonal features
        date = list(map(str,reader(mode="dates")))
        # date_info will have shape (4 ,days)
        if len(date) == buffer:
            date_info = extract_date_features_vectorized(date)[:,buffer-1:].unsqueeze(2).expand(-1, -1, stocks)
        else:
            date_info = extract_date_features_vectorized(date)[:,buffer-1:-1].unsqueeze(2).expand(-1, -1, stocks)
        # import ipdb; ipdb.set_trace()

        #AShare as well as low/high capital trend features
        
        feature_time_choice = []
        lowcap_indices = log_cap < -1
        highcap_indices = log_cap >= -1

        # use the total market return, liqudity, volume and amount to describe market trend
        dr_ret5 = reader("dr.ret_5")[buffer-1: -1, :]
        ret5_mean = torch.from_numpy(np.nanmean(dr_ret5, axis=-1)).unsqueeze(0)
        ret5_std = torch.from_numpy(np.nanstd(dr_ret5, axis=-1)).unsqueeze(0)
        highcap_ret5_mean = torch.from_numpy(np.nanmean(np.where(highcap_indices, dr_ret5, np.nan), axis=-1))
        lowcap_ret5_mean = torch.from_numpy(np.nanmean(np.where(lowcap_indices, dr_ret5, np.nan), axis=-1))
        highcap_ret5_std = torch.from_numpy(np.nanstd(np.where(highcap_indices, dr_ret5, np.nan), axis=-1))
        lowcap_ret5_std = torch.from_numpy(np.nanstd(np.where(lowcap_indices, dr_ret5, np.nan), axis=-1))
        feature_time_choice.append(ret5_mean)
        feature_time_choice.append(ret5_std)
        feature_time_choice.append(highcap_ret5_mean)
        feature_time_choice.append(lowcap_ret5_mean)
        feature_time_choice.append(highcap_ret5_std)
        feature_time_choice.append(lowcap_ret5_std)

        dr_ret10 = reader("dr.ret_10")[buffer-1: -1, :]
        ret10_mean = torch.from_numpy(np.nanmean(dr_ret10, axis=-1)).unsqueeze(0)
        ret10_std = torch.from_numpy(np.nanstd(dr_ret10, axis=-1)).unsqueeze(0)
        highcap_ret10_mean = torch.from_numpy(np.nanmean(np.where(highcap_indices, dr_ret10, np.nan), axis=-1))
        lowcap_ret10_mean = torch.from_numpy(np.nanmean(np.where(lowcap_indices, dr_ret10, np.nan), axis=-1))
        highcap_ret10_std = torch.from_numpy(np.nanstd(np.where(highcap_indices, dr_ret10, np.nan), axis=-1))
        lowcap_ret10_std = torch.from_numpy(np.nanstd(np.where(lowcap_indices, dr_ret10, np.nan), axis=-1))
        feature_time_choice.append(ret10_mean)
        feature_time_choice.append(ret10_std)
        feature_time_choice.append(highcap_ret10_mean)
        feature_time_choice.append(lowcap_ret10_mean)
        feature_time_choice.append(highcap_ret10_std)
        feature_time_choice.append(lowcap_ret10_std)


        dr_logliq5 = np.log(reader("dr.liq_5"))[buffer-1: -1, :]-14
        logliq5_mean = torch.from_numpy(np.nanmean(dr_logliq5, axis=-1)).unsqueeze(0)
        logliq5_std = torch.from_numpy(np.nanstd(dr_logliq5, axis=-1)).unsqueeze(0)
        highcap_logliq5_mean = torch.from_numpy(np.nanmean(np.where(highcap_indices, dr_logliq5, np.nan), axis=-1))
        lowcap_logliq5_mean = torch.from_numpy(np.nanmean(np.where(lowcap_indices, dr_logliq5, np.nan), axis=-1))
        highcap_logliq5_std = torch.from_numpy(np.nanstd(np.where(highcap_indices, dr_logliq5, np.nan), axis=-1))
        lowcap_logliq5_std = torch.from_numpy(np.nanstd(np.where(lowcap_indices, dr_logliq5, np.nan), axis=-1))
        feature_time_choice.append(logliq5_mean)
        feature_time_choice.append(logliq5_std)
        feature_time_choice.append(highcap_logliq5_mean)
        feature_time_choice.append(lowcap_logliq5_mean)
        feature_time_choice.append(highcap_logliq5_std)
        feature_time_choice.append(lowcap_logliq5_std)


        dr_logliq10 = np.log(reader("dr.liq_10"))[buffer-1: -1, :]-14
        logliq10_mean = torch.from_numpy(np.nanmean(dr_logliq10, axis=-1)).unsqueeze(0)
        logliq10_std = torch.from_numpy(np.nanstd(dr_logliq10, axis=-1)).unsqueeze(0)
        highcap_logliq10_mean = torch.from_numpy(np.nanmean(np.where(highcap_indices, dr_logliq10, np.nan), axis=-1))
        lowcap_logliq10_mean = torch.from_numpy(np.nanmean(np.where(lowcap_indices, dr_logliq10, np.nan), axis=-1))
        highcap_logliq10_std = torch.from_numpy(np.nanstd(np.where(highcap_indices, dr_logliq10, np.nan), axis=-1))
        lowcap_logliq10_std = torch.from_numpy(np.nanstd(np.where(lowcap_indices, dr_logliq10, np.nan), axis=-1))
        feature_time_choice.append(logliq10_mean)
        feature_time_choice.append(logliq10_std)
        feature_time_choice.append(highcap_logliq10_mean)
        feature_time_choice.append(lowcap_logliq10_mean)
        feature_time_choice.append(highcap_logliq10_std)
        feature_time_choice.append(lowcap_logliq10_std)

        all_volume = reader("IntervalFull.volume")
        volume = all_volume[buffer-1: -1, :, :]
        volume_sum = torch.tensor(np.log(np.nansum(np.nansum(volume, axis=-1), axis=-1))).unsqueeze(0) - 19
        feature_time_choice.append(volume_sum)

        all_amount = reader("IntervalFull.amount")
        amount = all_amount[buffer-1: -1, :, :]
        amount_total = np.nansum(np.nansum(amount, axis=-1), axis=-1)
        amount_sum = torch.tensor(np.log(np.nansum(np.nansum(amount, axis=-1), axis=-1))).unsqueeze(0) - 26
        feature_time_choice.append(amount_sum)

        all_close = reader("IntervalFull.close")

        borrowed_money_purchase = np.nansum(reader("AShareMarginTrade.S_MARGIN_PURCHWITHBORROWMONEY")[buffer-1: -1, :], axis=-1)
        bmp_ratio = torch.tensor(borrowed_money_purchase / amount_total).unsqueeze(0)
        feature_time_choice.append(bmp_ratio)

        daily_change = (reader("IntervalFull.close")[buffer-1: -1,:,-1] - reader("IntervalFull.open")[buffer-1: -1,:,0]).squeeze()

        # get the ratio of difference of uplimit number to the downlimit number and total number of stocks
        num_total = (1 - np.isnan(daily_change)).sum(axis=-1)

        num_uplim = (((daily_change * (self.SH_indices + self.SZ_indices)) >= 0.095) + ((daily_change * (self.gem_indices + self.star_indices)) >= 0.195)).sum(axis=-1)

        num_downlim = (((daily_change * (self.SH_indices + self.SZ_indices)) <= -0.095) + ((daily_change * (self.gem_indices + self.star_indices)) <= -0.195)).sum(axis=-1)

        up_down_ratio = (num_uplim - num_downlim) / num_total

        feature_time_choice.append(torch.from_numpy(up_down_ratio).unsqueeze(0))

        # for feature in feature_time_choice:
        #     print(feature.shape)

        # Industrial trend
        ind_factors = ["IndRotaComfactor.comFactor", "IndRotaFactors.indusMom", "IndRotaFactors.tassetstrateD", "IndRotaFactors.conroe", "IndRotaFactors.revY", "IndRotaFactors.opfTtmQoqZ", "IndRotaFactors.sentiment2"]
        ind_datas = [reader(factor) for factor in ind_factors]
        for i in range(len(ind_datas)):
            ind_datas[i] = torch.from_numpy(np.nan_to_num(ind_datas[i], np.nanmean(ind_datas[i])))[buffer-1: -1,:].unsqueeze(0)
        feature_num = len(feature_time_choice)
        days = feature_time_choice[0].shape[1]
        feature_time_choice = [torch.cat(feature_time_choice, dim=0).unsqueeze(2).expand(feature_num,days,stocks)]
        if self.training_dic.get("time_choice_block", False) or cfg.model.model_params.get("time_choice_method", None) not in ["barra_estimator", "moe_output"]:
            feature_time_choice = torch.cat(feature_time_choice+ind_datas, dim=0)
        else:
            feature_time_choice = feature_time_choice[0]

        extra_feature_list = self.training_dic.get("extra_feature_list", ["factor_Exp", "log_cap", "stock_block"])

        # down weight stocks reached limit price
        if len(dates) != buffer:
            if self.training_dic.get("down_weight_limited", False):
                actual_close = r.get_raw_data("IntervalFull.close", start_date, last_date)
                DnLimited = r.get_raw_data("DnLimPrice", start_date, last_date)
                DnLimited = torch.from_numpy(actual_close[:,:,tidx] <= DnLimited + 1e-3).unsqueeze(0)
                UpLimited = r.get_raw_data("UpLimPrice", start_date, last_date)
                UpLimited = torch.from_numpy(actual_close[:,:,tidx] >= UpLimited - 1e-3).unsqueeze(0)
                self.limited = ~ (DnLimited | UpLimited)

        # Barra Indices
        if "barra_indices" in extra_feature_list:
            barra_close = np.nansum((barra_indices * np.expand_dims(all_close[:,:,tidx], axis=0)), axis=-1) + 1.2e5
            barra_volume = np.nansum((barra_indices * np.expand_dims(all_volume[:,:,tidx], axis=0)), axis=-1) + 2e7
            barra_amount = np.nansum((barra_indices * np.expand_dims(all_amount[:,:,tidx], axis=0)), axis=-1) + 2e10

            barra_log_close = np.log(np.clip(barra_close, a_min=1e-10, a_max=None)) - 11.5
            barra_log_volume = np.log(np.clip(barra_volume, a_min=1e-10, a_max=None)) - 16.5
            barra_log_amount = np.log(np.clip(barra_amount, a_min=1e-10, a_max=None)) - 23.5

            barra_dim = barra_close.shape[0]

            barra_rolling_range = [5, 20]

            barra_features = {"close":barra_log_close, "volume":barra_log_volume, "amount": barra_log_amount}
            barra_feature_insert = []
            for feature in barra_features:
                barra_feature_insert.append(torch.from_numpy(barra_features[feature][:,buffer-1: -1]).unsqueeze(2).expand(barra_dim,days,stocks))
                for factor in barra_features[feature]:
                    for r in barra_rolling_range:
                        rolling_factor = RollingStatistics(factor)
                        barra_feature_insert.append(torch.from_numpy(rolling_factor.mean(r)[buffer-1: -1]).unsqueeze(1).expand(days,stocks).unsqueeze(0))
                        barra_feature_insert.append(torch.from_numpy(rolling_factor.std(r)[buffer-1: -1]).unsqueeze(1).expand(days,stocks).unsqueeze(0))

            feature_time_choice = torch.cat([feature_time_choice]+barra_feature_insert, dim=0)

        if self.training_dic.get("time_choice_block", False):
            self.time_choice_feature_num = feature_time_choice.shape[0]

        feature_list.append(feature_time_choice)
        
        IndicesName = ["000001.SH", "000016.SH", "000300.SH", "000510.SH", "000852.SH", "000903.SH", "000905.SH", "000906.SH", "399001.SZ", "399005.SZ", "399006.SZ", "399008.SZ", "399012.SZ", "399101.SZ", "399102.SZ", "399106.SZ", "399303.SZ", "932000.SH"]

        rolling_range = [5, 10, 20, 30, 60]

        if "indices_pctchange" in extra_feature_list:

            dq_pct_change = ".S_DQ_PCTCHANGE"

            indices_pctchange = [reader(IndexName+dq_pct_change) for IndexName in IndicesName]
            indices_pctchange_feature = torch.cat([torch.from_numpy(data[buffer-1: -1]).unsqueeze(1).expand(days,stocks).unsqueeze(0) for data in indices_pctchange], dim=0)

            pct_change_rolling_mean = []
            for pctchange in indices_pctchange:
                for r in rolling_range:
                    pct_change_rolling_mean.append(torch.from_numpy(RollingStatistics(pctchange).mean(r)[buffer-1: -1]).unsqueeze(1).expand(days,stocks).unsqueeze(0))
            pct_change_rolling_mean = torch.cat(pct_change_rolling_mean, dim=0)

            pct_change_rolling_std = []
            for pctchange in indices_pctchange:
                for r in rolling_range:
                    pct_change_rolling_std.append(torch.from_numpy(RollingStatistics(pctchange).std(r)[buffer-1: -1]).unsqueeze(1).expand(days,stocks).unsqueeze(0))
            pct_change_rolling_std = torch.cat(pct_change_rolling_std, dim=0)

            if self.training_dic.get("time_choice_block", False):
                self.time_choice_feature_num += indices_pctchange_feature.shape[0] + pct_change_rolling_mean.shape[0] + pct_change_rolling_std.shape[0] 

        if "indices_log_amount" in extra_feature_list:

            dq_amount = ".S_DQ_AMOUNT"

            log_amount = [(np.log(reader(IndexName+dq_amount))-17) for IndexName in IndicesName]
            log_amount_feature = torch.cat([torch.from_numpy(data[buffer-1: -1]).unsqueeze(1).expand(days,stocks).unsqueeze(0) for data in log_amount], dim=0)

            log_amount_rolling_mean = []
            for l_a in log_amount:
                for r in rolling_range:
                    log_amount_rolling_mean.append(torch.from_numpy(RollingStatistics(l_a).mean(r)[buffer-1: -1]).unsqueeze(1).expand(days,stocks).unsqueeze(0))
            log_amount_rolling_mean = torch.cat(log_amount_rolling_mean, dim=0)

            log_amount_rolling_std = []
            for l_a in log_amount:
                for r in rolling_range:
                    log_amount_rolling_std.append(torch.from_numpy(RollingStatistics(l_a).std(r)[buffer-1: -1]).unsqueeze(1).expand(days,stocks).unsqueeze(0))
            log_amount_rolling_std = torch.cat(log_amount_rolling_std, dim=0)

            if self.training_dic.get("time_choice_block", False):
                self.time_choice_feature_num += log_amount_feature.shape[0] + log_amount_rolling_mean.shape[0] + log_amount_rolling_std.shape[0]

        self.insert_num = 0

        if "indices_pctchange" in extra_feature_list:
            feature_list.append(indices_pctchange_feature)
            feature_list.append(pct_change_rolling_mean)
            feature_list.append(pct_change_rolling_std)
        if "indices_log_amount" in extra_feature_list:
            feature_list.append(log_amount_feature)
            feature_list.append(log_amount_rolling_mean)
            feature_list.append(log_amount_rolling_std)
        if "log_cap" in extra_feature_list:
            # self.insert_num += log_cap.shape[0]
            feature_list.append(log_cap)
        if "date_info" in extra_feature_list:
            self.insert_num += date_info.shape[0]
            feature_list.append(date_info)
        if "factor_Exp" in extra_feature_list:
            self.insert_num += factorExp.shape[0]
            feature_list.append(factorExp)
        if "stock_block" in extra_feature_list:
            self.insert_num += stock_block.shape[0]
            feature_list.append(stock_block)
        
        self.barra_dim = stock_block.shape[0] + factorExp.shape[0]

        # import ipdb; ipdb.set_trace()
        
        if "extra_feature" in cfg.training.model_params and cfg.training.model_params["extra_feature"]:
            self.feature_matrix = torch.cat(feature_list, dim=0)
        else:
            self.feature_matrix = None
        
        # print(self.feature_matrix.shape)

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

class KANRegressor(nn.Module):
    def __init__(
        self,
        input_dim,
        device = "cpu",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.device = device
        self.regressor_layers = [input_dim, 3*input_dim, 1]
        self.Regressor = nn.ModuleList([FastKANLayer(in_dim, out_dim, use_layernorm=False, base_activation=torch.nn.functional.gelu, spline_weight_init_scale=1e-3, dropout_rate=0, device=device) for in_dim, out_dim in zip(self.regressor_layers[:-1], self.regressor_layers[1:])])
    
    def forward(self, input):
        for layer in self.Regressor:
            input = layer(input)
        return input.squeeze()