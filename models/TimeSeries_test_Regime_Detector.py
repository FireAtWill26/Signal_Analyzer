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
from collections import Counter, defaultdict, deque
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import train_test_split
from multiprocessing.shared_memory import SharedMemory
from lion_pytorch import Lion
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from scipy.stats import skew, kurtosis
from numba import prange, njit
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import random
import math
import shutil
from prometheus.utils.cache_reader import CacheReader


from prometheus.utils.utils import WeightedCorrNp, calc_fcst_weighted_ic, weighted_mse_wrap, weighted_mse_eval_wrap, filter_eligible_data, filter_eligible_data_v2, operations_by_group
from prometheus.utils.torch_utils import select_gpu_with_minimum_memory, early_stopping_func, WarmupLR, DecayingCosineWarmRestarts, SharedMemDataset, SharedMemSeqDataset, SharedMemSeqDatasetRandom, SharedMemSeqDatasetDaily
from prometheus.ops import create_op_process
from prometheus.modelpool.basemodel import BaseModel, create_model
from prometheus.utils.registry_factory import TRAINING_REGISTRY, MODEL_REGISTRY
from prometheus.utils.speedup_package import calculate_stock_mean_3d_parallel
from prometheus.utils.pearson import niocorr
from prometheus.modelpool.Adversarial import Adversarial
from prometheus.utils.data_processing import RollingStatistics
from prometheus.riskmodel.barra.factor_dict import *
from prometheus.utils.weight_calculation import calculate_weights_cython, calculate_turnover_cython
from prometheus.utils.corr import calc_correlation, calc_rank_correlation
from prometheus.utils.muon import Muon, MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam
from prometheus.utils.date_info_extraction import *
from prometheus.utils.util_funcs import *
from prometheus.utils.cache_reader import CacheReader

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



@TRAINING_REGISTRY.register('TimeSeries_test_multiple_labels')
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

        self.num_labels = len(self.training_dic.get("blocks_sep", [["SH", "SZ", "gem", "star"]]))

        Path.mkdir(Path(self.training_dic["model_path"]) / model_name, exist_ok=True, parents=True)

        model_path_name = (self.training_dic["model_path"]).split("/")[-1]

        pit_tidx = model_name.split("_")
        pit_tidx = pit_tidx[1] + "_" + pit_tidx[2]

        if "saved_alpha_path" in self.training_dic:
            self.saved_alphas_loc = Path(self.training_dic["saved_alpha_path"]) / ("saved_train_alphas_" + pit_tidx + "_processed.npy")
        else:
            self.saved_alphas_loc = Path("/dfs/data/ksim/automation/20240116/readcache_new/data") / model_path_name / ("saved_train_alphas_" + pit_tidx + "_processed.npy")

        self.num_buckets_loc_list = [(Path(self.training_dic["model_path"]) / model_name / f'num_buckets_{i}.npy') for i in range(self.num_labels)]
        self.feature_buckets_loc_list = [(Path(self.training_dic["model_path"]) / model_name / f'feature_bucket_{i}.npy') for i in range(self.num_labels)]
        self.feature_list_loc_list = [(Path(self.training_dic["model_path"]) / model_name / f'feature_list_{i}.npy') for i in range(self.num_labels)]
        self.num_bags_loc_list = [(Path(self.training_dic["model_path"]) / model_name / f'num_bags_{i}.npy') for i in range(self.num_labels)]
        self.statistic_loc_list = [(Path(self.training_dic["model_path"]) / model_name / f'statistic_{i}.npy') for i in range(self.num_labels)]
        self.model_stats_loc_list = [(Path(self.training_dic["model_path"]) / model_name / f'model_stats_{i}.npy') for i in range(self.num_labels)]
        self.model_info_loc = Path(self.training_dic["model_path"]) / model_name / f'model_info.txt'
        self.stat_loc = Path(self.training_dic["model_path"]) / model_name / f'stat.npy'
        self.finish_flag_loc = Path(self.training_dic["model_path"]) / model_name / f"finished.npy"
        self.used_alphas_loc = Path(self.training_dic["model_path"]) / model_name / f'used_alphas.npy'

        self.model_state_path_list = [(Path(self.training_dic["model_path"]) / model_name / f'final_model_{i}.pt') for i in range(self.num_labels)]
        self.model_nd_state_path_list = [(Path(self.training_dic["model_path"]) / model_name / f'final_model_nd_{i}.pt') for i in range(self.num_labels)]
        self.optimizer_path_list = [(Path(self.training_dic["model_path"]) / model_name / f'final_optimizer_{i}.pt') for i in range(self.num_labels)]
        self.dict_path_list = [(Path(self.training_dic["model_path"]) / model_name / f'training_dict_{i}.pt') for i in range(self.num_labels)]
        self.log_path_list = [(Path(self.training_dic["model_path"]) / model_name / f'training_log_{i}.csv') for i in range(self.num_labels)]
        if "finetune" in self.training_dic and self.training_dic["finetune"]:
            self.log_path_finetune_list = [(Path(self.training_dic["model_path"]) / model_name / f'finetune_log_{i}.csv') for i in range(self.num_labels)]
        self.summary_path = Path("/dfs/data/tensorBoard") / cfg.train_basics["model_path_name"] / model_name
        if self.training_dic["eval_mask"]:
            self.model_mask_path_list = [(Path(self.training_dic["model_path"]) / model_name / f'final_model_mask_{i}.pt') for i in range(self.num_labels)]
            self.optimizer_mask_path_list = [(Path(self.training_dic["model_path"]) / model_name / f'final_optimizer_mask_{i}.pt') for i in range(self.num_labels)]
        self.model_most_path_list = [(Path(self.training_dic["model_path"]) / model_name / f'final_model_most_{i}.pt') for i in range(self.num_labels)]
        self.optimizer_most_path_list = [(Path(self.training_dic["model_path"]) / model_name / f'final_optimizer_most_{i}.pt') for i in range(self.num_labels)]

        if "feature_select" in self.training_dic and self.training_dic["feature_select"]:
            self.training_dic["features_delete"] = self.feature_selection(cfg)

        num_workers = max(4, min(16, torch.cuda.device_count() * 4))  # 根据GPU数量动态调整

        print("num_workers: ", num_workers)

        if Path(self.summary_path).exists():
            shutil.rmtree(self.summary_path)
        Path.mkdir(self.summary_path, exist_ok=True, parents=True)

        self.seq_len = self.training_dic["bptt"]
        
        self.label_handling()

        if "large_batch_size" not in self.training_dic:
            self.training_dic["large_batch_size"] = self.training_dic["batch_size"]

        days, stocks = self.indices[1].shape

        if self.training_dic.get("validation_by_return", False) or self.training_dic.get("cross_section", False):
            for i in range(days):
                if self.indices[1][i].sum() < 500:
                    self.indices[1][i] = False

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
            use_endpoints_eval = use_endpoints = False
            train_endpoints = eval_endpoints = None
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
            use_endpoints = True
            use_endpoints_eval = False
            eval_endpoints = None
        elif self.training_dic["sample_mode"] == "period":
            self.eval_period = self.training_dic.get("eval_period", 10)
            period_len = days // self.eval_period
            eval_days = math.floor(period_len * self.training_dic["eval_size"])
            eval_indices = np.zeros(indices[1].shape).astype(bool)
            pos_holder = np.zeros(indices[1].shape).astype(bool)
            for i in range(1, self.eval_period):
                for j in range(period_len * i - eval_days, period_len * i):
                    eval_indices[j] = indices[1][j]
                if "avoid_horizon" in self.training_dic and self.training_dic["avoid_horizon"]:
                    for l in range(max(0, period_len*i-eval_days-self.training_dic["horizon"]//2), min(days, period_len*i+self.training_dic["horizon"]//2)):
                        pos_holder[l] = indices[1][l]
                else:
                    for l in range(period_len * i - eval_days, period_len * i):
                        pos_holder[l] = indices[1][l]
            train_indices = indices[1] & ~pos_holder
            use_endpoints = use_endpoints_eval = False
            train_endpoints = eval_endpoints = None
        elif self.training_dic["sample_mode"] == "by_date":
            all_dates = np.unique(rows)
            _, eval_dates = train_test_split(range(len(all_dates)), test_size=self.training_dic["eval_size"], random_state=0, shuffle=True)
            eval_indices = np.zeros(indices[1].shape).astype(bool)
            place_holder = np.zeros(indices[1].shape).astype(bool)
            row_filter = np.unique(rows)[eval_dates]
            for i in range(indices[1].shape[0]):
                if i in row_filter:
                    eval_indices[i] = indices[1][i]
                    for j in range(max(0, i-self.seq_len), min(days, i+self.seq_len)):
                        place_holder[j] = indices[1][j]
            train_indices = indices[1] & ~place_holder
            use_endpoints = use_endpoints_eval = False
            train_endpoints = eval_endpoints = None
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
            use_endpoints = use_endpoints_eval = True
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
            use_endpoints = use_endpoints_eval = True
        
        self.train_dataset_list = []
        self.eval_dataset_list = []
        # import ipdb; ipdb.set_trace()

        extra_info = {"features": self.feature_matrix}
        if self.training_dic.get("validation_by_return", False):
            extra_info["ret_ratio"] = self.ret_ratio
        else:
            extra_info["ret_ratio"] = torch.zeros(1, days, stocks)
        if self.training_dic.get("weight_down_limited", False):
            extra_info["limited"] = self.limited
        else:
            extra_info["limited"] = torch.zeros(1, days, stocks)
        if self.training_dic.get("reduce_low_cap", False):
            extra_info["reduction_cap"] = self.reduction_cap

        self.batch_date = self.training_dic.get("batch_date", 1)

        for i in range(self.num_labels):            
            if self.training_dic.get("cross_section", False):
                if use_endpoints_eval:
                    self.train_dataset_list.append(SharedMemSeqDatasetDaily(model_name, self.train_input.shape, self.train_input.dtype, indices[0], train_indices & self.stocks_choice[i], self.train_output, self.seq_len, feature_matrix=extra_info, time_as_batch=False))
                else:
                    self.train_dataset_list.append(SharedMemSeqDatasetDaily(model_name, self.train_input.shape, self.train_input.dtype, indices[0], train_indices & self.stocks_choice[i], self.train_output, self.seq_len, feature_matrix=extra_info, time_as_batch=False))
            else:
                if use_endpoints:
                    self.train_dataset_list.append(SharedMemSeqDatasetRandom(model_name, self.train_input.shape, self.train_input.dtype, indices[0], train_indices & self.stocks_choice[i], self.train_output, self.seq_len, date_weight, use_endpoints, train_endpoints & self.stocks_choice[i], extra_info))
                else:
                    self.train_dataset_list.append(SharedMemSeqDatasetRandom(model_name, self.train_input.shape, self.train_input.dtype, indices[0], train_indices & self.stocks_choice[i], self.train_output, self.seq_len, date_weight, use_endpoints, None, extra_info))
            
            if self.training_dic.get("validation_by_return", False) or self.training_dic.get("cross_section", False):
                if use_endpoints_eval:
                    self.eval_dataset_list.append(SharedMemSeqDatasetDaily(model_name, self.train_input.shape, self.train_input.dtype, indices[0], eval_indices & self.stocks_choice[i], self.train_output, self.seq_len, feature_matrix=extra_info, time_as_batch=False))
                else:
                    self.eval_dataset_list.append(SharedMemSeqDatasetDaily(model_name, self.train_input.shape, self.train_input.dtype, indices[0], eval_indices & self.stocks_choice[i], self.train_output, self.seq_len, feature_matrix=extra_info, time_as_batch=False))
            else:
                if use_endpoints_eval:
                    self.eval_dataset_list.append(SharedMemSeqDatasetRandom(model_name, self.train_input.shape, self.train_input.dtype, indices[0], eval_indices & self.stocks_choice[i], self.train_output, self.seq_len, date_weight, use_endpoints_eval, eval_endpoints & self.stocks_choice[i], extra_info))
                else:
                    self.eval_dataset_list.append(SharedMemSeqDatasetRandom(model_name, self.train_input.shape, self.train_input.dtype, indices[0], eval_indices & self.stocks_choice[i], self.train_output, self.seq_len, date_weight, use_endpoints_eval, None, extra_info))

        if self.training_dic.get("cross_section", False):
            self.train_loader_list = [DataLoader(self.train_dataset_list[i], batch_size=1, shuffle=True, drop_last=True, prefetch_factor=4, pin_memory=True, num_workers=num_workers, persistent_workers=True, collate_fn=lambda x:x[0]) for i in range(self.num_labels)]
        else:
            self.train_loader_list = [DataLoader(self.train_dataset_list[i], batch_size=self.training_dic["large_batch_size"], shuffle=True, drop_last=True, prefetch_factor=4, pin_memory=True, num_workers=num_workers, persistent_workers=True) for i in range(self.num_labels)]

        if self.training_dic.get("validation_by_return", False) or self.training_dic.get("cross_section", False):
            self.eval_loader_list = [DataLoader(self.eval_dataset_list[i], batch_size=1, shuffle=False, drop_last=True, prefetch_factor=4, pin_memory=True, num_workers=num_workers, persistent_workers=True, collate_fn=lambda x:x[0]) for i in range(self.num_labels)]
        else:
            self.eval_loader_list = [DataLoader(self.eval_dataset_list[i], batch_size=self.training_dic["large_batch_size"]*2, shuffle=False, drop_last=True, prefetch_factor=4, pin_memory=True, num_workers=num_workers, persistent_workers=True) for i in range(self.num_labels)]

        if self.training_dic.get("cross_section", False) and self.batch_date != 1:
            self.train_loader_list = [ExteriorLoader(self.batch_date, self.train_loader_list[i]) for i in range(self.num_labels)]

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

            finetune_train_dataset_list = [SharedMemSeqDatasetRandom(model_name, self.train_input.shape, self.train_input.dtype, indices[0], train_indices & self.stocks_choice[i], self.train_output, self.seq_len, np.zeros(days), True, train_endpoints & self.stocks_choice[i], extra_info) for i in range(self.num_labels)]
            finetune_eval_dataset_list = [SharedMemSeqDatasetRandom(model_name, self.train_input.shape, self.train_input.dtype, indices[0], eval_indices & self.stocks_choice[i], self.train_output, self.seq_len, np.zeros(days), True, eval_endpoints & self.stocks_choice[i], extra_info) for i in range(self.num_labels)]

            self.finetune_train_loader_list = [DataLoader(finetune_train_dataset_list[i], batch_size=self.training_dic["batch_size"], shuffle=True, drop_last=True, prefetch_factor=4, pin_memory=True, num_workers=num_workers, persistent_workers=True) for i in range(self.num_labels)]
            self.finetune_eval_loader_list = [DataLoader(finetune_eval_dataset_list[i], batch_size=self.training_dic["batch_size"], shuffle=False, drop_last=True, prefetch_factor=4, pin_memory=True, num_workers=num_workers, persistent_workers=True) for i in range(self.num_labels)]

        for i in range(self.num_labels):
            print(f"For label {i}: len of train_data: {len(self.train_dataset_list[i])}, len of eval_data: {len(self.eval_dataset_list[i])}, len of train_loader: {len(self.train_loader_list[i])}, len of eval_loader: {len(self.eval_loader_list[i])}")

        if ("discrimination" in self.training_dic and self.training_dic["discrimination"]) or ("finetune" in self.training_dic and self.training_dic["finetune"]):
            for i in range(self.num_labels):
                print(f"For label {i}: len of finetune_train_data: {len(finetune_train_dataset_list[i])}, len of finetune_eval_data: {len(finetune_eval_dataset_list[i])}, len of finetune_train_loader: {len(self.finetune_train_loader_list[i])}, len of finetune_eval_loader: {len(self.finetune_eval_loader_list[i])}")

        self.input_dim = np.sum(indices[0])


        self.cfg_list = [deepcopy(cfg) for _ in range(self.num_labels)]

        if cfg.model.model_params.get("trident", False):
            self.buckets_info = deepcopy(self.training_dic["num_buckets"])
            if "saved_alpha_path" in self.training_dic:
                self.saved_alphas_loc = Path(self.training_dic["saved_alpha_path"]) / ("saved_train_alphas_" + pit_tidx + "_processed.npy")
            else:
                self.saved_alphas_loc = Path("/dfs/data/ksim/automation/20240116/readcache_new/data") / model_path_name / ("saved_train_alphas_" + pit_tidx + "_processed.npy")
            for label_idx in range(self.num_labels):
                _, feature_list = self.feature_bucketing(cfg, label_idx)
                self.cfg_list[label_idx].model.model_params["trident_heads"] = feature_list
                
        if "extra_feature" in self.training_dic and self.training_dic["extra_feature"]:
            self.input_dim += self.feature_matrix.shape[0]
        if cfg.model.type in ["KANGRU", "KANsformer"]:
            for i in range(self.num_labels):
                self.cfg_list[i].model.model_params["input_dim"] = self.input_dim
        if cfg.model.type == "KANGRU":
            hidden_dim = 1
            while hidden_dim < self.input_dim // 2:
                hidden_dim <<= 1
            for i in range(self.num_labels):
                self.cfg_list[i].model.model_params["hidden_dim"] = hidden_dim
                self.cfg_list[i].model.model_params["seq_len"] = self.seq_len
            embed_dim = 1
            while embed_dim < self.input_dim // 4:
                embed_dim <<= 1
            # embed_dim = self.input_dim // 4
            embed_dim //= 2
            for i in range(self.num_labels):
                if isinstance(self.cfg_list[i].model.model_params["embedding_dim"],list):
                    self.cfg_list[i].model.model_params["embedding_dim"][-1] = embed_dim
                else:
                    self.cfg_list[i].model.model_params["embedding_dim"] = embed_dim
                self.cfg_list[i].model.model_params["expand_dim"] = math.ceil(embed_dim * 1.5)
                self.cfg_list[i].model.model_params["head_dim"] = embed_dim // (self.cfg_list[i].model.model_params["num_heads"])
        if lock is not None:
            with lock:
                time.sleep(random.randint(1,30))
                super().gpu_helper(*args, **kwargs)
        if self.training_dic.get("feature_insertion", False):
            feature_insert = np.zeros(self.input_dim, dtype=bool)
            for i in range(self.insert_num):
                feature_insert[-1-i] = True
            for i in range(self.num_labels):
                self.cfg_list[i].model.model_params["feature_insert"] = feature_insert
        if self.training_dic.get("time_choice_block", False):
            time_choice_feature = np.zeros(self.input_dim, dtype=bool)
            for i in range(indices[0].sum(), indices[0].sum()+self.time_choice_feature_num):
                time_choice_feature[i] = True
            for i in range(self.num_labels):
                self.cfg_list[i].model.model_params["time_choice_feature"] = time_choice_feature
                self.cfg_list[i].model.model_params["barra_dim"] = self.barra_dim
        for i in range(self.num_labels):
            self.cfg_list[i].model.model_params["device"] = self.device

        self.model_list = [create_model(self.cfg_list[i].model).to(self.device) for _ in range(self.num_labels)]


        if self.training_dic.get("breakpoint_continue", False) and Path(self.finish_flag_loc).exists() and Path(self.used_alphas_loc).exists():
            used_alphas = np.load(self.used_alphas_loc)
            saved_alphas = np.load(self.saved_alphas_loc)
            if Path(self.dict_path_list[0]).exists():
                saved_cfg = torch.load(self.dict_path_list[0])
                if np.all(used_alphas==saved_alphas) and self.cfg_list[0]==saved_cfg:
                    return self.evaluation(self.cfg_list)

        for i in range(self.num_labels):
            torch.save(self.cfg_list[i], self.dict_path_list[i])


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

        if self.training_dic.get("sep_learning_rate", False):
            time_series_params_list = [[p for n, p in self.model_list[i].named_parameters() if "KANsformer" in n] for i in range(self.num_labels)]
            linear_params_list = [[p for n, p in self.model_list[i].named_parameters() if "KANsformer" not in n] for i in range(self.num_labels)]

        if self.training_dic['opt'] == "Adam":
            if self.training_dic.get("sep_learning_rate", False):
                self.optimizer_list = [torch.optim.Adam([dict(params=time_series_params_list[i], lr=lr, weight_decay=self.weight_decay), dict(params=linear_params_list[i], lr=lr/10, weight_decay=self.weight_decay)]) for i in range(self.num_labels)]
            else:
                self.optimizer_list = [torch.optim.Adam(self.model_list[i].parameters(), lr=lr, weight_decay=self.weight_decay) for i in range(self.num_labels)]
        elif self.training_dic['opt'] == "AdamW":
            if self.training_dic.get("sep_learning_rate", False):
                self.optimizer_list = [torch.optim.AdamW([dict(params=time_series_params_list[i], lr=lr, weight_decay=self.weight_decay), dict(params=linear_params_list[i], lr=lr/10, weight_decay=self.weight_decay)]) for i in range(self.num_labels)]
            else:
                self.optimizer_list = [torch.optim.AdamW(self.model_list[i].parameters(), lr=lr, weight_decay=self.weight_decay) for i in range(self.num_labels)]
        elif self.training_dic['opt'] == "RMSprop":
            if self.training_dic.get("sep_learning_rate", False):
                self.optimizer_list = [torch.optim.RMSprop([dict(params=time_series_params_list[i], lr=lr, weight_decay=self.weight_decay), dict(params=linear_params_list[i], lr=lr/10, weight_decay=self.weight_decay)]) for i in range(self.num_labels)]
            else:
                self.optimizer_list = [torch.optim.RMSprop(self.model_list[i].parameters(), lr=lr, weight_decay=self.weight_decay) for i in range(self.num_labels)]
        elif self.training_dic["opt"] == "Lion":
            if self.training_dic.get("sep_learning_rate", False):
                self.optimizer_list = [Lion([dict(params=time_series_params_list[i], lr=lr, weight_decay=self.weight_decay), dict(params=linear_params_list[i], lr=lr/10, weight_decay=self.weight_decay)]) for i in range(self.num_labels)]
            else:
                self.optimizer_list = [Lion(self.model_list[i].parameters(), lr=lr/10, weight_decay=self.weight_decay) for i in range(self.num_labels)]
        elif self.training_dic["opt"] == "Muon":
            muon_params_list = [[p for p in self.model_list[i].parameters() if p.ndim >= 2] for i in range(self.num_labels)]

            adamw_params_list = [[p for p in self.model_list[i].parameters() if p.ndim < 2] for i in range(self.num_labels)]
            self.optimizer_list = [SingleDeviceMuonWithAuxAdam([dict(params=muon_params_list[i], use_muon=True, lr=lr, weight_decay=self.weight_decay), dict(params=adamw_params_list[i], use_muon=False, lr=lr/10, weight_decay=self.weight_decay)]) for i in range(self.num_labels)]
        elif self.training_dic['opt'] == "SGD":
            if self.training_dic.get("sep_learning_rate", False):
                self.optimizer_list = [torch.optim.SGD([dict(params=time_series_params_list[i], lr=lr, weight_decay=self.weight_decay), dict(params=linear_params_list[i], lr=lr/10, weight_decay=self.weight_decay)]) for i in range(self.num_labels)]
            else:
                self.optimizer_list = [torch.optim.SGD(self.model_list[i].parameters(), lr=lr, weight_decay=self.weight_decay) for i in range(self.num_labels)]
        elif self.training_dic['opt'] == "LBFGS":
            if self.training_dic.get("sep_learning_rate", False):
                self.optimizer_list = [torch.optim.LBFGS([dict(params=time_series_params_list[i], lr=lr, weight_decay=self.weight_decay), dict(params=linear_params_list[i], lr=lr/10, weight_decay=self.weight_decay)]) for i in range(self.num_labels)]
            else:
                self.optimizer_list = [torch.optim.LBFGS(self.model_list[i].parameters(), lr=lr, history_size=10, tolerance_grad=1e-32, tolerance_change=1e-32) for i in range(self.num_labels)]
        else:
            raise ValueError("opt must be Adam, AdamW, SGD, LBFGS, or RMSprop.")

        writer = SummaryWriter(self.summary_path / "run")

        # import ipdb; ipdb.set_trace()

        result = []
        for i in range(self.num_labels):
            result.append({"train_loss": [], "eval_loss": [], "eval_mse_loss": [], "eval_correlation_loss": [], "eval_correlation_mask": [],  "good_points": []})
            if self.training_dic.get("validation_by_return", False) or self.training_dic.get("cross_section", False):
                if self.training_dic.get("validation_by_return", False):
                    result[i]["return"] = []
                result[i].pop("good_points")
                result[i].pop("eval_correlation_mask")


        scheduler_list = [torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_list[i], "min", factor=self.training_dic["scheduler_factor"], patience=self.training_dic["scheduler_patience"], threshold=self.training_dic["scheduler_threshold"]) for i in range(self.num_labels)]



        if "discrimination" in self.training_dic and self.training_dic["discrimination"]:
            for i in range(self.num_labels):
                result[i]["adv_loss"] = []

            self.adversarial_list = [Adversarial(self.cfg_list[i], self.device, self.input_dim) for i in range(self.num_labels)]

        if not "step_num" in self.training_dic:
            self.training_dic["step_num"] = 30

        for label_idx in range(self.num_labels):

            self.eval_steps = len(self.train_loader_list[label_idx]) // (self.training_dic["step_num"])
            self.model = self.model_list[label_idx]
            self.optimizer = self.optimizer_list[label_idx]
            scheduler = scheduler_list[label_idx]

            train_y = train_output[indices[1] & self.stocks_choice[label_idx]]

            var = np.var(train_y)
            std = np.std(train_y)
            mean = np.mean(train_y)
            skewness = skew(train_y)
            kurt = kurtosis(train_y)

            print(f"target stats| Mean: {mean.item()}| Variance: {var.item()}| Skewness: {skewness}| Kurtosis: {kurt}")

            pbar = tqdm(range(self.training_dic["epochs"]), desc="Training", ncols=200)

            best_eval_loss = float("inf")
            if self.training_dic["eval_mask"]:
                best_correlation_mask_loss = -float("inf")
            retrain_cnt = 1
            overfit_cnt = 0
            pred_mean = 0
            pred_std = 0
            best_points_cnt = 0
            total_step = 0

            if self.training_dic.get("validation_by_return", False):
                highest_ret = - float("inf")

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
                    if self.training_dic.get("cross_section", False):
                        dis_pred = torch.tensor([], device=self.device)
                        dis_data = torch.tensor([], device=self.device)
                        dis_target = torch.tensor([], device=self.device)
                epoch_training_loss = 0
                batch_num = 0
                step_cnt = 0
                adv_loss = 0    
                self.model.train()
                for data in self.train_loader_list[label_idx]:
                    # import ipdb; ipdb.set_trace()
                    if self.batch_date == 1 or not self.training_dic.get("cross_section", False):
                        extra_batch = data[-1]
                        if self.training_dic.get("cross_section", False):
                            if self.training_dic.get("reduce_low_cap", False):
                                reduction_cap = extra_batch["reduction_cap"].squeeze()
                                if self.training_dic.get("reduce_method", "addition") == "addition":
                                    target_batch = (data[1].squeeze() + reduction_cap).to(self.device, non_blocking=True, dtype=data[1].dtype)
                                elif self.training_dic.get("reduce_method", "addition") == "multiplication":
                                    target_batch = (data[1].squeeze() * reduction_cap).to(self.device, non_blocking=True, dtype=data[1].dtype)
                            else:                                
                                target_batch = data[1].squeeze().to(self.device, non_blocking=True, dtype=data[1].dtype)
                            extra_feat = extra_batch["features"]
                            if "extra_feature" in self.training_dic and self.training_dic["extra_feature"]:
                                # print(data[0].shape, data[-1].shape)
                                data_batch = torch.cat([data[0].squeeze(), extra_feat], dim=-1).to(self.device, non_blocking=True, dtype=data[0].dtype)
                            else:
                                data_batch = data[0].squeeze().to(self.device, non_blocking=True)
                            if self.training_dic.get("down_weight_limited", False):
                                limited = extra_batch["limited"].squeeze().to(self.device, non_blocking=True, dtype=data[1].dtype)
                            else:
                                limited = 1
                        else:
                            if self.training_dic.get("reduce_low_cap", False):
                                reduction_cap = extra_batch["reduction_cap"].squeeze()
                                if self.training_dic.get("reduce_method", "addition") == "addition":
                                    target_batch = (data[1] + reduction_cap).to(self.device, non_blocking=True, dtype=data[1].dtype)
                                elif self.training_dic.get("reduce_method", "addition") == "multiplication":
                                    target_batch = (data[1] * reduction_cap).to(self.device, non_blocking=True, dtype=data[1].dtype)
                            else:                                
                                target_batch = data[1].to(self.device, non_blocking=True, dtype=data[1].dtype)
                            extra_feat = extra_batch["features"]
                            if self.training_dic.get("down_weight_limited", False):
                                limited = extra_batch["limited"].squeeze().to(self.device, non_blocking=True, dtype=data[1].dtype)
                            else:
                                limited = 1
                        # import ipdb; ipdb.set_trace()
                            if "extra_feature" in self.training_dic and self.training_dic["extra_feature"]:
                                # print(data[0].shape, data[-1].shape)
                                data_batch = torch.cat([data[0], extra_feat], dim=-1).to(self.device, non_blocking=True, dtype=data[0].dtype)
                            else:
                                data_batch = data[0].to(self.device, non_blocking=True)
                        if self.training_dic.get("cross_section", False):
                            if "only_target" in self.training_dic and self.training_dic["only_target"]:
                                target_batch = target_batch[:,-1]
                            weight = torch.ones_like(target_batch, dtype=target_batch.dtype, device=target_batch.device)
                        else:
                            if "only_target" in self.training_dic and self.training_dic["only_target"]:
                                target_batch = target_batch[:,-1]
                                weight = 1 + data[2].squeeze()[:,-1].to(self.device, non_blocking=True)
                            else:
                                weight = 1 + data[2].squeeze().to(self.device, non_blocking=True)
                        with torch.no_grad():
                            pred = self.model.forward(data_batch)
                        pred = pred.detach().requires_grad_(True)
                        if "only_target" in self.training_dic and self.training_dic["only_target"]:
                            pred_used = pred[:,-1]
                        else:
                            pred_used = pred
                        # import ipdb; ipdb.set_trace()
                        if self.training_dic["weight_essential"]:
                            if "weight_pos_up" in self.training_dic:
                                upper_adjust_t = F.sigmoid(8 * (mean + self.training_dic["weight_pos_up"] * std - target_batch))
                                upper_adjust_p = F.sigmoid(8 * (pred_mean + self.training_dic["weight_pos_up"] * pred_std - pred))
                            else:
                                upper_adjust_t = upper_adjust_p = 1
                            weight += torch.max(F.sigmoid(8 * (target_batch - mean - self.training_dic["weight_pos"] * std)) * upper_adjust_t, F.sigmoid(8 * (pred_used - pred_mean - self.training_dic["weight_pos"] * pred_std)) * upper_adjust_p) * self.training_dic["weight_pow_over"] * limited
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
                        total_pred = pred
                        total_data = data_batch
                        total_target = target_batch
                    else:
                        train_loss = 0
                        total_pred = []
                        total_data = []
                        total_target = []
                        for chunk in data:
                            extra_batch = chunk[-1]
                            if self.training_dic.get("reduce_low_cap", False):
                                reduction_cap = extra_batch["reduction_cap"].squeeze()
                                if self.training_dic.get("reduce_method", "addition") == "addition":
                                    target_batch = (chunk[1].squeeze() + reduction_cap).to(self.device, non_blocking=True, dtype=chunk[1].dtype)
                                elif self.training_dic.get("reduce_method", "addition") == "multiplication":
                                    target_batch = (chunk[1].squeeze() * reduction_cap).to(self.device, non_blocking=True, dtype=chunk[1].dtype)
                            else:
                                target_batch = chunk[1].squeeze().to(self.device, non_blocking=True, dtype=chunk[1].dtype)
                            extra_feat = extra_batch["features"]
                            if "extra_feature" in self.training_dic and self.training_dic["extra_feature"]:
                                # print(data[0].shape, data[-1].shape)
                                data_batch = torch.cat([chunk[0].squeeze(), extra_feat], dim=-1).to(self.device, non_blocking=True, dtype=chunk[0].dtype)
                            else:
                                data_batch = chunk[0].squeeze().to(self.device, non_blocking=True)
                            if self.training_dic.get("down_weight_limited", False):
                                limited = extra_batch["limited"].squeeze().to(self.device, non_blocking=True, dtype=chunk[1].dtype)
                            else:
                                limited = 1
                            if "only_target" in self.training_dic and self.training_dic["only_target"]:
                                target_batch = target_batch[:,-1]
                            weight = torch.ones_like(target_batch, dtype=target_batch.dtype, device=target_batch.device)
                            with torch.no_grad():
                                pred = self.model.forward(data_batch)
                            pred = pred.detach().requires_grad_(True)
                            total_pred.append(pred)
                            total_data.append(data_batch)
                            total_target.append(target_batch)
                            if "only_target" in self.training_dic and self.training_dic["only_target"]:
                                pred_used = pred[:,-1]
                            else:
                                pred_used = pred
                            # import ipdb; ipdb.set_trace()
                            if self.training_dic["weight_essential"]:
                                if "weight_pos_up" in self.training_dic:
                                    upper_adjust_t = F.sigmoid(8 * (mean + self.training_dic["weight_pos_up"] * std - target_batch))
                                    upper_adjust_p = F.sigmoid(8 * (pred_mean + self.training_dic["weight_pos_up"] * pred_std - pred))
                                else:
                                    upper_adjust_t = upper_adjust_p = 1
                                weight += torch.max(F.sigmoid(8 * (target_batch - mean - self.training_dic["weight_pos"] * std)) * upper_adjust_t, F.sigmoid(8 * (pred_used - pred_mean - self.training_dic["weight_pos"] * pred_std)) * upper_adjust_p) * self.training_dic["weight_pow_over"] * limited
                            if self.training_dic["loss_fn"] in ["CCC", "Correlation", "RankCorrelation"]:
                                correlation_loss = loss_fn(pred_used, target_batch, weight)
                                reg_loss = reg_fn(pred_used, target_batch)
                                if self.training_dic["weight_essential"]:
                                    reg_loss = (reg_loss * weight)
                                reg_loss = reg_loss.mean()
                                train_loss += - self.correlation_ratio * correlation_loss + (1 - self.correlation_ratio) * reg_loss
                            else:
                                reg_loss = reg_fn(pred_used, target_batch)
                                if self.training_dic["weight_essential"]:
                                    reg_loss = (reg_loss * weight)
                                reg_loss = torch.mean(reg_loss)
                                train_loss += reg_loss
                        total_data = torch.cat(total_data, dim=0)
                        total_target = torch.cat(total_target, dim=0)
                    if self.training_dic["enable_l1"]:
                        l1_reg = self.training_dic["l1_ratio"] * self.l1_regularization()
                        train_loss += l1_reg
                    self.optimizer.zero_grad()
                    train_loss.backward()
                    # import ipdb; ipdb.set_trace()
                    if self.batch_date != 1:
                        intermediate_grad = torch.cat([pred.grad.detach().squeeze()/pred.shape[0]*self.training_dic["large_batch_size"] for pred in total_pred], dim=0)
                        length = sum([pred.shape[0] for pred in total_pred])
                    else:
                        intermediate_grad = pred.grad.detach().squeeze()/pred.shape[0]*self.training_dic["batch_size"]
                        length = total_pred.shape[0]
                    step_cnt += 1
                    for i in range(length // self.training_dic["batch_size"]):
                        grad_chunk = intermediate_grad[self.training_dic["batch_size"]*i: self.training_dic["batch_size"]*(i+1)]
                        y_chunk = torch.squeeze(self.model.forward(total_data[self.training_dic["batch_size"]*i: self.training_dic["batch_size"]*(i+1)]))
                        y_chunk.backward(gradient=grad_chunk)
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_norm)
                    self.optimizer.step()
                    if self.batch_date != 1:
                        for pred in total_pred:
                            pred.requires_grad_(False)
                        total_pred = torch.cat(total_pred, dim=0)
                    else:
                        pred.requires_grad_(False)
                    if "discrimination" in self.training_dic and self.training_dic["discrimination"]:
                        if not self.training_dic.get("cross_section", False):
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
                                    loss_dis = self.adversarial_list[label_idx].train_with_model(dis_pred, dis_data, dis_target)
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
                                loss_dis = self.adversarial_list[label_idx].train_with_model(pred, data_batch[:,-1,:], target_batch)
                                adv_loss += loss_dis
                        else:
                            unfinished_len = total_pred.shape[0]
                            pos = 0
                            len_diff = cfg.dis_train.model_params["dis_target_size"] - dis_pred.shape[0]
                            # import ipdb; ipdb.set_trace()
                            while unfinished_len > 0:
                                unfinished_len -= len_diff
                                if "only_target" not in self.training_dic or not self.training_dic["only_target"]:
                                    dis_target = torch.cat([dis_target, total_target[pos:len_diff+pos, -1]], dim=0)
                                dis_pred = torch.cat([dis_pred, total_pred[pos:len_diff+pos, -1]], dim=0)
                                dis_data = torch.cat([dis_data, total_data[pos:len_diff+pos,-1,:]], dim=0)
                                pos = len_diff + pos
                                if cfg.dis_train.model_params["dis_target_size"] == dis_pred.shape[0]:
                                    loss_dis = self.adversarial_list[label_idx].train_with_model(dis_pred, dis_data, dis_target)
                                    len_diff = cfg.dis_train.model_params["dis_target_size"]
                                    dis_pred = torch.tensor([], device=self.device)
                                    dis_data = torch.tensor([], device=self.device)
                                    dis_target = torch.tensor([], device=self.device)
                    epoch_training_loss += train_loss.item()
                    batch_num += 1

                    if step_cnt == self.eval_steps:
                        
                        step_cnt = 0
                        total_step += 1

                        result[label_idx]["train_loss"].append(epoch_training_loss / batch_num)

                        if "discrimination" in self.training_dic and self.training_dic["discrimination"]:
                            result[label_idx]["adv_loss"].append(adv_loss/batch_num)


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
                            if self.training_dic.get("validation_by_return", False) or self.training_dic.get("cross_section", False):
                                if self.training_dic.get("validation_by_return", False):
                                    total_return = 0
                                start_time = time.time()
                                for data in self.eval_loader_list[label_idx]:
                                    batch_pred = []
                                    extra_batch = data[-1]
                                    # import ipdb; ipdb.set_trace()
                                    if self.training_dic.get("reduce_low_cap", False):
                                        reduction_cap = extra_batch["reduction_cap"].squeeze()
                                        if self.training_dic.get("reduce_method", "addition") == "addition":
                                            target_batch = (data[1] + reduction_cap).to(self.device, non_blocking=True, dtype=data[1].dtype)
                                        elif self.training_dic.get("reduce_method", "addition") == "multiplication":
                                            target_batch = (data[1] * reduction_cap).to(self.device, non_blocking=True, dtype=data[1].dtype)
                                    else:
                                        target_batch = data[1].squeeze().to(self.device, non_blocking=True, dtype=data[1].dtype)
                                    extra_feat = extra_batch["features"]
                                    if self.training_dic.get("down_weight_limited", False):
                                        limited = extra_batch["limited"].squeeze().to(self.device, non_blocking=True, dtype=data[1].dtype)[:,-1]
                                    else:
                                        limited = 1
                                    target_batch = target_batch.squeeze()[:,-1]
                                    if "extra_feature" in self.training_dic and self.training_dic["extra_feature"]:
                                        # print(data[0].shape, data[-1].shape)
                                        data_batch = torch.cat([data[0].squeeze(), extra_feat], dim=-1).to(self.device, non_blocking=True, dtype=data[0].dtype)
                                    else:
                                        data_batch = data[0].squeeze().to(self.device, non_blocking=True)
                                    # target_batch = data[1][:,-1].squeeze().to(self.device, non_blocking=True)
                                    for i in range(math.ceil(data_batch.shape[0]/(self.training_dic["large_batch_size"]*2))):
                                        batch_pred.append(self.model.forward(data_batch[i*2*self.training_dic["large_batch_size"]:(i+1)*2*self.training_dic["large_batch_size"]], infer=True)[:,1])
                                    pred = torch.cat(batch_pred, dim=0)
                                    if self.training_dic.get("validation_by_return", False):
                                        ret_ratio = extra_batch["ret_ratio"].squeeze().to(self.device, non_blocking=True, dtype=data[1].dtype)[:,-1]
                                    # import ipdb; ipdb.set_trace()
                                    total_pred.append(pred.detach().squeeze())
                                    total_target.append(target_batch.squeeze())
                                    weight = torch.ones_like(pred, device=data_batch.device)
                                    if self.training_dic["weight_essential"]:
                                        if "mask_pos_up" in self.training_dic:
                                            upper_adjust_p = F.sigmoid(8 * (pred_mean + self.training_dic["mask_pos_up"] * pred_std - pred))
                                        else:
                                            upper_adjust_p = 1
                                        weight += F.sigmoid(8 * (pred - pred_mean - self.training_dic["mask_pos"] * pred_std)) * upper_adjust_p * self.training_dic["weight_pow_under"] * limited
                                    # import ipdb; ipdb.set_trace()
                                    new_mse_loss = F.mse_loss(pred, target_batch).item()
                                    mse_loss += new_mse_loss
                                    if self.training_dic["loss_fn"] in ["CCC", "Correlation", "RankCorrelation"]:
                                        new_correlation_loss = loss_fn_eval(pred, target_batch, weight).item()
                                        correlation_loss += new_correlation_loss
                                        eval_loss += -self.correlation_ratio * new_correlation_loss + (1 - self.correlation_ratio) * new_mse_loss
                                    else:
                                        eval_loss += new_mse_loss
                                        correlation_loss += self.correlation_loss(pred, target_batch).item()
                                    if self.training_dic.get("validation_by_return", False):
                                        pred -= pred.mean(dim=0)
                                        abs_sum = torch.sum(torch.abs(pred))
                                        total_return += ((pred * ret_ratio).sum() / abs_sum).item()
                                eval_loss /= len(self.eval_loader_list[label_idx])
                                mse_loss /= len(self.eval_loader_list[label_idx])
                                correlation_loss /= len(self.eval_loader_list[label_idx])

                                total_pred = torch.cat(total_pred, dim=0)
                                total_target = torch.cat(total_target, dim=0)

                                pred_mean = torch.mean(total_pred)
                                pred_std = torch.std(total_pred)
                                
                                if self.training_dic.get("validation_by_return", False):
                                    if not result[label_idx]["return"] or total_return > highest_ret:
                                        # best_correlation_loss = correlation_loss
                                        # if eval_loss < best_eval_loss:
                                        highest_ret = total_return
                                        torch.save(self.model.to("cpu").state_dict(), self.model_state_path_list[label_idx])
                                        torch.save(self.optimizer.state_dict(), self.optimizer_path_list[label_idx])
                                        self.model.to(self.device)
                                else:
                                    if not result[label_idx]["eval_loss"] or (mse_loss - 10 * correlation_loss < best_eval_loss):
                                        best_eval_loss = mse_loss - 10 * correlation_loss
                                        torch.save(self.model.to("cpu").state_dict(), self.model_state_path_list[label_idx])
                                        torch.save(self.optimizer.state_dict(), self.optimizer_path_list[label_idx])
                                        self.model.to(self.device)
                                
                                if self.training_dic.get("validation_by_return", False):
                                    if result[label_idx]["return"] and total_return < max(result[label_idx]["return"]) + self.eps and total_step >= self.training_dic["overfit_threshold"]:
                                        overfit_cnt += 1
                                    else:
                                        overfit_cnt = 0

                                    result[label_idx]["eval_loss"].append(mse_loss - 10 * correlation_loss)

                                    pbar.set_description("Label: %d| Epoch :%d|Train Loss: %.2e|Evaluation Ret: %.2e|MSE Loss: %.2e|Correlation: %.2e|LR: %.2e|Retrain_LR: %.2e|%.2e" % (label_idx+1, epoch+1, result[label_idx]["train_loss"][-1], total_return, mse_loss, correlation_loss, self.optimizer.param_groups[0]["lr"], self.training_dic["retrain_lr"], time.time()-start_time))


                                    writer.add_scalar("Training_Stats/Train Loss", train_loss, total_step)
                                    writer.add_scalar("Training_Stats/Evaluation Ret", total_return, total_step)
                                    writer.add_scalar("Training_Stats/MSE Loss", mse_loss, total_step)
                                    writer.add_scalar("Training_Stats/Correlation", correlation_loss, total_step)
                                    writer.add_scalar("Training_Stats/Learning rate", self.optimizer.param_groups[0]["lr"], total_step)

                                    result[label_idx]["eval_mse_loss"].append(mse_loss)
                                    result[label_idx]["eval_correlation_loss"].append(correlation_loss)
                                    result[label_idx]["return"].append(total_return)
                                else:
                                    if result[label_idx]["eval_loss"] and mse_loss - 10 * correlation_loss > min(result[label_idx]["eval_loss"]) + self.eps and total_step >= self.training_dic["overfit_threshold"]:
                                        overfit_cnt += 1
                                    else:
                                        overfit_cnt = 0

                                    result[label_idx]["eval_loss"].append(mse_loss - 10 * correlation_loss)

                                    pbar.set_description("Label: %d| Epoch :%d|Train Loss: %.2e|Evaluation Loss: %.2e|MSE Loss: %.2e|Correlation: %.2e|LR: %.2e|Retrain_LR: %.2e|%.2e" % (label_idx+1, epoch+1, result[label_idx]["train_loss"][-1], eval_loss, mse_loss, correlation_loss, self.optimizer.param_groups[0]["lr"], self.training_dic["retrain_lr"], time.time()-start_time))


                                    writer.add_scalar("Training_Stats/Train Loss", train_loss, total_step)
                                    writer.add_scalar("Training_Stats/Evaluation Loss", eval_loss, total_step)
                                    writer.add_scalar("Training_Stats/MSE Loss", mse_loss, total_step)
                                    writer.add_scalar("Training_Stats/Correlation", correlation_loss, total_step)
                                    writer.add_scalar("Training_Stats/Learning rate", self.optimizer.param_groups[0]["lr"], total_step)

                                    result[label_idx]["eval_mse_loss"].append(mse_loss)
                                    result[label_idx]["eval_correlation_loss"].append(correlation_loss)


                                if self.training_dic["retrain"] and ((overfit_cnt >= self.training_dic["overfit_patience"]) or ((self.optimizer.param_groups[0]["lr"] <= self.training_dic["retrain_lr"]/self.threshold_ratio))):
                                    if self.optimizer.param_groups[0]["lr"] <= self.training_dic["retrain_lr"]/self.threshold_ratio:
                                        self.training_dic["retrain_lr"] *= self.training_dic["retrain_lr_factor"]
                                    overfit_cnt = 0
                                    skip_reset = False
                                    if self.optimizer.param_groups[0]["lr"]/2 >= self.training_dic["retrain_lr"]:
                                        self.training_dic["retrain_lr"] = self.optimizer.param_groups[0]["lr"]/2
                                        skip_reset = True
                                    self.model.load_state_dict(torch.load(self.model_state_path_list[label_idx], map_location=self.device))
                                    self.optimizer.load_state_dict(torch.load(self.optimizer_path_list[label_idx]))
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
                                df = pd.DataFrame(result[label_idx])
                                df.to_csv(self.log_path_list[label_idx], index=False, header = True)
                            else:
                                start_time = time.time()
                                for data in self.eval_loader_list[label_idx]:
                                    extra_batch = data[-1]
                                    if self.training_dic.get("reduce_low_cap", False):
                                        reduction_cap = extra_batch["reduction_cap"].squeeze()
                                        if self.training_dic.get("reduce_method", "addition") == "addition":
                                            target_batch = (data[1] + reduction_cap).to(self.device, non_blocking=True, dtype=data[1].dtype)
                                        elif self.training_dic.get("reduce_method", "addition") == "multiplication":
                                            target_batch = (data[1] * reduction_cap).to(self.device, non_blocking=True, dtype=data[1].dtype)
                                    else:
                                        target_batch = data[1].to(self.device, non_blocking=True, dtype=data[1].dtype)
                                    extra_feat = extra_batch["features"]
                                    if "extra_feature" in self.training_dic and self.training_dic["extra_feature"]:
                                        # print(data[0].shape, data[-1].shape)
                                        data_batch = torch.cat([data[0], extra_feat], dim=-1).to(self.device, non_blocking=True, dtype=data[0].dtype)
                                    else:
                                        data_batch = data[0].to(self.device, non_blocking=True)
                                    if self.training_dic.get("down_weight_limited", False):
                                        limited = extra_batch["limited"].squeeze().to(self.device, non_blocking=True, dtype=data[1].dtype)[:, -1]
                                    else:
                                        limited = 1
                                    target_batch = target_batch[:, -1]
                                    weight = 1 + data[2][:,-1].to(self.device, non_blocking=True)
                                    pred = self.model.forward(data_batch, infer=True)[:,-1]
                                    new_mse_loss = F.mse_loss(pred, target_batch).item()
                                    mse_loss += new_mse_loss
                                    if self.training_dic["weight_essential"]:
                                        if "mask_pos_up" in self.training_dic:
                                            upper_adjust_p = F.sigmoid(8 * (pred_mean + self.training_dic["mask_pos_up"] * pred_std - pred))
                                        else:
                                            upper_adjust_p = 1
                                        weight += F.sigmoid(8 * (pred - pred_mean - self.training_dic["mask_pos"] * pred_std)) * upper_adjust_p * self.training_dic["weight_pow_under"] * limited
                                    if self.training_dic["loss_fn"] in ["CCC", "Correlation", "RankCorrelation"]:
                                        new_correlation_loss = loss_fn_eval(pred, target_batch, weight).item()
                                        correlation_loss += new_correlation_loss
                                        eval_loss += -self.correlation_ratio * new_correlation_loss + (1-self.correlation_ratio) * new_mse_loss
                                    else:
                                        eval_loss += new_mse_loss
                                        correlation_loss += self.correlation_loss(pred, target_batch).item()
                                    total_pred.append(pred.squeeze())
                                    total_target.append(target_batch.squeeze())
                                eval_loss /= len(self.eval_loader_list[label_idx])
                                mse_loss /= len(self.eval_loader_list[label_idx])
                                correlation_loss /= len(self.eval_loader_list[label_idx])

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
                                result[label_idx]["good_points"].append(torch.sum((total_pred > pred_mean + self.training_dic["mask_pos"] * pred_std)&(total_target > mean + self.training_dic["mask_pos"] * std)).item())
                                
                                if not result[label_idx]["eval_loss"] or (mse_loss - 10 * correlation_loss < best_eval_loss):
                                    # best_correlation_loss = correlation_loss
                                    # if eval_loss < best_eval_loss:
                                    best_eval_loss = mse_loss - 10 * correlation_loss
                                    torch.save(self.model.to("cpu").state_dict(), self.model_state_path_list[label_idx])
                                    torch.save(self.optimizer.state_dict(), self.optimizer_path_list[label_idx])
                                    self.model.to(self.device)
                                
                                if result[label_idx]["eval_loss"] and mse_loss - 10 * correlation_loss > min(result[label_idx]["eval_loss"]) -self.eps and total_step >= self.training_dic["overfit_threshold"]:
                                    overfit_cnt += 1
                                else:
                                    overfit_cnt = 0

                                result[label_idx]["eval_loss"].append(mse_loss - 10 * correlation_loss)

                                pbar.set_description("Label: %d| Epoch :%d|Train Loss: %.2e|Evaluation Loss: %.2e|MSE Loss: %.2e|Correlation: %.2e|LR: %.2e|Retrain_LR: %.2e|%.2e" % (label_idx+1, epoch+1, result[label_idx]["train_loss"][-1], eval_loss, mse_loss, correlation_loss, self.optimizer.param_groups[0]["lr"], self.training_dic["retrain_lr"], time.time()-start_time))


                                writer.add_scalar("Training_Stats/Train Loss", train_loss, total_step)
                                writer.add_scalar("Training_Stats/Evaluation Loss", eval_loss, total_step)
                                writer.add_scalar("Training_Stats/MSE Loss", mse_loss, total_step)
                                writer.add_scalar("Training_Stats/Correlation", correlation_loss, total_step)
                                writer.add_scalar("Training_Stats/Learning rate", self.optimizer.param_groups[0]["lr"], total_step)

                                result[label_idx]["eval_mse_loss"].append(mse_loss)
                                result[label_idx]["eval_correlation_loss"].append(correlation_loss)
                                if self.training_dic["eval_mask"]:
                                    result[label_idx]["eval_correlation_mask"].append(correlation_mask)


                                if self.training_dic["retrain"] and ((overfit_cnt >= self.training_dic["overfit_patience"]) or ((self.optimizer.param_groups[0]["lr"] <= self.training_dic["retrain_lr"]/self.threshold_ratio))):
                                    if self.optimizer.param_groups[0]["lr"] <= self.training_dic["retrain_lr"]/self.threshold_ratio:
                                        self.training_dic["retrain_lr"] *= self.training_dic["retrain_lr_factor"]
                                    overfit_cnt = 0
                                    skip_reset = False
                                    if self.optimizer.param_groups[0]["lr"]/2 >= self.training_dic["retrain_lr"]:
                                        self.training_dic["retrain_lr"] = self.optimizer.param_groups[0]["lr"]/2
                                        skip_reset = True
                                    self.model.load_state_dict(torch.load(self.model_state_path_list[label_idx], map_location=self.device))
                                    self.optimizer.load_state_dict(torch.load(self.optimizer_path_list[label_idx]))
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

                                df = pd.DataFrame(result[label_idx])
                                df.to_csv(self.log_path_list[label_idx], index=False, header = True)

        del self.train_loader_list[label_idx]
        del self.train_dataset_list[label_idx]
        del self.model_list[label_idx]
        del self.optimizer_list[label_idx]
        del scheduler_list[label_idx]

        gc.collect()
        torch.cuda.empty_cache()

        if "discrimination" in self.training_dic and self.training_dic["discrimination"]:
            for label_idx in range(self.num_labels):
                self.adversarial_list[label_idx].adv_against_model(self.model_state_path_list[label_idx], self.optimizer_path_list[label_idx], self.finetune_train_loader_list[label_idx], self.finetune_eval_loader_list[label_idx])

            del self.finetune_train_loader_list
            del self.finetune_eval_loader_list
            del finetune_train_dataset_list
            del finetune_eval_dataset_list


        pearson_ic, spearman_ic, mse_loss = self.evaluation(self.cfg_list)

        del self.eval_loader_list
        del self.eval_dataset_list

        train_input_buffer.close()

        gc.collect()
        torch.cuda.empty_cache()

        saved_alphas = np.load(self.saved_alphas_loc)
        np.save(self.used_alphas_loc, saved_alphas)
        np.save(self.finish_flag_loc, [])

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

    def evaluation(self, cfg_list):
        # load the best model from the training process
        model_result_list = [create_model(cfg_list[i].model) for i in range(self.num_labels)]
        for i in range(self.num_labels):
            model_result_list[i].load_state_dict(torch.load(self.model_state_path_list[i]))
        for i in range(self.num_labels):
            model_result_list[i] = model_result_list[i].to(self.device)
            model_result_list[i].eval()
        total_pred = None
        np_target = None
        with torch.no_grad():
            for label_idx in range(self.num_labels):
                for data in self.eval_loader_list[label_idx]:
                    extra_batch = data[-1]
                    if self.training_dic.get("reduce_low_cap", False):
                        reduction_cap = extra_batch["reduction_cap"]
                        if self.training_dic.get("reduce_method", "addition") == "addition":
                            target_batch = (data[1].squeeze() + reduction_cap).to(non_blocking=True, dtype=data[1].dtype)
                        elif self.training_dic.get("reduce_method", "addition") == "multiplication":
                            target_batch = (data[1].squeeze() * reduction_cap).to(non_blocking=True, dtype=data[1].dtype)
                    else:
                        target_batch = data[1].squeeze().to(non_blocking=True, dtype=data[1].dtype)                    
                    extra_feat = extra_batch["features"]
                    if "extra_feature" in self.training_dic and self.training_dic["extra_feature"]:
                        # print(data[0].shape, data[-1].shape)
                        data_batch = torch.cat([data[0].squeeze(), extra_feat], dim=-1).to(self.device, non_blocking=True, dtype=data[0].dtype)
                    else:
                        data_batch = data[0].squeeze().to(self.device, non_blocking=True)
                    if self.training_dic["denormalizer"]:
                        batch_mean = torch.unsqueeze(torch.mean(data_batch, dim=1), 1)
                        batch_std = torch.unsqueeze(torch.std(data_batch, dim=1), 1)
                        data_batch = torch.cat([data_batch, batch_mean, batch_std.pow(0.5)], dim=1).to(self.device, non_blocking=True)
                    if "horizontal_norm" in self.training_dic and self.training_dic["horizontal_norm"]:
                        data_batch = F.normalize(data_batch, p=2.0, dim=1)
                    target_batch = target_batch[:,-1].to(non_blocking=True)
                    pred = model_result_list[label_idx].forward(data_batch)[:,-1]
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

        if isinstance(self.training_dic["num_buckets"], int) and self.training_dic["num_buckets"]==1:
            feature_buckets_indices = [self.indices[0]]
            feature_list_indices = [np.concatenate([np.ones(np.sum(self.indices[0])), np.ones(self.feature_matrix.shape[0])]).astype(bool)]
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
        self.dict_path = Path(model_path) / model_name 
        self.model_state_dict_path = Path(model_path) / model_name 
        return 1

    def predict(self, x, cfg, valid_alphas=None, non_valid_alphas=None, date=None, test_stage=True, lock=None, valid_stock=None, *args, **kwargs):
        x = x[..., valid_alphas]
        start_time = time.time()

        ProcessDataForTrainingIntermediate = {}
        if cfg.get("data_x", None):
            for op in cfg.get("data_x"):
                ProcessDataForTrainingIntermediate[op] = create_op_process(cfg.data_x[op])            
            for i in range(x.shape[1]):
                for op in ProcessDataForTrainingIntermediate:
                    ProcessDataForTrainingIntermediate[op].apply(x[:,i,:].squeeze(), test_stage=True)
        # super().predict_helper(x,cfg)
        print(f"data processing takes {time.time()-start_time}")

        """
        if cfg.model.type == "FastKAN":
            for i in range(len(cfg.model.model_params["layers_hidden"])-1):
                cfg.model.model_params["layers_hidden"][i] *= input_dim
        """
        
        self.num_labels = len(self.training_dic.get("blocks_sep", [["SH", "SZ", "gem", "star"]]))

        self.dict_path_list = [(Path(self.dict_path) / f"training_dict_{i}.pt") for i in range(self.num_labels)]

        self.model_state_dict_path_list = [(Path(self.model_state_dict_path) / f"final_model_{i}.pt") for i in range(self.num_labels)]

        cfg_list = [torch.load(self.dict_path_list[i]) for i in range(self.num_labels)]

        if not "denormalizer" in cfg_list[0].training.model_params:
            cfg_list[0].training.model_params["denormalizer"] = False

        if "feature_select" in cfg_list[0].training.model_params and cfg_list[0].training.model_params["feature_select"]:
            # print(x.shape, len(cfg_list[0].training.model_params["features_delete"]))
            x = x[:,:,cfg_list[0].training.model_params["features_delete"]]

        # if "mode" not in cfg.model.model_params:
        #     cfg.model.model_params["mode"] = "default"
        # self.vertical_norm = cfg.training.model_params["vertical_norm"]
        # self.horizontal_norm = "horizontal_norm" in cfg.training.model_params and cfg.training.model_params["horizontal_norm"]
        # cfg.model.model_params["device"] = self.device
        
        if "extra_feature" in cfg.training.model_params and cfg.training.model_params["extra_feature"]:
            self.feature_matrix = ((self.feature_matrix.squeeze().transpose(2,0))[valid_stock]).numpy()
            x = np.concatenate([x, self.feature_matrix], axis=-1)

        x = np.nan_to_num(x)
        
        # print(f"the shape of x is {x.shape}")
        # if cfg.training.model_params["denormalizer"]:
        #     x = np.concatenate([x, np.mean(x, axis=1).reshape(-1, 1), np.power(np.std(x, axis=1), 0.5).reshape(-1, 1)], axis=1)

        self.label_handling()

        if lock is not None:
            with lock:
                time.sleep(10)
                super().gpu_helper(*args, **kwargs)

        start_time = time.time()

        self.batch_size = cfg.training.model_params["batch_size"] * 4

        stocks = x.shape[0]
        steps = math.ceil(stocks / self.batch_size)

        for i in range(self.num_labels):
            cfg_list[i].model.model_params["device"] = self.device
            cfg_list[i].model.model_params["infer"] = True
        model = [create_model(cfg_list[i].model) for i in range(self.num_labels)]
        for i in range(self.num_labels):
            model[i].load_state_dict(torch.load(self.model_state_dict_path_list[i], map_location=self.device))
            model[i].eval()
        print(f"it takes {time.time()-start_time} to initiate")

        if self.training_dic.get("ensemble_in_date", False):
            date_ratio = self.training_dic.get("date_ratio", torch.linspace(0.2, 1, steps=self.training_dic["bptt"], device=self.device))

        start_time = time.time()
        input = []
        for i in range(self.num_labels):
            self.stocks_choice[i] = (self.stocks_choice[i].transpose(1,0)).squeeze()[valid_stock][:,-1]
            input.append(x[self.stocks_choice[i]])
            # print(self.stocks_choice[i][:,-1].sum())
        result = torch.zeros((x.shape[0]), dtype=torch.float32)
        if self.training_dic.get("label_ensemble", False):
            label_ensemble_ratio = np.array(self.training_dic.get("label_ensemble_ratio", [self.stocks_choice[i].sum() for i in range(self.num_labels)])).astype(np.float32)
            label_ensemble_ratio /= label_ensemble_ratio.sum()
            with torch.no_grad():
                steps = math.ceil(stocks / self.batch_size)
                for label_idx in range(self.num_labels):
                    pred = []
                    for i in range(steps):
                        pred_cur = model[label_idx].forward(torch.from_numpy(x[i*self.batch_size:(i+1)*self.batch_size,:,:]).contiguous().to(torch.float32).to(self.device))
                        if self.training_dic.get("ensemble_in_date", False):
                            pred.append(((pred_cur * date_ratio).sum(dim=-1) * label_ensemble_ratio[label_idx]).to("cpu").detach())
                        else:
                            pred.append((pred_cur[:, -1] * label_ensemble_ratio[label_idx]).to("cpu").detach())
                    pred = torch.cat(pred, dim=0)
                    result += pred
        else:
            with torch.no_grad():
                for label_idx in range(self.num_labels):
                    pred = []
                    steps = math.ceil(input[label_idx].shape[0] / self.batch_size)
                    for i in range(steps):
                        # print(x[i*self.batch_size:(i+1)*self.batch_size,:,:].shape)
                        data_input = torch.from_numpy(input[label_idx][i*self.batch_size:(i+1)*self.batch_size,:,:]).contiguous().to(torch.float32).to(self.device)
                        if self.training_dic.get("ensemble_in_date", False):
                            pred.append(((model[label_idx].forward(data_input) * date_ratio).sum(dim=-1)).to("cpu").detach())
                        else:
                            pred.append((model[label_idx].forward(data_input)[:, -1]).to("cpu").detach())
                    pred = torch.cat(pred, dim=0)
                    result[self.stocks_choice[label_idx]] = pred
            del data_input

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
        result = result.numpy()
        
        sub_universe_selected = self.training_dic.get("stock_select", None)

        if sub_universe_selected is not None:
            stock_selection = np.zeros(x.shape[0], dtype=np.uint8)
            for sub_universe in self.sub_universe_dict:
                if sub_universe in sub_universe_selected:
                    stock_selection += self.sub_universe_dict[sub_universe].squeeze().transpose(1,0)[:,-1][valid_stock]
            stock_selection = stock_selection.astype(bool)
            result[~stock_selection] = np.nan

        del model, pred
        gc.collect()
        torch.cuda.empty_cache()

        return result

    def label_handling(self):
        self.stocks_choice = []
        days, stocks = self.SH_indices.shape[0], self.SH_indices.shape[1]
        for blocks in self.training_dic.get("blocks_sep", [["SH", "SZ", "gem", "star"]]):
            stocks_choice = np.zeros((days, stocks), dtype=bool)
            if "SH" in blocks:
                stocks_choice += self.SH_indices
            if "SZ" in blocks:
                stocks_choice += self.SZ_indices
            if "gem" in blocks:
                stocks_choice += self.gem_indices
            if "star" in blocks:
                stocks_choice += self.star_indices
            self.stocks_choice.append(stocks_choice.astype(bool))
             
    def create_raw_features(self, cfg, model_name, reader):

        self.training_dic = deepcopy(cfg).training.model_params

        tidx = int(model_name.split("_")[-1])

        buffer = cfg.train_basics.get("cache_reader_buffer", 0)

        dates = reader(mode="dates")

        seq_len = self.training_dic.get("bptt", 1) - 1

        if len(dates) != buffer + seq_len:
            start_date = dates[buffer]
            r = CacheReader(cfg.env.root_directory)
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
        if len(date) == buffer + seq_len:
            date_info = extract_date_features_vectorized(date)[:,buffer-1:].unsqueeze(2).expand(-1, -1, stocks)
        else:
            date_info = extract_date_features_vectorized(date)[:,buffer-1:-1].unsqueeze(2).expand(-1, -1, stocks)
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
        if len(dates) != buffer + seq_len:
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

