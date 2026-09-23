import numpy as np
from pathlib import Path
from os.path import isfile, join, dirname
from os import listdir
from scipy.stats import rankdata, spearmanr, skew, kurtosis
import scipy.stats
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import math
import time
from copy import deepcopy
from collections import Counter, defaultdict
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import train_test_split
from multiprocessing.shared_memory import SharedMemory
from lion_pytorch import Lion
# from muon import Muon, MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import progressbar

from prometheus.utils.utils import WeightedCorrNp, calc_fcst_weighted_ic, weighted_mse_wrap, weighted_mse_eval_wrap, filter_eligible_data, filter_eligible_data_v2, operations_by_group
from prometheus.utils.torch_utils import select_gpu_with_minimum_memory, early_stopping_func, WarmupLR, DecayingCosineWarmRestarts, SharedMemDataset, SharedMemSeqDataset
from prometheus.ops import create_op_process
from prometheus.modelpool.basemodel import BaseModel, create_model
from prometheus.utils.registry_factory import TRAINING_REGISTRY, MODEL_REGISTRY
from prometheus.utils.speedup_package import calculate_stock_mean_3d_parallel
from prometheus.utils.pearson import niocorr

class Train_Dataset(Dataset):
    def __init__(self, x, y, vertical_norm=False):
        if vertical_norm:
            self.x = F.normalize(torch.from_numpy(x).to(torch.float32), p=2.0, dim=0)
        else:
            self.x = F.normalize(torch.from_numpy(x).to(torch.float32), p=2.0, dim=1)
        self.y = torch.from_numpy(y).to(torch.float32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]



@TRAINING_REGISTRY.register('Feature_Selection_test')
class training_test(BaseModel):
    def __init__(self, *args, **kwargs):
        # super().__init__(*args, **kwargs)
        pass

    def fit(self, cfg, indices, train_output, sample_stocks_date, sample_stocks_seccode, model_name, lock=None, *args, **kwargs):

        super().fit_helper(cfg, indices, train_output, sample_stocks_date=sample_stocks_date, model_name=model_name, *args, **kwargs)
        if lock is not None:
            with lock:
                time.sleep(10)
                super().gpu_helper(*args, **kwargs)

        train_input_buffer = SharedMemory(model_name)
        train_input = np.ndarray((indices[0].shape[0], indices[1].shape[0], indices[1].shape[1]), dtype=np.float32, buffer=train_input_buffer.buf)

        features = np.where(indices[0])[0]

        selection_res = [True] * np.sum(indices[0])

        # Mark the position of the features after dataloader

        alpha_indices = {}

        for pos, idx in enumerate(features):
            alpha_indices[idx] = pos

        train_y = train_output[indices[1]]

        spearman_dict = defaultdict(list)
        
        to_delete = set()

        # Pick the features that doesn't correlated to the target or have a low information ratio

        # import ipdb; ipdb.set_trace()

        horizon = cfg.training.model_params["horizon"]

        for j in features:
            # for i in range(indices[1].shape[0]-horizon):
            #     if np.sum(indices[1][i,:]>0):
            #         # import ipdb; ipdb.set_trace()
            #         spearman_dict[j].append(WeightedCorrNp(x=train_input[j][i][indices[1][i,:]], y=train_output[i][indices[1][i,:]], w=np.ones(np.sum(indices[1][i,:])))("spearman"))
            #         # print(f"finished {i, j}")
            spearman_dict[j] = niocorr(train_input[j][:-horizon,:], train_output[:-horizon,:])
            print(f"finished feature {j}")

        alpha_score= {}

        for i in spearman_dict:
            mean = np.abs(np.mean(np.nan_to_num(spearman_dict[i])))
            std = np.std(np.nan_to_num(spearman_dict[i]))
            alpha_score[i] = mean / std
            if mean < 0.01 or alpha_score[i] < 0.3:
                to_delete.add(i)

        import ipdb; ipdb.set_trace()

        # Pick features which perform worse on the previous test from pairs of features that too correlated

        pair_ic = {}

        with ProcessPoolExecutor() as Executor:
            for i in range(len(features)-1):
                for j in range(i+1, len(features)):
                    if features[i] not in to_delete and features[j] not in to_delete:
                        # pair_ic[(i,j)] = np.mean([WeightedCorrNp(x=train_input[features[i]][k][indices[1][k,:]], y=train_input[features[j]][k][indices[1][k,:]], w=np.ones(np.sum(indices[1][k,:])))("spearman") for k in range(indices[1].shape[0]-horizon)])
                        pair_ic[(i,j)] = Executor.submit(niocorr, train_input[i][:-horizon,:], train_input[j][:-horizon,:])
                    # print(f"finished {i,j}")

        pair_ic[(i,j)] = np.mean(pair_ic[(i,j)].result())

        # import ipdb; ipdb.set_trace()

        if pair_ic:
            for i, j in pair_ic.keys():
                if i not in to_delete and j not in to_delete and pair_ic[(i,j)] > 0.7:
                    if alpha_score[i] > alpha_score[j]:
                        to_delete.add(j)
                    else:
                        to_delete.add(i)
                    print(f"finished {i, j}")

        # Add more feature selection logics here

        # Eliminate the features that are picked and mark them in selection result to inform the inference module

        for i in to_delete:
            indices[i] = False
            selection_res[alpha_indices[i]] = False
        
        import ipdb; ipdb.set_trace()
        

    def predict(self, x, cfg, date=None, test_stage=True, lock=None, *args, **kwargs):
        return 
        