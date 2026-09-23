import numpy as np
from pathlib import Path
from os.path import isfile, join
from os import listdir
from scipy.stats import rankdata
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import gc
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from multiprocessing.shared_memory import SharedMemory

from prometheus.utils.utils import WeightedCorrNp, calc_fcst_weighted_ic, weighted_mse_wrap, weighted_mse_eval_wrap, filter_eligible_data, filter_eligible_data_v2, operations_by_group
from prometheus.ops import create_op_process
from prometheus.modelpool.basemodel import BaseModel, create_model
from prometheus.utils.registry_factory import TRAINING_REGISTRY, MODEL_REGISTRY
from prometheus.utils.cache_reader import *
from prometheus.riskmodel.barra.factor_dict import *
from copy import deepcopy
from prometheus.label_engineering.base_labels import *
from prometheus.utils.cache_reader import CacheReader
from prometheus.utils.corr import *


@TRAINING_REGISTRY.register('test_targets')
class test_targets(BaseModel):

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

        model_path_name = (self.training_dic["model_path"]).split("/")[-1]

        pit_tidx = model_name.split("_")
        pit_tidx = pit_tidx[1] + "_" + pit_tidx[2]
        tidx = int(model_name.split("_")[-1])

        if "saved_alpha_path" in self.training_dic:
            self.saved_alphas_loc = Path(self.training_dic["saved_alpha_path"]) / ("saved_train_alphas_" + pit_tidx + "_processed.npy")
        else:
            self.saved_alphas_loc = Path("/dfs/data/ksim/automation/20240116/readcache_new/data") / model_path_name / ("saved_train_alphas_" + pit_tidx + "_processed.npy")

        saved_alphas = np.load(self.saved_alphas_loc)

        valphas = ["valpha" in alpha for alpha in saved_alphas]

        alphas = self.train_input[indices[0]]
        alphas = alphas[valphas]

        Corrs = [WeightedCorrNp(x=alphas[pos][indices[1]], y=train_output[indices[1]], w=np.ones(len(train_output[indices[1]])))("spearman") for pos in range(len(alphas))]
        import ipdb; ipdb.set_trace()

        target = []
        returns = []
        for i in range(train_output.shape[0]):
            if indices[1][i,:].sum() != 0:
                target.append(train_output[i][indices[1][i,:]])
                returns.append(self.returns[i][indices[1][i,:]])
        import ipdb; ipdb.set_trace()
        Corrs = [WeightedCorrNp(x=returns[i], y=target[i], w=np.ones(len(target[i]))) for i in range(len(target))]
        reader = CacheReader("/data/datacache/commoncache")
        import ipdb; ipdb.set_trace()
        return 1, 1, 0

    def load(*args, **kwargs):
        pass

    def predict(self, x, cfg, date=None, test_stage=True, lock=None, valid_stock=None, *args, **kwargs):

        return self.returns.squeeze()[valid_stock]
        


    def create_raw_features(self, cfg, model_name, reader):

        self.reader = reader

        self.training_dic = deepcopy(cfg).training.model_params

        self.returns = create_labels(cfg, model_name, reader)

        # self.training_dic = deepcopy(cfg).training.model_params

        # self.return_type = self.training_dic.get("return_type", "raw")

        # tidx = int(model_name.split("_")[-1])

        # buffer = cfg.train_basics.get("cache_reader_buffer", 0)

        # horizon = self.training_dic.get("horizon", 1)

        # dates = reader(mode="dates")
        # current_date = dates[-1]
        # r = CacheReader("/data/datacache/commoncache")
        # date_idx = np.where(r.dates == current_date)[0][0] + 1
        # current_date = r.dates[date_idx]
        # target_date = r.dates[date_idx + horizon]
        # close = r.get_raw_data("IntervalFull.close", current_date, target_date)[:,:,tidx]
        # # close = np.concatenate((close, np.zeros((1, close.shape[1]), dtype=np.float32)), axis=0)
        # adj_fct = r.get_raw_data("adjfactor", current_date, target_date)
        # for i in range(horizon):
        #     close[0,:] = close[0,:] * adj_fct[i,:]
        # base_price = close[0,:]
        # raw_ret = np.nan_to_num(close[-1,:] / (base_price + 1e-8) - 1, 0)
        # self.raw_ret = raw_ret - raw_ret.mean(axis=-1)

        # stocks = close.shape[1]
        
        # dsrt_return = r.get_raw_data(f"CNE5Ret.DSRT.d{horizon}", target_date, target_date)[:,:,tidx]
        # dsrt_return -= np.nanmean(dsrt_return, axis=-1, keepdims=True)
        # self.dsrt_return = np.nan_to_num(dsrt_return, 0)


        # # load CNE5 factor list (including style, industry and contry)
        # self.styleFactorDataNames = [f"CNE5D_RISK.{f}" for f in STYLE_FACTORS["cne5"]]
        # self.industryFactorDataNames = INDUSTRY_COV_ORDER["cne5"]
        # self.factorList = self.styleFactorDataNames + self.industryFactorDataNames + ["COUNTRY"]

        # # load CNE5 factor covariance matrix
        # barra_cne5_factor_cov = r.get_raw_data("CNE5D_COV", current_date, current_date)
        
        # date = barra_cne5_factor_cov.shape[0]

        # factorExp = np.zeros((len(self.factorList), date, stocks), dtype=np.float32)
        # raw_ret = np.expand_dims(raw_ret, axis=0)
        # self.dsrt_return = np.zeros_like(raw_ret)
        
        # for i in range(len(self.styleFactorDataNames)):
        #     names = [f.strip() for f in self.styleFactorDataNames[i].split(",")]
        #     factorExp[i, :, :] = r.get_raw_data(names[0], current_date, current_date)
        
        # for i in range(len(self.industryFactorDataNames)):
        #     indLabel = i + 1
        #     stockInd = r.get_raw_data("CNE5S_RISK.IND", current_date, current_date) == indLabel
        #     factorExp[len(self.styleFactorDataNames)+i, :, :] = stockInd

        # factorExp[len(self.factorList) - 1, :, :] = 1

        # if self.training_dic.get("weight_method", None) == "sqrt_cap":
        #     weights = np.sqrt(r.get_raw_data("cap", current_date, current_date))
        # elif self.training_dic.get("weight_method", None) == "volatility":
        #     dsrt_for_weight = r.get_raw_data(f"CNE5Ret.DSRT.d1", dates[0], r.dates[date_idx+1])[:,:,tidx]
        #     residual_sqrt = dsrt_for_weight ** 2
        #     valid_mask = ~np.isnan(residual_sqrt)
        #     weights = np.zeros([date, stocks])
        #     half_life = 63
        #     min_periods = 20
        #     decay = 0.5 ** (1.0 / half_life)
        #     time_weights = decay ** np.arange(buffer - 1, -1, -1)
        #     time_weights = np.expand_dims(time_weights, axis=1)
        #     sum_W = np.sum(time_weights, axis=0)
        #     valid_stocks = (np.sum(valid_mask, axis=0) >= min_periods) & (sum_W > 0)
        #     for i in range(-1, -date-1, -1):
        #         weights[i, valid_stocks] = np.nansum(residual_sqrt[i-buffer:i, valid_stocks] * time_weights, axis=0) / sum_W
        #         # import ipdb; ipdb.set_trace()
        # else:
        #     weights = np.zeros_like(raw_ret)
        #     for i in range(weights.shape[0]):
        #         weights[i, ~np.isnan(raw_ret[i])] = 1

        # weights /= np.nansum(weights, axis=-1, keepdims=True)

        # # import ipdb; ipdb.set_trace()

        # for i in range(date):
        #     X_i = factorExp[:,i,:].T
        #     Y_i = raw_ret[i,:]
        #     valid_mask = ~np.isnan(Y_i).any(axis=0) & ~np.isnan(X_i).any(axis=1) & ~np.isnan(weights[i])
            
        #     if valid_mask.sum() <= X_i.shape[1]:
        #         continue
        #     X_i = X_i[valid_mask]
        #     Y_i = Y_i[valid_mask] 

        #     X_w = X_i * np.expand_dims(weights[i,valid_mask], axis=1)
        #     Y_w = Y_i * weights[i,valid_mask]
        
        #     factor_returns, _, _, _ = np.linalg.lstsq(X_w, Y_w, rcond=None)

        #     dsrt_return = X_i @ factor_returns

        #     dsrt_return = Y_i - dsrt_return

        #     self.dsrt_return[i, valid_mask] = np.nan_to_num(dsrt_return - np.nanmean(dsrt_return, axis=-1), 0)

        #     # import ipdb; ipdb.set_trace()
