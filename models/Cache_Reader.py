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
from prometheus.utils.cache_reader import CacheReader
from prometheus.utils.data_processing import RollingStatistics
from prometheus.riskmodel.barra.factor_dict import *
from prometheus.utils.date_info_extraction import *


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



@TRAINING_REGISTRY.register('Cache_Reader')
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
        self.train_input = np.ndarray((indices[0].shape[0], indices[1].shape[0], indices[1].shape[1]), dtype=np.float32, buffer=train_input_buffer.buf)

        self.indices = indices
        
        import ipdb; ipdb.set_trace()

        self.train_output = train_output[indices[1]]
        import ipdb; ipdb.set_trace()
        return 0, 0, 0
        

    def predict(self, x, cfg, date=None, test_stage=True, lock=None, *args, **kwargs):
        return 
        
      
        
    def create_raw_features(self, cfg, model_name, reader):

        self.training_dic = deepcopy(cfg).training.model_params

        tidx = int(model_name.split("_")[-1])

        buffer = cfg.train_basics.get("cache_reader_buffer", 0)

        if self.training_dic.get("validation_by_return", False):
            # close = reader("IntervalFull.close")[buffer-2:-1,:,tidx]
            if self.training_dic.get("return_type", "raw") == "raw" or self.training_dic.get("loss_fn") == "ret_raw":
                close = reader("IntervalFull.close")[buffer:,:,tidx+1]
                close = np.concatenate((close, np.zeros((1, close.shape[1]), dtype=np.float32)), axis=0)
                adj_fct = reader("adjfactor")[buffer+1:,:]
                adj_fct = np.concatenate((adj_fct, np.ones((1, adj_fct.shape[1]), dtype=np.float32)), axis=0)
                base_price = close[:-1,:] * adj_fct
                self.ret_ratio = np.nan_to_num(close[1:,:] / (base_price + 1e-8) - 1, 0)
            elif self.training_dic.get("return_type", "raw") == "dsrt" or self.training_dic.get("loss_fn") == "ret_dsrt":
                dsrt_return = reader("CNE5Ret.DSRT.d1")[buffer+1:,:,tidx+1]
                self.ret_ratio = np.nan_to_num(np.concatenate((dsrt_return, np.zeros((1, dsrt_return.shape[1]), dtype=np.float32)), axis=0), 0)

            self.ret_ratio = torch.tensor(self.ret_ratio).unsqueeze(0)
            

        feature_list = []

        # grab log_cap from cache and use it to get the reduction for low cap stocks when needed
        log_cap = torch.from_numpy(np.log(reader("cap")[buffer-1:-1,:].astype(np.float32))).unsqueeze(0)

        if self.training_dic.get("reduce_low_cap", False):
            reduce_ratio = self.training_dic.get("reduce_ratio", 1)
            if self.training_dic.get("reduce_method", "addition") == "addition":
                self.reduction = (reduce_ratio * (F.sigmoid(log_cap - 13) - 1))
            elif self.training_dic.get("reduce_method", "addition") == "multiplication":
                self.reduction = ((F.sigmoid(log_cap - 13))+reduce_ratio) / (reduce_ratio + 1)

            # self.feature_matrix = reduction

        log_cap -= 14

        # grab block information from cache
        block4 = reader("block4n")

        # secind = reader("secoindustry")
        # secind_num = secind.max() + 1

        days, stocks = block4.shape[0]-buffer, block4.shape[1]

        self.SH_indices = block4[buffer-1:-1,:] == 0
        self.SZ_indices = block4[buffer-1:-1,:] == 1
        self.gem_indices = block4[buffer-1:-1,:] == 2
        self.star_indices = block4[buffer-1:-1,:] == 3

        self.ZZ2000_indices = reader("ZZ2000")[buffer-1:-1,:]
        self.ZZ1000_indices = reader("ZZ1000_")[buffer-1:-1,:]
        self.ZZ500_indices = reader("ZZ500_")[buffer-1:-1,:]
        self.HS300_indices = reader("HS300")[buffer-1:-1,:]
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

        factorExp = torch.from_numpy(factorExp[:,buffer-1:-1,:])

        barra_invert_indices = -deepcopy(barra_indices[:10])

        barra_indices = np.concatenate([barra_invert_indices, barra_indices], axis=0)

        for i in range(20):
            barra_indices[i] = np.tanh(barra_indices[i])

        #retrieve date information from cache and use these info to create seasonal features
        date = list(map(str,reader(mode="dates")))
        # date_info will have shape (4 ,days)
        date_info = extract_date_features_vectorized(date)[:,buffer-1:].unsqueeze(2).expand(-1, -1, stocks)
        # import ipdb; ipdb.set_trace()

        #AShare as well as low/high capital trend features
        
        feature_time_choice = []
        lowcap_indices = log_cap < -1
        highcap_indices = log_cap >= -1

        # use the total market return, liqudity, volume and amount to describe market trend
        dr_ret5 = reader("dr.ret_5")[buffer-1:-1, :]
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

        dr_ret10 = reader("dr.ret_10")[buffer-1:-1, :]
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


        dr_logliq5 = np.log(reader("dr.liq_5"))[buffer-1:-1, :]-14
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


        dr_logliq10 = np.log(reader("dr.liq_10"))[buffer-1:-1, :]-14
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
        volume = all_volume[buffer-1:-1, :, :]
        volume_sum = torch.tensor(np.log(np.nansum(np.nansum(volume, axis=-1), axis=-1))).unsqueeze(0) - 19
        feature_time_choice.append(volume_sum)

        all_amount = reader("IntervalFull.amount")
        amount = reader("IntervalFull.amount")[buffer-1:-1, :, :]
        amount_total = np.nansum(np.nansum(amount, axis=-1), axis=-1)
        amount_sum = torch.tensor(np.log(np.nansum(np.nansum(amount, axis=-1), axis=-1))).unsqueeze(0) - 26
        feature_time_choice.append(amount_sum)

        all_close = reader("IntervalFull.close")

        borrowed_money_purchase = np.nansum(reader("AShareMarginTrade.S_MARGIN_PURCHWITHBORROWMONEY")[buffer-1:-1, :], axis=-1)
        bmp_ratio = torch.tensor(borrowed_money_purchase / amount_total).unsqueeze(0)
        feature_time_choice.append(bmp_ratio)


        daily_change = (reader("IntervalFull.close")[buffer-1:-1,:,-1] - reader("IntervalFull.open")[buffer-1:-1,:,0]).squeeze()

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
            ind_datas[i] = torch.from_numpy(np.nan_to_num(ind_datas[i], np.nanmean(ind_datas[i])))[buffer-1:-1,:].unsqueeze(0)
        feature_num = len(feature_time_choice)
        days = feature_time_choice[0].shape[1]
        feature_time_choice = [torch.cat(feature_time_choice, dim=0).unsqueeze(2).expand(feature_num,days,stocks)]
        feature_time_choice = torch.cat(feature_time_choice+ind_datas, dim=0)

        
        if self.training_dic.get("time_choice_block", False):
            self.time_choice_feature_num = feature_time_choice.shape[0]

        feature_list.append(feature_time_choice)
        
        extra_feature_list = self.training_dic.get("extra_feature_list", ["factor_Exp", "log_cap", "stock_block"])

        IndicesName = ["000001.SH", "000016.SH", "000300.SH", "000510.SH", "000852.SH", "000903.SH", "000905.SH", "000906.SH", "399001.SZ", "399005.SZ", "399006.SZ", "399008.SZ", "399012.SZ", "399101.SZ", "399102.SZ", "399106.SZ", "399303.SZ", "932000.SH"]

        rolling_range = [5, 10, 20, 30, 60]        

        if "indices_pctchange" in extra_feature_list:

            dq_pct_change = ".S_DQ_PCTCHANGE"

            indices_pctchange = [reader(IndexName+dq_pct_change) for IndexName in IndicesName]
            indices_pctchange_feature = torch.cat([torch.from_numpy(data[buffer-1:-1]).unsqueeze(1).expand(days,stocks).unsqueeze(0) for data in indices_pctchange], dim=0)

            pct_change_rolling_mean = []
            for pctchange in indices_pctchange:
                for r in rolling_range:
                    pct_change_rolling_mean.append(torch.from_numpy(RollingStatistics(pctchange).mean(r)[buffer-1:-1]).unsqueeze(1).expand(days,stocks).unsqueeze(0))
            pct_change_rolling_mean = torch.cat(pct_change_rolling_mean, dim=0)

            pct_change_rolling_std = []
            for pctchange in indices_pctchange:
                for r in rolling_range:
                    pct_change_rolling_std.append(torch.from_numpy(RollingStatistics(pctchange).std(r)[buffer-1:-1]).unsqueeze(1).expand(days,stocks).unsqueeze(0))
            pct_change_rolling_std = torch.cat(pct_change_rolling_std, dim=0)

            if self.training_dic.get("time_choice_block", False):
                self.time_choice_feature_num += indices_pctchange_feature.shape[0] + pct_change_rolling_mean.shape[0] + pct_change_rolling_std.shape[0] 

        if "indices_log_amount" in extra_feature_list:

            dq_amount = ".S_DQ_AMOUNT"

            log_amount = [(np.log(reader(IndexName+dq_amount))-17) for IndexName in IndicesName]
            log_amount_feature = torch.cat([torch.from_numpy(data[buffer-1:-1]).unsqueeze(1).expand(days,stocks).unsqueeze(0) for data in log_amount], dim=0)

            log_amount_rolling_mean = []
            for l_a in log_amount:
                for r in rolling_range:
                    log_amount_rolling_mean.append(torch.from_numpy(RollingStatistics(l_a).mean(r)[buffer-1:-1]).unsqueeze(1).expand(days,stocks).unsqueeze(0))
            log_amount_rolling_mean = torch.cat(log_amount_rolling_mean, dim=0)

            log_amount_rolling_std = []
            for l_a in log_amount:
                for r in rolling_range:
                    log_amount_rolling_std.append(torch.from_numpy(RollingStatistics(l_a).std(r)[buffer-1:-1]).unsqueeze(1).expand(days,stocks).unsqueeze(0))
            log_amount_rolling_std = torch.cat(log_amount_rolling_std, dim=0)

            if self.training_dic.get("time_choice_block", False):
                self.time_choice_feature_num += log_amount_feature.shape[0] + log_amount_rolling_mean.shape[0] + log_amount_rolling_std.shape[0]

        barra_close = np.nansum((barra_indices * np.expand_dims(all_close[:,:,tidx], axis=0)), axis=-1) + 1.2e5
        barra_volume = np.nansum((barra_indices * np.expand_dims(all_volume[:,:,tidx], axis=0)), axis=-1) + 2e7
        barra_amount = np.nansum((barra_indices * np.expand_dims(all_amount[:,:,tidx], axis=0)), axis=-1) + 2e10

        barra_log_close = np.log(barra_close) - 11.5
        barra_log_volume = np.log(barra_volume) - 16.5
        barra_log_amount = np.log(barra_amount) - 23.5

        barra_dim = barra_close.shape[0]

        barra_rolling_range = [5, 20, 60]

        barra_features = {"close":barra_log_close*3, "volume":barra_log_volume*3, "amount": barra_log_amount*3}
        barra_feature_insert = []
        for feature in barra_features:
            barra_feature_insert.append(torch.from_numpy(barra_features[feature][:,buffer-1:-1]).unsqueeze(2).expand(barra_dim,days,stocks))
            for factor in barra_features[feature]:
                for r in barra_rolling_range:
                    rolling_factor = RollingStatistics(factor)
                    barra_feature_insert.append(torch.from_numpy(rolling_factor.mean(r)[buffer-1:-1]).unsqueeze(1).expand(days,stocks).unsqueeze(0))
                    barra_feature_insert.append(torch.from_numpy(rolling_factor.std(r)[buffer-1:-1]).unsqueeze(1).expand(days,stocks).unsqueeze(0))

        import ipdb; ipdb.set_trace()

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

        # features_dic["feature_SH000001_leading_return"] = SH000001_leading_return[buffer-1:-1]
        # features_dic["feature_SZ399001_leading_return"] = SZ399001_leading_return[buffer-1:-1]
        # features_dic["feature_SH000300_leading_return"] = SH000300_leading_return[buffer-1:-1]
        # features_dic["feature_SH000905_leading_return"] = SH000905_leading_return[buffer-1:-1]
        # features_dic["feature_SH000906_leading_return"] = SH000906_leading_return[buffer-1:-1]

        # features_dic["feature_SH000001_leading_return_mean_5"] = RollingStatistics(SH000001_leading_return).mean(5)[buffer-1:-1]
        # features_dic["feature_SZ399001_leading_return_mean_5"] = RollingStatistics(SZ399001_leading_return).mean(5)[buffer-1:-1]
        # features_dic["feature_SH000300_leading_return_mean_5"] = RollingStatistics(SH000300_leading_return).mean(5)[buffer-1:-1]
        # features_dic["feature_SH000905_leading_return_mean_5"] = RollingStatistics(SH000905_leading_return).mean(5)[buffer-1:-1]
        # features_dic["feature_SH000906_leading_return_mean_5"] = RollingStatistics(SH000906_leading_return).mean(5)[buffer-1:-1]

        # features_dic["feature_SH000001_leading_return_mean_10"] = RollingStatistics(SH000001_leading_return).mean(10)[buffer-1:-1]
        # features_dic["feature_SZ399001_leading_return_mean_10"] = RollingStatistics(SZ399001_leading_return).mean(10)[buffer-1:-1]
        # features_dic["feature_SH000300_leading_return_mean_10"] = RollingStatistics(SH000300_leading_return).mean(10)[buffer-1:-1]
        # features_dic["feature_SH000905_leading_return_mean_10"] = RollingStatistics(SH000905_leading_return).mean(10)[buffer-1:-1]
        # features_dic["feature_SH000906_leading_return_mean_10"] = RollingStatistics(SH000906_leading_return).mean(10)[buffer-1:-1]

        # features_dic["feature_SH000001_leading_return_mean_20"] = RollingStatistics(SH000001_leading_return).mean(20)[buffer-1:-1]
        # features_dic["feature_SZ399001_leading_return_mean_20"] = RollingStatistics(SZ399001_leading_return).mean(20)[buffer-1:-1]
        # features_dic["feature_SH000300_leading_return_mean_20"] = RollingStatistics(SH000300_leading_return).mean(20)[buffer-1:-1]
        # features_dic["feature_SH000905_leading_return_mean_20"] = RollingStatistics(SH000905_leading_return).mean(20)[buffer-1:-1]
        # features_dic["feature_SH000906_leading_return_mean_20"] = RollingStatistics(SH000906_leading_return).mean(20)[buffer-1:-1]
        
        # features_dic["feature_SH000001_leading_return_mean_30"] = RollingStatistics(SH000001_leading_return).mean(30)[buffer-1:-1]
        # features_dic["feature_SZ399001_leading_return_mean_30"] = RollingStatistics(SZ399001_leading_return).mean(30)[buffer-1:-1]
        # features_dic["feature_SH000300_leading_return_mean_30"] = RollingStatistics(SH000300_leading_return).mean(30)[buffer-1:-1]
        # features_dic["feature_SH000905_leading_return_mean_30"] = RollingStatistics(SH000905_leading_return).mean(30)[buffer-1:-1]
        # features_dic["feature_SH000906_leading_return_mean_30"] = RollingStatistics(SH000906_leading_return).mean(30)[buffer-1:-1]
        
        # features_dic["feature_SH000001_leading_return_mean_60"] = RollingStatistics(SH000001_leading_return).mean(60)[buffer-1:-1]
        # features_dic["feature_SZ399001_leading_return_mean_60"] = RollingStatistics(SZ399001_leading_return).mean(60)[buffer-1:-1]
        # features_dic["feature_SH000300_leading_return_mean_60"] = RollingStatistics(SH000300_leading_return).mean(60)[buffer-1:-1]
        # features_dic["feature_SH000905_leading_return_mean_60"] = RollingStatistics(SH000905_leading_return).mean(60)[buffer-1:-1]
        # features_dic["feature_SH000906_leading_return_mean_60"] = RollingStatistics(SH000906_leading_return).mean(60)[buffer-1:-1]

        # features_dic["feature_SH000001_leading_return_std_5"] = RollingStatistics(SH000001_leading_return).std(5)[buffer-1:-1]
        # features_dic["feature_SZ399001_leading_return_std_5"] = RollingStatistics(SZ399001_leading_return).std(5)[buffer-1:-1]
        # features_dic["feature_SH000300_leading_return_std_5"] = RollingStatistics(SH000300_leading_return).std(5)[buffer-1:-1]
        # features_dic["feature_SH000905_leading_return_std_5"] = RollingStatistics(SH000905_leading_return).std(5)[buffer-1:-1]
        # features_dic["feature_SH000906_leading_return_std_5"] = RollingStatistics(SH000906_leading_return).std(5)[buffer-1:-1]

        # features_dic["feature_SH000001_leading_return_std_10"] = RollingStatistics(SH000001_leading_return).std(10)[buffer-1:-1]
        # features_dic["feature_SZ399001_leading_return_std_10"] = RollingStatistics(SZ399001_leading_return).std(10)[buffer-1:-1]
        # features_dic["feature_SH000300_leading_return_std_10"] = RollingStatistics(SH000300_leading_return).std(10)[buffer-1:-1]
        # features_dic["feature_SH000905_leading_return_std_10"] = RollingStatistics(SH000905_leading_return).std(10)[buffer-1:-1]
        # features_dic["feature_SH000906_leading_return_std_10"] = RollingStatistics(SH000906_leading_return).std(10)[buffer-1:-1]
        
        # features_dic["feature_SH000001_leading_return_std_20"] = RollingStatistics(SH000001_leading_return).std(20)[buffer-1:-1]
        # features_dic["feature_SZ399001_leading_return_std_20"] = RollingStatistics(SZ399001_leading_return).std(20)[buffer-1:-1]
        # features_dic["feature_SH000300_leading_return_std_20"] = RollingStatistics(SH000300_leading_return).std(20)[buffer-1:-1]
        # features_dic["feature_SH000905_leading_return_std_20"] = RollingStatistics(SH000905_leading_return).std(20)[buffer-1:-1]
        # features_dic["feature_SH000906_leading_return_std_20"] = RollingStatistics(SH000906_leading_return).std(20)[buffer-1:-1]        
        
        # features_dic["feature_SH000001_leading_return_std_30"] = RollingStatistics(SH000001_leading_return).std(30)[buffer-1:-1]
        # features_dic["feature_SZ399001_leading_return_std_30"] = RollingStatistics(SZ399001_leading_return).std(30)[buffer-1:-1]
        # features_dic["feature_SH000300_leading_return_std_30"] = RollingStatistics(SH000300_leading_return).std(30)[buffer-1:-1]
        # features_dic["feature_SH000905_leading_return_std_30"] = RollingStatistics(SH000905_leading_return).std(30)[buffer-1:-1]
        # features_dic["feature_SH000906_leading_return_std_30"] = RollingStatistics(SH000906_leading_return).std(30)[buffer-1:-1]       
        
        # features_dic["feature_SH000001_leading_return_std_60"] = RollingStatistics(SH000001_leading_return).std(60)[buffer-1:-1]
        # features_dic["feature_SZ399001_leading_return_std_60"] = RollingStatistics(SZ399001_leading_return).std(60)[buffer-1:-1]
        # features_dic["feature_SH000300_leading_return_std_60"] = RollingStatistics(SH000300_leading_return).std(60)[buffer-1:-1]
        # features_dic["feature_SH000905_leading_return_std_60"] = RollingStatistics(SH000905_leading_return).std(60)[buffer-1:-1]
        # features_dic["feature_SH000906_leading_return_std_60"] = RollingStatistics(SH000906_leading_return).std(60)[buffer-1:-1]

        # features_dic["feature_SH000001_amount_mean_pct_5"] = (RollingStatistics(SH000001_amount).mean(5) / (SH000001_amount+1e-8))[buffer-1:-1]
        # features_dic["feature_SZ399001_amount_mean_pct_5"] = (RollingStatistics(SZ399001_amount).mean(5) / (SZ399001_amount+1e-8))[buffer-1:-1]
        # features_dic["feature_SH000300_amount_mean_pct_5"] = (RollingStatistics(SH000300_amount).mean(5) / (SH000300_amount+1e-8))[buffer-1:-1]
        # features_dic["feature_SH000905_amount_mean_pct_5"] = (RollingStatistics(SH000905_amount).mean(5) / (SH000905_amount+1e-8))[buffer-1:-1]
        # features_dic["feature_SH000906_amount_mean_pct_5"] = (RollingStatistics(SH000906_amount).mean(5) / (SH000906_amount+1e-8))[buffer-1:-1]

        # features_dic["feature_SH000001_amount_mean_pct_10"] = (RollingStatistics(SH000001_amount).mean(10) / (SH000001_amount+1e-8))[buffer-1:-1]
        # features_dic["feature_SZ399001_amount_mean_pct_10"] = (RollingStatistics(SZ399001_amount).mean(10) / (SZ399001_amount+1e-8))[buffer-1:-1]
        # features_dic["feature_SH000300_amount_mean_pct_10"] = (RollingStatistics(SH000300_amount).mean(10) / (SH000300_amount+1e-8))[buffer-1:-1]
        # features_dic["feature_SH000905_amount_mean_pct_10"] = (RollingStatistics(SH000905_amount).mean(10) / (SH000905_amount+1e-8))[buffer-1:-1]
        # features_dic["feature_SH000906_amount_mean_pct_10"] = (RollingStatistics(SH000906_amount).mean(10) / (SH000906_amount+1e-8))[buffer-1:-1]

        # features_dic["feature_SH000001_amount_mean_pct_20"] = (RollingStatistics(SH000001_amount).mean(20) / (SH000001_amount+1e-8))[buffer-1:-1]
        # features_dic["feature_SZ399001_amount_mean_pct_20"] = (RollingStatistics(SZ399001_amount).mean(20) / (SZ399001_amount+1e-8))[buffer-1:-1]
        # features_dic["feature_SH000300_amount_mean_pct_20"] = (RollingStatistics(SH000300_amount).mean(20) / (SH000300_amount+1e-8))[buffer-1:-1]
        # features_dic["feature_SH000905_amount_mean_pct_20"] = (RollingStatistics(SH000905_amount).mean(20) / (SH000905_amount+1e-8))[buffer-1:-1]
        # features_dic["feature_SH000906_amount_mean_pct_20"] = (RollingStatistics(SH000906_amount).mean(20) / (SH000906_amount+1e-8))[buffer-1:-1]

        # features_dic["feature_SH000001_amount_mean_pct_30"] = (RollingStatistics(SH000001_amount).mean(30) / (SH000001_amount+1e-8))[buffer-1:-1]
        # features_dic["feature_SZ399001_amount_mean_pct_30"] = (RollingStatistics(SZ399001_amount).mean(30) / (SZ399001_amount+1e-8))[buffer-1:-1]
        # features_dic["feature_SH000300_amount_mean_pct_30"] = (RollingStatistics(SH000300_amount).mean(30) / (SH000300_amount+1e-8))[buffer-1:-1]
        # features_dic["feature_SH000905_amount_mean_pct_30"] = (RollingStatistics(SH000905_amount).mean(30) / (SH000905_amount+1e-8))[buffer-1:-1]
        # features_dic["feature_SH000906_amount_mean_pct_30"] = (RollingStatistics(SH000906_amount).mean(30) / (SH000906_amount+1e-8))[buffer-1:-1]

        # features_dic["feature_SH000001_amount_mean_pct_60"] = (RollingStatistics(SH000001_amount).mean(60) / (SH000001_amount+1e-8))[buffer-1:-1]
        # features_dic["feature_SZ399001_amount_mean_pct_60"] = (RollingStatistics(SZ399001_amount).mean(60) / (SZ399001_amount+1e-8))[buffer-1:-1]
        # features_dic["feature_SH000300_amount_mean_pct_60"] = (RollingStatistics(SH000300_amount).mean(60) / (SH000300_amount+1e-8))[buffer-1:-1]
        # features_dic["feature_SH000905_amount_mean_pct_60"] = (RollingStatistics(SH000905_amount).mean(60) / (SH000905_amount+1e-8))[buffer-1:-1]
        # features_dic["feature_SH000906_amount_mean_pct_60"] = (RollingStatistics(SH000906_amount).mean(60) / (SH000906_amount+1e-8))[buffer-1:-1]

        # features_dic["feature_SH000001_amount_std_pct_5"] = (RollingStatistics(SH000001_amount).std(5) / (SH000001_amount+1e-8))[buffer-1:-1]
        # features_dic["feature_SZ399001_amount_std_pct_5"] = (RollingStatistics(SZ399001_amount).std(5) / (SZ399001_amount+1e-8))[buffer-1:-1]
        # features_dic["feature_SH000300_amount_std_pct_5"] = (RollingStatistics(SH000300_amount).std(5) / (SH000300_amount+1e-8))[buffer-1:-1]
        # features_dic["feature_SH000905_amount_std_pct_5"] = (RollingStatistics(SH000905_amount).std(5) / (SH000905_amount+1e-8))[buffer-1:-1]
        # features_dic["feature_SH000906_amount_std_pct_5"] = (RollingStatistics(SH000906_amount).std(5) / (SH000906_amount+1e-8))[buffer-1:-1]

        # features_dic["feature_SH000001_amount_std_pct_10"] = (RollingStatistics(SH000001_amount).std(10) / (SH000001_amount+1e-8))[buffer-1:-1]
        # features_dic["feature_SZ399001_amount_std_pct_10"] = (RollingStatistics(SZ399001_amount).std(10) / (SZ399001_amount+1e-8))[buffer-1:-1]
        # features_dic["feature_SH000300_amount_std_pct_10"] = (RollingStatistics(SH000300_amount).std(10) / (SH000300_amount+1e-8))[buffer-1:-1]
        # features_dic["feature_SH000905_amount_std_pct_10"] = (RollingStatistics(SH000905_amount).std(10) / (SH000905_amount+1e-8))[buffer-1:-1]
        # features_dic["feature_SH000906_amount_std_pct_10"] = (RollingStatistics(SH000906_amount).std(10) / (SH000906_amount+1e-8))[buffer-1:-1]

        # features_dic["feature_SH000001_amount_std_pct_20"] = (RollingStatistics(SH000001_amount).std(20) / (SH000001_amount+1e-8))[buffer-1:-1]
        # features_dic["feature_SZ399001_amount_std_pct_20"] = (RollingStatistics(SZ399001_amount).std(20) / (SZ399001_amount+1e-8))[buffer-1:-1]
        # features_dic["feature_SH000300_amount_std_pct_20"] = (RollingStatistics(SH000300_amount).std(20) / (SH000300_amount+1e-8))[buffer-1:-1]
        # features_dic["feature_SH000905_amount_std_pct_20"] = (RollingStatistics(SH000905_amount).std(20) / (SH000905_amount+1e-8))[buffer-1:-1]
        # features_dic["feature_SH000906_amount_std_pct_20"] = (RollingStatistics(SH000906_amount).std(20) / (SH000906_amount+1e-8))[buffer-1:-1]

        # features_dic["feature_SH000001_amount_std_pct_30"] = (RollingStatistics(SH000001_amount).std(30) / (SH000001_amount+1e-8))[buffer-1:-1]
        # features_dic["feature_SZ399001_amount_std_pct_30"] = (RollingStatistics(SZ399001_amount).std(30) / (SZ399001_amount+1e-8))[buffer-1:-1]
        # features_dic["feature_SH000300_amount_std_pct_30"] = (RollingStatistics(SH000300_amount).std(30) / (SH000300_amount+1e-8))[buffer-1:-1]
        # features_dic["feature_SH000905_amount_std_pct_30"] = (RollingStatistics(SH000905_amount).std(30) / (SH000905_amount+1e-8))[buffer-1:-1]
        # features_dic["feature_SH000906_amount_std_pct_30"] = (RollingStatistics(SH000906_amount).std(30) / (SH000906_amount+1e-8))[buffer-1:-1]

        # features_dic["feature_SH000001_amount_std_pct_60"] = (RollingStatistics(SH000001_amount).std(60) / (SH000001_amount+1e-8))[buffer-1:-1]
        # features_dic["feature_SZ399001_amount_std_pct_60"] = (RollingStatistics(SZ399001_amount).std(60) / (SZ399001_amount+1e-8))[buffer-1:-1]
        # features_dic["feature_SH000300_amount_std_pct_60"] = (RollingStatistics(SH000300_amount).std(60) / (SH000300_amount+1e-8))[buffer-1:-1]
        # features_dic["feature_SH000905_amount_std_pct_60"] = (RollingStatistics(SH000905_amount).std(60) / (SH000905_amount+1e-8))[buffer-1:-1]
        # features_dic["feature_SH000906_amount_std_pct_60"] = (RollingStatistics(SH000906_amount).std(60) / (SH000906_amount+1e-8))[buffer-1:-1]

        # sorted_keys = sorted(features_dic.keys())
        # feature_matrix = np.column_stack([features_dic[k] for k in sorted_keys]).transpose(1, 0)
        # self.feature_matrix = torch.tensor(feature_matrix).unsqueeze(2).expand(-1, -1, stocks)
        # for i in range(self.feature_matrix.shape[1]):
        #     if cfg.get("data_x", None):
        #         for op in cfg.get('data_x'):
        #             ProcessDataForTrainingIntermediate = create_op_process(cfg.data_x[op])
        #             ProcessDataForTrainingIntermediate.apply(self.feature_matrix[:,i,:].squeeze(), test_stage=True)
        # import ipdb; ipdb.set_trace()
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
        
        if "extra_feature" in cfg.training.model_params and cfg.training.model_params["extra_feature"]:
            self.feature_matrix = torch.cat(feature_list, dim=0)
        else:
            self.feature_matrix = None
        
        # print(self.feature_matrix.shape)

        return

