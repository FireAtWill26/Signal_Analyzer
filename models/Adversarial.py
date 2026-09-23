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
from prometheus.utils.corr import calc_correlation

class Adversarial():
    def __init__(self, cfg, device, input_dim):
        self.device = device
        self.cfg = cfg
        self.input_dim = input_dim
        self.training_dic = cfg.training.model_params
        self.dis_dic = cfg.dis_train.model_params
        if "corr_importance" not in self.dis_dic:
            self.dis_dic["corr_importance"] = 5
        if "dis_loss_fn" in self.dis_dic and self.dis_dic["dis_loss_fn"] == "Wasserstain":
            self.adv_loss_fn = lambda x, y: WassersteinLoss(x, y)
        else:
            self.adv_loss_fn = nn.BCEWithLogitsLoss()
        if "dis_type" not in self.dis_dic:
            self.dis_dic["dis_type"] = "default"
        if self.dis_dic["dis_type"] == "only_target":
            self.real_label = torch.ones(1).to(self.device)
            self.fake_label = torch.zeros(1).to(self.device)
            if "FastKAN" in self.cfg.discriminator.type:
                for i in range(len(self.cfg.discriminator.model_params["layers_hidden"])-self.dis_dic["dis_plain_layers"]):
                    self.cfg.discriminator.model_params["layers_hidden"][i] = math.ceil(self.cfg.discriminator.model_params["layers_hidden"][i] * self.dis_dic["dis_target_size"])
        else:
            self.real_label = torch.ones(self.dis_dic["dis_target_size"], 1).to(self.device)
            self.fake_label = torch.zeros(self.dis_dic["dis_target_size"], 1).to(self.device)
            if "FastKAN" in self.cfg.discriminator.type:
                for i in range(len(self.cfg.discriminator.model_params["layers_hidden"])-3):
                    self.cfg.discriminator.model_params["layers_hidden"][i] = math.ceil(self.cfg.discriminator.model_params["layers_hidden"][i] * (self.input_dim + 1))
        self.discriminator = create_model(cfg.discriminator).to(self.device)
        self.discriminator.train()
        self.discriminator_optimizer=torch.optim.AdamW(self.discriminator.parameters(), lr=self.dis_dic["dis_lr"], weight_decay=self.dis_dic["weight_decay"])
        self.current_round = 0
        if "gp_round" in self.dis_dic:
            self.gp_round = self.dis_dic["gp_round"]
        else:
            self.gp_round = 16

    def train_with_model(self, pred, train_data, train_target):
        self.discriminator_optimizer.zero_grad()
        # print(pred.shape, train_target.shape)
        self.current_round += 1
        if self.dis_dic["dis_type"] == "only_target":
            loss_dis = self.adv_loss_fn(self.discriminator.forward(pred.detach()), self.fake_label) + self.adv_loss_fn(self.discriminator.forward(train_target.detach()), self.real_label)
            if self.dis_dic["gradient_penalty"] and self.current_round == self.gp_round:
                gp = self.Compute_Gradient_Penalty(pred.detach(), train_target.detach()) * self.gp_round
                loss_dis += gp
                self.current_round = 0
        else:
            train_target = torch.cat([train_data.detach(), train_target.detach().view(-1,1)], dim=1)
            pred = torch.cat([train_data.detach(), pred.detach().view(-1,1)], dim=1)
            loss_dis = self.adv_loss_fn(self.discriminator.forward(train_target), self.real_label) + self.adv_loss_fn(self.discriminator.forward(pred), self.fake_label)
            if self.dis_dic["gradient_penalty"] and self.current_round == self.gp_round:
                gp = self.Compute_Gradient_Penalty(pred, train_target) * self.gp_round
                loss_dis += gp
                self.current_round = 0
        loss_dis /= 2
        loss_dis.backward()
        self.discriminator_optimizer.step()
        if "dis_clip_weights" in self.dis_dic and self.dis_dic["dis_clip_weights"]:
            for param in self.discriminator.parameters():
                param.data.clamp_(-self.dis_dic["dis_clip_bound"], self.dis_dic["dis_clip_bound"])
        del train_data, train_target
        gc.collect()
        torch.cuda.empty_cache()
        return loss_dis.item()

    def adv_against_model(self, model_path, optimizer_path, train_dataloader, eval_dataloader, prev_models=None, feature_list=None):
        log_path = model_path.with_name("adv_log.csv")
        adv_res = {"Adv_loss": [], "MSE_loss": [], "Corr_Loss": [], "LR": [], "Dis_LR": []}
        self.model = create_model(self.cfg.model).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        params_to_update = []
        if "bptt" not in self.training_dic:
            layer_len = len(self.cfg.model.model_params["layers_hidden"]) - 1
            if "dis_finetune_layer" not in self.dis_dic:
                self.dis_dic["dis_finetune_layer"] = 0
            layer_threshold = layer_len - self.dis_dic["dis_finetune_layer"]

            for name, params in self.model.named_parameters():
                if "stem_component" in self.cfg.model.model_params and self.cfg.model.model_params["stem_component"]:
                    if "stem_component" in name:
                        params.requires_grad = False
                    else:
                        params.requires_grad = True
                        params_to_update.append(params)
                else:
                    if len(name.split(".")) > 1 and name.split(".")[1].isdigit() and int(name.split(".")[1]) < layer_threshold:
                        params.requires_grad = False
                    else:
                        params.requires_grad = True
                        params_to_update.append(params)
        else:
            for name, params in self.model.named_parameters():
                if self.cfg.model.type == "KANsformer":
                    if "Embedding" in name:
                        params.requires_grad = False
                    else:
                        params.requires_grad = True
                        params_to_update.append(params)
                else:
                    if "out" in name:
                        params.requires_grad = True
                        params_to_update.append(params)
                    else:
                        params.requires_grad = False

        self.optimizer = torch.optim.AdamW(params_to_update, lr=self.dis_dic["model_lr"], weight_decay=self.dis_dic["weight_decay"])
        self.discriminator_optimizer = torch.optim.AdamW(self.discriminator.parameters(), lr=self.dis_dic["dis_lr"]/self.dis_dic["dis_adv_ratio"], weight_decay=self.dis_dic["weight_decay"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, "min", factor=self.dis_dic["scheduler_factor"], patience=self.dis_dic["scheduler_patience"], threshold=self.dis_dic["scheduler_threshold"])

        if isinstance(self.dis_dic["dis_step_num"], int):
            dis_step = len(train_dataloader) // self.dis_dic["dis_step_num"]
        else:
            dis_step = int(self.dis_dic["dis_step_num"])

        pbar_dis = tqdm(range(self.dis_dic["dis_epoch"]), desc="discrimination", ncols=160)

        self.model.eval()

        mse_loss = 0
        correlation_loss = 0
        eval_batch_num = 0
        if "weight_essential" in self.dis_dic and self.dis_dic["weight_essential"]:
            total_pred = []
            pred_mean = 0
            pred_std = 0.2
        with torch.no_grad():
            for data in eval_dataloader:
                extra_dict = data[-1]
                if self.training_dic.get("reduce_low_cap", False):
                    reduction_cap = extra_dict["reduction_cap"].to(dtype=data[0].dtype)
                extra_feat = extra_dict["features"].squeeze()
                if "extra_feature" in self.training_dic and self.training_dic["extra_feature"]:
                    if "bptt" in self.training_dic:
                        extra_feat = extra_feat
                    data_batch = torch.cat([data[0], extra_feat], dim=-1).to(self.device, non_blocking=True, dtype=data[0].dtype)
                else:
                    data_batch = data[0].to(self.device, non_blocking=True)
                if self.training_dic.get("reduce_low_cap", False):
                    if self.training_dic.get("reduce_method", "addition") == "addition":
                        target_batch = (data[1] + reduction_cap)
                    elif self.training_dic.get("reduce_method", "addition") == "multiplication":
                        target_batch = (data[1] * reduction_cap)
                else:
                    target_batch = data[1]

                if len(target_batch.shape) > 1:
                    target_batch = target_batch[:,-1].to(self.device, non_blocking=True)
                    if prev_models is not None:
                        for model, features in zip(prev_models, feature_list[:-1]):
                            target_batch -= model.forward(data_batch[:,:,features])[:,-1]
                    if feature_list is not None:
                        data_batch = data_batch[:, :, feature_list[-1]]
                    pred = self.model.forward(data_batch)[:,-1]
                else:
                    target_batch = target_batch.to(self.device, non_blocking=True)
                    if prev_models is not None:
                        for model, features in zip(prev_models, feature_list[:-1]):
                            target_batch -= model.forward(data_batch[:,features]).squeeze()
                    if feature_list is not None:
                        data_batch = data_batch[:, feature_list[-1]]
                    pred = torch.squeeze(self.model.forward(data_batch))
                if "weight_essential" in self.dis_dic and self.dis_dic["weight_essential"]:
                    total_pred.append(pred)
                weight = torch.ones_like(pred)
                if "weight_essential" in self.dis_dic and self.dis_dic["weight_essential"]:
                    if "weight_pos_up" in self.dis_dic:
                        upper_adjust_p = F.sigmoid(8 * (pred_mean + self.dis_dic["weight_pos_up"] * pred_std - pred))
                    else:
                        upper_adjust_p = 1
                    weight += F.sigmoid(8 * pred - pred_mean - self.dis_dic["weight_pos"] * pred_std) * upper_adjust_p * self.dis_dic["weight_pow"]
                mse_loss += torch.mean(F.mse_loss(pred, target_batch)).item()
                correlation_loss += calc_correlation(pred, target_batch, weight).item()
                eval_batch_num += 1
        mse_loss /= eval_batch_num
        correlation_loss /= eval_batch_num
        if "weight_essential" in self.dis_dic and self.dis_dic["weight_essential"]:
            total_pred = torch.cat(total_pred, dim=0)
            pred_mean = total_pred.mean()
            pred_std = total_pred.std()

        
        torch.save(self.optimizer.state_dict(), optimizer_path)
        adv_res["Adv_loss"].append(0)
        adv_res["MSE_loss"].append(mse_loss)
        adv_res["Corr_Loss"].append(correlation_loss)
        adv_res["LR"].append(self.optimizer.param_groups[-1]["lr"])
        adv_res["Dis_LR"].append(self.discriminator_optimizer.param_groups[0]["lr"])

        best_eval = mse_loss - self.dis_dic["corr_importance"] * correlation_loss

        dis_overfit_cnt = 0

        if "jumpback" in self.dis_dic and self.dis_dic["jumpback"]:
            unsaved_steps = 0

        for epoch in pbar_dis:
            self.discriminator.train()
            total_adv_loss = 0
            batch_num = 0
            step_cnt = 0
            if self.training_dic["batch_size"] != self.dis_dic["dis_target_size"]:
                target_len = 0
                total_target = []
                total_pred = []
                total_data = []
            for data in train_dataloader:
                extra_dict = data[-1]
                if self.training_dic.get("reduce_low_cap", False):
                    reduction_cap = extra_dict["reduction_cap"].squeeze().to(dtype=data[0].dtype)
                extra_feat = extra_dict["features"].squeeze()
                self.discriminator_optimizer.zero_grad()
                if "extra_feature" in self.training_dic and self.training_dic["extra_feature"]:
                    data_batch = torch.cat([data[0], extra_feat], dim=-1).to(self.device, non_blocking=True, dtype=data[0].dtype)
                else:
                    data_batch = data[0].to(self.device, non_blocking=True)
                if self.training_dic.get("reduce_low_cap", False):
                    if self.training_dic.get("reduce_method", "addition") == "addition":
                        target_batch = (data[1] + reduction_cap)
                    elif self.training_dic.get("reduce_method", "addition") == "multiplication":
                        target_batch = (data[1] * reduction_cap)
                else:
                    target_batch = data[1]
                if len(target_batch.shape) > 1:
                    target_batch = target_batch[:,-1].to(self.device, non_blocking=True)
                    if prev_models is not None:
                        for model, features in zip(prev_models, feature_list[:-1]):
                            target_batch -= model.forward(data_batch[:, :, features])[:,-1]
                    if feature_list is not None:
                        data_batch = data_batch[:, :, feature_list[-1]]
                    pred = self.model.forward(data_batch)[:,-1]
                    data_attach = data_batch[:,-1,:]
                else:
                    target_batch = target_batch.to(self.device, non_blocking=True)
                    if prev_models is not None:
                        for model, features in zip(prev_models, feature_list[:-1]):
                            target_batch -= model.forward(data_batch[:, features]).squeeze()
                    if feature_list is not None:
                        data_batch = data_batch[:, feature_list[-1]]
                    pred = torch.squeeze(self.model.forward(data_batch))
                    data_attach = data_batch
                    data = data_batch
                if self.training_dic["batch_size"] != self.dis_dic["dis_target_size"]:
                    target_len += target_batch.shape[0]
                    total_target.append(target_batch.detach())
                    total_pred.append(pred.detach())
                    total_data.append(data_batch.detach())
                if (self.training_dic["batch_size"] == self.dis_dic["dis_target_size"]) or (self.training_dic["batch_size"] != self.dis_dic["dis_target_size"] and target_len >= self.dis_dic["dis_target_size"]):
                    if self.training_dic["batch_size"] != self.dis_dic["dis_target_size"]:
                        target_batch = torch.cat(total_target, dim=0)
                        pred = torch.cat(total_pred, dim=0)
                        # data_attach = torch.cat(total_data, dim=0)
                        data_attach = total_data
                        if len(data_attach[0].shape) > 2:
                            data = torch.cat(data_attach, dim=0)[:,-1,:]
                        else:
                            data = torch.cat(data_attach, dim=0)
                        target_len = 0
                        total_target = []
                        total_pred = []
                        total_data = []
                    if self.dis_dic["dis_type"] == "only_target":
                        adv_loss = self.adv_loss_fn(self.discriminator.forward(pred.detach()), self.fake_label) + self.adv_loss_fn(self.discriminator.forward(target_batch.detach()), self.real_label)
                        if self.dis_dic["gradient_penalty"]:
                            gp = self.Compute_Gradient_Penalty(pred, target_batch)
                            adv_loss += gp
                    else:
                        target_batch = torch.cat([data.detach(), target_batch.detach().view(-1,1)], dim=1)
                        pred = torch.cat([data.detach(), pred.detach().view(-1,1)], dim=1)
                        adv_loss = self.adv_loss_fn(self.discriminator.forward(target_batch), self.real_label) + self.adv_loss_fn(self.discriminator.forward(pred), self.fake_label)
                        if self.dis_dic["gradient_penalty"]:
                            gp = self.Compute_Gradient_Penalty(pred, target_batch)
                            adv_loss += gp
                    adv_loss /= 2
                    adv_loss.backward()
                    total_adv_loss += adv_loss.item()
                    batch_num += 1
                    step_cnt += 1
                    self.discriminator_optimizer.step()
                    if "dis_clip_weights" in self.dis_dic and self.dis_dic["dis_clip_weights"]:
                        for param in self.discriminator.parameters():
                            param.data.clamp_(-self.dis_dic["dis_clip_bound"], self.dis_dic["dis_clip_bound"])
                if step_cnt == dis_step:
                    step_cnt = 0
                    self.discriminator.eval()
                    self.model.train()
                    if isinstance(data_attach, list):
                        pred = []
                        with torch.no_grad():
                            for i in range(len(data_attach)):
                                pred.append(torch.squeeze(self.model.forward(data_attach[i])))
                        pred = torch.cat(pred, dim=0)
                        pred.requires_grad_(True)
                    else:
                        pred = torch.squeeze(self.model.forward(data_attach))
                    # pred = torch.squeeze(self.model.forward(data_attach))
                    if len(pred.shape) > 1:
                        pred_last = pred[:,-1]
                    else:
                        pred_last = pred
                    if self.dis_dic["dis_type"] == "only_target":
                        dis_loss = self.adv_loss_fn(self.discriminator.forward(pred_last), self.real_label)
                    else:
                        dis_loss = self.adv_loss_fn(self.discriminator.forward(torch.cat([data, pred_last.view(-1,1)], dim=1)), self.real_label)  
                    self.optimizer.zero_grad()                      
                    dis_loss.backward()
                    # import ipdb; ipdb.set_trace()
                    if isinstance(data_attach, list):
                        intermediate_grad = pred.grad.detach()
                        for i in range(len(data_attach)):
                            grad_chunk = intermediate_grad[self.training_dic["batch_size"]*i: self.training_dic["batch_size"]*(i+1)]
                            y_chunk = torch.squeeze(self.model.forward(data_attach[i]))
                            y_chunk.backward(gradient=grad_chunk)
                    self.optimizer.step()
                    self.model.eval()
                    mse_loss = 0
                    correlation_loss = 0
                    eval_batch_num = 0
                    if "weight_essential" in self.dis_dic and self.dis_dic["weight_essential"]:
                        total_pred = []
                    with torch.no_grad():
                        for data in eval_dataloader:
                            extra_dict = data[-1]
                            if self.training_dic.get("reduce_low_cap", False):
                                reduction_cap = extra_dict["reduction_cap"].squeeze().to(dtype=data[0].dtype)
                            extra_feat = extra_dict["features"].squeeze()
                            if "extra_feature" in self.training_dic and self.training_dic["extra_feature"]:
                                data_batch = torch.cat([data[0], extra_feat], dim=-1).to(self.device, non_blocking=True, dtype=data[0].dtype)
                            else:
                                data_batch = data[0].to(self.device, non_blocking=True)
                            if self.training_dic.get("reduce_low_cap", False):
                                if self.training_dic.get("reduce_method", "addition") == "addition":
                                    target_batch = (data[1] + reduction_cap)
                                elif self.training_dic.get("reduce_method", "addition") == "multiplication":
                                    target_batch = (data[1] * reduction_cap)
                            else:
                                target_batch = data[1]
                            if len(target_batch.shape) > 1:
                                target_batch = target_batch[:,-1].to(self.device, non_blocking=True)
                                if prev_models is not None:
                                    for model, features in zip(prev_models, feature_list):
                                        target_batch -= model.forward(data_batch[:,:,features])[:,-1]
                                if feature_list is not None:
                                    data_batch = data_batch[:, :, feature_list[-1]]
                                pred = self.model.forward(data_batch)[:,-1]
                            else:
                                target_batch = target_batch.to(self.device, non_blocking=True)
                                if prev_models is not None:
                                    for model, features in zip(prev_models, feature_list):
                                        target_batch -= model.forward(data_batch[:, features]).squeeze()
                                if feature_list is not None:
                                    data_batch = data_batch[:, feature_list[-1]]
                                pred = torch.squeeze(self.model.forward(data_batch))
                            weight = torch.ones_like(pred)
                            if "weight_essential" in self.dis_dic and self.dis_dic["weight_essential"]:
                                if "weight_pos_up" in self.dis_dic:
                                    upper_adjust_p = F.sigmoid(8 * (pred_mean + self.dis_dic["weight_pos_up"] * pred_std - pred))
                                else:
                                    upper_adjust_p = 1
                                weight += F.sigmoid(8 * pred - pred_mean - self.dis_dic["weight_pos"] * pred_std) * upper_adjust_p * self.dis_dic["weight_pow"]
                            mse_loss += torch.mean(F.mse_loss(pred, target_batch)).item()
                            correlation_loss += calc_correlation(pred, target_batch, weight=weight).item()
                            if "weight_essential" in self.dis_dic and self.dis_dic["weight_essential"]:
                                total_pred.append(pred)
                            eval_batch_num += 1
                    mse_loss /= eval_batch_num
                    self.discriminator.train()
                    correlation_loss /= eval_batch_num
                    if "weight_essential" in self.dis_dic and self.dis_dic["weight_essential"]:
                        total_pred = torch.cat(total_pred, dim=0)
                        pred_mean = total_pred.mean()
                        pred_std = total_pred.std()
                        total_pred = []
                    pbar_dis.set_description("Epoch :%d|Adv Loss: %.2e|MSE Loss: %.2e|Correlation Loss: %.2e" % (epoch+1, total_adv_loss/batch_num, mse_loss, correlation_loss))
                    adv_loss = total_adv_loss/batch_num
                    adv_res["Adv_loss"].append(adv_loss)
                    adv_res["MSE_loss"].append(mse_loss)
                    adv_res["Corr_Loss"].append(correlation_loss)
                    adv_res["LR"].append("%.5e" % self.optimizer.param_groups[0]["lr"])
                    adv_res["Dis_LR"].append("%.5e" % self.discriminator_optimizer.param_groups[0]["lr"])
                    df_adv = pd.DataFrame(adv_res)
                    df_adv.to_csv(log_path, index=False, header=True)

                    if best_eval > mse_loss - self.dis_dic["corr_importance"] * correlation_loss:
                        best_eval = mse_loss - self.dis_dic["corr_importance"] * correlation_loss
                        torch.save(self.model.state_dict(), model_path)
                        torch.save(self.optimizer.state_dict(), optimizer_path)
                        if "jumpback" in self.dis_dic and self.dis_dic["jumpback"]:
                            unsaved_steps = 0
                    else:
                        dis_overfit_cnt += 1
                        if "jumpback" in self.dis_dic and self.dis_dic["jumpback"]:
                            unsaved_steps += 1

                    if "dis_retrain" in self.dis_dic and self.dis_dic["dis_retrain"]:
                        if dis_overfit_cnt >= self.dis_dic["dis_overfit_threshold"]:
                            dis_overfit_cnt = 0
                            self.model.load_state_dict(torch.load(model_path))
                            self.dis_dic["retrain_lr"] = max(self.optimizer.param_groups[0]["lr"]/2, self.dis_dic["retrain_lr"])
                            # Re-set requires_grad for parameters after loading state_dict
                            # params_to_update = []
                            # if "bptt" not in self.training_dic:
                            #     layer_len = len(self.cfg.model.model_params["layers_hidden"]) - 1
                            #     if "dis_finetune_layer" not in self.dis_dic:
                            #         self.dis_dic["dis_finetune_layer"] = 0
                            #     layer_threshold = layer_len - self.dis_dic["dis_finetune_layer"]

                            #     for name, params in self.model.named_parameters():
                            #         if int(name.split(".")[1]) < layer_threshold:
                            #             params.requires_grad = False
                            #         else:
                            #             params.requires_grad = True
                            #             params_to_update.append(params)
                            # else:
                            #     for name, params in self.model.named_parameters():
                            #         if "Embedding" in name:
                            #             params.requires_grad = False
                            #         else:
                            #             params.requires_grad = True
                            #             params_to_update.append(params)
                            
                            # # Re-create optimizer with updated parameters
                            # self.optimizer = torch.optim.AdamW(params_to_update, lr=self.dis_dic["retrain_lr"], weight_decay=self.dis_dic["weight_decay"])
                            self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.device))
                            for param_group in self.optimizer.param_groups:
                                param_group["lr"] = self.dis_dic["retrain_lr"]
                            
                            # self.model.to(self.device)
                            del scheduler
                            gc.collect()
                            torch.cuda.empty_cache()
                            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, "min", factor=self.dis_dic["scheduler_factor"], patience=self.dis_dic["scheduler_patience"], threshold=self.dis_dic["scheduler_threshold"])
                            self.dis_dic["retrain_lr"] = max(self.dis_dic["retrain_lr"]*self.dis_dic["retrain_lr_factor"], self.dis_dic["scheduler_threshold"])
                            if "jumpback" in self.dis_dic and self.dis_dic["jumpback"]:
                                if unsaved_steps >= self.dis_dic["jumpback_step"]:
                                    self.dis_dic["model_lr"] *= self.dis_dic["jumpback_ratio"]
                                    self.dis_dic["retrain_lr"] = max(self.dis_dic["retrain_lr"], self.dis_dic["model_lr"])
                                    unsaved_steps = 0
                        else:
                            scheduler.step(mse_loss - self.dis_dic["corr_importance"] * correlation_loss)
                    else:
                        scheduler.step(mse_loss - self.dis_dic["corr_importance"] * correlation_loss)
        del self.model
        gc.collect()
        torch.cuda.empty_cache()
    
    def Compute_Gradient_Penalty(self, pred, target, weight=None):
        alpha = torch.rand(pred.size(0), 1, device=pred.device)
        interpolates = alpha * target + (1 - alpha) * pred
        interpolates.requires_grad_(True)
        disc_interpolates = self.discriminator(interpolates)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                        grad_outputs=torch.ones_like(disc_interpolates),
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.shape[0], -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.dis_dic["gp_weight"]
        return gradient_penalty

def WassersteinLoss(input, label):
    return torch.mean(input * (2 * label - 1))