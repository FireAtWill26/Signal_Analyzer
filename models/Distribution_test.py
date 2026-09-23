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
from collections import Counter
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import train_test_split
from multiprocessing.shared_memory import SharedMemory
from lion_pytorch import Lion
# from muon import Muon, MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from scipy.stats import skew, kurtosis

from prometheus.utils.utils import WeightedCorrNp, calc_fcst_weighted_ic, weighted_mse_wrap, weighted_mse_eval_wrap, filter_eligible_data, filter_eligible_data_v2, operations_by_group
from prometheus.utils.torch_utils import select_gpu_with_minimum_memory, early_stopping_func, WarmupLR, DecayingCosineWarmRestarts, SharedMemDataset, SharedMemSeqDataset
from prometheus.ops import create_op_process
from prometheus.modelpool.basemodel import BaseModel, create_model
from prometheus.utils.registry_factory import TRAINING_REGISTRY, MODEL_REGISTRY
from prometheus.utils.speedup_package import calculate_stock_mean_3d_parallel

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



@TRAINING_REGISTRY.register('Distribution_test')
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
        self.train_output = train_output

        self.training_dic = cfg.training.model_params
        Path.mkdir(Path(self.training_dic["model_path"]) / model_name, exist_ok=True, parents=True)

        self.model_state_path = Path(self.training_dic["model_path"]) / model_name / 'final_model.pt'
        self.optimizer_path = Path(self.training_dic["model_path"]) / model_name / 'final_optimizer.pt'
        self.dict_path = Path(self.training_dic["model_path"]) / model_name / 'training_dict.pt'
        self.log_path = Path(self.training_dic["model_path"]) / model_name / 'training_log.csv'
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
        
        train_y = train_output[indices[1]]

        self.backet_max, self.backet_min = np.max(train_y), np.min(train_y)

        self.backet_num = cfg.model.model_params["layers_hidden"][-1]
        self.backet_width = (self.backet_max - self.backet_min) / self.backet_num

        self.training_dic["backet_max"] = self.backet_max
        self.training_dic["backet_min"] = self.backet_min
        self.training_dic["backet_num"] = self.backet_num

        days, stocks = self.indices[1].shape
        if not "by_date" in self.training_dic:
            self.training_dic["by_date"] = False        
        rows, cols = np.where(indices[1])
        if self.training_dic["by_date"]:
            all_dates = np.unique(rows)
            train_dates, eval_dates = train_test_split(all_dates, test_size=self.training_dic["eval_size"], random_state=0, shuffle=True)
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
        else:
            train_idx, eval_idx = train_test_split(range(len(rows)), test_size=self.training_dic["eval_size"], random_state=0, shuffle=True)
            train_indices = np.zeros(indices[1].shape).astype(bool)
            for i in train_idx:
                train_indices[rows[i]][cols[i]] = True
            eval_indices = np.zeros(indices[1].shape).astype(bool)
            for i in eval_idx:
                eval_indices[rows[i]][cols[i]] = True
        
        if "date_weight" in self.training_dic and self.training_dic["date_weight"]:
            date_weight = torch.arange(days)
            date_weight = torch.sigmoid(date_weight-(days-20*self.training_dic["date_weight_month"])) * self.training_dic["date_weight_pow"]
        else:
            date_weight = torch.zeros(days)

        self.train_dataset = SharedMemDataset(model_name, self.train_input.shape, self.train_input.dtype, self.indices[0], train_indices, self.train_output,date_weight)
        self.eval_dataset = SharedMemDataset(model_name, self.train_input.shape, self.train_input.dtype, self.indices[0], eval_indices, self.train_output,date_weight)

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.training_dic["batch_size"], shuffle=True, drop_last=True, prefetch_factor=4, pin_memory=True, num_workers=num_workers, persistent_workers=True)
        self.eval_loader = DataLoader(self.eval_dataset, batch_size=self.training_dic["batch_size"], shuffle=False, drop_last=True, prefetch_factor=4, pin_memory=True, num_workers=num_workers, persistent_workers=True)

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
                self.cfg.model.model_params["layer_hidden"][i] *= self.input_dim
        self.model = create_model(self.cfg.model).to(self.device)
        self.tanh_output = nn.Tanh()


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

        print(f"target stats| Mean: {mean.item()}| Variance: {var.item()}| Skewness: {skewness}| Kurtosis: {kurt}")

        pbar = tqdm(range(self.training_dic['epochs']), desc='Training', ncols=160)

        if self.training_dic["loss_fn"] == "CCC":
            loss_fn = lambda x, y, w: self.ccc(x,y,w)
        elif self.training_dic["loss_fn"] == "Correlation":
            loss_fn = lambda x, y, w: self.correlation_loss(x,y,w)
        else:
            loss_fn = lambda x, y, w: 0

        if self.training_dic["weight_essential"]:
            self.weights = torch.arange(self.backet_num, dtype=torch.float32).to(self.device)
            self.weights = torch.sigmoid(self.weights - self.backet_num/2 - self.training_dic["weight_pos"]) * self.training_dic["weight_ratio"] + 1

            loss_class = nn.CrossEntropyLoss(weight=self.weights)
        else:
            loss_class = nn.CrossEntropyLoss()

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
        
        result = {"train_loss": [], "eval_loss": [], "eval_mse_loss": [], "eval_correlation_loss": [], "eval_correlation_mask": [],  "good_points": []}

        if self.training_dic["opt"] == "LBFGS":
            def closure():
                self.optimizer.zero_grad()
                pred = self.model.forward(data_batch)
                train_loss = loss_fn(pred, target_batch)
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
            for data_batch, target_batch, weight in self.train_loader:
                step_cnt += 1
                # 提前批量移动数据到设备
                if self.training_dic["denormalizer"]:
                    data_batch = data_batch.to(self.device)
                    batch_mean = torch.unsqueeze(torch.mean(data_batch, dim=1), 1)
                    batch_std = torch.unsqueeze(torch.std(data_batch, dim=1), 1)
                    data_batch = torch.cat([data_batch, 2*torch.tanh(batch_mean), 2*torch.tanh(batch_std)], dim=1).to(self.device, non_blocking=True)
                else:
                    data_batch = data_batch.to(self.device, non_blocking=True)
                target_batch = target_batch.to(self.device, non_blocking=True)
                target_batch = torch.floor((target_batch-self.backet_min)/ self.backet_width)
                target_batch = torch.clip(target_batch, 0, self.backet_num -1).to(self.device, non_blocking=True)
                if self.training_dic["soft_one_hot"]:
                    target_batch = torch.unsqueeze(target_batch, 1)
                    temp = torch.arange(self.backet_num, dtype=torch.float32).to(self.device)
                    target_batch = nn.Softmax(dim=1)(-((temp - target_batch)/self.training_dic["half_radius"]).pow(2))

                self.optimizer.zero_grad()
                # with autocast():
                pred = self.model.forward(data_batch)
                # assert torch.isnan(train_loss).sum() == 0, print(train_loss)
                # scaler.scale(train_loss).backward()

                if self.training_dic["soft_one_hot"]:
                    loss = loss_class(pred, target_batch)
                else:
                    loss = loss_class(pred, target_batch.to(torch.int64))


                if self.training_dic["enable_l1"]:
                    loss += self.training_dic["l1_ratio"] * self.l1_regularization()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                # scaler.step(self.optimizer)
                # scaler.update()
                self.optimizer.step()
                # assert torch.isnan(self.model.parameters()).sum() == 0, print(self.model.parameters())
                epoch_training_loss += loss.item()
                batch_num += 1

                if step_cnt == self.training_dic["eval_steps"]:
                    step_cnt = 0

                    train_loss = epoch_training_loss / batch_num
                    result["train_loss"].append(train_loss)

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
                                data_batch = torch.cat([data_batch, 2*torch.tanh(batch_mean), 2*torch.tanh(batch_std)], dim=1).to(self.device, non_blocking=True)
                            else:
                                data_batch = data_batch.to(self.device, non_blocking=True)
                            target_batch = target_batch.to(self.device, non_blocking=True)
                            weight = torch.ones_like(target_batch) + date_weight.to(self.device, non_blocking=True) 
                            if self.training_dic["weight_essential"]:
                                # mask = target_batch >= mean + self.training_dic["weight_pos"] * std
                                # weight = torch.where(mask, 2 * torch.ones_like(target_batch), weight)
                                weight += F.sigmoid(8 * (pred - pred_mean - self.training_dic["mask_pos"] * pred_std)) * self.training_dic["weight_pow_under"]
                            pred = self.model.forward(data_batch)
                            target_val = torch.linspace(self.backet_min, self.backet_max-self.backet_width, self.backet_num).to(self.device)
                            pred = nn.Softmax(dim=1)(pred) * target_val
                            pred = torch.sum(pred, dim=1)

                            new_mse_loss = nn.MSELoss()(pred, target_batch).item()
                            mse_loss += new_mse_loss
                            eval_loss += new_mse_loss
                            correlation_loss += self.correlation_loss(pred, target_batch, weight).item()
                            if not total_pred:
                                total_pred = [pred.detach()]
                                total_target = [target_batch.detach()]
                            else:
                                total_pred.append(pred.detach())
                                total_target.append(target_batch.detach())

                    eval_loss /= len(self.eval_loader)
                    mse_loss /= len(self.eval_loader)
                    correlation_loss /= len(self.eval_loader)
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

                    df = pd.DataFrame(result)
                    df.to_csv(self.log_path, index=False, header = True)

        del self.train_loader
        gc.collect()
        torch.cuda.empty_cache()

        pearson_ic, spearman_ic, mse_loss = self.evaluation(self.cfg)

        del self.eval_loader
        gc.collect()
        torch.cuda.empty_cache()

        return pearson_ic, spearman_ic, mse_loss

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
        self.backet_max = self.training_dic["backet_max"]
        self.backet_min = self.training_dic["backet_min"]
        self.backet_num = self.training_dic["backet_num"]
        self.backet_width = (self.backet_max - self.backet_min) / self.backet_num

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
                data_batch = torch.cat([data_batch, 2*torch.tanh(batch_mean), 2*torch.tanh(batch_std)], dim=1).to(self.device, non_blocking=True)
            else:
                data_batch = data_batch.to(self.device, non_blocking=True)
            target_batch = target_batch.to(non_blocking=True)
            pred = self.model.forward(data_batch)
            target_val = torch.linspace(self.backet_min, self.backet_max-self.backet_width, self.backet_num).to(self.device)
            pred = nn.Softmax(dim=1)(pred) * target_val
            pred = torch.sum(pred, dim=1)
            if not total_pred:
                total_pred = [pred.to("cpu").detach().numpy()]
                np_target = [target_batch.numpy()]
            else:
                total_pred.append(pred.to("cpu").detach().numpy())
                np_target.append(target_batch.numpy())
        total_pred = np.concatenate(total_pred, axis=0)
        np_target = np.concatenate(np_target, axis=0)

        # evaluate the model on the evaluation dataset. Return and print the correlation coefficients as well as mse loss
        # total_pred = model_result.forward(self.eval_dataset.x).detach().numpy()
        # total_pred = np.squeeze(total_pred)
        # assert np.isnan(total_pred).sum() == 0, print(total_pred)
        # np_target = self.eval_dataset.y.numpy()
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

        self.infer_mode = cfg.training.model_params["infer_mode"]
        self.infer_power = cfg.training.model_params["infer_power"]
        self.infer_half_radius = cfg.training.model_params["infer_half_radius"]

        cfg = torch.load(self.dict_path)
        if not "denormalizer" in cfg.training.model_params:
            cfg.training.model_params["denormalizer"] = False
        self.backet_max = cfg.training.model_params["backet_max"]
        self.backet_min = cfg.training.model_params["backet_min"]
        self.backet_num = cfg.training.model_params["backet_num"]
        self.backet_width = (self.backet_max - self.backet_min) / self.backet_num

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
        model = create_model(cfg.model)
        model.load_state_dict(torch.load(self.model_state_dict_path))
        model = model.to(self.device)
        model.eval()
        x = np.nan_to_num(x)

        if cfg.training.model_params["denormalizer"]:
            x = np.concatenate([x, 2*np.tanh(np.mean(x, axis=1)).reshape(-1, 1), 2*np.tanh(np.std(x, axis=1)).reshape(-1, 1)], axis=1)

        with torch.no_grad():
            if self.vertical_norm:
                pred = model.forward(F.normalize(torch.from_numpy(x), p=2.0, dim=0).to(torch.float32).to(self.device))
            else:
                pred = model.forward(torch.from_numpy(x).to(torch.float32).to(self.device))
            weight = torch.ones(self.backet_num).to(self.device)
            if self.infer_mode == "aggressive":
                weight += torch.tanh(torch.relu((torch.arange(self.backet_num).to(self.device)-self.backet_num//2)/(self.backet_num/self.infer_half_radius))).pow(3) * self.infer_power
            elif self.infer_mode == "passive":
                weight += torch.tanh(torch.relu((self.backet_num//2-torch.arange(self.backet_num).to(self.device))/(self.backet_num/self.infer_half_radius))).pow(3) * self.infer_power
            elif self.infer_mode == "two_side":
                weight += torch.tanh(torch.abs((self.backet_num//2-torch.arange(self.backet_num).to(self.device))/(self.backet_num/self.infer_half_radius))).pow(3) * self.infer_power
            target_val = torch.linspace(self.backet_min, self.backet_max-self.backet_width, self.backet_num).to(self.device)
            pred = nn.Softmax(dim=1)(pred) * weight
            pred /= torch.sum(pred)
            pred *= target_val
            pred = torch.sum(pred, dim=1)

        return pred.squeeze().to("cpu").numpy()