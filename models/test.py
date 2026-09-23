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
from copy import deepcopy
from collections import Counter
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import train_test_split
from lion_pytorch import Lion
# from muon import Muon, MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from scipy.stats import skew, kurtosis





from docs.Physics.Physics_2A_conservation_law import pred_fn
from docs.Physics.Physics_2B_conservation_law_2D import reg_loss
from prometheus.utils.utils import WeightedCorrNp
from prometheus.ops import create_data_process
from prometheus.modelpool.basemodel import BaseModel, create_model
from prometheus.utils.registry_factory import TRAINING_REGISTRY, MODEL_REGISTRY

class Train_Dataset(Dataset):
    def __init__(self, x, y, vertical_norm=False, denormalizer=False):
        self.x = torch.from_numpy(x).to(torch.float32)
        if denormalizer:
            mean = torch.unsqueeze(torch.mean(self.x, dim=1), 1)
            std = torch.unsqueeze(torch.std(self.x, dim=1), 1)
            self.x = torch.cat([self.x, 2*torch.tanh(mean), 2*torch.tanh(std)], dim=1)
        if vertical_norm:
            self.x = F.normalize(self.x, p=2.0, dim=0)
        else:
            self.x = F.normalize(self.x, p=2.0, dim=1)
        self.y = torch.from_numpy(y).to(torch.float32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]



@TRAINING_REGISTRY.register('test')
class training_test(BaseModel):
    def __init__(self, *args, **kwargs):
        # super().__init__(*args, **kwargs)
        pass

    def fit(self, cfg, train_x, train_y, sample_stocks_date, sample_stocks_seccode, model_name, *args, **kwargs):
        """
        print(sample_stocks_date)
        trainx = pd.DataFrame(train_x[:5000,:])
        trainy = pd.DataFrame(train_y[:5000])

        file_path = '/dfs/data'

        trainx.to_csv(file_path + 'trainx.csv', index=False)
        trainy.to_csv(file_path + 'trainy.csv', index=False)

        import ipdb; ipdb.set_trace()
        """

        super().fit_helper(cfg, train_x, train_y, sample_stocks_date=sample_stocks_date, model_name=model_name, *args, **kwargs)

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


        Path.mkdir(self.summary_path, exist_ok=True, parents=True)
        
        writer = SummaryWriter(self.summary_path / "run")

        if not "denormalizer" in self.training_dic:
            self.training_dic["denormalizer"] = False


        self.input_dim = train_x.shape[1]
        if self.training_dic["denormalizer"]:
            self.input_dim += 1
        self.cfg = deepcopy(cfg)
        if self.cfg.model.type == "FastKAN":
            for i in range(len(self.cfg.model.model_params["layers_hidden"])-self.training_dic["plain_layer"]):
                self.cfg.model.model_params["layers_hidden"][i] *= self.input_dim
        self.model = create_model(self.cfg.model).to(self.device)
        self.tanh_output = nn.Tanh()

        scaler = GradScaler()

        train_x, self.eval_x, train_y, self.eval_y = train_test_split(train_x, train_y, test_size=self.training_dic["eval_size"], random_state=self.training_dic["seed"])


        # if only_eval set to true in config, call the quick_evaluation function which will not train the model but only do the evaluation.

        if self.training_dic["only_eval"]:
            return self.quick_evaluation()
            
        torch.save(self.cfg, self.dict_path)
        
        self.vertical_norm = self.training_dic["vertical_norm"]
        train_dataset = Train_Dataset(train_x, train_y, self.vertical_norm, self.training_dic["denormalizer"])
        eval_dataset = Train_Dataset(self.eval_x, self.eval_y, self.vertical_norm, self.training_dic["denormalizer"])
        

        num_workers = max(4, min(16, torch.cuda.device_count() * 4))  # 根据GPU数量动态调整


        self.train_loader = DataLoader(train_dataset, batch_size=self.training_dic["batch_size"], shuffle=True, drop_last=True, prefetch_factor=4, pin_memory=True, num_workers=num_workers, persistent_workers=True)
        self.eval_loader = DataLoader(eval_dataset, batch_size=self.training_dic["batch_size"], shuffle=False, prefetch_factor=4, pin_memory=True, num_workers=num_workers, persistent_workers=True)

        var = torch.var(eval_dataset.y)
        std = torch.std(eval_dataset.y)
        mean = torch.mean(eval_dataset.y)
        skewness = skew(self.eval_y)
        kurt = kurtosis(self.eval_y)

        print(f"target stats| Mean: {mean.item()}| Variance: {var.item()}| Skewness: {skewness}| Kurtosis: {kurt}")

        pbar = tqdm(range(self.training_dic['epochs']), desc='Training', ncols=160)

        if self.training_dic["loss_fn"] == "CCC":
            loss_fn = loss_fn_eval = lambda x, y: self.ccc(x,y)
        elif self.training_dic["loss_fn"] == "Correlation":
            loss_fn = loss_fn_eval = lambda x, y: self.correlation_loss(x,y)        
        else:
            loss_fn = loss_fn_eval = lambda x, y: 0
        if self.training_dic["reg_loss"] == "Huber":
            reg_fn = nn.HuberLoss(reduction="none", delta=self.training_dic["huber_delta"])
        else:
            reg_fn = nn.MSELoss(reduction="none")

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
            self.optimizer = Lion(self.model.parameters(), lr=lr, weight_decay=self.weight_decay)
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
        
        result = {"train_loss": [], "eval_loss": [], "eval_mse_loss": [], "eval_correlation_loss": [], "eval_correlation_mask": [],  "good_points": [], "huber_delta": []}

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
            for data_batch, target_batch in self.train_loader:
                # 提前批量移动数据到设备
                data_batch = data_batch.to(self.device, non_blocking=True)
                target_batch = target_batch.to(self.device, non_blocking=True)
                
                if self.training_dic["opt"] == "LBFGS":
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    train_loss = self.optimizer.step(closure)
                else:
                    self.optimizer.zero_grad()
                    # with autocast():
                    pred = torch.squeeze(self.model.forward(data_batch))
                    train_mean = torch.mean(pred)
                    train_std = torch.std(pred)
                    if self.training_dic["ignore_negative_target"]:
                        mask = (target_batch <= self.training_dic["ignore_threshold"]) | (pred <= self.training_dic["ignore_threshold"])
                        pred = pred[mask]
                        target_batch = target_batch[mask]
                    if self.training_dic["weight_essential"]:
                        mask1 = ((target_batch < mean + self.training_dic["weight_pos"] * std) & (pred > pred_mean + self.training_dic["weight_pos"] * pred_std)) 
                        mask2 = (target_batch > mean + self.training_dic["weight_pos"] * std)
                        # & (pred < pred_mean + self.training_dic["weight_pos"] * pred_std)
                        weight = torch.ones_like(target_batch)
                        weight = torch.where(mask1, (1 + mean + self.training_dic["weight_pos"] * std - target_batch).pow(self.training_dic["weight_pow_over"]), weight)
                        weight = torch.where(mask2, (1 + target_batch - mean - self.training_dic["weight_pos"] * std).pow(self.training_dic["weight_pow_under"]), weight)
                    if self.training_dic["loss_fn"] == "CCC" or self.training_dic["loss_fn"] == "Correlation":
                        correlation_loss = loss_fn(pred, target_batch)
                        reg_loss = reg_fn(pred, target_batch)
                        if self.training_dic["weight_essential"]:
                            reg_loss = (reg_loss * weight)
                        reg_loss = reg_loss.mean()
                        train_loss = -self.correlation_ratio * correlation_loss + (1 - self.correlation_ratio) * reg_loss
                    else:
                        reg_loss = reg_fn(pred, target_batch)
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
                    # assert torch.isnan(self.model.parameters()).sum() == 0, print(self.model.parameters())
                epoch_training_loss += train_loss.item()
                batch_num += 1
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
                for data_batch, target_batch in self.eval_loader:
                    data_batch = data_batch.to(self.device, non_blocking=True)
                    target_batch = target_batch.to(self.device, non_blocking=True)
                    pred = torch.squeeze(self.model.forward(data_batch))

                    if self.training_dic["ignore_negative_target"]:
                        mask = (target_batch <= self.training_dic["ignore_threshold"]) | (pred <= self.training_dic["ignore_threshold"])
                        pred = pred[mask]
                        target_batch = target_batch[mask]
                    new_mse_loss = nn.MSELoss()(pred, target_batch).item()
                    mse_loss += new_mse_loss
                    if self.training_dic["loss_fn"] == "CCC" or self.training_dic["loss_fn"] == "Correlation":
                        new_correlation_loss = loss_fn_eval(pred, target_batch).item()
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



            if result["eval_loss"] and eval_loss > result["eval_loss"][-1] and epoch > self.training_dic["overfit_threshold"]:
                overfit_cnt += 1
                if overfit_cnt == self.training_dic["overfit_patience"]:
                    break
            else:
                overfit_cnt = 0
           
            # save the model weights if the evaluation loss is improved
            if not result["eval_loss"] or (mse_loss - 10 * correlation_loss < best_eval_loss):
                # best_correlation_loss = correlation_loss
                # if eval_loss < best_eval_loss:
                best_eval_loss = mse_loss - 10 * correlation_loss
                torch.save(self.model.to("cpu").state_dict(), self.model_state_path)
                torch.save(self.optimizer.state_dict(), self.optimizer_path)
                self.model.to(self.device)
            if self.training_dic["eval_mask"]:
                if not result["eval_correlation_mask"] or -correlation_mask + 10 * correlation_loss> best_correlation_mask_loss:
                    best_correlation_mask_loss = -correlation_mask + 10 * correlation_loss
                    torch.save(self.model.to("cpu").state_dict(), self.model_mask_path)
                    torch.save(self.optimizer.state_dict(), self.optimizer_mask_path)
                    self.model.to(self.device)
            if not result["good_points"] or result["good_points"][-1] > best_points_cnt:
                best_points_cnt = result["good_points"][-1]
                torch.save(self.model.to("cpu").state_dict(), self.model_most_path)
                torch.save(self.optimizer.state_dict(), self.optimizer_most_path)
                self.model.to(self.device)




            result["eval_loss"].append(eval_loss)

            pbar.set_description("Epoch :%d | Train Loss: %.2e | Evaluation Loss: %.2e | MSE Loss: %.2e | Correlation: %.2e" % (epoch+1, train_loss, eval_loss, mse_loss, correlation_loss))
            result["eval_mse_loss"].append(mse_loss)
            result["eval_correlation_loss"].append(correlation_loss)
            if self.training_dic["eval_mask"]:
                result["eval_correlation_mask"].append(correlation_mask)



            
            if self.training_dic["retrain"] and retrain_cnt == self.training_dic["retrain_period"]:
                self.training_dic["retrain_lr"] = max(self.optimizer.param_groups[0]["lr"], self.training_dic["retrain_lr"])
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
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, "min", factor=self.training_dic["scheduler_factor"], patience=self.training_dic["scheduler_patience"], threshold=self.training_dic["scheduler_threshold"])
                self.training_dic["retrain_period"] = max(self.training_dic["retrain_period"]-self.training_dic["retrain_step"], self.training_dic["retrain_floor"])
                self.training_dic["retrain_lr"] = max(self.training_dic["retrain_lr"]*self.training_dic["retrain_lr_factor"], self.training_dic["scheduler_threshold"])
                retrain_cnt = 1
            else:
                if self.training_dic["loss_fn"] == "CCC" or self.training_dic["loss_fn"] == "Correlation":
                    scheduler.step(mse_loss - 10 * correlation_loss)
                else:
                    scheduler.step(eval_loss)
                retrain_cnt += 1
                
            if self.training_dic["update_delta"] and self.training_dic["reg_loss"] == "Huber":
                if not (epoch+1) % self.training_dic["update_period"]:
                    if total_pred.shape[0] > 0:
                        self.training_dic["huber_delta"] = np.percentile(np.abs(total_pred.cpu().numpy()-total_target.cpu().numpy()), 90)
                        reg_fn = nn.HuberLoss(delta=self.training_dic["huber_delta"])
            result["huber_delta"].append(self.training_dic["huber_delta"])

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
        eval_dataset = Train_Dataset(self.eval_x, self.eval_y, self.vertical_norm, self.training_dic["denormalizer"])

        num_workers = max(4, min(16, torch.cuda.device_count() * 4))  # 根据GPU数量动态调整
        self.eval_loader = DataLoader(eval_dataset, batch_size=self.training_dic["batch_size"], shuffle=False, prefetch_factor=2, pin_memory=True, num_workers=num_workers, persistent_workers=False)

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
        for data_batch, target_batch in self.eval_loader:
            data_batch = data_batch.to(self.device, non_blocking=True)
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
        # total_pred = model_result.forward(eval_dataset.x).detach().numpy()
        # total_pred = np.squeeze(total_pred)
        # assert np.isnan(total_pred).sum() == 0, print(total_pred)
        # np_target = eval_dataset.y.numpy()
        delta_candidate = np.percentile(np.abs(total_pred-np_target), 90)
        mse_loss = mean_squared_error(total_pred, np_target)
        pearson_ic = WeightedCorrNp(x=total_pred, y=np_target, w=np.ones(len(np_target)))('pearson')
        spearman_ic = WeightedCorrNp(x=total_pred, y=np_target, w=np.ones(len(np_target)))('spearman')
        print(f"peason_ic: {pearson_ic} | spearman_ic: {spearman_ic} | mse_loss: {mse_loss} | std of pred: {np.std(total_pred)} | mean of pred: {np.mean(total_pred)} | max of pred: {np.max(total_pred)} | min of pred: {np.min(total_pred)}")
        print(f"according to the evaluation, the suggested delta value is {delta_candidate}")
        return pearson_ic, spearman_ic, mse_loss


    def correlation_loss(self, pred, target):
        # correlation_ratio = self.training_dic["correlation_ratio"]

        pred_centered = pred - pred.mean()
        target_centered = target - target.mean()

        corr_nume = torch.sum(pred_centered * target_centered)
        corr_denom = torch.sqrt((torch.sum(pred_centered ** 2) + 1e-7) * (torch.sum(target_centered ** 2) + 1e-7))

        corr = corr_nume / corr_denom
        # return - correlation_ratio * corr + (1-correlation_ratio) * torch.nn.MSELoss()(pred, target)

        return corr

    def ccc(self, pred, target):
        # compute the combined loss of correlation loss and mse loss with correlation_ratio in the training dict
        # correlation_ratio = self.training_dic["correlation_ratio"]
        sxy = torch.mean((pred - torch.mean(pred)) * (target - torch.mean(target)))
        ccc_loss = 2 * sxy /(torch.var(pred) + torch.var(target) + (pred.mean() - target.mean()) ** 2 + 1e-7)
        # return - correlation_ratio * ccc_loss + (1-correlation_ratio) * torch.nn.MSELoss()(pred, target)
        return ccc_loss



    def load(self, model_path, model_name, **kwargs):
        if model_path is None:
            assert self.model_path is not None
            model_path = self.model_path
        self.model_name = model_name
        self.dict_path = Path(model_path) / model_name / "training_dict.pt"
        self.model_state_dict_path = Path(model_path) / model_name / "final_model.pt"
        return 1

    def predict(self, x, cfg, *args, **kwargs):
        if cfg.get("data_x", None):
            for op in cfg.get('data_x'):
                ProcessDataForTrainingIntermediate = create_data_process(cfg.data_x[op])
                ProcessDataForTrainingIntermediate.apply(x, test_stage=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_dim = x.shape[1]

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
                pred = model.forward(F.normalize(torch.from_numpy(x), p=2.0, dim=1).to(torch.float32).to(self.device))
        return pred.squeeze().to("cpu").numpy()