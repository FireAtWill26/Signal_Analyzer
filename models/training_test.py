from nt import error
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
from collections import Counter
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import train_test_split
from lion_pytorch import Lion
from muon import Muon
from torch.cuda.amp import autocast, GradScaler
import gc



from prometheus.utils.utils import WeightedCorrNp
from prometheus.ops import create_data_process
from prometheus.modelpool.basemodel import BaseModel, create_model
from prometheus.utils.registry_factory import TRAINING_REGISTRY, MODEL_REGISTRY

class Train_Dataset(Dataset):
    def __init__(self, x, y):
        self.x = F.normalize(torch.from_numpy(x).to(torch.float32), p=2.0, dim=1)
        self.y = torch.from_numpy(y).to(torch.float32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]



@TRAINING_REGISTRY.register('training_test')
class training_test(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

        self.input_dim = train_x.shape[1]
        if cfg.model.type == "FastKAN":
            for i in range(len(cfg.model.model_params["layers_hidden"])-1):
                cfg.model.model_params["layers_hidden"][i] *= self.input_dim
        self.model = create_model(cfg.model).to(self.device)
        self.tanh_output = nn.Tanh()

        scaler = GradScaler()

        torch.save(cfg, self.dict_path)

        train_x, eval_x, train_y, eval_y = train_test_split(train_x, train_y, test_size=51200, random_state=42)

        train_dataset = Train_Dataset(train_x, train_y)
        eval_dataset = Train_Dataset(eval_x, eval_y)

        num_workers = max(4, min(16, torch.cuda.device_count() * 4))  # 根据GPU数量动态调整


        train_loader = DataLoader(train_dataset, batch_size=self.training_dic["batch_size"], shuffle=True, drop_last=True, prefetch_factor=4, pin_memory=True, num_workers=num_workers, persistent_workers=True)
        eval_loader = DataLoader(eval_dataset, batch_size=self.training_dic["batch_size"], shuffle=False, prefetch_factor=4, pin_memory=True, num_workers=num_workers, persistent_workers=True)



        pbar = tqdm(range(self.training_dic['epochs']), desc='Training', ncols=150)

        if self.training_dic["loss_fn"] == "CCC":
            loss_fn = loss_fn_eval = lambda x, y: self.ccc(x,y)
        elif self.training_dic["loss_fn"] == "Correlation":
            loss_fn = loss_fn_eval = lambda x, y: self.correlation_loss(x,y)        
        else:
            loss_fn = loss_fn_eval = nn.MSELoss()


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
        elif self.training_dic["opt"] == "Muon":
            self.optimizer = Muon(self.model.parameters(), lr=lr, weight_decay=self.weight_decay)
        elif self.training_dic['opt'] == "SGD":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=self.weight_decay)
        elif self.training_dic['opt'] == "LBFGS":
            self.optimizer = torch.optim.LBFGS(self.model.parameters(), lr=lr, history_size=10, tolerance_grad=1e-32, tolerance_change=1e-32)
        else:
            raise ValueError("opt must be Adam, AdamW, SGD, LBFGS, or RMSprop.")

        
        result = {"train_loss": [], "eval_loss": []}

        if self.training_dic["opt"] == "LBFGS":
            def closure():
                self.optimizer.zero_grad()
                pred = self.model.forward(data_batch)
                train_loss = loss_fn(pred, target_batch)
                train_loss.backward()
                return train_loss

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, "min", factor=self.training_dic["scheduler_factor"], patience=self.training_dic["scheduler_patience"])

        for epoch in pbar:
            self.model.train()           
            epoch_training_loss = 0
            batch_num = 0
            best_eval_loss = float("inf")
            best_correlation_loss = -float("inf")

            for data_batch, target_batch in train_loader:
                # 提前批量移动数据到设备
                data_batch = data_batch.to(self.device, non_blocking=True)
                target_batch = target_batch.to(self.device, non_blocking=True)
                
                if self.training_dic["opt"] == "LBFGS":
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    train_loss = self.optimizer.step(closure)
                else:
                    self.optimizer.zero_grad()
                    # with autocast():
                    pred = self.model.forward(data_batch)
                    if self.training_dic["loss_fn"] == "CCC" or self.training_dic["loss_fn"] == "Correlation":

                        correlation_loss = loss_fn(torch.squeeze(pred), target_batch)
                        train_loss = -self.correlation_ratio * correlation_loss + (1 - self.correlation_ratio) * nn.MSELoss()(torch.squeeze(pred), target_batch)
                    else:
                        train_loss = loss_fn(torch.squeeze(pred), target_batch)
                    assert torch.isnan(train_loss).sum() == 0, print(train_loss)
                    # scaler.scale(train_loss).backward()
                    train_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    # scaler.step(self.optimizer)
                    # scaler.update()
                    self.optimizer.step()
                    assert torch.isnan(self.model.parameters()).sum() == 0, print(self.model.parameters())
                epoch_training_loss += train_loss.item()
                batch_num += 1
            train_loss = epoch_training_loss / batch_num
            result["train_loss"].append(train_loss)

            # evaluation after each of the epochs
            eval_loss = 0
            mse_loss = 0
            correlation_loss = 0

            self.model.eval()
            with torch.no_grad():
                for data_batch, target_batch in eval_loader:
                    data_batch = data_batch.to(self.device, non_blocking=True)
                    target_batch = target_batch.to(self.device, non_blocking=True)
                    pred = self.model.forward(data_batch)
                    new_mse_loss = nn.MSELoss()(torch.squeeze(pred), target_batch).item()
                    mse_loss += new_mse_loss
                    if self.training_dic["loss_fn"] == "CCC" or self.training_dic["loss_fn"] == "Correlation":
                        new_correlation_loss = loss_fn_eval(torch.squeeze(pred), target_batch).item()
                        correlation_loss += new_correlation_loss
                        eval_loss += -self.correlation_ratio * new_correlation_loss + (1-self.correlation_ratio) * new_mse_loss
                    # compute both of the combined loss and the mse loss
                    else:
                        eval_loss += new_mse_loss
            eval_loss /= len(eval_loader)
            mse_loss /= len(eval_loader)
            correlation_loss /= len(eval_loader)
            result["eval_loss"].append(eval_loss)

            
            # save the model weights if the evaluation loss is improved
            if self.training_dic["loss_fn"] == "CCC" or self.training_dic["loss_fn"] == "Correlation":
                if not result["eval_loss"] or (correlation_loss > best_correlation_loss and mse_loss < 1.):
                    best_correlation_loss = correlation_loss
                    torch.save(self.model.to("cpu").state_dict(), self.model_state_path)
                    torch.save(self.optimizer.state_dict(), self.optimizer_path)
                    self.model.to(self.device)
            else:
                if not result["eval_loss"] or eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    torch.save(self.model.to("cpu").state_dict(), self.model_state_path)
                    torch.save(self.optimizer.state_dict(), self.optimizer_path)
                    self.model.to(self.device)

            result["eval_loss"].append(eval_loss.item())

            pbar.set_description("Epoch :%d | Train Loss: %.2e | Evaluation Loss: %.2e | MSE Loss: %.2e | Correlation Loss: %.2e" % (epoch+1, train_loss, eval_loss, mse_loss, correlation_loss))

            scheduler.step(eval_loss)

        del train_loader
        del eval_loader

        gc.collect()
        torch.cuda.empty_cache()

        return self.evaluation(cfg, eval_dataset)

    def l1_regularization(self):
        l1_loss = 0
        for param in self.model.parameters():
            l1_loss += torch.sum(torch.abs(param))
        return l1_loss


    def evaluation(self, cfg, eval_dataset):
        # load the best model from the training process
        model_result = create_model(cfg.model)
        model_result.load_state_dict(torch.load(self.model_state_path))
        model_result = model_result.to("cpu")
        model_result.eval()
        # evaluate the model on the evaluation dataset. Return and print the correlation coefficients as well as mse loss
        total_pred = model_result.forward(eval_dataset.x).detach().numpy()
        total_pred = np.squeeze(total_pred)
        assert np.isnan(total_pred).sum() == 0, print(total_pred)
        np_target = eval_dataset.y.numpy()
        mse_loss = mean_squared_error(total_pred, np_target)
        pearson_ic = WeightedCorrNp(x=total_pred, y=np_target, w=np.ones(len(np_target)))('pearson')
        spearman_ic = WeightedCorrNp(x=total_pred, y=np_target, w=np.ones(len(np_target)))('spearman')
        print(f"peason_ic: {pearson_ic} | spearman_ic: {spearman_ic} | mse_loss: {mse_loss}")
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
        self.model_state_dict_path = Path(model_path) / model_name / "final_model.pt"
        self.dict_path = Path(model_path) / model_name / "training_dict.pt"
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
        cfg = torch.load(self.dict_path)

        model = create_model(cfg.model)
        model.load_state_dict(torch.load(self.model_state_dict_path))
        model = model.to(self.device)
        model.eval()
        x = np.nan_to_num(x)
        with torch.no_grad():
            pred = model.forward(F.normalize(torch.from_numpy(x), p=2.0, dim=0).to(torch.float32).to(self.device))
        return pred.squeeze().to("cpu").numpy()