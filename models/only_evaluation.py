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
    def __init__(self, x, y, vertical_norm=False):
        if vertical_norm:
            self.x = F.normalize(torch.from_numpy(x).to(torch.float32), p=2.0, dim=0)
        else:
            self.x = torch.from_numpy(x).to(torch.float32)
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

        train_x, eval_x, train_y, eval_y = train_test_split(train_x, train_y, test_size=self.training_dic["eval_size"], random_state=42)

        self.vertical_norm = self.training_dic["vertical_norm"]
        train_dataset = Train_Dataset(train_x, train_y, self.vertical_norm)
        eval_dataset = Train_Dataset(eval_x, eval_y, self.vertical_norm)

        num_workers = max(4, min(16, torch.cuda.device_count() * 4))  # 根据GPU数量动态调整


        train_loader = DataLoader(train_dataset, batch_size=self.training_dic["batch_size"], shuffle=True, drop_last=True, prefetch_factor=4, pin_memory=True, num_workers=num_workers, persistent_workers=True)
        eval_loader = DataLoader(eval_dataset, batch_size=self.training_dic["batch_size"], shuffle=False, prefetch_factor=4, pin_memory=True, num_workers=num_workers, persistent_workers=True)

        cfg = torch.load(self.dict_path)
        self.training_dic = cfg.training.model_params

        del train_loader



        pearson_ic, spearman_ic, mse_loss = self.evaluation(cfg, eval_loader)

        del eval_loader
        gc.collect()
        torch.cuda.empty_cache()

        return pearson_ic, spearman_ic, mse_loss


    def l1_regularization(self):
        l1_loss = 0
        for param in self.model.parameters():
            l1_loss += torch.sum(torch.abs(param))
        return l1_loss


    def evaluation(self, cfg, eval_dataloader):
        # load the best model from the training process
        model_result = create_model(cfg.model)
        model_result.load_state_dict(torch.load(self.model_state_path))
        model_result = model_result.to(self.device)
        model_result.eval()
        total_pred = [np.array([])]
        np_target = [np.array([])]
        for data_batch, target_batch in eval_dataloader:
            data_batch = data_batch.to(self.device, non_blocking=True)
            target_batch = target_batch.to(non_blocking=True)
            pred = model_result.forward(data_batch)
            if not total_pred:
                total_pred = [torch.squeeze(pred).to("cpu").detach().numpy()]
                np_target = [target_batch.numpy()]
            else:
                total_pred = total_pred.append(torch.squeeze(pred).to("cpu").detach().numpy())
                np_target = np_target.append(target_batch.numpy())
        total_pred = np.concatenate(total_pred, axis=0)
        np_target = np.concatenate(np_target, axis=0)

        # evaluate the model on the evaluation dataset. Return and print the correlation coefficients as well as mse loss
        # total_pred = model_result.forward(eval_dataset.x).detach().numpy()
        # total_pred = np.squeeze(total_pred)
        # assert np.isnan(total_pred).sum() == 0, print(total_pred)
        # np_target = eval_dataset.y.numpy()
        mse_loss = mean_squared_error(total_pred, np_target)
        pearson_ic = WeightedCorrNp(x=total_pred, y=np_target, w=np.ones(len(np_target)))('pearson')
        spearman_ic = WeightedCorrNp(x=total_pred, y=np_target, w=np.ones(len(np_target)))('spearman')
        print(f"peason_ic: {pearson_ic} | spearman_ic: {spearman_ic} | mse_loss: {mse_loss} | std of pred: {np.std(total_pred)} | mean of pred: {np.mean(total_pred)}")

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
        self.vertical_norm = cfg.training.model_params["vertical_norm"]

        with torch.no_grad():
            if self.vertical_norm:
                pred = model.forward(F.normalize(torch.from_numpy(x), p=2.0, dim=0).to(torch.float32).to(self.device))
            else:
                pred = model.forward(torch.from_numpy(x).to(torch.float32).to(self.device))
        return pred.squeeze().to("cpu").numpy()