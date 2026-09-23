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
        
        super().fit_helper(cfg, train_x, train_y, sample_stocks_date=sample_stocks_date, model_name=model_name, *args, **kwargs)

        self.training_dic = cfg.training.model_params
        Path.mkdir(Path(self.training_dic["model_path"]) / model_name, exist_ok=True, parents=True)


        self.input_dim = train_x.shape[1]

        self.cfg = deepcopy(cfg)
        self.models = []
        for i in range(len(self.cfg.model)):
            if self.cfg.model[i].type == "FastKAN":
                for j in range(len(self.cfg.model[i].model_params["layers_hidden"])-self.cfg.model[i].training_params["plain_layer"]):
                    self.cfg.model[i].model_params["layers_hidden"][j] *= self.input_dim
            self.models.append(create_model(self.cfg.model[i]).to(self.device))
        
        self.optimizers = []

        for i in range(len(self.models)):
            if self.cfg.model[i].optimizer == "Adam":
                self.optimizers.append(torch.optim.Adam(self.models[i].parameters(), lr=self.cfg.model[i].training_params["lr"], weight_decay=self.cfg.model[i].training_params["weight_decay"]))
            elif self.cfg.model[i].optimizer == "AdamW":
                self.optimizers.append(torch.optim.AdamW(self.models[i].parameters(), lr=self.cfg.model[i].training_params["lr"], weight_decay=self.cfg.model[i].training_params["weight_decay"])) 
            elif self.cfg.model[i].optimizer == "SGD":
                self.optimizers.append(torch.optim.SGD(self.models[i].parameters(), lr=self.cfg.model[i].training_params["lr"], weight_decay=self.cfg.model[i].training_params["weight_decay"]))
            elif self.cfg.model[i].optimizer == "Lion":
                self.optimizers.append(Lion(self.models[i].parameters(), lr=self.cfg.model[i].training_params["lr"], weight_decay=self.cfg.model[i].training_params["weight_decay"]))

        