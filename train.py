import os
import sys
sys.path.append('..')

import random
import numpy as np
import pandas as pd
import folktables
from folktables import ACSDataSource, ACSIncome, ACSEmployment

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import models, transforms
from FullyConnected import FullyConnected
import pytorch_lightning as pl

SEED = 0
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

dataset = "German"
data_df = pd.read_parquet("hf://datasets/cestwc/german-credit/data/train-00000-of-00001.parquet").dropna()
target = 'class'
sensitive_features = ['personal status and sex']

# dataset = "Adult"
# data_df = pd.read_parquet('hf://datasets/cestwc/census-income/data/train-00000-of-00001.parquet').dropna()
# target = 'income'
# sensitive_features = ['sex']

# dataset = 'Compas'
# from datasets import load_dataset
# data_df = load_dataset("cestwc/compas-temporary")['train'].to_pandas().fillna(0)
# target = 'is_violent_recid'
# sensitive_features = ['race']

# dataset = 'Bank'
# data_df = pd.read_parquet("hf://datasets/cestwc/bank-marketing/data/train-00000-of-00001.parquet")
# target = 'y'
# sensitive_features = ['age']


int_cols = data_df.select_dtypes(include=['int', 'int64', 'int32']).columns
data_df[int_cols] = data_df[int_cols].astype('object')


# Data loaders
import folk_utils
if(dataset in ["Employ", "Folk", "Insurance", "Coverage"]):
    X_train, X_test, X_val, y_train, y_test, y_val, lp_epsilon, sr_epsilon = folk_utils.get_dataset(dataset)
    f_epsilon = lp_epsilon
else:
    X_train, X_test, X_val, y_train, y_test, y_val, lp_epsilon, sr_epsilon = folk_utils.get_UCI_dataset(dataset, data_df, target, sensitive_features)
    f_epsilon = lp_epsilon


mode = "FAIR-DRO"

if(mode == "FAIR-GLOB"):
    lr = 0.001 #-10
    eps = 1.0
    alpha = 0.02#1
    bs = 256 
    glob_adv = 100
if(mode == "FAIR-IBP"):
    lr = 0.001
    eps = 0.02
    alpha = 0.02#2
    bs = 256 
    glob_adv = 0
elif(mode == "FAIR-IBPG"):
    lr = 0.001
    eps = 0.075
    alpha = 0.02
    bs = 256 
    glob_adv = 0
elif(mode == "FAIR-PGD"):
    lr = 0.001
    eps = 0.02
    alpha = 0.02
    bs = 256
    glob_adv = 0
elif(mode == "FAIR-DRO"):
    lr = 0.001
    eps = 0.02
    alpha = 0.02
    bs = 256
    glob_adv = 16
elif(mode == "SGD"):
    lr = 0.001
    eps = 0.00
    alpha = 0.0
    bs = 128
    glob_adv = 0
if(dataset == "German" and mode != "SGD"):
    alpha = 1.5
    eps *= 6
    #eps *= 3.5 

class custDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.Tensor(X).float()
        self.y = y
        print(X.shape, y.shape, '#############')
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return self.X.shape[0]
        
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    

CustTrain = custDataset(X_train, y_train)  
CustVal = custDataset(X_val, y_val) 
CustTest = custDataset(X_val, y_val)

class CustomDataModule(pl.LightningDataModule):
    def __init__(self, train, val, test, batch_size=bs):
        super().__init__()
        self.train_data = train
        self.val_data = val
        self.test_data = test
        self.batch_size = batch_size
        
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)
    
dm = CustomDataModule(CustTrain, CustVal, CustVal)





print(np.round(lp_epsilon, 2))
print(len(lp_epsilon))
print()


import time
if(mode == "FAIR-GLOB" or mode == "FAIR-IBPG"):
    mode_id = "FAIR-IBP"
else:
    mode_id = mode
    
if(dataset == "Coverage"):
    lr *= 2
DEPTH = 2
WIDTH = 2048
#model = FullyConnected(hidden_lay=2, hidden_dim=256,
model = FullyConnected(hidden_lay=DEPTH, hidden_dim=WIDTH,
                       learning_rate = lr, mode=mode_id, 
                       epsilon=eps, alpha=alpha,
                       glob_advs=glob_adv, dataset=dataset, in_dim=X_val.shape[1])
model.set_fair_interval(lp_epsilon)
# used to be 30, 35
model.MAX_EPOCHS = 5
start = time.time()
trainer = pl.Trainer(max_epochs=10, accelerator="cpu", devices=1)
trainer.fit(model, datamodule=dm)
result = trainer.test(model, datamodule=dm)
end = time.time()


print("Time: ", end - start)
print("Est: ", (end - start)*30)

torch_model = torch.hub.load('cat-claws/nn', 'simplecnn', convs = [], linears = [X_val.shape[1], WIDTH,  WIDTH], num_classes = 2)


key_mapping = dict(zip(
    ['layers.1.weight', 'layers.1.bias', 'layers.3.weight', 'layers.3.bias', 'layers.5.weight', 'layers.5.bias'],
    ['lays.0.weight', 'lays.0.bias', 'lays.1.weight', 'lays.1.bias', 'lays.2.weight', 'lays.2.bias']
))


new_state_dict = {target_key:model.state_dict()[source_key] for target_key, source_key in key_mapping.items()}


torch_model.load_state_dict(new_state_dict)
print(torch_model)