import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.utils.data as Data
from matplotlib import pyplot as plt

from distance import *
from dataloader import load_data
from train import train

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
Data from source and target domains are dictions whose keys are datetime and values are lists of daily data.
"""
with open('source_price_5min.pkl', 'rb') as handle:
    data_source = pickle.load(handle)
with open('target_price_5min.pkl', 'rb') as handle:
    data_target = pickle.load(handle)
# print("The source domain has {} days and the target domain has {} days.".format(len(data_source), len(data_target)))

X_train, y_train, y_scaler_train = load_data(data_source, 'price')
X_test, y_test, y_scaler_test = load_data(data_target, 'price')
X_train = torch.from_numpy(X_train).type(torch.Tensor).to(device)
y_train = torch.from_numpy(y_train).type(torch.Tensor).to(device)
X_test = torch.from_numpy(X_test).type(torch.Tensor).to(device)
y_test = torch.from_numpy(y_test).type(torch.Tensor).to(device)
dataset_train = Data.TensorDataset(X_train, y_train)
dataset_test = Data.TensorDataset(X_test, y_test)

"""
Diction mod stores attributes of the model.
model: could be LSTM, DA, GC, for LSTM, DAN(CORAL resp.) and CDAN respectively.
"""
mod = {'model':'GC', 'hidden_dim': 64, 'num_layers': 1,
       'patience': 20, 'n_batch': 7, 'num_epoch': 100,
       'source': dataset_train, 'target': dataset_test, 'batch_size': 1024,
       'X_test': X_test,'y_test':y_test, 'y_scaler_train': y_scaler_train, 'y_scaler_test': y_scaler_test,
       'method': 'KL', 't1': 0.1, 't2': 0.1
       }

a, b = train(mod)
