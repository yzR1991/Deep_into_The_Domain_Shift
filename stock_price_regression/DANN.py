# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import scipy
import scipy.stats
import torch
import torch.nn as nn
import torch.utils.data as Data
import time
import math
import datetime
import collections
import pickle
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
from sklearn.metrics import mean_squared_error

"""## Base Modules

### Feature Extractor
"""

class FeatureExtractor(nn.Module):
    """
    Feature Extractor
    input: batch_size*n_periods*n_stocks
    output: batch_size*hidden_dim
    """
    def __init__(self, input_dim=22, hidden_dim=64, num_layers=1):
        super(FeatureExtractor, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.layer = nn.Sequential(nn.BatchNorm1d(hidden_dim) ,nn.Linear(hidden_dim, hidden_dim))

    def forward(self, x, hc=(None, None)):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        if hc[0] != None and hc[1] != None:
            out, (hn, cn) = self.lstm(x, (hc[0], hc[1]))
        else:
            out, (hn, cn) = self.lstm(x, (h0.detach().to(device), c0.detach().to(device)))
        
        out = self.layer(out[:,-1,:]) 
        return out

"""### Classifier"""

class Classifier(nn.Module):
    """
    Classifier
    """
    def __init__(self, input_dim=64, output_dim=22):
        super(Classifier, self).__init__()
        self.layer = nn.Sequential(nn.Linear(input_dim, input_dim // 2), nn.ReLU(),
                                   nn.Linear(input_dim // 2, output_dim))
    
    def forward(self, x):
        out = self.layer(x)
        return out

"""### Discriminator"""

class Discriminator(nn.Module):
    """
    Discriminator
    """
    def __init__(self, input_dim=64, output_dim=1):
        super(Discriminator, self).__init__()
        self.layer = nn.Sequential(nn.Linear(input_dim, input_dim // 2), nn.ReLU(),
                                   nn.Linear(input_dim // 2, input_dim //4), nn.ReLU(),
                                   nn.Linear(input_dim // 4, output_dim), nn.Sigmoid())
    def forward(self, x):
        out = self.layer(x)
        return out

"""## Data Preparation"""

device = torch.device('cpu')

with open('source_price_5min.pkl', 'rb') as handle:
    data_source = pickle.load(handle)
with open('target_price_5min.pkl', 'rb') as handle:
    data_target = pickle.load(handle)
print("The source domain has {} days and the target domain has {} days.".format(len(data_source), len(data_target)))

def package_data(input,look_back = 12):
    data = []
    for index in range(input.shape[0] - look_back + 1):
        data.append(input[index:index + look_back])
    data = np.array(data)
    return data

def load_data(dict_data, data_type='price'):
    values_list = list(dict_data.values())
    x_all, y_all = values_list[0][0], values_list[0][1]
    for i in range(1,len(values_list)):
        x_all = np.concatenate((x_all,values_list[i][0]))
        y_all = np.concatenate((y_all,values_list[i][1]))
    if data_type == 'return':
        y_scaler = StandardScaler()
        x_scaler = StandardScaler().fit(x_all)
        y_pieces = y_scaler.fit_transform(y_all)
    elif data_type == 'price':
        y_scaler = MinMaxScaler(feature_range=(-1,1))
        x_scaler = MinMaxScaler(feature_range=(-1,1)).fit(x_all)
        y_pieces = y_scaler.fit_transform(y_all)
    elif data_type == 'scale':
        y_scaler = MaxAbsScaler()
        x_scaler = MaxAbsScaler().fit(x_all)
        y_pieces = y_scaler.fit_transform(y_all)
    # y_pieces = StandardScaler().fit_transform(y_all.reshape(-1,1))
    
    x_pieces = package_data(x_scaler.transform(values_list[0][0]))
    for i in range(1,len(values_list)):
        x_piece = package_data(x_scaler.transform(values_list[i][0]))
        x_pieces = np.concatenate((x_pieces, x_piece), axis=0)
    return x_pieces, y_pieces, y_scaler
X_train, y_train, y_scaler_train = load_data(data_source, 'price')

X_test, y_test, y_scaler_test = load_data(data_target, 'price')
X_train = torch.from_numpy(X_train).type(torch.Tensor).to(device)
y_train = torch.from_numpy(y_train).type(torch.Tensor).to(device)
X_test = torch.from_numpy(X_test).type(torch.Tensor).to(device)
y_test = torch.from_numpy(y_test).type(torch.Tensor).to(device)
dataset_train = Data.TensorDataset(X_train, y_train)
dataset_test = Data.TensorDataset(X_test, y_test)
dataloader_train = Data.DataLoader(dataset_train, batch_size = 1024, shuffle = True, drop_last=True)
dataloader_test = Data.DataLoader(dataset_test, batch_size = 1024, shuffle = True, drop_last=True)

"""## Unified train()"""

def get_lambda(epoch, max_epoch):
    p = epoch / max_epoch
    return 2. / (1+np.exp(-10.*p)) - 1.

mod = {'batch_size':1024, 'hidden_dim':64, 'num_layers':1, 'num_epochs':100,
       'patience':20, 'trade_off':1,
       'src': (X_train, y_train) , 'tgt': (X_test, y_test),
       }

def train(mod):
    step = 0
    ll_c, ll_d = [], [] # loss for classifier and discriminator
    rmse_test = []
    # rmse_test_marginal = []
    mape_test = []
    # mape_test_marginal = []

    dataset_src = Data.TensorDataset(mod['src'][0], mod['src'][1])
    dataset_tgt = Data.TensorDataset(mod['tgt'][0], mod['tgt'][1])
    dataloader_src = Data.DataLoader(dataset_src, batch_size = mod['batch_size'], shuffle = True, drop_last=True)
    dataloader_tgt = Data.DataLoader(dataset_tgt, batch_size = mod['batch_size'], shuffle = True, drop_last=True)
    iter_tgt = iter(dataloader_tgt)
    log_interval = len(dataloader_tgt)

    D_src = torch.ones(mod['batch_size'], 1).to(device) # Discriminator Label to real
    D_tgt = torch.zeros(mod['batch_size'], 1).to(device) # Discriminator Label to fake
    D_labels = torch.cat([D_src, D_tgt], dim=0)

    input_dim = mod['src'][0].shape[-1]
    hidden_dim = mod['hidden_dim']
    output_dim = mod['src'][1].shape[-1]

    F = FeatureExtractor(input_dim=input_dim, hidden_dim=mod['hidden_dim'], num_layers=mod['num_layers']).to(device)
    C = Classifier(input_dim=mod['hidden_dim'], output_dim=output_dim).to(device)
    D = Discriminator(input_dim=mod['hidden_dim']).to(device)

    F_opt = torch.optim.Adam(F.parameters(), lr=0.01)
    C_opt = torch.optim.Adam(C.parameters(), lr=0.01)
    D_opt = torch.optim.Adam(D.parameters(), lr=0.01)

    bce = nn.BCELoss()
    xe = nn.MSELoss()

    count = 0
    best_rmse = 1e2
    best_mape = 0
    iterator = tqdm(range(mod['num_epochs']))
    for epoch in iterator:
        for idx, (src_x, src_y) in enumerate(dataloader_src):
            try:
                tgt_x, tgt_y = iter_tgt.next()
            except Exception as err:
                iter_tgt = iter(dataloader_test)
                tgt_x, tgt_y = iter_tgt.next()
            
            # training Discriminator
            x = torch.cat([src_x, tgt_x], dim=0)
            h = F(x)
            y = D(h.detach())

            Ld = bce(y, D_labels)
            D.zero_grad()
            Ld.backward()
            D_opt.step()

            c = C(h[:mod['batch_size']])
            y = D(h)
            Lc = torch.sqrt(xe(c, src_y))
            Ld = bce(y, D_labels)
            lamda = mod['trade_off']*get_lambda(epoch, mod['num_epochs'])
            Ltot = Lc - lamda*Ld

            F.zero_grad()
            C.zero_grad()
            D.zero_grad()
            
            Ltot.backward()
            
            C_opt.step()
            F_opt.step()
            step += 1
            if step%log_interval == 0:
                ll_c.append(Lc)
                ll_d.append(Ld)
                F.eval()
                C.eval()
                y_test_pred = C(F(mod['tgt'][0]))
                y_test_pred_true = y_scaler_test.inverse_transform(y_test_pred.cpu().detach().numpy())
                y_test_true = y_scaler_test.inverse_transform(y_test.cpu().detach().numpy())
                cur_test_rmse = torch.sqrt(xe(y_test_pred, mod['tgt'][1])).item()
                cur_test_rmse_marginal = torch.sqrt(torch.mean((y_test_pred-mod['tgt'][1])**2, axis=0)).detach().numpy()
                cur_test_mape = np.mean(np.abs((y_test_pred_true - y_test_true) / y_test_true))
                cur_test_mape_marginal = np.mean(np.abs((y_test_pred_true - y_test_true) / y_test_true), axis=0)
                rmse_test.append(cur_test_rmse)
                mape_test.append(cur_test_mape)
                # rmse_test_marginal.append(cur_test_rmse_marginal)
                # mape_test_marginal.append(cur_test_mape_marginal)
                F.train()
                C.train()
                print('Epoch: {}/{}, Step: {}, D Loss: {:.4f}, C Loss: {:.4f}, Test RMSE: {:.4f}, Test RE: {:.4f}, lambda: {:.4f}'.format(epoch, mod['num_epochs'], step, Ld.item(), Lc.item(), cur_test_rmse, cur_test_mape, lamda))

                if cur_test_rmse < best_rmse:
                    best_rmse = cur_test_rmse
                    best_mape = cur_test_mape
                    best_rmse_marginal = cur_test_rmse_marginal
                    best_mape_marginal = cur_test_mape_marginal
                    count = 0
                else:
                    count += 1
        if (count >= mod['patience']) or (count < mod['patience'] and epoch == mod['num_epochs']-1):
            plt.plot(range(len(ll_c)), ll_c, label='Prediction Error')
            plt.plot(range(len(ll_d)), ll_d, label='Discriminator Error')
            plt.legend()
            plt.show()

            plt.plot(range(len(rmse_test)), rmse_test, label='Test RMSE')
            plt.plot(range(len(mape_test)), mape_test, label='Test MAPE')
            plt.legend()
            plt.show()

            print('Training ends at {}-th iteration with best test RMSE {} and MAPE {}.'.format(step, best_rmse, best_mape))
            iterator.close()
            break
    return [best_rmse, best_mape, best_rmse_marginal, best_mape_marginal], [ll_c, ll_d, rmse_test, mape_test]

r, d = train(mod)

