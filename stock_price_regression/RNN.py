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
import seaborn as sns
import multiprocessing as mp
from tqdm import tqdm
from google.colab import files
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
from sklearn.metrics import mean_squared_error

from distance import *

with open('source_price_5min.pkl', 'rb') as handle:
    data_source = pickle.load(handle)
with open('target_price_5min.pkl', 'rb') as handle:
    data_target = pickle.load(handle)
# print("The source domain has {} days and the target domain has {} days.".format(len(data_source), len(data_target)))

"""## Preparation data"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def package_data(input,look_back = 12):
    data = []
    for index in range(input.shape[0] - look_back + 1):
        data.append(input[index:index + look_back])
    data = np.array(data)
    return data

def load_data(dict_data, data_type='return'):
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
dataloader_train = Data.DataLoader(dataset_train, batch_size = 250, shuffle = True, drop_last=True)
dataloader_test = Data.DataLoader(dataset_test, batch_size = 250, shuffle = True, drop_last=True)

"""### RNN"""

class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(RNN, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.num_layers = num_layers

        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True, nonlinearity='relu')

        # Readout layer
        self.fc = nn.Sequential(nn.BatchNorm1d(hidden_dim) ,nn.Linear(hidden_dim, hidden_dim // 4),nn.ReLU())
        # self.fc1 = nn.Sequential(nn.BatchNorm1d(hidden_dim // 2) ,nn.Linear(hidden_dim // 2, hidden_dim // 4),nn.ReLU())
        self.fc2 = nn.Sequential(nn.BatchNorm1d(hidden_dim // 4) ,nn.Linear(hidden_dim // 4, output_dim))

    def forward(self, x, hidden_state = None):
        # Initialize hidden state with zeros

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        # out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        out, hidden_state = self.rnn(x, hidden_state)

        # Index hidden state of last time step
        # out.size()
        # out[:, -1, :] --> just want last time step hidden states! 
        out = self.fc(out[:, -1, :]) 
        # out = self.fc1(out)
        out = self.fc2(out)
        # out.size()
        return out, hidden_state

"""### RNN"""

input_dim = X_train.shape[-1]
output_dim = y_train.shape[-1]
hidden_dim = 64
num_layers = 1
num_epochs = 100
num_experiments = 100

train_iter = iter(dataloader_train)
test_iter = iter(dataloader_test)

n_batch =  min(len(dataloader_train), len(dataloader_test))

best_loss_test = 1e2
patience = 20
count = 0

RNN_loss_train = []
RNN_loss_test = []
RNN_mape_train = []
RNN_mape_test = []
RNN_mape_test_marginal = []
RNN_loss_history = []
RNN_mape_history = []

for _ in tqdm(range(num_experiments)):

    best_loss_test, best_loss_train = 1e2, 1e2
    best_mape_test, best_mape_train = 1e2, 1e2

    model = RNN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
    loss_fn = torch.nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
    hstate = None

    count = 0

    hist = np.zeros(num_epochs)
    hist_error = np.zeros(num_epochs)
    hist_mape_train = np.zeros(num_epochs)
    hist_mape_test = np.zeros(num_epochs)

    for t in range(num_epochs):
        # model.hidden = model.init_hidden()
        cur_loss = 0
        cur_error = 0
        cur_mape = 0
        for i in range(n_batch):
            try:
                x_train_iter, y_train_iter = train_iter.next()
            except Exception as err:
                train_iter = iter(dataloader_train)
                x_train_iter, y_train_iter = train_iter.next()
            # try:
            #     x_test_iter, y_test_iter = test_iter.next()
            # except Exception as err:
            #     test_iter = iter(dataloader_test)
            #     x_test_iter, y_test_iter = test_iter.next()
            if hstate != None: hstate = hstate.data
            y_train_pred, hstate = model(x_train_iter, hstate) # update the hidden state each time

            loss = torch.sqrt(loss_fn(y_train_pred, y_train_iter))
            # loss = loss_fn(y_train_pred, y_train_iter)
            cur_loss += loss.item()
            y_train_iter_true = y_scaler_train.inverse_transform(y_train_iter.detach().numpy())
            y_train_pred_true = y_scaler_train.inverse_transform(y_train_pred.detach().numpy())
            cur_mape += np.mean(np.abs((y_train_iter_true - y_train_pred_true) / y_train_iter_true))
            if t % 10 == 0 and i % 50 == 0:
                print("Epoch ", t, "Batch ", i, "RMSE: ", loss.item())
            # cur_mape += np.median((torch.abs((y_train_iter - y_train_pred) / y_train_iter)).detach().numpy())
            # hstate = hstate.detach()
            # Zero out gradient, else they will accumulate between epochs
            optimiser.zero_grad()

            # Backward pass
            loss.backward()

            # Update parameters
            optimiser.step()
        model.eval()
        y_test_pred, _ = model(X_test, None)
        y_test_pred_true = y_scaler_test.inverse_transform(y_test_pred.detach().numpy())
        y_test_true = y_scaler_test.inverse_transform(y_test.detach().numpy())
        # cur_error = loss_fn(y_test_pred, y_test)
        cur_error = torch.sqrt(loss_fn(y_test_pred, y_test))
        cur_mape_test = np.mean(np.abs((y_test_pred_true - y_test_true) / y_test_true))
        cur_mape_test_marginal = np.mean(np.abs((y_test_pred_true - y_test_true) / y_test_true), axis=0)
        model.train()
        if t % 10 == 0:
            print("Epoch ", t, "Batch ", i, "MAPE: ", cur_mape_test)

        hist[t] = cur_loss / n_batch
        hist_error[t] = cur_error
        hist_mape_train[t] = cur_mape / n_batch
        hist_mape_test[t] = cur_mape_test

        if cur_error < best_loss_test:
            count = 0
            best_loss_test = cur_error.item()
            best_loss_train = cur_loss / n_batch
            best_mape_test = cur_mape_test
            best_mape_train = cur_mape / n_batch
            best_mape_test_marginal = cur_mape_test_marginal
        else:
            count += 1
        if count >= patience or (count < patience and t == num_epochs - 1):
            RNN_loss_train.append(best_loss_train)
            RNN_loss_test.append(best_loss_test)
            RNN_mape_train.append(best_mape_train)
            RNN_mape_test.append(best_mape_test)
            RNN_mape_test_marginal.append(best_mape_test_marginal)
            RNN_loss_history.append(hist_error)
            RNN_mape_history.append(hist_mape_test)

            print('Early stop at {}-th training with best test RMSE {}'.format(t, best_loss_test))
            plt.plot(hist[hist > 0], label="Train Loss")
            plt.plot(hist_error[hist_error > 0], label="Test Loss")
            plt.legend()
            plt.show()

            plt.plot(hist_mape_train[hist_mape_train > 0], label="Train MAPE")
            plt.plot(hist_mape_test[hist_mape_test > 0], label="Test MAPE")
            plt.legend()
            plt.show()
            break

dic = {'rmse':np.array(RNN_loss_test),
       'mape':np.array(RNN_mape_test),
       'mape_marginal': np.array(RNN_mape_test_marginal),
       'rmse_history':np.array(RNN_loss_history),
       'mape_history':np.array(RNN_mape_history)}
with open('rnn.pkl', 'wb') as handle:
    pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
