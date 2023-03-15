import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# from multiprocessing import Process, Queue
from scipy import stats
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
from distance import *
from tqdm import tqdm
from domain_models import *
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, roc_auc_score, r2_score
import pickle
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def MLP(learning_rate_MLP, num_experiments, src_x, src_y, tgt_x, tgt_y, scaler=False):
    if not os.path.isdir('./MLP_xy_list'):
        os.mkdir('./MLP_xy_list')
    # MLP
    mod = { 
        'model': 'MLP', 'input': 11, 'hidden': 8, 'output': 1, 'batch_size': 128, 'patience': 10,
        'src': (src_x, src_y), 'tgt': (tgt_x, tgt_y),
        'iteration': 10000,
        'trade_off1': 1, 'trade_off2': 1
    }
    start = time.time()
    res = {}
    acc_list = []
    re_list = []
    for i in tqdm(range(num_experiments)):
        rslt, list_xf, list_yf, log_interval = train(mod, learning_rate_MLP, src_x, src_y, tgt_x, tgt_y, scaler=scaler)
        joblib.dump(list_xf, './MLP_xy_list/list_xf_' + str(i) + '_interval='+str(log_interval))
        joblib.dump(list_yf, './MLP_xy_list/list_yf_' + str(i) + '_interval='+str(log_interval))
        res[i] = rslt
        acc_list.append(rslt['roc'])
        if scaler: re_list.append(rslt['re'])
        print("The {}-th experiment ends with best accuaracy {}".format(i + 1, rslt['roc'][0]))
    end = time.time()
    print("Total training costs {} seconds".format(end - start))
    acc_mean = np.mean(acc_list)
    acc_std = np.std(acc_list)
    if scaler:
        re_mean = np.mean(re_list)
        re_std = np.std(re_list)
    df = pd.DataFrame(res)
    if not scaler:
        return df, acc_mean, acc_std
    else:
        return df, acc_mean, acc_std, re_mean, re_std
    # with open('file.pkl', 'wb') as handle:
    #     pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)

def AFN(learning_rate_MLP, num_experiments, src_x, src_y, tgt_x, tgt_y, scaler=False):
    if not os.path.isdir('./AFN_xy_list'):
        os.mkdir('./AFN_xy_list')
    # AFN
    mod = {
        'model': 'AFN', 'input': 11, 'hidden': 8, 'output': 1, 'batch_size': 128, 'patience': 10,
        'src': (src_x, src_y), 'tgt': (tgt_x, tgt_y),
        'iteration': 10000, 'weight': 1, 'radius': 0.1,
        'trade_off1': 10, 'trade_off2': 1
    }
    start = time.time()
    res = {}
    acc_list = []
    re_list = []
    for i in tqdm(range(num_experiments)):
       rslt, list_xf, list_yf, log_interval = train(mod, learning_rate_DAN, src_x, src_y, tgt_x, tgt_y, scaler=scaler)
       joblib.dump(list_xf, './AFN_xy_list/list_xf_' + str(i) + '_interval='+str(log_interval))
       joblib.dump(list_yf, './AFN_xy_list/list_yf_' + str(i) + '_interval='+str(log_interval))
       res[i] = rslt
       acc_list.append(rslt['roc'])
       if scaler: re_list.append(rslt['re'])
       print("The {}-th experiment ends with best accuaracy {}".format(i+1, rslt['roc'][0]))
    end = time.time()
    print("Total training costs {} seconds".format(end - start))
    acc_mean = np.mean(acc_list)
    acc_std = np.std(acc_list)
    if scaler:
        re_mean = np.mean(re_list)
        re_std = np.std(re_list)
    df = pd.DataFrame(res)
    if not scaler:
        return df, acc_mean, acc_std
    else:
        return df, acc_mean, acc_std, re_mean, re_std

def MCD(learning_rate_MLP, num_experiments, src_x, src_y, tgt_x, tgt_y, scaler=False):
    if not os.path.isdir('./MCD_xy_list'):
        os.mkdir('./MCD_xy_list')
    mod = {
        'model': 'MCD', 'input': 11, 'hidden': 8, 'output': 1, 'batch_size': 128, 'patience': 10,
        'src': (src_x, src_y), 'tgt': (tgt_x, tgt_y),
        'iteration': 10000,
        'trade_off1': 1, 'trade_off2': 1
    }
    start = time.time()
    res = {}
    acc_list = []
    re_list = []
    for i in tqdm(range(num_experiments)):
       rslt, list_xf, list_yf, log_interval = train(mod, learning_rate_DAN, src_x, src_y, tgt_x, tgt_y, scaler=scaler)
       joblib.dump(list_xf, './MCD_xy_list/list_xf_' + str(i) + '_interval='+str(log_interval))
       joblib.dump(list_yf, './MCD_xy_list/list_yf_' + str(i) + '_interval='+str(log_interval))
       res[i] = rslt
       acc_list.append(rslt['roc'])
       if scaler: re_list.append(rslt['re'])
       print("The {}-th experiment ends with best accuaracy {}".format(i+1, rslt['roc'][0]))
    end = time.time()
    print("Total training costs {} seconds".format(end - start))
    acc_mean = np.mean(acc_list)
    acc_std = np.std(acc_list)
    if scaler:
        re_mean = np.mean(re_list)
        re_std = np.std(re_list)
    df = pd.DataFrame(res)
    if not scaler:
        return df, acc_mean, acc_std
    else:
        return df, acc_mean, acc_std, re_mean, re_std

def DANN(learning_rate_DAN, num_experiments, src_x, src_y, tgt_x, tgt_y, scaler=False):
    if not os.path.isdir('./DANN_xy_list'):
        os.mkdir('./DANN_xy_list')
    mod = {
        'model': 'DANN', 'input': 11, 'hidden': 8, 'output': 1, 'batch_size': 128, 'patience': 10,
        'src': (src_x, src_y), 'tgt': (tgt_x, tgt_y),
        'iteration': 10000,
        'trade_off1': 0.01, 'trade_off2': 1
    }
    start = time.time()
    res = {}
    acc_list = []
    re_list = []
    for i in tqdm(range(num_experiments)):
       rslt, list_xf, list_yf, log_interval = train(mod, learning_rate_DAN, src_x, src_y, tgt_x, tgt_y, scaler=scaler)
       joblib.dump(list_xf, './DANN_xy_list/list_xf_' + str(i) + '_interval='+str(log_interval))
       joblib.dump(list_yf, './DANN_xy_list/list_yf_' + str(i) + '_interval='+str(log_interval))
       res[i] = rslt
       acc_list.append(rslt['roc'])
       if scaler: re_list.append(rslt['re'])
       print("The {}-th experiment ends with best accuaracy {}".format(i+1, rslt['roc'][0]))
    end = time.time()
    print("Total training costs {} seconds".format(end - start))
    acc_mean = np.mean(acc_list)
    acc_std = np.std(acc_list)
    if scaler:
        re_mean = np.mean(re_list)
        re_std = np.std(re_list)
    df = pd.DataFrame(res)
    if not scaler:
        return df, acc_mean, acc_std
    else:
        return df, acc_mean, acc_std, re_mean, re_std

def DAN(learning_rate_DAN, num_experiments, src_x, src_y, tgt_x, tgt_y, scaler=False):
    if not os.path.isdir('./DAN_xy_list'):
        os.mkdir('./DAN_xy_list')
    # DAN
    mod_1 = {
       'model': 'DAN', 'input': 11, 'hidden': 8, 'output': 1, 'batch_size':128, 'patience':10,
       'src':(src_x,src_y), 'tgt':(tgt_x,tgt_y),
       'iteration':10000,
       'trade_off1':0.1, 'trade_off2':1
    }
    start = time.time()
    res = {}
    acc_list = []
    re_list = []
    for i in tqdm(range(num_experiments)):
       rslt, list_xf, list_yf, log_interval = train(mod_1, learning_rate_DAN, src_x, src_y, tgt_x, tgt_y, scaler=scaler)
       joblib.dump(list_xf, './DAN_xy_list/list_xf_' + str(i) + '_interval='+str(log_interval))
       joblib.dump(list_yf, './DAN_xy_list/list_yf_' + str(i) + '_interval='+str(log_interval))
       res[i] = rslt
       acc_list.append(rslt['roc'])
       if scaler: re_list.append(rslt['re'])
       print("The {}-th experiment ends with best accuaracy {}".format(i+1, rslt['roc'][0]))
    end = time.time()
    print("Total training costs {} seconds".format(end - start))
    acc_mean = np.mean(acc_list)
    acc_std = np.std(acc_list)
    if scaler:
        re_mean = np.mean(re_list)
        re_std = np.std(re_list)
    df = pd.DataFrame(res)
    if not scaler:
        return df, acc_mean, acc_std
    else:
        return df, acc_mean, acc_std, re_mean, re_std
    # with open('file.pkl','wb') as handle:
    #    pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
def CORAL(learning_rate_CORAL, num_experiments, src_x, src_y, tgt_x, tgt_y, scaler=False):
    if not os.path.isdir('./CORAL_xy_list'):
        os.mkdir('./CORAL_xy_list')
    # CORAL
    mod_2 = {
       'model': 'CORAL', 'input': 11, 'hidden': 8, 'output': 1, 'batch_size':128, 'patience':10,
       'src':(src_x,src_y), 'tgt':(tgt_x,tgt_y),
       'iteration':10000,
       'trade_off1':1, 'trade_off2':1
    }
    start = time.time()
    res = {}
    acc_list = []
    re_list = []
    for i in tqdm(range(num_experiments)):
       rslt, list_xf, list_yf, log_interval = train(mod_2, learning_rate_CORAL, src_x, src_y, tgt_x, tgt_y, scaler=scaler)
       joblib.dump(list_xf, './CORAL_xy_list/list_xf_' + str(i) + '_interval='+str(log_interval))
       joblib.dump(list_yf, './CORAL_xy_list/list_yf_' + str(i) + '_interval='+str(log_interval))
       res[i] = rslt
       acc_list.append(rslt['roc'])
       if scaler: re_list.append(rslt['re'])
       print("The {}-th experiment ends with best accuaracy {}".format(i+1, rslt['roc'][0]))
    end = time.time()
    print("Total training costs {} seconds".format(end - start))
    acc_mean = np.mean(acc_list)
    acc_std = np.std(acc_list)
    if scaler:
        re_mean = np.mean(re_list)
        re_std = np.std(re_list)
    df = pd.DataFrame(res)
    if not scaler:
        return df, acc_mean, acc_std
    else:
        return df, acc_mean, acc_std, re_mean, re_std
def CDAN(learning_rate_CDAN, num_experiments, src_x, src_y, tgt_x, tgt_y, scaler=False):
    if not os.path.isdir('./CDAN_xy_list'):
        os.mkdir('./CDAN_xy_list')
    # CDAN
    mod_3 = {
       'model': 'CDAN', 'input': 11, 'hidden': 8, 'output': 1, 'batch_size':128, 'patience':10,
       'src':(src_x,src_y), 'tgt':(tgt_x,tgt_y),
       'iteration':10000,
       'trade_off1':1, 'trade_off2': 1
    }
    start = time.time()
    res={}
    acc_list = []
    re_list = []
    for i in tqdm(range(num_experiments)):
       rslt, list_xf, list_yf, log_interval = train(mod_3, learning_rate_CDAN, src_x, src_y, tgt_x, tgt_y, scaler=scaler)
       joblib.dump(list_xf, './CDAN_xy_list/list_xf_' + str(i) + '_interval='+str(log_interval))
       joblib.dump(list_yf, './CDAN_xy_list/list_yf_' + str(i) + '_interval='+str(log_interval))
       res[i] = rslt
       acc_list.append(rslt['roc'])
       if scaler: re_list.append(rslt['re'])
       print("The {}-th experiment ends with best accuaracy {}".format(i+1, rslt['roc'][0]))
    end = time.time()
    print("Total training costs {} seconds".format(end - start))
    acc_mean = np.mean(acc_list)
    acc_std = np.std(acc_list)
    if scaler:
        re_mean = np.mean(re_list)
        re_std = np.std(re_list)
    df = pd.DataFrame(res)
    if not scaler:
        return df, acc_mean, acc_std
    else:
        return df, acc_mean, acc_std, re_mean, re_std
if __name__ == '__main__':
    src_x, src_y, tgt_x, tgt_y, scaler_y = feature_load(scaler=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_experiments = 100
    learning_rate_MLP = 0.01
    learning_rate_DAN = 0.01
    learning_rate_CORAL = 0.01
    learning_rate_CDAN = 0.01
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20
    start = time.time()
    print('MLP start')
    df_1, acc_mean_MLP, acc_std_MLP, re_mean_MLP, re_std_MLP = MLP(learning_rate_MLP, num_experiments, src_x, src_y, tgt_x, tgt_y, scaler=scaler_y)
    print('AFN start')
    df_5, acc_mean_AFN, acc_std_AFN, re_mean_AFN, re_std_AFN = AFN(learning_rate_MLP, num_experiments, src_x, src_y, tgt_x, tgt_y, scaler=scaler_y)
    print('MCD start')
    df_6, acc_mean_MCD, acc_std_MCD, re_mean_MCD, re_std_MCD = MCD(learning_rate_MLP, num_experiments, src_x, src_y, tgt_x, tgt_y, scaler=scaler_y)
    print('DANN start')
    df_7, acc_mean_DANN, acc_std_DANN, re_mean_DANN, re_std_DANN = DANN(learning_rate_MLP, num_experiments, src_x, src_y, tgt_x, tgt_y, scaler=scaler_y)
    print('DAN start')
    df_2, acc_mean_DAN, acc_std_DAN, re_mean_DAN, re_std_DAN = DAN(learning_rate_DAN, num_experiments, src_x, src_y, tgt_x, tgt_y, scaler=scaler_y)
    print('CORAL start')
    df_3, acc_mean_CORAL, acc_std_CORAL, re_mean_CORAL, re_std_CORAL = CORAL(learning_rate_CORAL, num_experiments, src_x, src_y, tgt_x, tgt_y, scaler=scaler_y)
    print('CDAN start')
    df_4, acc_mean_CDAN, acc_std_CDAN, re_mean_CDAN, re_std_CDAN = CDAN(learning_rate_CDAN, num_experiments, src_x, src_y, tgt_x, tgt_y, scaler=scaler_y)
    newdf = pd.concat((df_1, df_5, df_6, df_7, df_2, df_3, df_4))
    newdf.to_csv('wr_detail.csv')
    end = time.time()
    df_result = pd.DataFrame(columns=['mean', 'std', 're', 're_std'], index=['MLP', 'AFN', 'MCD', 'DANN', 'DAN', 'CORAL', 'CDAN'])
    df_result.loc['MLP', 'mean'], df_result.loc['MLP', 'std'], df_result.loc['MLP','re'], df_result.loc['MLP', 're_std'] = acc_mean_MLP, acc_std_MLP, re_mean_MLP, re_std_MLP
    df_result.loc['AFN', 'mean'], df_result.loc['AFN', 'std'], df_result.loc['AFN','re'], df_result.loc['AFN', 're_std'] = acc_mean_AFN, acc_std_AFN, re_mean_AFN, re_std_AFN
    df_result.loc['MCD', 'mean'], df_result.loc['MCD', 'std'], df_result.loc['MCD','re'], df_result.loc['MCD', 're_std'] = acc_mean_MCD, acc_std_MCD, re_mean_MCD, re_std_MCD
    df_result.loc['DANN', 'mean'], df_result.loc['DANN', 'std'], df_result.loc['DANN','re'], df_result.loc['DANN', 're_std'] = acc_mean_DANN, acc_std_DANN, re_mean_DANN, re_std_DANN
    df_result.loc['DAN', 'mean'], df_result.loc['DAN', 'std'], df_result.loc['DAN','re'], df_result.loc['DAN', 're_std'] = acc_mean_DAN, acc_std_DAN, re_mean_DAN, re_std_DAN
    df_result.loc['CORAL', 'mean'], df_result.loc['CORAL', 'std'], df_result.loc['CORAL','re'], df_result.loc['CORAL', 're_std'] = acc_mean_CORAL, acc_std_CORAL, re_mean_CORAL, re_std_CORAL
    df_result.loc['CDAN', 'mean'], df_result.loc['CDAN', 'std'], df_result.loc['CDAN','re'], df_result.loc['CDAN', 're_std'] = acc_mean_CDAN, acc_std_CDAN, re_mean_CDAN, re_std_CDAN
    df_result.to_csv('/results/white_red.csv')
    print('All experiments cost {:.4} seconds.'.format(end-start))