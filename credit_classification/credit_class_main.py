import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import MinMaxScaler
from multiprocessing import Process, Queue
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
def MLP(learning_rate_MLP, num_experiments, src_x, src_y, tgt_x, tgt_y):
    if not os.path.isdir('./MLP_xy_list'):
        os.mkdir('./MLP_xy_list')
    # MLP
    mod = {
        'model': 'MLP', 'input': 69, 'hidden': 128, 'output': 2, 'batch_size': 1024, 'patience': 5,
        'src': (src_x, src_y), 'tgt': (tgt_x, tgt_y),
        'iteration': 10000,
        'trade_off1': 1, 'trade_off2': 1
    }
    start = time.time()
    res = {}
    acc_list = []
    for i in tqdm(range(num_experiments)):
        rslt, list_xf, list_yf, log_interval = train(mod, learning_rate_MLP, src_x, src_y, tgt_x, tgt_y)
        joblib.dump(list_xf, './MLP_xy_list/list_xf_' + str(i) + '_interval='+str(log_interval))
        joblib.dump(list_yf, './MLP_xy_list/list_yf_' + str(i) + '_interval='+str(log_interval))
        res[i] = rslt
        acc_list.append(rslt['roc'])
        print("The {}-th experiment ends with best accuaracy {}".format(i + 1, rslt['roc'][0]))
    end = time.time()
    print("Total training costs {} seconds".format(end - start))
    acc_mean = np.mean(acc_list)
    acc_std = np.std(acc_list)
    df = pd.DataFrame(res)
    return df, acc_mean, acc_std
    # with open('file.pkl', 'wb') as handle:
    #     pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
def AFN(learning_rate_MLP, num_experiments, src_x, src_y, tgt_x, tgt_y):
    if not os.path.isdir('./AFN_xy_list'):
        os.mkdir('./AFN_xy_list')
    # AFN
    mod = {
        'model': 'AFN', 'input': 69, 'hidden': 128, 'output': 2, 'batch_size': 1024, 'patience': 5,
        'src': (src_x, src_y), 'tgt': (tgt_x, tgt_y),
        'iteration': 10000, 'weight': 0.1, 'radius': 25,
        'trade_off1': 1, 'trade_off2': 1
    }
    start = time.time()
    res = {}
    acc_list = []
    for i in tqdm(range(num_experiments)):
       rslt, list_xf, list_yf, log_interval = train(mod, learning_rate_DAN, src_x, src_y, tgt_x, tgt_y)
       joblib.dump(list_xf, './AFN_xy_list/list_xf_' + str(i) + '_interval='+str(log_interval))
       joblib.dump(list_yf, './AFN_xy_list/list_yf_' + str(i) + '_interval='+str(log_interval))
       res[i] = rslt
       acc_list.append(rslt['roc'])
       print("The {}-th experiment ends with best accuaracy {}".format(i+1, rslt['roc'][0]))
    end = time.time()
    print("Total training costs {} seconds".format(end - start))
    acc_mean = np.mean(acc_list)
    acc_std = np.std(acc_list)
    df = pd.DataFrame(res)
    return df, acc_mean, acc_std

def MCD(learning_rate_MLP, num_experiments, src_x, src_y, tgt_x, tgt_y):
    if not os.path.isdir('./MCD_xy_list'):
        os.mkdir('./MCD_xy_list')
    mod = {
        'model': 'MCD', 'input': 69, 'hidden': 128, 'output': 2, 'batch_size': 1024, 'patience': 5,
        'src': (src_x, src_y), 'tgt': (tgt_x, tgt_y),
        'iteration': 10000,
        'trade_off1': 1, 'trade_off2': 1
    }
    start = time.time()
    res = {}
    acc_list = []
    for i in tqdm(range(num_experiments)):
       rslt, list_xf, list_yf, log_interval = train(mod, learning_rate_DAN, src_x, src_y, tgt_x, tgt_y)
       joblib.dump(list_xf, './MCD_xy_list/list_xf_' + str(i) + '_interval='+str(log_interval))
       joblib.dump(list_yf, './MCD_xy_list/list_yf_' + str(i) + '_interval='+str(log_interval))
       res[i] = rslt
       acc_list.append(rslt['roc'])
       print("The {}-th experiment ends with best accuaracy {}".format(i+1, rslt['roc'][0]))
    end = time.time()
    print("Total training costs {} seconds".format(end - start))
    acc_mean = np.mean(acc_list)
    acc_std = np.std(acc_list)
    df = pd.DataFrame(res)
    return df, acc_mean, acc_std

def DANN(learning_rate_DAN, num_experiments, src_x, src_y, tgt_x, tgt_y):
    if not os.path.isdir('./DANN_xy_list'):
        os.mkdir('./DANN_xy_list')
    mod = {
        'model': 'DANN', 'input': 69, 'hidden': 128, 'output': 2, 'batch_size': 1024, 'patience': 5,
        'src': (src_x, src_y), 'tgt': (tgt_x, tgt_y),
        'iteration': 10000,
        'trade_off1': 0.1, 'trade_off2': 1
    }
    start = time.time()
    res = {}
    acc_list = []
    for i in tqdm(range(num_experiments)):
       rslt, list_xf, list_yf, log_interval = train(mod, learning_rate_DAN, src_x, src_y, tgt_x, tgt_y)
       joblib.dump(list_xf, './DANN_xy_list/list_xf_' + str(i) + '_interval='+str(log_interval))
       joblib.dump(list_yf, './DANN_xy_list/list_yf_' + str(i) + '_interval='+str(log_interval))
       res[i] = rslt
       acc_list.append(rslt['roc'])
       print("The {}-th experiment ends with best accuaracy {}".format(i+1, rslt['roc'][0]))
    end = time.time()
    print("Total training costs {} seconds".format(end - start))
    acc_mean = np.mean(acc_list)
    acc_std = np.std(acc_list)
    df = pd.DataFrame(res)
    return df, acc_mean, acc_std

def DAN(learning_rate_DAN, num_experiments, src_x, src_y, tgt_x, tgt_y):
    if not os.path.isdir('./DAN_xy_list'):
        os.mkdir('./DAN_xy_list')
    # DAN
    mod_1 = {
       'model': 'DAN', 'input': 69, 'hidden': 64, 'output': 2, 'batch_size':1024, 'patience':5,
       'src':(src_x,src_y), 'tgt':(tgt_x,tgt_y),
       'iteration':10000,
       'trade_off1':1, 'trade_off2':1
    }
    start = time.time()
    res = {}
    acc_list = []
    for i in tqdm(range(num_experiments)):
       rslt, list_xf, list_yf, log_interval = train(mod_1, learning_rate_DAN, src_x, src_y, tgt_x, tgt_y)
       joblib.dump(list_xf, './DAN_xy_list/list_xf_' + str(i) + '_interval='+str(log_interval))
       joblib.dump(list_yf, './DAN_xy_list/list_yf_' + str(i) + '_interval='+str(log_interval))
       res[i] = rslt
       acc_list.append(rslt['roc'])
       print("The {}-th experiment ends with best accuaracy {}".format(i+1, rslt['roc'][0]))
    end = time.time()
    print("Total training costs {} seconds".format(end - start))
    acc_mean = np.mean(acc_list)
    acc_std = np.std(acc_list)
    df = pd.DataFrame(res)
    return df, acc_mean, acc_std
    # with open('file.pkl','wb') as handle:
    #    pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
def CORAL(learning_rate_CORAL, num_experiments, src_x, src_y, tgt_x, tgt_y):
    if not os.path.isdir('./CORAL_xy_list'):
        os.mkdir('./CORAL_xy_list')
    # CORAL
    mod_2 = {
       'model': 'CORAL', 'input': 69, 'hidden': 64, 'output': 2, 'batch_size':1024, 'patience':5,
       'src':(src_x,src_y), 'tgt':(tgt_x,tgt_y),
       'iteration':10000,
       'trade_off1':0.1, 'trade_off2':1
    }
    start = time.time()
    res = {}
    acc_list = []
    for i in tqdm(range(num_experiments)):
       rslt, list_xf, list_yf, log_interval = train(mod_2, learning_rate_CORAL, src_x, src_y, tgt_x, tgt_y)
       joblib.dump(list_xf, './CORAL_xy_list/list_xf_' + str(i) + '_interval='+str(log_interval))
       joblib.dump(list_yf, './CORAL_xy_list/list_yf_' + str(i) + '_interval='+str(log_interval))
       res[i] = rslt
       acc_list.append(rslt['roc'])
       print("The {}-th experiment ends with best accuaracy {}".format(i+1, rslt['roc'][0]))
    end = time.time()
    print("Total training costs {} seconds".format(end - start))
    acc_mean = np.mean(acc_list)
    acc_std = np.std(acc_list)
    df = pd.DataFrame(res)
    return df, acc_mean, acc_std
def CDAN(learning_rate_CDAN, num_experiments, src_x, src_y, tgt_x, tgt_y):
    if not os.path.isdir('./CDAN_xy_list'):
        os.mkdir('./CDAN_xy_list')
    # CDAN
    mod_3 = {
       'model': 'CDAN', 'input': 69, 'hidden': 128, 'output': 2, 'batch_size':1024, 'patience':5,
       'src':(src_x,src_y), 'tgt':(tgt_x,tgt_y),
       'iteration':10000,
       'trade_off1':1, 'trade_off2': 0.01
    }
    start = time.time()
    res={}
    acc_list = []
    for i in tqdm(range(num_experiments)):
       rslt, list_xf, list_yf, log_interval = train(mod_3, learning_rate_CDAN, src_x, src_y, tgt_x, tgt_y)
       joblib.dump(list_xf, './CDAN_xy_list/list_xf_' + str(i) + '_interval='+str(log_interval))
       joblib.dump(list_yf, './CDAN_xy_list/list_yf_' + str(i) + '_interval='+str(log_interval))
       res[i] = rslt
       acc_list.append(rslt['roc'])
       print("The {}-th experiment ends with best accuaracy {}".format(i+1, rslt['roc'][0]))
    end = time.time()
    print("Total training costs {} seconds".format(end - start))
    acc_mean = np.mean(acc_list)
    acc_std = np.std(acc_list)
    df = pd.DataFrame(res)
    return df, acc_mean, acc_std
if __name__ == '__main__':
    source_file = 'M1905.csv'
    tgt_file = 'M1906.csv'
    src_x, src_y = feature_load(filename=source_file)  # source
    tgt_x, tgt_y = feature_load(filename=tgt_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_experiments = 10
    learning_rate_MLP = 0.01
    learning_rate_DAN = 0.01
    learning_rate_CORAL = 0.01
    learning_rate_CDAN = 0.01
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20
    # print('MLP start')
    # df_1, acc_mean_MLP, acc_std_MLP = MLP(learning_rate_MLP, num_experiments, src_x, src_y, tgt_x, tgt_y)
    # print('AFN start')
    # df_5, acc_mean_AFN, acc_std_AFN = AFN(learning_rate_MLP, num_experiments, src_x, src_y, tgt_x, tgt_y)
    print('MCD start')
    df_6, acc_mean_MCD, acc_std_MCD = MCD(learning_rate_MLP, num_experiments, src_x, src_y, tgt_x, tgt_y)
    # print('DANN start')
    # df_7, acc_mean_DANN, acc_std_DANN = DANN(learning_rate_MLP, num_experiments, src_x, src_y, tgt_x, tgt_y)
    # print('DAN start')
    # df_2, acc_mean_DAN, acc_std_DAN = DAN(learning_rate_DAN, num_experiments, src_x, src_y, tgt_x, tgt_y)
    # print('CORAL start')
    # df_3, acc_mean_CORAL, acc_std_CORAL = CORAL(learning_rate_CORAL, num_experiments, src_x, src_y, tgt_x, tgt_y)
    # print('CDAN start')
    # df_4, acc_mean_CDAN, acc_std_CDAN = CDAN(learning_rate_CDAN, num_experiments, src_x, src_y, tgt_x, tgt_y)
    # newdf = pd.concat((df_1, df_2, df_3, df_4))
    # newdf.to_csv(source_file+'_'+tgt_file+'_detail.csv')
    df_result = pd.DataFrame(columns=['mean', 'std'], index=['MLP', 'AFN', 'MCD', 'DANN', 'DAN', 'CORAL', 'CDAN'])
    # df_result.loc['MLP', 'mean'], df_result.loc['MLP', 'std'] = acc_mean_MLP, acc_std_MLP
    # df_result.loc['AFN', 'mean'], df_result.loc['AFN', 'std'] = acc_mean_AFN, acc_std_AFN
    df_result.loc['MCD', 'mean'], df_result.loc['MCD', 'std'] = acc_mean_MCD, acc_std_MCD
    # df_result.loc['DANN', 'mean'], df_result.loc['DANN', 'std'] = acc_mean_DANN, acc_std_DANN
    # df_result.loc['DAN', 'mean'], df_result.loc['DAN', 'std'] = acc_mean_DAN, acc_std_DAN
    # df_result.loc['CORAL', 'mean'], df_result.loc['CORAL', 'std'] = acc_mean_CORAL, acc_std_CORAL
    # df_result.loc['CDAN', 'mean'], df_result.loc['CDAN', 'std'] = acc_mean_CDAN, acc_std_CDAN
    df_result.to_csv(source_file+'_'+tgt_file+'.csv')
