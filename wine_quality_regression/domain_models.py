import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
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
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, roc_auc_score, r2_score
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20

# def feature_load(filename = 'm1912.csv'):
#     dt_credit = pd.read_csv(filename,index_col = 0)
#     dt_x = dt_credit.iloc[:,2:].fillna(0).to_numpy()
#     dt_y = dt_credit.iloc[:,1].replace(['0+','30+','60+','90+','120+'],0)
#     dt_y = dt_y.replace(['payoff'],1).to_numpy()
#     scaler_input = MinMaxScaler(feature_range=(0, 1))
#     x_trans = scaler_input.fit_transform(dt_x)
#     return x_trans, dt_y

# def feature_load_tgt_unbalanced(filename, frac):
#     dt_credit = pd.read_csv(filename,index_col = 0)
#     dt_x = dt_credit.iloc[:,2:].fillna(0).to_numpy()
#     dt_y = dt_credit.iloc[:,1].replace(['0+','30+','60+','90+','120+'],0)
#     dt_y = dt_y.replace(['payoff'],1).to_numpy()
#     scaler_input = MinMaxScaler(feature_range=(0, 1))
#     x_trans = scaler_input.fit_transform(dt_x)
#     idx_1 = np.where(dt_y==1)[0]
#     idx_0 = np.where(dt_y==0)[0]
#     idx_0_extracted = np.random.choice(len(idx_0), size=int(frac*len(idx_0)), replace=False)
#     combine = np.concatenate((idx_1, idx_0_extracted))
#     return x_trans[combine, :], dt_y[combine]

def feature_load(src_path='/data/winequality-white.csv', tgt_path='/data/winequality-red.csv', scaler=False):
    df_src = pd.read_csv(src_path, sep = ';')
    df_tgt = pd.read_csv(tgt_path, sep = ';')

    src_size = df_src.shape[0]
    df_total = pd.concat([df_src, df_tgt])
    scaler_x, scaler_y = MinMaxScaler(feature_range=(0,1)), MinMaxScaler(feature_range=(0,1))
    scaled_input = scaler_x.fit_transform(df_total.values[:,:-1])
    y = scaler_y.fit_transform(df_total.values[:,-1].reshape(-1,1)).flatten()
    if not scaler:
        return scaled_input[:src_size], y[:src_size], scaled_input[src_size:], y[src_size:]
    else:
        return scaled_input[:src_size], y[:src_size], scaled_input[src_size:], y[src_size:], scaler_y 

class net_MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, drop_rate=0.1):
        super(net_MLP, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.ReLU())

        self.layer2 = nn.Sequential(nn.Linear(hidden_dim,output_dim))
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x
    def forward_ft(self,x,y):
        x_f = self.layer1(x)
        y_f = self.layer1(y)
        return x_f,y_f
    def predict(self, y):
        y = self.layer1(y)
        y = self.layer2(y)
        return y
    
class net_AFN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, weight=0.1, radius=25):
        super(net_AFN, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.ReLU())

        self.layer2 = nn.Sequential(nn.Linear(hidden_dim,output_dim))
        self.weight = weight
        self.radius = radius
    def get_L2norm_loss_self_driven(self, x):
        l = (x.norm(p=2, dim=1).mean() - self.radius) ** 2
        return self.weight * l
    def forward(self, x):
        x = self.layer1(x)
        div = self.get_L2norm_loss_self_driven(x)
        x = self.layer2(x)
        return div, x
    def forward_ft(self,x,y):
        x_f = self.layer1(x)
        y_f = self.layer1(y)
        return x_f,y_f
    def predict(self, y):
        y = self.layer1(y)
        y = self.layer2(y)
        return y

class net_MCD(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(net_MCD, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.ReLU())

        self.layer2 = nn.Sequential(nn.Linear(hidden_dim,output_dim))
        
        self.F1 = ClassifierHead(input_dim = hidden_dim, output_dim = output_dim)
        self.F2 = ClassifierHead(input_dim = hidden_dim, output_dim = output_dim)
    
    def forward(self, x):
        mediate = self.layer1(x)
        y = self.layer2(mediate)
        return mediate, y
    
    def predict(self, y):
        y = self.layer1(y)
        y = self.layer2(y)
        return y
    
    def forward_ft(self, x, y):
        x_f = self.layer1(x)
        y_f = self.layer1(y)
        return x_f,y_f

class net_DANN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(net_DANN, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.ReLU())

        self.layer2 = nn.Sequential(nn.Linear(hidden_dim,output_dim))
        self.classifier = DomainClassifier(input_dim = hidden_dim, output_dim = 1)
    
    def forward(self, x):
        mediate = self.layer1(x)
        y = self.layer2(mediate)
        return mediate, y
    
    def predict(self, y):
        y = self.layer1(y)
        y = self.layer2(y)
        return y
    
    def forward_ft(self, x, y):
        x_f = self.layer1(x)
        y_f = self.layer1(y)
        return x_f,y_f

class net_DAN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, drop_rate=0.1, method='DAN'):
        super(net_DAN, self).__init__()
        self.method = method
        self.layer1 = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(hidden_dim, output_dim))

    def forward(self, x, y):
        x,y = self.layer1(x), self.layer1(y)
        if self.method == 'DAN':
            total_div = MMD(x, y)
        elif self.method == 'CORAL':
            total_div = self.CORAL(x,y)
        x = self.layer2(x)
        return total_div, x
    def predict(self, y):
        y = self.layer1(y)
        y = self.layer2(y)
        return y
    def CORAL(self, source, target):
        d = source.data.shape[1]
        ns, nt = source.data.shape[0], target.data.shape[0]
        # source covariance
        xm = torch.mean(source, 0, keepdim=True) - source
        xc = xm.t() @ xm / (ns - 1)
        # target covariance
        xmt = torch.mean(target, 0, keepdim=True) - target
        xct = xmt.t() @ xmt / (nt - 1)
        # frobenius norm between source and target
        loss = torch.mul((xc - xct), (xc - xct))
        loss = torch.sum(loss) / (4*d*d)
        return loss
    def forward_ft(self,x,y):
        x_f = self.layer1(x)
        y_f = self.layer1(y)
        return x_f,y_f
class net_CDAN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, drop_rate=0.1):
        super(net_CDAN, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(hidden_dim, output_dim))


    def forward(self, x, y):
        x, y = self.layer1(x), self.layer1(y)
        marginal_div = self.marginal_div(x, y)
        copula_distance = self.copula_distance(x, y)
        x = self.layer2(x)
        return marginal_div, copula_distance, x

    def predict(self, y):
        y = self.layer1(y)
        y = self.layer2(y)
        return y

    def marginal_div(self, X, Y, loss_metric='MMD'):
        if loss_metric == 'MMD':
            marginal_loss = MD_MMD(X, Y)
        else:
            marginal_loss = 0
        return marginal_loss

    def copula_distance(self, X, Y, loss_metric='KL'):
        if loss_metric == 'Frobenius':
            copula_loss = CD_Frobenius(X, Y)
        elif loss_metric == 'KL':
            copula_loss = CD_KL(X, Y)
        return copula_loss
    def forward_ft(self,x,y):
        x_f = self.layer1(x)
        y_f = self.layer1(y)
        return x_f,y_f
    
def train(mod, learning_rate, src_x, src_y, tgt_x, tgt_y, scaler = None):
    if mod['model']=='MLP':
        model = net_MLP(input_dim = mod['input'], hidden_dim = mod['hidden'], output_dim = mod['output']).to(device)
    elif mod['model'] == 'AFN': 
        model = net_AFN(input_dim = mod['input'], hidden_dim = mod['hidden'], output_dim = mod['output'], weight
                        = mod['weight'], radius = mod['radius']).to(device)
    elif mod['model'] == 'MCD':
        model = net_MCD(input_dim = mod['input'], hidden_dim = mod['hidden'], output_dim = mod['output']).to(device)
    elif mod['model'] == 'DANN':
        model = net_DANN(input_dim = mod['input'], hidden_dim = mod['hidden'], output_dim = mod['output']).to(device)
    elif mod['model']=='DAN':
        model = net_DAN(input_dim = mod['input'], hidden_dim = mod['hidden'], output_dim = mod['output']).to(device)
    elif mod['model'] == 'CORAL':
        model = net_DAN(input_dim = mod['input'], hidden_dim = mod['hidden'], output_dim = mod['output'], method='CORAL').to(device)
    elif mod['model']=='CDAN':
        model = net_CDAN(input_dim = mod['input'], hidden_dim = mod['hidden'], output_dim = mod['output']).to(device)

    src_dataset = Data.TensorDataset(torch.tensor(mod['src'][0]).float().to(device),torch.tensor(mod['src'][1]).float().to(device))
    src_loader = Data.DataLoader(src_dataset,batch_size = mod['batch_size'],shuffle=True,num_workers=0,drop_last=True)
    tgt_dataset = Data.TensorDataset(torch.tensor(mod['tgt'][0]).float().to(device),torch.tensor(mod['tgt'][1]).float().to(device))
    tgt_loader = Data.DataLoader(tgt_dataset,batch_size = mod['batch_size'],shuffle=True,num_workers=0,drop_last=True)

    loss_func = torch.nn.MSELoss()
    log_interval = len(tgt_loader)
    rslt = {'l_src':[],'domain_div':[], 'total_div':[], 'copula_distance':[],'roc':[],'time':[], 're': []}
    src_iter = iter(src_loader)
    tgt_iter = iter(tgt_loader)
    roc = -10
    rel_error = 10
    list_xf, list_yf, list_loss_src,list_domain_div,list_copula_distance, list_total_div =[],[],[],[],[],[]
    time_start = time.time()
    count = 0
    iterator = tqdm(range(1, mod['iteration']+1))
    for i in iterator:
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        try:
            src_data, src_label = src_iter.next()
        except Exception as err:
            src_iter=iter(src_loader)
            src_data, src_label = src_iter.next()
        try:
            tgt_data, tgt_label = tgt_iter.next()
        except Exception as err:
            tgt_iter=iter(tgt_loader)
            tgt_data, tgt_label = tgt_iter.next()

        optimizer.zero_grad()
        if mod['model'] == 'MLP':
            src_out = model(src_data)
            loss = loss_func(src_out.flatten(), src_label)
        elif mod['model'] == 'AFN':
            # src_div/tgt_div is the L2-norm between features and a given const vector, radius
            src_div, src_out = model(src_data)
            tgt_div, tgt_out = model(tgt_data)
            total_div = src_div + tgt_div
            loss = loss_func(src_out.flatten(), src_label) + mod['trade_off1'] * total_div
            list_domain_div.append(total_div.cpu().data.numpy())
        elif mod['model'] == 'MCD':
            # prediction_1/2 is tgt predictors, the diff should be small
            # adversarial method
            _, src_out = model(src_data)
            tgt_mediate, tgt_out = model(tgt_data)
            prediction_1 = model.F1(tgt_mediate)
            prediction_2 = model.F2(tgt_mediate)
            total_div = classifier_discrepancy(prediction_1, prediction_2)
            loss = loss_func(src_out.flatten(), src_label) - mod['trade_off1'] * total_div
            list_domain_div.append(total_div.cpu().data.numpy())
        elif mod['model'] == 'DANN':
            src_mediate, src_out = model(src_data)
            tgt_mediate, tgt_out = model(tgt_data)
            d_src = torch.ones((src_mediate.size(0),1)).to(device)
            d_tgt = torch.zeros((tgt_mediate.size(0),1)).to(device)
            total_div = F.binary_cross_entropy(model.classifier(src_mediate), d_src) + F.binary_cross_entropy(model.classifier(tgt_mediate), d_tgt)
            loss = loss_func(src_out.flatten(), src_label) - mod['trade_off1'] * total_div
            list_domain_div.append(total_div.cpu().data.numpy())
        elif mod['model'] == 'DAN' or mod['model'] == 'CORAL':
            total_div, src_out = model(src_data, tgt_data)
            loss = loss_func(src_out.flatten(), src_label) + mod['trade_off1'] * total_div
            list_total_div.append(total_div.cpu().data.numpy())
        elif mod['model'] == 'CDAN':
            marginal_div, copula_distance, src_out = model(src_data, tgt_data)
            loss = loss_func(src_out.flatten(), src_label) + mod['trade_off1'] * marginal_div + mod[
                'trade_off2'] * copula_distance
            list_domain_div.append(marginal_div.cpu().data.numpy())
            list_copula_distance.append(copula_distance.cpu().data.numpy())
        list_loss_src.append(loss.cpu().data.numpy())
        if mod['model'] == 'DANN':
            loss.backward(retain_graph=True)
        else: 
            loss.backward()
        optimizer.step()
        if mod['model'] == 'MCD':
            # MCD adversarial, logic due to
            # https://github.com/mil-tokyo/MCD_DA/blob/master/classification/solver.py
            for _ in range(5):
                optimizer_F = torch.optim.Adam([{"params":model.F1.parameters()}, {"params":model.F2.parameters()}],
                                               lr=learning_rate)
                tgt_mediate, tgt_out = model(tgt_data)
                prediction_1 = model.F1(tgt_mediate)
                prediction_2 = model.F2(tgt_mediate)
                total_div = classifier_discrepancy(prediction_1, prediction_2)
                total_div.backward()
                optimizer_F.step()
        elif mod['model'] == 'DANN':
            # DANN to train the domain classifier
            for _ in range(5):
                optimizer_F = torch.optim.Adam(model.classifier.parameters(),
                                                lr=learning_rate)
                optimizer_F.step()
        model.eval()
        x_f, y_f = model.forward_ft(src_data,tgt_data)
        list_xf.append(x_f.cpu().data.numpy())
        list_yf.append(y_f.cpu().data.numpy())
        if i % log_interval == 0:
            rslt['l_src'].append(np.average(list_loss_src))
            if list_domain_div:
                rslt['domain_div'].append(np.average(list_domain_div))
            else:
                rslt['domain_div'].append(0)
            if list_total_div:
                rslt['total_div'].append(np.average(list_total_div))
            else:
                rslt['total_div'].append(0)
            if list_copula_distance:
                rslt['copula_distance'].append(np.average(list_copula_distance))
            else:
                rslt['copula_distance'].append(0)
            print('Train iter: {} [({:.0f}%)]\tLoss: {:.6f}'.format(i, 100. * i / mod['iteration'],
                                                                    np.average(list_loss_src)))
            list_loss_src,list_domain_div,list_total_div,list_copula_distance = [],[],[],[]
            tgt_pred = model.predict(torch.tensor(tgt_x).float().to(device))
            tgt_loss = loss_func(tgt_pred.flatten(), torch.tensor(tgt_y).float().to(device))
            tgt_pred = tgt_pred.detach().numpy().flatten()
            roc_update = r2_score(tgt_y, tgt_pred)
            if scaler:
                tgt_y_true = scaler.inverse_transform(tgt_y.reshape(-1,1)).flatten()
                tgt_pred_true = scaler.inverse_transform(tgt_pred.reshape(-1,1)).flatten()
                rel_error_update = np.mean(np.abs(tgt_y_true - tgt_pred_true) / tgt_y_true)
            if roc_update > roc:
                roc = roc_update
                if scaler:
                    rel_error = rel_error_update
            else:
                count += 1
            print('\n Target loss: {:.4f}, ROC: {}\n'.format(
                tgt_loss, roc))
            if count >= mod['patience']:
                iterator.close()
                print("Training stops at {}-th loop with best roc {}".format(i, roc))
                break
        time_end = time.time()
    rslt['roc'].append(roc)
    rslt['re'].append(rel_error)
    rslt['time'].append(time_end-time_start)
    return rslt, list_xf, list_yf, log_interval