from distance import *
import numpy as np
import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
LSTM works as the benchmark
"""
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout = 0):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(nn.BatchNorm1d(hidden_dim) ,nn.Linear(hidden_dim, hidden_dim // 4),nn.ReLU())
        self.fc1 = nn.Sequential(nn.BatchNorm1d(hidden_dim // 4) ,nn.Linear(hidden_dim // 4, output_dim))

    def forward(self, x, hc=(None, None)):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        if hc[0] != None and hc[1] != None:
            out, (hn, cn) = self.lstm(x, (hc[0], hc[1]))
        else:
            out, (hn, cn) = self.lstm(x, (h0.detach().to(device), c0.detach().to(device)))
        out = self.fc(out[:, -1, :]) 
        out = self.fc1(out)
        return out, (hn, cn)
    
    def predict(self,x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (_,__) = self.lstm(x, (h0.detach().to(device), c0.detach().to(device)))
        out = self.fc(out[:,-1,:])
        out = self.fc1(out)
        return out


"""    
DALSTM is LSTM combined with DAN or CORAL, depending on the choice of the attribute 'method' 
"""
class DALSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0, method='CORAL'):
        super(DALSTM, self).__init__()
        self.method = method
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(nn.BatchNorm1d(hidden_dim) ,nn.Linear(hidden_dim, hidden_dim // 2),nn.ReLU(),
                                nn.BatchNorm1d(hidden_dim // 2), nn.Linear(hidden_dim // 2, hidden_dim // 4))
        self.fc1 = nn.Sequential(nn.BatchNorm1d(hidden_dim // 4) ,nn.Linear(hidden_dim // 4, output_dim))

    def forward(self, x, y, hc=(None, None)):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        if hc[0] != None and hc[1] != None:
            out, (hn, cn) = self.lstm(x, (hc[0], hc[1]))
            out_, (_, __) = self.lstm(y, (hc[0], hc[1]))
        else:
            out, (hn, cn) = self.lstm(x, (h0.detach().to(device), c0.detach().to(device)))
            out_, (_, __) = self.lstm(y, (h0.detach().to(device), c0.detach().to(device)))
        if self.method == 'CORAL':
            total_div = self.CORAL(out[:,-1,:], out_[:,-1,:])
        else:
            total_div = MMD(out[:,-1,:], out_[:,-1,:])
        out = self.fc(out[:, -1, :]) 
        out = self.fc1(out)
        return total_div, out, (hn, cn)
    
    def predict(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (_,__) = self.lstm(x, (h0.detach().to(device), c0.detach().to(device)))
        out = self.fc(out[:,-1,:])
        out = self.fc1(out)
        return out
    
    def CORAL(self, source, target):
        d = source.data.shape[1]
        ns, nt = source.data.shape[0], target.data.shape[0]
        xm = torch.mean(source, 0, keepdim=True) - source
        xc = xm.t() @ xm / (ns - 1)
        xmt = torch.mean(target, 0, keepdim=True) - target
        xct = xmt.t() @ xmt / (nt - 1)
        loss = torch.mul((xc - xct), (xc - xct))
        loss = torch.sum(loss) / (4*d*d)
        return loss

    
"""
GCLSTM is LSTM+CDAN, which is the abbreviation of Gaussian-Copua based LSTM.
"""
class GCLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout = 0, method='KL'):
        super(GCLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(nn.BatchNorm1d(hidden_dim) ,nn.Linear(hidden_dim, hidden_dim // 2))
        self.fc1 = nn.Sequential(nn.BatchNorm1d(hidden_dim // 2) ,nn.Linear(hidden_dim // 2, output_dim))

        self.method = method

    def forward(self, x, y, hc=(None, None)):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        if hc[0] != None and hc[1] != None:
            out, (hn, cn) = self.lstm(x, (hc[0], hc[1]))
            out_, (_, __) = self.lstm(y, (hc[0], hc[1]))
        else:
            out, (hn, cn) = self.lstm(x, (h0.detach().to(device), c0.detach().to(device)))
            out_, (_, __) = self.lstm(y, (h0.detach().to(device), c0.detach().to(device)))
        marginal_div = self.marginal_div(out[:,-1,:], out_[:,-1,:])
        copula_dist = self.copula_distance(out[:,-1,:], out_[:,-1,:], self.method)
        out = self.fc(out[:, -1, :]) 
        out = self.fc1(out)
        return marginal_div, copula_dist, out, (hn, cn)

    def marginal_div(self, x, y):
        return MD_MMD(x,y)

    def copula_distance(self, x, y, method):
        size = x.shape[0] // 2
        cd = 0
        if method == 'Frobenius':
            return CD_Frobenius(x,y)
        elif method == 'KL':
            return CD_KL(x,y)

    def predict(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        # out, (_,__) = self.lstm(x, (h0.detach(), c0.detach()))
        out, (_,__) = self.lstm(x, (h0.detach().to(device), c0.detach().to(device)))
        out = self.fc(out[:,-1,:])
        out = self.fc1(out)
        return out
