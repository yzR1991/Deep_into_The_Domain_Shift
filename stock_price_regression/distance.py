import torch
import math
#DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def MMD(source, target, kernel_mul=2.0, kernel_num=2, fix_sigma=1.0):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    kernels = sum(kernel_val)
    batch_size = int(source.size()[0])
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size] 
    loss = torch.mean(XX + YY - XY -YX)
    return loss

def MD_MMD(x, y):
    """
    Calculate the summation of marginal divergences
    For each column vector of 1-dim, change it to a 2-dim matrix first
    Then apply the gaussian kernel function 
    """
    md = 0
    for i in range(x.shape[1]):
        md += MMD(x[:,i].view(x.shape[0],1), y[:,i].view(y.shape[0],1))
    return md

def Kendall_tau(x, y, size):
    """
    To return the copula coefficient estimated by Kendall Tau
    """
    x_a, x_b = x[:size], x[size:2*size]
    y_a, y_b = y[:size], y[size:2*size]
    res = torch.tanh(5*(x_a - x_b) * (y_a - y_b))
    return torch.sin(torch.mean(res) * math.pi / 2)

def CD_Frobenius(x, y):
    """
    First we use Kendall's tau to estimate parameters of gaussian copula,
    namely the correlation matrix,
    where torch.sign() is replaced by torch.tanh()
    Then the distance between two copulas is given by
    the Frobenius norm of the differences between correlation matrices
    """
    size = x.shape[0] // 2
    cd = 0
    for i in range(x.shape[1]-1):
        for j in range(i,x.shape[1]):
            # calculate ij-th entries of x,y copulas
            cx_ij = Kendall_tau(x[:,i], x[:,j], size)
            cy_ij = Kendall_tau(y[:,i], y[:,j], size)
            cd += 2*(cx_ij - cy_ij)**2
    return torch.sqrt(cd)

def CD_KL(x,y):
    """
    First we use Kendall's tau to estimate parameters of gaussian copula,
    namely the correlation matrix,
    where torch.sign() is replaced by torch.tanh()
    Then the distance between two copulas is given by
    the KL divergence
    """
    size = x.shape[0] // 2
    cd = 0
    for i in range(x.shape[1]-1):
        for j in range(i,x.shape[1]):
            # calculate ij-th entries of x,y copulas
            # then apply the copula distance for KL divergence
            cx_ij = Kendall_tau(x[:,i], x[:,j], size)
            cy_ij = Kendall_tau(y[:,i], y[:,j], size)
            cd += torch.abs(torch.log(1 - cx_ij**2) - torch.log(1 - cy_ij**2))
    return cd
