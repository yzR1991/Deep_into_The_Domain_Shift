import numpy as np
import torch
import torch.utils.data as Data
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt

from models import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(mod):
    """
    This will return two lists.
    The first list, res, contains the best results when training process is terminated.
    Namely, the best values of loss function on two domains, the best relative error on two domains, the best RMSEs and relative errors of 22 stocks.
    The second list, detail, contains the historical results of the training for each epoch.
    Namely, the values of loss function for target domain, the RMSE for two domains, the relative errors for two domains, and domain divergences and copula distances.
    """
    input_dim, output_dim = mod['source'][0][0].shape[-1], mod['source'][0][-1].shape[0]
    if mod['model'] == 'LSTM':
        model = LSTM(input_dim=input_dim, hidden_dim = mod['hidden_dim'], output_dim=output_dim, num_layers=mod['num_layers']).to(device)
    elif mod['model'] == 'DA':
        model = DALSTM(input_dim=input_dim, hidden_dim = mod['hidden_dim'], output_dim=output_dim, num_layers=mod['num_layers'], method=mod['method']).to(device)
    else:
        model = GCLSTM(input_dim=input_dim, hidden_dim = mod['hidden_dim'], output_dim=output_dim, num_layers=mod['num_layers'], method=mod['method']).to(device)
    loss_fn = torch.nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

    best_loss_test, best_loss_train = 1e2, 1e2
    best_mape_test, best_mape_train = 1e2, 1e2

    source_loader = Data.DataLoader(mod['source'], batch_size = mod['batch_size'], shuffle = True, drop_last=True)
    target_loader = Data.DataLoader(mod['target'], batch_size = mod['batch_size'], shuffle = True, drop_last=True)
    train_iter = iter(source_loader)
    test_iter = iter(target_loader)

    hstate = None
    cstate = None

    hist_loss = np.zeros(mod['num_epoch']) # total train loss
    hist_error_train = np.zeros(mod['num_epoch']) # 
    hist_error_test = np.zeros(mod['num_epoch'])
    hist_mape_train = np.zeros(mod['num_epoch'])
    hist_mape_test = np.zeros(mod['num_epoch'])
    hist_error_test_marginal = np.zeros([mod['num_epoch'], output_dim])
    hist_mape_test_marginal = np.zeros([mod['num_epoch'], output_dim])
    hist_div = np.zeros(mod['num_epoch'])
    hist_dist = np.zeros(mod['num_epoch'])

    iterator = tqdm(range(mod['num_epoch']))
    for t in iterator:
        cur_loss = 0
        cur_pred_error = 0
        cur_mape = 0   
        for i in range(mod['n_batch']):
            try:
                x_train_iter, y_train_iter = train_iter.next()
            except Exception as err:
                train_iter = iter(source_loader)
                x_train_iter, y_train_iter = train_iter.next()
            try:
                x_test_iter, y_test_iter = test_iter.next()
            except Exception as err:
                test_iter = iter(target_loader)
                x_test_iter, y_test_iter = test_iter.next()
            if hstate != None: hstate = hstate.detach()
            if cstate != None: cstate = cstate.detach() 
            
            if mod['model'] == 'LSTM':
                y_train_pred, (hstate, cstate) = model(x_train_iter, (hstate,cstate))
            elif mod['model'] == 'DA':
                total_div, y_train_pred, (hstate, cstate) = model(x_train_iter, x_test_iter, (hstate,cstate))
            else:
                marginal_div, copula_dist, y_train_pred, (hstate, cstate) = model(x_train_iter, x_test_iter, (hstate,cstate))
            
            pred_error = torch.sqrt(loss_fn(y_train_pred, y_train_iter))
            cur_pred_error += pred_error
            if mod['model'] == 'LSTM':
                loss = pred_error
            elif mod['model'] == 'DA':
                loss = pred_error + mod['t1']*total_div
            else:
                loss = pred_error + mod['t1']*marginal_div + mod['t2']*copula_dist
            
            cur_loss += loss
            y_train_iter_true = mod['y_scaler_train'].inverse_transform(y_train_iter.cpu().detach().numpy())
            y_train_pred_true = mod['y_scaler_train'].inverse_transform(y_train_pred.cpu().detach().numpy())
            cur_mape += np.mean(np.abs((y_train_iter_true - y_train_pred_true) / y_train_iter_true))
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        
        model.eval()
        y_test_pred = model.predict(mod['X_test'])
        y_test_pred_true = mod['y_scaler_test'].inverse_transform(y_test_pred.cpu().detach().numpy())
        y_test_true = mod['y_scaler_test'].inverse_transform(mod['y_test'].cpu().detach().numpy())
        cur_error = torch.sqrt(loss_fn(y_test_pred, mod['y_test']))
        cur_mape_test = np.mean(np.abs((y_test_pred_true - y_test_true) / y_test_true))
        cur_error_marginal = torch.sqrt(torch.mean((y_test_pred-mod['y_test'])**2, axis=0)).cpu().detach().numpy()
        cur_mape_test_marginal = np.mean(np.abs((y_test_pred_true - y_test_true) / y_test_true), axis=0)
        model.train()

        if t % 10 == 0:
                print("Epoch ", t, "Train RMSE: ", cur_pred_error.item() / mod['n_batch'], "Train MAPE: ", cur_mape / mod['n_batch'],
                      "\n Test RMSE: ", cur_error.item(), "Test MAPE: ", cur_mape_test)

        hist_loss[t] = cur_loss / mod['n_batch']
        hist_error_train[t] = cur_pred_error / mod['n_batch']
        hist_error_test[t] = cur_error
        hist_mape_train[t] = cur_mape / mod['n_batch']
        hist_mape_test[t] = cur_mape_test
        hist_error_test_marginal[t] = cur_error_marginal
        hist_mape_test_marginal[t] = cur_mape_test_marginal
        if mod['model'] == 'DA':
            hist_div[t] = total_div
        elif mod['model'] == 'GC':
            hist_div[t] = marginal_div
            hist_dist[t] = copula_dist
        
        if cur_error < best_loss_test:
            count = 0
            best_loss_test = cur_error.item()
            best_loss_train = cur_loss / mod['n_batch']
            best_mape_test = cur_mape_test
            best_mape_train = cur_mape / mod['n_batch']
            best_loss_test_marginal = cur_error_marginal
            best_mape_test_marginal = cur_mape_test_marginal
        else:
            count += 1
        
        if count >= mod['patience'] or (count < mod['patience'] and t == mod['num_epoch'] - 1):
            print('Training stop at {}-th training with best test RMSE {} and MAPE {}'.format(t, best_loss_test, best_mape_test))
            plt.plot(hist_error_train[hist_error_train > 0], label="Train RMSE")
            plt.plot(hist_error_test[hist_error_test > 0], label="Test RMSE")
            plt.legend()
            plt.show()

            plt.plot(hist_mape_train[hist_mape_train > 0], label="Train MAPE")
            plt.plot(hist_mape_test[hist_mape_test > 0], label="Test MAPE")
            plt.legend()
            plt.show()
            iterator.close()
            break
    res = [best_loss_train.item(), best_loss_test, best_mape_train, best_mape_test, best_loss_test_marginal, best_mape_test_marginal]
    detail = [hist_loss,hist_error_train,hist_error_test,hist_mape_train,hist_mape_test,hist_div,hist_dist]
    return res, detail
