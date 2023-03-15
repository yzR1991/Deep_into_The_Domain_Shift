import pickle
import numpy as np
import torch
import torch.utils.data as Data
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler

def package_data(input, look_back = 12):
	"""
	To package the data with length 12
	"""
	data = []
	for index in range(input.shape[0] - look_back + 1):
		data.append(input[index:index + look_back])
	data = np.array(data)
	return data

def load_data(dict_data, data_type='price'):
	"""
	Normalizations. The input is a diction whose keys are data of type Datetime.date and values are lists of daily data.
	For return, it is (X-X.mean(axis=0))/X.std(axis=0)
	For price, it is (X - X.max(axis=0))/(X.max(axis=0) - X.min(axis=0))
	For scale, it is X/X.max(axis=0)
	"""
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
