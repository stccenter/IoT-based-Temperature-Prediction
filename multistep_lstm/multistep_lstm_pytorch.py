"""
Created on  8/27/20
@author: Jingchao Yang
"""
from platform import python_version
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import math
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from statistics import mean
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime, date, timedelta
from functools import reduce

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.linear = nn.Linear(hidden_size, output_size)

    def reset_hidden_state(self):
        self.hidden = (torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device),
                       torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device))

    def forward(self, x):
        # input shape: (batch, seq_len, input_size) (how many sequences, train window, how many inputs)
        # output shape: (seq_len, output_size, input_size)
        self.batch_size = x.size(0)
        self.reset_hidden_state()
        output, self.hidden = self.lstm(x, self.hidden)
        # Decode the hidden state of the last time step
        y_pred = self.linear(output)[:, -1, :]
        return y_pred  # (seq_len, output_size)


class Dataset:
    def __init__(self, dataset, minmax, train_window, output_size, test_station=False):
        '''
        Normalize (bool, optional): optional normalization
        '''
        self.keys = dataset.columns
        self.min = minmax[0]
        self.max = minmax[1]
        self.test_station = test_station
        self.data = []

        for key in self.keys:  # each station
            single_column = dataset[key].values
            dataX, dataY = [], []
            single_column = (single_column - self.min) / (self.max - self.min)
            dataX, dataY = create_dataset(single_column, train_window, output_size)

            # np.array/tensor size will be [seq_len, time_window] rather than [seq_len, time_window, 1]
            if test_station:  # For testing stations
                self.data.append([dataX, dataY])
            else:  # For training stations: split data into 70% training and 30% validation sets
                trainX, valX = traintest(dataX, 0.7)
                trainY, valY = traintest(dataY, 0.7)
                self.data.append([trainX, trainY, valX, valY])

    def __len__(self):
        return len(self.data)

    # access Dataset as list items, dictionary entries, array elements etc.
    # support the indexing such that data[i] can be used to get ith sample
    def __getitem__(self, idx):
        # i is the key index, data[i] idx is the index of the matrix of the data[i]
        # return self.data[idx]
        if self.test_station:
            # return x (seq_len, time_window) and y (seq_len, output_size)
            testX = self.data[idx][0].unsqueeze(2).float()
            testY = self.data[idx][1].unsqueeze(2).float()
            return testX, testY
        else:
            # return trainX, trainY, valX, valY
            trainX = self.data[idx][0].unsqueeze(2).float()
            trainY = self.data[idx][1].unsqueeze(2).float()
            valX = self.data[idx][2].unsqueeze(2).float()
            valY = self.data[idx][3].unsqueeze(2).float()
            return trainX, trainY, valX, valY


class Dataset_multivariate:
    def __init__(self, dataset, minmax, train_window, output_size, ext_data, ext_name, iot_wu_match_df, test_station=False):
        '''
        Normalize (bool, optional): optional normalization
        '''
        self.keys = dataset.columns
        self.min = minmax[0]
        self.max = minmax[1]
        self.test_station = test_station
        self.data = []

        for key in self.keys:  # each station
            # single_column = dataset[key].values
            # single_column = (single_column - self.min) / (self.max - self.min)

            merged = dataset[[key]]
            wu_match = iot_wu_match_df.loc[(iot_wu_match_df['Geohash'] == key)]['WU_ind'].values[0]
            ext_match = []
            # ext_name = ['humidity', 'pressure', 'windSpeed']

            for ext in range(len(ext_data)):
                match = ext_data[ext][[str(wu_match)]]
                match.columns = [ext_name[ext]]
                ext_match.append(match)
            ext_match = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True), ext_match)
            ext_match.index = pd.to_datetime(ext_match.index, format='%m/%d/%Y %H:%M').strftime('%Y-%m-%d %H:%M:%S')
            merged = merged.join(ext_match)
            merged = merged.dropna()
            merged = merged.sort_index().values
            merged[:, 0] = (merged[:, 0] - self.min) / (self.max - self.min)

            dataX, dataY = [], []
            dataX, dataY = create_dataset(merged, train_window, output_size, multivar=True)

            # np.array/tensor size will be [seq_len, time_window] rather than [seq_len, time_window, 1]
            if test_station:  # For testing stations
                self.data.append([dataX, dataY])
            else:  # For training stations: split data into 70% training and 30% validation sets
                trainX, valX = traintest(dataX, 0.7)
                trainY, valY = traintest(dataY, 0.7)
                self.data.append([trainX, trainY, valX, valY])

    def __len__(self):
        return len(self.data)

    # access Dataset as list items, dictionary entries, array elements etc.
    # support the indexing such that data[i] can be used to get ith sample
    def __getitem__(self, idx):
        # i is the key index, data[i] idx is the index of the matrix of the data[i]
        # return self.data[idx]
        if self.test_station:
            # return x (seq_len, time_window) and y (seq_len, output_size)
            testX = self.data[idx][0].unsqueeze(2).float()
            testY = self.data[idx][1].unsqueeze(2).float()
            return testX, testY
        else:
            # return trainX, trainY, valX, valY
            trainX = self.data[idx][0].unsqueeze(2).float()
            trainY = self.data[idx][1].unsqueeze(2).float()
            valX = self.data[idx][2].unsqueeze(2).float()
            valY = self.data[idx][3].unsqueeze(2).float()
            return trainX, trainY, valX, valY


def initial_model(input_size=1, hidden_size=30, num_layers=2, learning_rate=0.05, output_size=12):
    loss_func = torch.nn.MSELoss()  # mean-squared error for regression
    model = LSTM(input_size, hidden_size, num_layers, output_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return loss_func, model, optimizer


def train_LSTM(dataloader, model, loss_func, optimizer, epoch):
    model.train()
    loss_list = []
    for idx, data in enumerate(dataloader):
        y_pred = model(data[0])
        optimizer.zero_grad()
        # obtain the loss function
        loss = loss_func(y_pred, data[1].reshape(y_pred.shape))
        loss.backward()
        optimizer.step()
        # record loss
        loss_list.append(loss.item())
    return loss_list


def test_LSTM(dataloader, model, loss_func, optimizer, epoch):
    model.eval()
    loss_list = []
    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            y_pred = model(data[0])
            loss = loss_func(y_pred, data[1].reshape(y_pred.shape))
            loss_list.append(loss.item())
    return loss_list


def univariate_data(dataset, start_index, end_index, history_size, target_size, tensor=True):
    # The parameter history_size is the size of the past window of information.
    # The target_size is how far in the future does the model need to learn to predict.
    # The target_size is the label that needs to be predicted.
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i - history_size, i)
        # Reshape data from (history_size,) to (history_size, 1)
        data.append(np.reshape(dataset[indices], (history_size, 1)))
        labels.append(dataset[i: i + target_size])

    data = np.array(data)
    labels = np.array(labels)

    if tensor:
        data = torch.from_numpy(data).float().to(device)
        labels = torch.from_numpy(labels).float().to(device)

    return data, labels


def traintest(dataset, train_slice, return_size=False):
    # split into train and test sets
    train_size = int(len(dataset) * train_slice)
    train, test = dataset[:train_size], dataset[train_size:]

    if return_size:  # return train_size to retrieve x axis
        return train_size, train, test
    else:
        return train, test


def create_dataset(dataset, train_window, output_size, tensor=True, multivar=False):
    dataX, dataY = [], []
    L = len(dataset)
    for i in range(L - train_window - output_size + 1):
        _x = dataset[i:i + train_window]
        _y = dataset[i + train_window: (i + train_window + output_size)]
        dataX.append(_x)
        if multivar:
            # only using target attribute for y, expand dimension
            dataY.append(np.expand_dims(_y[:, 0], axis=1))
        else:
            dataY.append(_y)

    dataX = np.array(dataX)
    dataY = np.array(dataY)

    if tensor:
        # dataX = torch.from_numpy(dataX).float().to(device)
        # dataY = torch.from_numpy(dataY).float().to(device)
        dataX = torch.from_numpy(dataX).float()
        dataY = torch.from_numpy(dataY).float()

    return dataX, dataY
