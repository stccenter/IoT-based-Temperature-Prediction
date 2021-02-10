"""
Created on  10/19/2020
@author: Jingchao Yang
"""
import matplotlib.pyplot as plt
import time
from statistics import mean
from torch.utils.data import TensorDataset, DataLoader
# from multistep_lstm import multistep_lstm_pytorch
import multistep_lstm_pytorch
from sklearn import preprocessing
import numpy as np
from numpy import isnan
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import math
from numpy import array
from numpy import nanmedian


def fill_missing(values):
    """
    fill missing values with a value at the same time one day ago

    :param values:
    :return:
    """
    one_day = 24
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            if isnan(values[row, col]):
                values[row, col] = values[row - one_day, col]


# impute missing data
def impute_missing(train_chunks, rows, hours, series, col_ix):
    # impute missing using the median value for hour in all series
    imputed = list()
    for i in range(len(series)):
        if isnan(series[i]):
            # collect all rows across all chunks for the hour
            all_rows = list()
            for rows in train_chunks:
                [all_rows.append(row) for row in rows[rows[:, 2] == hours[i]]]
            # calculate the central tendency for target
            all_rows = array(all_rows)
            # fill with median value
            value = nanmedian(all_rows[:, col_ix])
            if isnan(value):
                value = 0.0
            imputed.append(value)
        else:
            imputed.append(series[i])
    return imputed


def min_max_scaler(df):
    """

    :param df:
    :return:
    """
    min_max_scaler = preprocessing.MinMaxScaler()
    df_np = df.values
    df_np_scaled = min_max_scaler.fit_transform(df_np)
    df_scaled = pd.DataFrame(df_np_scaled)

    df_scaled.index = df.index
    df_scaled.columns = df.columns

    return df_scaled


def result_to_celsius(fahrenheit):
    celsius = (fahrenheit - 32) * 5.0 / 9.0
    return celsius


def start_training(model, train_data, device, loss_func, optimizer, num_epochs, epoch_interval=1):
    train_loss, test_loss, mean_loss_train, mean_test_loss = [], [], [], []
    min_val_loss, mean_min_val_loss = np.Inf, np.Inf
    n_epochs_stop = 3
    epochs_no_improve = 0
    early_stop = False

    start = time.time()
    # train the model
    for epoch in range(num_epochs):
        running_loss_train = []
        running_loss_val = []
        loss2 = 0
        for idx in range(len(train_data)):
            train_loader = DataLoader(TensorDataset(train_data[idx][0][:, :, 0, :].to(device),
                                                    train_data[idx][1][:, :, 0, :].to(device)),
                                      shuffle=True, batch_size=1000, drop_last=True)
            val_loader = DataLoader(TensorDataset(train_data[idx][2][:, :, 0, :].to(device),
                                                  train_data[idx][3][:, :, 0, :].to(device)),
                                    shuffle=True, batch_size=400, drop_last=True)
            loss1 = multistep_lstm_pytorch.train_LSTM(train_loader, model, loss_func, optimizer,
                                                      epoch)  # calculate train_loss
            loss2 = multistep_lstm_pytorch.test_LSTM(val_loader, model, loss_func, optimizer,
                                                     epoch)  # calculate test_loss
            running_loss_train.append(sum(loss1))
            running_loss_val.append(sum(loss2))
            train_loss.extend(loss1)
            test_loss.extend(loss2)

            if mean(loss2) < min_val_loss:
                # Save the model
                # torch.save(model)
                epochs_no_improve = 0
                min_val_loss = mean(loss2)

            else:
                epochs_no_improve += 1

            if epoch > 5 and epochs_no_improve == n_epochs_stop:
                print('Early stopping!')
                early_stop = True
                break
            else:
                continue

        mean_loss_train.append(mean(running_loss_train))
        mean_test_loss.append(mean(running_loss_val))
        if epoch % epoch_interval == 0:
            print(
                "Epoch: %d, train_loss: %1.5f, val_loss: %1.5f" % (
                    epoch, mean(running_loss_train), mean(running_loss_val)))
            if mean(running_loss_val) < mean_min_val_loss:
                mean_min_val_loss = mean(running_loss_val)
            else:
                print('Early stopping!')
                early_stop = True
        if early_stop:
            print("Stopped")
            break

    end = time.time()
    print(end - start)

    print(model)

    # plt.plot(train_loss)
    # plt.plot(test_loss)
    # plt.show()
    #
    # plt.plot(mean_loss_train)
    # plt.plot(mean_test_loss)
    # plt.show()

    return model


def result_evaluation(model_output, gpu=True):
    rmse_by_station, mae_by_station = dict(), dict()
    rmse_by_hour, mae_by_hour = [], []
    for key in model_output.keys():
        if gpu:
            pred_key = result_to_celsius(model_output[key][0].data.cpu().numpy())
            orig_key = result_to_celsius(model_output[key][1].data.cpu().numpy())
        else:
            pred_key = result_to_celsius(model_output[key][0])
            orig_key = result_to_celsius(model_output[key][1])

        rmse_by_hour_temp = np.sqrt(mean_squared_error(pred_key, orig_key, multioutput='raw_values'))
        rmse_by_hour.append(rmse_by_hour_temp)
        rmse_by_station[key] = math.sqrt(mean_squared_error(pred_key, orig_key))

        mae_by_hour_temp = mean_absolute_error(pred_key, orig_key, multioutput='raw_values')
        mae_by_hour.append(mae_by_hour_temp)
        mae_by_station[key] = mean_absolute_error(pred_key, orig_key)

    return pd.DataFrame.from_dict(rmse_by_station, orient='index', columns=['value']), \
           pd.DataFrame.from_dict(mae_by_station, orient='index', columns=['value']), \
           np.average(rmse_by_hour, axis=0), np.average(mae_by_hour, axis=0)
