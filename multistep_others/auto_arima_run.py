"""
Created on  12/7/2020
@author: Jingchao Yang
"""
from pmdarima.arima import auto_arima
from multistep_others import data_helper
from multistep_lstm import model_train
import numpy as np
import pandas as pd
from multistep_lstm import multistep_lstm_pytorch
from tqdm import tqdm


train_window, output_size = 24, 12

'''getting data'''
iot_sensors, iot_df = data_helper.get_data()
while iot_df.isnull().values.any():
    model_train.fill_missing(iot_df.values)
print('NaN value in IoT df?', iot_df.isnull().values.any())
# iot_df = iot_df.dropna()

# test_sensor = iot_sensors[0]
# data_re = iot_df[test_sensor]

'''separate train and test stations'''
train_stations = set(np.random.choice(iot_sensors, int(len(iot_sensors) * 0.7), replace=False))
test_stations = set(iot_sensors) - train_stations

train_data_raw = iot_df[train_stations]
test_data_raw = iot_df[test_stations]

print(train_data_raw.shape)
print(test_data_raw.shape)
print(train_data_raw.columns)

'''last 30% from test stations for prediction test'''
# test_data_raw_test = test_data_raw[-int(0.3 * test_data_raw.shape[0]):]
dataX, dataY = multistep_lstm_pytorch.create_dataset(test_data_raw, train_window, output_size,
                                                     tensor=False, multivar=False)
print(dataX.shape)
print(dataY.shape)

'''auto arima'''
rmse_all, r2_all = [], []

test_pred_orig_dict = dict()
for i in tqdm(range(dataX.shape[-1])):
    station = test_data_raw.columns[i]
    preds, origs = [], []
    for j in range(dataX.shape[0]):
        X = dataX[j][:, i]
        y = dataY[j][:, i]

        modl = auto_arima(X, start_p=1, start_q=1, start_P=1, start_Q=1,
                          max_p=5, max_q=5, max_P=5, max_Q=5, seasonal=True,
                          stepwise=True, suppress_warnings=True, D=10, max_D=10,
                          error_action='ignore')
        # Create predictions for the future, evaluate on test
        preds.append(modl.predict(n_periods=output_size))
        origs.append(y)

    test_pred_orig_dict[station] = (np.array(preds), np.array(origs))

    rmse_by_station_test, mae_by_station_test, rmse_by_hour_test, mae_by_hour_test = model_train.result_evaluation(
        test_pred_orig_dict, gpu=False)

    path = r'..\result'
    rmse_by_station_test.to_csv(path + r'\arima_trainScores_C.csv')
    np.savetxt(path + r'\arima_trainScores_C_by_hour.csv', rmse_by_hour_test, delimiter=",")