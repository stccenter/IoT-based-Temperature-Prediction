"""
Created on  12/7/2020
@author: Jingchao Yang
"""
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
import pandas as pd
from sklearn.metrics import mean_squared_error as MSE
import numpy as np
from multistep_others import data_helper
from multistep_lstm import model_train
from multistep_lstm import multistep_lstm_pytorch
from tqdm import tqdm
from functools import reduce


def add_variate(station, iot_wu_match_df, ext_data, ext_name):
    wu_match = iot_wu_match_df.loc[(iot_wu_match_df['Geohash'] == col)]['WU_ind'].values[0]
    ext_match = []
    # ext_name = ['humidity', 'pressure', 'windSpeed']

    for ext in range(len(ext_data)):
        match = ext_data[ext][[str(wu_match)]]
        match.columns = [ext_name[ext]]
        ext_match.append(match)
    ext_match = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True), ext_match)
    ext_match.index = pd.to_datetime(ext_match.index, format='%m/%d/%Y %H:%M').strftime('%Y-%m-%d %H:%M:%S')
    ext_match.index.drop_duplicates(keep='first')
    station = station.join(ext_match)
    station = station.dropna()
    station = station.sort_index().values

    return station


train_window, output_size = 24, 12
multi_variate_mode = False
'''getting data'''
iot_sensors, iot_df = data_helper.get_data()
while iot_df.isnull().values.any():
    model_train.fill_missing(iot_df.values)
print('NaN value in IoT df?', iot_df.isnull().values.any())

'''separate train and test stations'''
train_stations = set(np.random.choice(iot_sensors, int(len(iot_sensors) * 0.7), replace=False))
test_stations = set(iot_sensors) - train_stations

train_data_raw = iot_df[train_stations]
test_data_raw = iot_df[test_stations]

print(train_data_raw.shape)
print(test_data_raw.shape)

'''multivariate'''
if multi_variate_mode:
    exp_path = r'D:\IoT_HeatIsland\iotTemp_exp_bak\exp_data\LA'
    wu_path = exp_path + r'\WU'
    ext_name = ['humidity', 'windSpeed', 'dewPoint', 'precipProbability', 'pressure', 'cloudCover', 'uvIndex']
    ext_data_path = wu_path + r'\byAttributes'
    ext_data_ls = []
    for ext in ext_name:
        ext_df = pd.read_csv(ext_data_path + f'\{ext}.csv', index_col=['datetime'])
        while ext_df.isnull().values.any():
            model_train.fill_missing(ext_df.values)
        print(f'NaN value in {ext} df?', ext_df.isnull().values.any())
        ext_data_ls.append(model_train.min_max_scaler(ext_df))

    iot_wu_match_df = pd.read_csv(exp_path + r'\iot_wu_colocate.csv', index_col=0)

    # train_data_raw = add_variate(train_data_raw, iot_wu_match_df, ext_data_ls, ext_name)
    # test_data_raw = add_variate(train_data_raw, iot_wu_match_df, ext_data_ls, ext_name)

'''xgboost'''
xgb_r = MultiOutputRegressor(xgb.XGBRegressor(objective='reg:squarederror',
                                              n_estimators=10,
                                              seed=123))
# train model
for col in tqdm(train_data_raw.columns):
    X = train_data_raw[col]
    if multi_variate_mode:
        X = add_variate(train_data_raw[[col]], iot_wu_match_df, ext_data_ls, ext_name)

    train_X, train_y = multistep_lstm_pytorch.create_dataset(X,
                                                             train_window,
                                                             output_size,
                                                             tensor=False,
                                                             multivar=multi_variate_mode)
    if multi_variate_mode:
        train_X = train_X.reshape(-1, train_window*(len(ext_name)+1))
        train_y = train_y.reshape(-1, output_size)
    xgb_r.fit(train_X, train_y)

# model predict
test_pred_orig_dict = dict()
for col in tqdm(test_data_raw.columns):
    X = test_data_raw[col]
    if multi_variate_mode:
        X = add_variate(test_data_raw[[col]], iot_wu_match_df, ext_data_ls, ext_name)

    test_X, test_y = multistep_lstm_pytorch.create_dataset(X,
                                                           train_window,
                                                           output_size,
                                                           tensor=False,
                                                           multivar=multi_variate_mode)
    if multi_variate_mode:
        test_X = test_X.reshape(-1, train_window*(len(ext_name)+1))
        test_y = test_y.reshape(-1, output_size)
    preds = xgb_r.predict(test_X)
    test_pred_orig_dict[col] = (np.array(preds), np.array(test_y))


rmse_by_station_test, mae_by_station_test, rmse_by_hour_test, mae_by_hour_test = model_train.result_evaluation(
    test_pred_orig_dict, gpu=False)

path = r'..\result'
rmse_by_station_test.to_csv(path + r'\xgboost_testScores_C.csv')
np.savetxt(path + r'\xgboost_testScores_C_by_hour.csv', rmse_by_hour_test, delimiter=",")