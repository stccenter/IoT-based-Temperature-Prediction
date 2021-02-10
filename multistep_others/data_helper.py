"""
Created on  1/4/2021
@author: Jingchao Yang
"""
import pandas as pd
from numpy import isnan


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


def get_data():
    geohash_df = pd.read_csv(
        r'..\data\LA\IoT\nodes_missing_5percent.csv',
        usecols=['Geohash'])
    iot_sensors = geohash_df.values.reshape(-1)
    iot_df = pd.read_csv(r'..\data\LA\IoT\preInt_matrix_full.csv',
                         usecols=['datetime'] + iot_sensors.tolist(), index_col=['datetime'])
    fill_missing(iot_df.values)

    return iot_sensors, iot_df