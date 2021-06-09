"""
Created on  9/4/2020
@author: Jingchao Yang

pairing geotab coors to wu coors using 1-nearest-neighbor"""
import pandas as pd
import numpy as np

iot_coors = pd.read_csv('../data/LA/IoT/nodes_missing_5percent.csv',
                        usecols=['Geohash', 'Longitude_SW', 'Latitude_SW'])
wu_coors = pd.read_csv('../data/LA/WU/coor_keys.csv',
                       index_col=False, usecols=['lon', 'lat']).to_numpy()


P = np.add.outer(np.sum(iot_coors[['Longitude_SW', 'Latitude_SW']].to_numpy()**2, axis=1), np.sum(wu_coors**2, axis=1))
N = np.dot(iot_coors[['Longitude_SW', 'Latitude_SW']].to_numpy(), wu_coors.T)
dists = np.sqrt(P - 2*N)
print(dists.shape)
print(dists)

minInRows = np.amin(dists, axis=1)
minInRows_ind = np.argmin(dists, axis=1)

coor_match_df = pd.DataFrame(iot_coors['Geohash'])
coor_match_df['WU_ind'] = minInRows_ind

coor_match_df.to_csv('../data/LA/iot_wu_colocate.csv')


