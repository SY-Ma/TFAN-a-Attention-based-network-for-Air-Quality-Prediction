# python3.7
# -*- coding: utf-8 -*-
# @Project_Name  : Air Quality prediction Beijing
# @File          : data_processer.py
# @Time          : 2022/11/8 14:04
# @Author        : SY.M
# @Software      : PyCharm
import sys

from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
import os
import math
from sklearn.preprocessing import StandardScaler

class DataProcesser(Dataset):
    def __init__(self, scalar, dataset_type, seq_len, pred_len, sequential=True,
                 AQ_path='choose your airquality.csv absolutely path',
                 ML_path='choose your meteorology.csv absolutely path',
                 station_belong_path='choose your station.csv absolutely path'):
        super(DataProcesser, self).__init__()

        # random completely, two for train, one for val and one for test, loop until last id
        self.train_id = [1001, 1002, 1005, 1006, 1009, 1010, 1013, 1014, 1017, 1018, 1021, 1025, 1026,
                         1029, 1030, 1033, 1034]
        self.val_id = [1003, 1007, 1011, 1015, 1019, 1023, 1027, 1031, 1035]
        self.test_id = [1004, 1008, 1012, 1016, 1020, 1024, 1028, 1032, 1036]

        # test
        # self.train_id = [1001, ]
        # self.val_id = [1002, ]
        # self.test_id = [1003, ]

        self.dataset_type = dataset_type
        self.station_list = self.train_id + self.val_id + self.test_id
        self.AQ_dict = None
        self.ML_dict = None
        self.district_list = None
        self.station_data = None
        self.scalar = None
        self.sequential = sequential
        '''when sequential==False，we just use the data to analysis and visualization, 
           and did not build dataset, if you want to train model, please set sequential==True'''
        self.stamp_colums = ['month', 'day', 'weekday', 'hour']
        self.data_columns = ['PM25_Concentration', 'PM10_Concentration', 'NO2_Concentration',
                        'CO_Concentration', 'O3_Concentration', 'SO2_Concentration',
                        'temperature', 'pressure', 'humidity', 'wind_speed',
                        'weather', 'wind_direction']
        self.build_data_dict(AQ_path, ML_path, station_belong_path, scalar)
        # data colunms:
        # Index(['month', 'day', 'weekday', 'hour', 'PM25_Concentration',
        #    'PM10_Concentration', 'NO2_Concentration', 'CO_Concentration',
        #    'O3_Concentration', 'SO2_Concentration', 'temperature', 'pressure',
        #    'humidity', 'wind_speed', 'weather', 'wind_direction'],
        #   dtype='object')
        if self.sequential:
            self.X, self.X_stamp, self.Y, self.Y_stamp = \
                self.build_dataset(seq_len=seq_len, pred_len=pred_len, dataset_type=dataset_type)

    def build_data_dict(self, AQ_path, ML_path, station_belong_path, scalar):
        all_AQ_data = pd.read_csv(AQ_path)
        all_ML_data = pd.read_csv(ML_path)
        district_belong = pd.read_csv(station_belong_path, index_col='station_id').district_id[:36].to_dict()
        self.district_list = set([district_belong[station_id] for station_id in self.station_list])
        self.AQ_dict = {}
        self.ML_dict = {}
        self.station_data = {}
        # get air quality data   key:station id   value:AQ data
        for station_id in self.station_list:
            self.AQ_dict[station_id] = all_AQ_data[all_AQ_data.station_id == station_id]
        # get meteorology data   key:district id   value:ML data
        for district_id in self.district_list:
            self.ML_dict[district_id] = all_ML_data[all_ML_data.id == district_id]
        # merge AQ and ML data by station id
        for station_id in self.AQ_dict.keys():
            AQ_data = self.AQ_dict[station_id]
            ML_data = self.ML_dict[district_belong[station_id]]
            # merge data and miss data interpolation
            AQ_ML_data = pd.merge(AQ_data, ML_data, on='time', how='left')\
                .fillna(method='ffill').fillna(method='bfill').fillna(0)
            # transform date to month,day,weekday,hour
            AQ_ML_data = self.get_global_timestamps(AQ_ML_data)
            self.station_data[station_id] = AQ_ML_data

        if scalar:
            self.scalar = StandardScaler()
            # features need to be transformed
            # transform_feature = ['PM25_Concentration', 'PM10_Concentration', 'NO2_Concentration',
            #                      'CO_Concentration', 'O3_Concentration', 'SO2_Concentration',
            #                      'temperature', 'pressure', 'humidity', 'wind_speed']
            transform_feature = ['PM25_Concentration', 'PM10_Concentration', 'NO2_Concentration',
                                 'CO_Concentration', 'O3_Concentration', 'SO2_Concentration',
                                 'temperature', 'pressure', 'humidity', 'wind_speed', 'weather', 'wind_direction']
            # get all train data
            train_part = []
            for station_id in self.train_id:
                train_part.append(self.station_data[station_id])
            train_part = pd.concat(train_part, axis=0)[transform_feature].values
            # fit train data
            self.scalar.fit(train_part)
            # transform all data
            for station_id in self.station_list:
                transformed_data = self.scalar.transform(self.station_data[station_id][transform_feature].values)
                # transformed_data = pd.DataFrame(transformed_data, columns=transform_feature)
                self.station_data[station_id][transform_feature] = transformed_data

        if self.sequential:
            # make time sequential
            for station_id in self.station_list:
                AQ_ML_data = self.station_data[station_id]
                AQ_ML_data = self.make_sequential(AQ_ML_data)
                self.station_data[station_id] = AQ_ML_data

    def get_global_timestamps(self, AQ_ML_data):
        time = AQ_ML_data.time
        time = pd.to_datetime(time)
        sensor = []
        sensor.append(time.apply(lambda date: date.month))
        sensor.append(time.apply(lambda date: date.day))
        sensor.append(time.apply(lambda date: date.weekday()))
        sensor.append(time.apply(lambda date: date.hour))
        stamp = pd.concat(sensor, axis=1)
        # reorder columns
        stamp.columns = self.stamp_colums
        AQ_ML_data = AQ_ML_data[self.data_columns]
        # cat stamp and data
        AQ_ML_data = pd.concat([stamp, AQ_ML_data], axis=1)
        return AQ_ML_data

    def make_sequential(self, AQ_ML_data):
        """
        Make data slice continuous in time and throw them into a list
        every item in list is used to obtain sample
        :return:
        """
        time_point = AQ_ML_data.hour[0]  # 'hour' value of the first line
        start = 0
        end = 0
        sequential_list = []
        for hour in AQ_ML_data.hour:
            # print(hour, time_point)
            if hour != time_point:
                sequential_list.append(AQ_ML_data[start:end])
                start = end
                time_point = hour
            end += 1
            time_point = (time_point + 1) % 24
        return sequential_list

    def build_dataset(self, seq_len, pred_len, dataset_type):
        X = []
        Y = []
        X_stamp = []
        Y_stamp = []
        assert dataset_type in ['train', 'val', 'test'], "Parameter 'Dataset_type' Value ERROR"
        id_list = {'train': self.train_id, 'val': self.val_id, 'test': self.test_id}
        for station_id in id_list[dataset_type]:
            sample_num = 0
            station_AQ_ML_list = self.station_data[station_id]
            for sequential_slice in station_AQ_ML_list:
                start = 0
                data_len = sequential_slice.shape[0]  # 连续无缺失数据片段的长度
                if data_len < seq_len + pred_len:
                    continue
                while (start + seq_len + pred_len) <= data_len:
                    sample_x = sequential_slice[start:start + seq_len][self.data_columns]
                    sample_x_stamp = sequential_slice[start:start + seq_len][self.stamp_colums]
                    sample_y = sequential_slice[start + seq_len:start + seq_len + pred_len][self.data_columns]
                    sample_y_stamp = sequential_slice[start + seq_len:start + seq_len + pred_len][self.stamp_colums]
                    start += 1
                    X.append(sample_x.values)
                    X_stamp.append(sample_x_stamp.values)
                    Y.append(sample_y.values)
                    Y_stamp.append(sample_y_stamp.values)
                    sample_num += 1
            print(f'stationID:{station_id}\t sample_num:{sample_num}')
        X = np.stack(X, axis=0)
        Y = np.stack(Y, axis=0)
        X_stamp = np.stack(X_stamp, axis=0)
        Y_stamp = np.stack(Y_stamp, axis=0)
        print(f'{dataset_type} X:{X.shape}\tX_stamp:{X_stamp.shape}\tY:{Y.shape}\tY_stamp:{Y_stamp.shape}')
        return X, X_stamp, Y, Y_stamp

    def __getitem__(self, item):
        return self.X[item], self.X_stamp[item], self.Y[item], self.Y_stamp[item]

    def __len__(self):
        return self.X.shape[0]