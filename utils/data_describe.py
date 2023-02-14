# python3.7
# -*- coding: utf-8 -*-
# @Project_Name  : Air Quality prediction Beijing
# @File          : data_describe.py
# @Time          : 2022/11/12 21:41
# @Author        : SY.M
# @Software      : PyCharm

from data_process.data_processer import DataProcesser
import pandas as pd
pd.set_option('display.max_columns', 100) #a就是你要设置显示的最大列数参数
pd.set_option('display.max_rows', 10) #b就是你要设置显示的最大的行数参数
pd.set_option('display.width', 99999999999999999) #x就是你要设置的显示的宽度，防止轻易换行


dper = DataProcesser(scalar=False, seq_len=96, pred_len=24, dataset_type='train', sequential=False)

all_data = []
for staion_id, station_data in dper.station_data.items():
    all_data.append(station_data[dper.data_columns])
all_data = pd.concat(all_data, axis=0)
print(all_data.describe())
