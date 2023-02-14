# python3.7
# -*- coding: utf-8 -*-
# @Project_Name  : Air Quality prediction Beijing
# @File          : MLP.py
# @Time          : 2022/11/10 16:36
# @Author        : SY.M
# @Software      : PyCharm

from sklearn.neural_network import MLPRegressor

from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    from data_process.data_processer import DataProcesser
    import torch
    from utils.random_seed import setup_seed
    from utils.early_stop import Eearly_stop
    from torch.utils.data import DataLoader
    import numpy as np
    from utils.visualization import draw_line
    import os

    feature_index = {'PM2.5': 0, 'PM10': 1, 'NO2': 2, 'CO': 3, 'O3': 4, 'SO2': 5,
                     'TEMP': 6, 'PRES': 7, 'humidity': 8, 'wind_speed': 9, 'weather': 10, 'WD': 11}
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    start_seed = 30
    end_seed = 30

    BATCHSIZE = 256
    EPOCH = 10000
    test_interval = 1
    update_interval = 5
    save_model_flag = True
    target_feature = 'PM2.5'
    pred_target = feature_index[target_feature]
    seq_len = 168
    pred_len = 72
    dropout = 0.2

    scalar = True
    print('loading data...')
    train_dataset = DataProcesser(scalar=scalar, seq_len=seq_len, pred_len=pred_len, dataset_type='train')
    val_dataset = DataProcesser(scalar=scalar, seq_len=seq_len, pred_len=pred_len, dataset_type='val')
    test_dataset = DataProcesser(scalar=scalar, seq_len=seq_len, pred_len=pred_len, dataset_type='test')
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCHSIZE, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCHSIZE, shuffle=False)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCHSIZE, shuffle=False)
    standard_scalar = train_dataset.scalar
    if standard_scalar is not None:
        f_mean = standard_scalar.mean_[pred_target]
        f_std = np.sqrt(standard_scalar.var_[pred_target])

    train_X, train_Y = train_dataset.X[:, :, pred_target], train_dataset.Y[:, :, pred_target]
    val_X, val_Y = val_dataset.X[:, :, pred_target], val_dataset.Y[:, :, pred_target]
    test_X, test_Y = test_dataset.X[:, :, pred_target], test_dataset.Y[:, :, pred_target]

    def train(test_X, test_Y, seed):
        # build and train model
        regr = MLPRegressor(hidden_layer_sizes=(256, 128, 256), shuffle=True,
                            activation='relu', solver='adam', early_stopping=True,
                            random_state=seed, max_iter=10000).fit(train_X, train_Y)

        pre = regr.predict(test_X)
        loss = np.mean(np.abs(pre - test_Y)).item()
        loss_rmse = np.sqrt(np.mean((pre - test_Y) ** 2).item())
        loss = round(loss, 4)
        loss_rmse = round(loss_rmse, 4)

        pre = pre * f_std + f_mean
        test_Y = test_Y * f_std + f_mean
        loss_raw = np.mean(np.abs(pre - test_Y)).item()
        loss_rmse_raw = np.sqrt(np.mean((pre - test_Y) ** 2).item())
        loss_raw = round(loss_raw, 4)
        loss_rmse_raw = round(loss_rmse_raw, 4)
        print(f'MAE:{loss}/RMSE:{loss_rmse}||MAE:{loss_raw}/RMSE:{loss_rmse_raw}')
        if save_model_flag:
            root_path = 'G:\PyCharmWorkSpace\PyCharmProjects\Air quality\Air Quality prediction Beijing\saved_model'
            model_name = 'MLP'
            save_path = os.path.join(root_path, model_name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            file_name = f'{model_name} seed={seed} {target_feature} {seq_len}_{pred_len} {(loss, loss_rmse, loss_raw, loss_rmse_raw)}.pt'
            file_path = os.path.join(save_path, file_name)
            torch.save(regr, file_path)
            print('save model successful!')
        return loss, loss_rmse, loss_raw, loss_rmse_raw

    def run(start_seed, end_seed):
        result = {}
        for seed in range(start_seed, end_seed + 1):
            print(f'use device:{DEVICE}\t\trandom seed:{seed}')
            loss, loss_rmse, loss_raw, loss_rmse_raw = train(test_X, test_Y, seed)
            result[seed] = (loss, loss_rmse, loss_raw, loss_rmse_raw)

        for item in result.items():
            print(f'result on random seed{item}')

    run(start_seed=start_seed, end_seed=end_seed)