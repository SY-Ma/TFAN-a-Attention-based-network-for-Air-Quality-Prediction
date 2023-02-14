# python3.7
# -*- coding: utf-8 -*-
# @Project_Name  : Air Quality prediction Beijing
# @File          : Repeat.py
# @Time          : 2022/11/10 16:03
# @Author        : SY.M
# @Software      : PyCharm


import torch


class Repeat(torch.nn.Module):
    def __init__(self, d_output):
        super(Repeat, self).__init__()
        self.d_output = d_output

    def forward(self, x, feature_idx=0):
        pre = x[:, -1, feature_idx].unsqueeze(-1)
        pre = pre.expand(pre.shape[0], self.d_output)
        return pre


if __name__ == '__main__':
    from data_process.data_processer import DataProcesser
    import torch
    from utils.random_seed import setup_seed
    from utils.early_stop import Eearly_stop
    from torch.utils.data import DataLoader
    import numpy as np
    from utils.visualization import draw_line

    feature_index = {'PM2.5': 0, 'PM10': 1, 'NO2': 2, 'CO': 3, 'O3': 4, 'SO2': 5,
                     'TEMP': 6, 'PRES': 7, 'humidity': 8, 'wind_speed': 9, 'weather': 10, 'WD': 11}
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    seed = 30
    print(f'use device:{DEVICE}\t\trandom seed:{seed}')
    setup_seed(seed)
    BATCHSIZE = 256
    EPOCH = 10000
    test_interval = 1
    update_interval = 999
    save_model_flag = False
    pred_target = feature_index['SO2']
    seq_len = 168
    pred_len = 96

    scalar = False
    print('loading data...')
    # train_dataset = DataProcesser(scalar=scalar, seq_len=seq_len, pred_len=pred_len, dataset_type='train')
    # val_dataset = DataProcesser(scalar=scalar, seq_len=seq_len, pred_len=pred_len, dataset_type='val')
    test_dataset = DataProcesser(scalar=scalar, seq_len=seq_len, pred_len=pred_len, dataset_type='test')
    # train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCHSIZE, shuffle=True)
    # val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCHSIZE, shuffle=False)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCHSIZE, shuffle=False)
    standard_scalar = test_dataset.scalar

    # Build optimizer and loss_function
    loss_func = torch.nn.L1Loss(reduction='sum')
    # loss_func_mse = torch.nn.MSELoss(reduction='sum')

    net = Repeat(d_output=pred_len)

    # Test
    def test(dataloader, test_net):
        net.eval()
        pres = []
        ys = []
        sample_num = 0
        with torch.no_grad():
            for x, x_stamp, y, y_stamp in dataloader:
                sample_num += x.shape[0]
                pre = test_net(x.to(DEVICE), feature_idx=pred_target)
                # loss = torch.sum(torch.abs(pre-y.to(DEVICE))).item()

                pre = pre.reshape(-1, pre.shape[-1])
                y = y[:, :, pred_target]
                pres.append(pre.detach().cpu().numpy())
                ys.append(y.detach().cpu().numpy())
            pres = np.concatenate(pres, axis=0)
            ys = np.concatenate(ys, axis=0)
            loss = np.mean(np.abs(pres - ys)).item()
            loss_rmse = np.sqrt(np.mean((pres - ys) ** 2).item())
            loss = round(loss, 4)
            loss_rmse = round(loss_rmse, 4)
            return loss, loss_rmse

    loss, loss_rmse = test(dataloader=test_dataloader, test_net=net)
    print(f'MAE={loss}\t\tRMSE={loss_rmse}')