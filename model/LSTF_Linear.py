# python3.7
# -*- coding: utf-8 -*-
# @Project_Name  : Air Quality prediction Beijing
# @File          : LSTF_Linear.py
# @Time          : 2022/11/9 18:43
# @Author        : SY.M
# @Software      : PyCharm

import torch
import torch.nn as nn
import numpy as np
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class LSTF_Linear(torch.nn.Module):
    def __init__(self, feature_num, seq_len, pred_len):
        super(LSTF_Linear, self).__init__()
        self.linear = torch.nn.Linear(in_features=seq_len, out_features=pred_len)

    def forward(self, x):
        x = self.linear(x.transpose(-1, -2))
        return x.permute(0, 2, 1)  # [Batch, Sequence_length, channels]


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class LSTF_DLinear(nn.Module):
    """
    Decomposition-Linear
    """

    def __init__(self, seq_len, pred_len, individual, enc_in):
        super(LSTF_DLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.individual = individual
        self.channels = enc_in

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))

                # Use this two lines if you want to visualize the weights
                # self.Linear_Seasonal[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                # self.Linear_Trend[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

            # Use this two lines if you want to visualize the weights
            # self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
            # self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
                                       dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)  # to [Batch, Output length, Channel]


class LSTF_Nlinear(nn.Module):
    """
    Normalization-Linear
    """

    def __init__(self, seq_len, pred_len):
        super(LSTF_Nlinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.Linear = nn.Linear(self.seq_len, self.pred_len)
        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last
        x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + seq_last
        return x  # [Batch, Output length, Channel]


# Test
def test(dataloader, test_net, pred_target, f_std=None,
        DEVICE='cuda:0' if torch.cuda.is_available() else 'cpu'):
    test_net.eval().to(DEVICE)
    pres = []
    ys = []
    sample_num = 0
    with torch.no_grad():
        for x, x_stamp, y, y_stamp in dataloader:
            x = x.float().to(DEVICE)
            y = y.float()[:, :, pred_target].to(DEVICE)
            x_stamp = x_stamp.float().to(DEVICE)
            y_stamp = y_stamp.float().to(DEVICE)

            pre = test_net(x)[:, :, pred_target]

            pre = pre.reshape(-1, pre.shape[-1])
            y = y.reshape(-1, y.shape[-1])
            pres.append(pre.detach().cpu().numpy())
            ys.append(y.detach().cpu().numpy())
        pres = np.concatenate(pres, axis=0)
        ys = np.concatenate(ys, axis=0)
        loss = np.mean(np.abs(pres - ys)).item()
        loss_rmse = np.sqrt(np.mean((pres - ys) ** 2).item())
        loss = round(loss, 4)
        loss_rmse = round(loss_rmse, 4)
        if f_std is not None:
            return loss, loss_rmse, round(loss * f_std, 4), round(loss_rmse * f_std, 4)
        return loss, loss_rmse


if __name__ == '__main__':
    from data_process.data_processer import DataProcesser
    import torch
    from utils.random_seed import setup_seed
    from utils.early_stop import Eearly_stop
    from torch.utils.data import DataLoader
    import numpy as np
    from utils.visualization import draw_line
    from utils.save_model import save_model

    feature_index = {'PM2.5': 0, 'PM10': 1, 'NO2': 2, 'CO': 3, 'O3': 4, 'SO2': 5,
                     'TEMP': 6, 'PRES': 7, 'humidity': 8, 'wind_speed': 9, 'weather': 10, 'WD': 11}
    start_seed = 30
    end_seed = 30

    BATCHSIZE = 256
    EPOCH = 10000
    test_interval = 1
    update_interval = 999
    save_model_flag = True
    target_feature = 'CO'
    pred_target = feature_index[target_feature]
    seq_len = 168
    pred_len = 120

    scalar = True
    print('loading data...')
    train_dataset = DataProcesser(scalar=scalar, seq_len=seq_len, pred_len=pred_len, dataset_type='train')
    val_dataset = DataProcesser(scalar=scalar, seq_len=seq_len, pred_len=pred_len, dataset_type='val')
    test_dataset = DataProcesser(scalar=scalar, seq_len=seq_len, pred_len=pred_len, dataset_type='test')
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCHSIZE, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCHSIZE, shuffle=False)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCHSIZE, shuffle=False)
    standard_scalar = train_dataset.scalar if scalar else None
    if standard_scalar is not None:
        f_mean = standard_scalar.mean_[pred_target]
        f_std = np.sqrt(standard_scalar.var_[pred_target])
    d_feature = train_dataset.X.shape[-1]


    # Train
    def train(seed):
        # build model
        # net = LSTF_Linear(feature_num=d_feature, pred_len=pred_len, seq_len=seq_len).to(DEVICE)
        # net = LSTF_Nlinear(pred_len=pred_len, seq_len=seq_len).to(DEVICE)
        net = LSTF_DLinear(enc_in=d_feature, pred_len=pred_len, seq_len=seq_len, individual=True).to(DEVICE)
        early_stop = Eearly_stop(patience=10, save_model_flag=save_model_flag)
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
        loss_func = torch.nn.L1Loss(reduction='sum')

        train_loss_list = []
        val_loss_list = []
        test_loss_list = []
        # pbar = tqdm(total=EPOCH)
        for epoch_idx in range(EPOCH):
            # pbar.update()
            net.train()
            epoch_loss = 0.0
            for x, x_stamp, y, y_stamp in train_dataloader:
                optimizer.zero_grad()
                x = x.float().to(DEVICE)
                y = y.float()[:, :, pred_target].to(DEVICE)
                x_stamp = x_stamp.float().to(DEVICE)
                y_stamp = y_stamp.float().to(DEVICE)
                pre = net(x)[:, :, pred_target]
                loss = loss_func(pre, y)
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
            train_loss = round(epoch_loss / train_dataset.X.shape[0] / pred_len, 4)
            print(f'EPOCH:{epoch_idx}\t\tTRAIN_LOSS:{round(train_loss, 4)}')

            if epoch_idx % test_interval == 0:
                val_loss, _ = test(dataloader=val_dataloader, test_net=net, pred_target=pred_target)
                test_loss, test_loss_rmse = test(dataloader=test_dataloader, test_net=net, pred_target=pred_target)
                print(f'EPOCH:{epoch_idx}\t\tVAL_LOSS:{val_loss}'
                      f'\t\tTEST_LOSS:{test_loss}/{test_loss_rmse}||'
                      f'{round(test_loss * f_std, 4)}/{round(test_loss_rmse * f_std, 4)}', end='\t\t')
                if early_stop(val_loss, net, type='e'):
                    val_loss, val_loss_rmse = test(dataloader=val_dataloader, test_net=early_stop.net, pred_target=pred_target)
                    test_loss, test_loss_rmse = test(dataloader=test_dataloader, test_net=early_stop.net, pred_target=pred_target)
                    train_loss, train_loss_rmse = test(dataloader=train_dataloader, test_net=early_stop.net, pred_target=pred_target)
                    print(f'accuracy on Train:\033[0;34mMAE={train_loss}/{train_loss_rmse}||'
                          f'{round(train_loss * f_std, 4)}/{round(train_loss_rmse * f_std, 4)}\033[0m\r\n'
                          f'accuracy on Val:\033[0;34mMAE={val_loss}/{val_loss_rmse}||'
                          f'{round(val_loss * f_std, 4)}/{round(val_loss_rmse * f_std, 4)}\033[0m\r\n'
                          f'accuracy on Test:\033[0;34mMAE={test_loss}/{test_loss_rmse}||'
                          f'{round(test_loss * f_std, 4)}/{round(test_loss_rmse * f_std, 4)}\033[0m\r\n')
                    if save_model_flag:
                        result = (test_loss, test_loss_rmse, round(test_loss * f_std, 2), round(test_loss_rmse * f_std, 2))
                        early_stop.save_model(model_name='LSTF-Linear', target_feature=target_feature, seed=seed,
                                              result=result, pred_len=pred_len, seq_len=seq_len,
                                              state_dict={'d_feature': d_feature, 'pred_len': pred_len,
                                                          'seq_len': seq_len, 'pred_target': pred_target})
                    return test_loss, test_loss_rmse

            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)
            test_loss_list.append(test_loss)

            if (epoch_idx + 1) % update_interval == 0:
                draw_line(title="LSTF-Linear", train_loss=train_loss_list,
                          val_loss=val_loss_list, test_loss=test_loss_list)

    def run(start_seed, end_seed):
        result = {}
        for seed in range(start_seed, end_seed + 1):
            print(f'use device:{DEVICE}\t\trandom seed:{seed}')
            setup_seed(seed)
            mae, rmse = train(seed)
            result[seed] = (mae, rmse, round(mae*f_std, 4), round(rmse*f_std, 4))
        print(f'Predicted Target:{target_feature}\tSeq_len:{seq_len}\tPred_len:{pred_len}')
        for item in result.items():
            print(f'result on random seed{item}')

    run(start_seed=start_seed, end_seed=end_seed)
