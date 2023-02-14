# python3.7
# -*- coding: utf-8 -*-
# @Project_Name  : Air Quality prediction Beijing
# @File          : LSTM.py
# @Time          : 2022/11/11 15:23
# @Author        : SY.M
# @Software      : PyCharm

import torch


class LSTM_Model(torch.nn.Module):
    def __init__(self, seq_len, pred_len, feature_nums, hidden_len, num_layers):
        super(LSTM_Model, self).__init__()
        self.lstm = torch.nn.LSTM(hidden_size=hidden_len, input_size=feature_nums,
                                  num_layers=num_layers, batch_first=True)
        self.output_layer = torch.nn.Linear(in_features=hidden_len * seq_len, out_features=pred_len)

    def forward(self, x):
        # h_0 is zero tensor if it is not defined during initialization
        output, _ = self.lstm(x)
        output = self.output_layer(output.reshape(output.shape[0], -1))

        return output

# Test
def test(dataloader, test_net, pred_target, f_std=None,
        DEVICE='cuda:0' if torch.cuda.is_available() else 'cpu'):
    test_net.eval()
    pres = []
    ys = []
    sample_num = 0
    with torch.no_grad():
        for x, x_stamp, y, y_stamp in dataloader:
            x = x.float().to(DEVICE)
            y = y.float()[:, :, pred_target].to(DEVICE)
            x_stamp = x_stamp.float().to(DEVICE)
            y_stamp = y_stamp.float().to(DEVICE)

            pre = test_net(x)

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

    feature_index = {'PM2.5': 0, 'PM10': 1, 'NO2': 2, 'CO': 3, 'O3': 4, 'SO2': 5,
                     'TEMP': 6, 'PRES': 7, 'humidity': 8, 'wind_speed': 9, 'weather': 10, 'WD': 11}
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # experiment settings
    seed_list = [30, ]
    seq_len_list = [168, ]
    pred_len_list = [24, 72, 120, ]
    save_model_situation = [12, 24, 48, 72, 96, 120, 144]

    target_feature = 'O3'
    pred_target = feature_index[target_feature]

    scalar = True
    BATCHSIZE = 256
    EPOCH = 10000
    test_interval = 1
    update_interval = 999
    hidden_len = 512
    num_layers = 1
    d_feature = 12

    # Train
    def train(seed, pred_len, seq_len, train_dataloader, test_dataloader,
              val_dataloader, f_std, save_model_flag):
        net = LSTM_Model(feature_nums=d_feature, pred_len=pred_len,
                         seq_len=seq_len, hidden_len=hidden_len, num_layers=num_layers).to(DEVICE)
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
                pre = net(x)
                loss = loss_func(pre, y)
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
            train_loss = round(epoch_loss / train_dataloader.dataset.X.shape[0] / pred_len, 4)
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
                        early_stop.save_model(model_name='LSTM', target_feature=target_feature, seed=seed,
                                              result=result, pred_len=pred_len, seq_len=seq_len,
                                              state_dict={'feature_nums': d_feature, 'pred_len': pred_len, 'pred_target': pred_target,
                                              'seq_len': seq_len, 'hidden_len': hidden_len, 'num_layers': num_layers})
                    return test_loss, test_loss_rmse

            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)
            test_loss_list.append(test_loss)

            if epoch_idx % update_interval == 0:
                draw_line(title="LSTM", train_loss=train_loss_list,
                          val_loss=val_loss_list, test_loss=test_loss_list)


    def run(seed_list, seq_len_list, pred_len_list):
        result = {}
        for seq_len in seq_len_list:
            for pred_len in pred_len_list:
                # building dataset
                save_model_flag = True if pred_len in save_model_situation else False

                print('loading data...')
                train_dataset = DataProcesser(scalar=scalar, seq_len=seq_len, pred_len=pred_len, dataset_type='train')
                val_dataset = DataProcesser(scalar=scalar, seq_len=seq_len, pred_len=pred_len, dataset_type='val')
                test_dataset = DataProcesser(scalar=scalar, seq_len=seq_len, pred_len=pred_len, dataset_type='test')
                train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCHSIZE, shuffle=True)
                val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCHSIZE, shuffle=False)
                test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCHSIZE, shuffle=False)
                standard_scalar = train_dataset.scalar if scalar else None
                if scalar:
                    f_mean = standard_scalar.mean_[pred_target]
                    f_std = np.sqrt(standard_scalar.var_[pred_target])
                else:
                    f_mean = f_std = 1

                for seed in seed_list:
                    print(f'Predicted Target:{target_feature}\r\n'
                          f'\r\n Seq_len={seq_len} Pred_len={pred_len}')
                    print(f'use device:{DEVICE}\t\trandom seed:{seed}')
                    setup_seed(seed)
                    mae, rmse = train(seed, pred_len=pred_len, seq_len=seq_len, train_dataloader=train_dataloader,
                                      val_dataloader=val_dataloader, test_dataloader=test_dataloader,
                                      f_std=f_std, save_model_flag=save_model_flag)
                    if scalar:
                        result[f'{target_feature} seq_len={seq_len} pred_len={pred_len} seed={seed}'] = (mae, rmse, round(mae * f_std, 4), round(rmse * f_std, 4))
                    else:
                        result[f'{target_feature} seq_len={seq_len} pred_len={pred_len} seed={seed}'] = (mae, rmse)
        for item in result.items():
                    print(f'result on random seed{item}')

    run(seed_list=seed_list, seq_len_list=seq_len_list, pred_len_list=pred_len_list)
