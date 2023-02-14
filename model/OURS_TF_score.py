# python3.7
# -*- coding: utf-8 -*-
# @Project_Name  : ETDdata Forcasting
# @File          : OURS_TF_score.py
# @Time          : 2022/11/6 20:51
# @Author        : SY.M
# @Software      : PyCharm


import torch
import math
import numpy as np


class OURS(torch.nn.Module):
    def __init__(self, seq_len, d_feature, d_embedding, d_hidden, pred_len,
                 dropout, q, k, v, h, N, pe=True, mask=True):
        super(OURS, self).__init__()

        self.d_time = seq_len + pred_len
        self.d_output = pred_len
        self.standard_layer = torch.nn.BatchNorm1d(num_features=d_feature)
        # self.standard_layer = torch.nn.InstanceNorm1d(num_features=d_feature)

        self.input_time = Input_Layer(second_dim=self.d_time, third_dim=d_feature,
                                      d_embedding=d_embedding, pe=pe, dropout=dropout)
        self.input_feature = Input_Layer(second_dim=d_feature, third_dim=self.d_time,
                                         d_embedding=d_embedding, pe=pe, dropout=dropout)
        self.encoder_time = torch.nn.ModuleList([Encoder(d_embedding=d_embedding,
                                                         d_hidden=d_hidden, dropout=dropout,
                                                         q=q, k=k, v=v, h=h, mask=mask) for _ in range(N[0])])
        self.encoder_feature = torch.nn.ModuleList([Encoder(d_embedding=d_embedding,
                                                            d_hidden=d_hidden, dropout=dropout,
                                                            q=q, k=k, v=v, h=h, mask=mask) for _ in range(N[0])])
        self.encoder_TF = torch.nn.ModuleList([Encoder(d_embedding=d_embedding,
                                                       d_hidden=d_hidden, dropout=dropout,
                                                       q=q, k=k, v=v, h=h, mask=mask) for _ in range(N[1])])
        self.encoder_FT = torch.nn.ModuleList([Encoder(d_embedding=d_embedding,
                                                       d_hidden=d_hidden, dropout=dropout,
                                                       q=q, k=k, v=v, h=h, mask=mask) for _ in range(N[1])])
        self.output = Output_layer(d_embedding=d_embedding, d_feature=d_feature, d_time=self.d_time, d_output=pred_len)

    def forward(self, enc_in, enc_stamp, dec_in, dec_stamp, pred_target):

        mean = torch.mean(enc_in[:, :, pred_target], dim=-1).reshape(enc_in.shape[0], 1)
        # mean = x[:, -1, -1].reshape(x.shape[0], 1)

        # cat 0
        enc_in = dec_in
        enc_stamp = dec_stamp

        enc_in = self.standard_layer(enc_in.transpose(-1, -2)).transpose(-1, -2)

        x_time = self.input_time(enc_in, x_mask=enc_stamp)
        x_feature = self.input_feature(enc_in.transpose(-1, -2))

        for encoder in self.encoder_time:
            x_time, t_score = encoder(x_time, x_time, x_time)
        for encoder in self.encoder_feature:
            x_feature, f_score = encoder(x_feature, x_feature, x_feature)
        for encoder in self.encoder_TF:
            x_time_feature, tf_score = encoder(x_time, x_feature, x_feature)
        for encoder in self.encoder_FT:
            x_feature_time, ft_score = encoder(x_feature, x_time, x_time)

        x = self.output(x_time=x_time_feature, x_time_feature=x_feature_time, message=mean)

        return x, (t_score, f_score, tf_score, ft_score)


class Input_Layer(torch.nn.Module):
    def __init__(self, second_dim, third_dim, d_embedding, dropout, pe=True):
        super(Input_Layer, self).__init__()
        # self.token_embedding = torch.nn.Linear(in_features=third_dim, out_features=d_embedding)
        self.token_embedding = TokenEmbedding(second_dim=second_dim, third_dim=third_dim,
                                              d_embedding=d_embedding, dropout=dropout)
        # self.token_embedding = Multi_Channel_Embedding(second_dim=second_dim, third_dim=third_dim, d_embedding=d_embedding)
        # self.time_embedding = torch.nn.Linear(in_features=4, out_features=d_embedding)
        self.time_embedding = TemporalEmbedding(d_embedding=d_embedding)
        self.pe = pe

    def forward(self, x, x_mask=None):
        x = self.token_embedding(x)
        if self.pe:
            x += self.position_encode(x)
        if x_mask is not None:
            global_embedding = self.time_embedding(x_mask)
            x += global_embedding
        return x

    def position_encode(self, x):
        pe = torch.ones_like(x[0])
        position = torch.arange(0, x.shape[1]).unsqueeze(-1)
        temp = torch.Tensor(range(0, x.shape[-1], 2))
        temp = temp * -(math.log(10000) / x.shape[-1])
        temp = torch.exp(temp).unsqueeze(0)
        temp = torch.matmul(position.float(), temp)  # shape:[input, d_model/2]
        pe[:, 0::2] = torch.sin(temp)
        pe[:, 1::2] = torch.cos(temp)
        return pe


class TokenEmbedding(torch.nn.Module):
    def __init__(self, second_dim, third_dim, d_embedding, dropout):
        super(TokenEmbedding, self).__init__()
        self.embedding = torch.nn.Conv1d(in_channels=third_dim, out_channels=d_embedding, kernel_size=1)
        self.dropout = torch.nn.Dropout(dropout)
        self.norm = torch.nn.LayerNorm(d_embedding)

    def forward(self, x):
        x = self.embedding(x.transpose(-1, -2)).transpose(-1, -2)
        x = self.dropout(x)
        # x = self.norm(x)
        return x


class TemporalEmbedding(torch.nn.Module):
    def __init__(self, d_embedding, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = torch.nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_embedding)
        self.hour_embed = Embed(hour_size, d_embedding)
        self.weekday_embed = Embed(weekday_size, d_embedding)
        self.day_embed = Embed(day_size, d_embedding)
        self.month_embed = Embed(month_size, d_embedding)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        out = hour_x + weekday_x + day_x + month_x + minute_x
        return out


# class Time_Feature_Score(torch.nn.Module):
#     def __init__(self, d_model, q, k, v, h, mask=False):
#         super(Time_Feature_Score, self).__init__()
#         self.h = h
#         self.mask = mask
#         self.Q = torch.nn.Linear(d_model, q * h)
#         self.K = torch.nn.Linear(d_model, q * h)
#
#     def forward(self, time_ebed, feature_ebed, feature_value):
#
#         Qs = torch.stack(torch.chunk(self.Q(time_ebed), self.h, dim=-1), dim=1)
#         Ks = torch.stack(torch.chunk(self.Q(time_ebed), self.h, dim=-1), dim=1)
#
#         score = torch.matmul(Qs, Ks)
#
#         if self.mask:
#             mask = torch.ones_like(score[0])
#             mask = mask.tril(diagonal=0)
#             score = torch.where(mask > 0, score, (torch.ones_like(mask) * self.inf).to(DEVICE))
#
#         score = torch.cat(torch.chunk(score, self.h, dim=1), dim=-1)


class Encoder(torch.nn.Module):
    def __init__(self, d_embedding, d_hidden, q, k, v, h, dropout, mask=True):
        super(Encoder, self).__init__()
        self.MHA = MultiHeadAttention(d_embedding=d_embedding, q=q, k=k, v=v, h=h, mask=mask)
        self.relu = torch.nn.ReLU()
        self.gelu = torch.nn.GELU()
        self.dropout = torch.nn.Dropout(dropout)
        self.layernorm_1 = torch.nn.LayerNorm(d_embedding)
        self.layernorm_2 = torch.nn.LayerNorm(d_embedding)

        self.feedforward = torch.nn.Sequential(torch.nn.Dropout(dropout),
                                               torch.nn.Conv1d(in_channels=d_embedding, out_channels=d_hidden,
                                                               kernel_size=1),
                                               torch.nn.ReLU(),
                                               torch.nn.Dropout(dropout),
                                               torch.nn.Conv1d(in_channels=d_hidden, out_channels=d_embedding,
                                                               kernel_size=1))

    def forward(self, query, key, value):
        residual = query
        x, score = self.MHA(query, key, value)
        x = self.dropout(x)
        y = x = self.layernorm_1(x + residual)
        # x = self.relu(x)

        y = self.feedforward(y.transpose(-1, -2)).transpose(-1, -2)
        x = self.layernorm_2(x + y)

        return x, score


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_embedding, q, k, v, h, mask=True):
        super(MultiHeadAttention, self).__init__()

        self.W_Q = torch.nn.Linear(in_features=d_embedding, out_features=h * q)
        self.W_K = torch.nn.Linear(in_features=d_embedding, out_features=h * k)
        self.W_V = torch.nn.Linear(in_features=d_embedding, out_features=h * v)

        self.mask = mask
        self.h = h
        self.inf = -2 ** 32 + 1

        self.out_linear = torch.nn.Linear(v * h, d_embedding)

    def forward(self, query, key, value):
        Q = torch.cat(torch.chunk(self.W_Q(query), self.h, dim=-1), dim=0)
        K = torch.cat(torch.chunk(self.W_K(key), self.h, dim=-1), dim=0)
        V = torch.cat(torch.chunk(self.W_V(value), self.h, dim=-1), dim=0)

        score = torch.matmul(Q, K.transpose(-1, -2))

        if self.mask:
            mask = torch.ones_like(score[0])
            mask = mask.tril(diagonal=0)
            score = torch.where(mask > 0, score, (torch.ones_like(mask) * self.inf).to(DEVICE))

        score = torch.softmax(score, dim=-1)

        attention = torch.cat(torch.chunk(torch.matmul(score, V), self.h, dim=0), dim=-1)

        out = self.out_linear(attention)

        return out, score


class Output_layer(torch.nn.Module):
    def __init__(self, d_embedding, d_feature, d_time, d_output):
        super(Output_layer, self).__init__()
        # 分别预测并加权求和
        self.output_time = torch.nn.Linear(in_features=d_embedding * d_time, out_features=d_output)
        self.output_time_feature = torch.nn.Linear(in_features=d_embedding * d_feature, out_features=d_output)
        self.weight = torch.nn.Parameter(torch.Tensor([0.33, 0.33, 0.33, 0]))

        # 拼接并进行映射
        self.output = torch.nn.Linear(in_features=d_embedding * (d_time + d_feature), out_features=d_output)

    def forward(self, x_time, x_time_feature, message):
        # 分别预测并加权求和
        x_time = self.output_time(x_time.reshape(x_time.shape[0], -1))
        x_time_feature = self.output_time_feature(x_time_feature.reshape(x_time_feature.shape[0], -1))
        x = x_time_feature * self.weight[0] + x_time * self.weight[1] + self.weight[-1] + message * self.weight[2]

        # 拼接并进行映射
        # x = torch.cat([x_time, x_feature], dim=1)
        # x = torch.einsum('bcd->be', x)
        # x = self.output(x)

        return x


# Test
def test(dataloader, test_net, pred_len, d_token, pred_target, f_std=None,
        DEVICE='cuda:0' if torch.cuda.is_available() else 'cpu'):
    test_net.eval()
    pres = []
    ys = []
    with torch.no_grad():
        for x, x_stamp, y, y_stamp in dataloader:
            x = x.float().to(DEVICE)
            y = y.float()[:, :, pred_target].to(DEVICE)
            x_stamp = x_stamp.float().to(DEVICE)
            y_stamp = y_stamp.float().to(DEVICE)

            enc_in = x
            enc_in_stamp = x_stamp
            # padding = torch.cat([torch.zeros(x.shape[0], pred_len, 10), -torch.ones(x.shape[0], pred_len, 2)], dim=-1)
            padding = torch.zeros(x.shape[0], pred_len, x.shape[-1])
            dec_in = torch.cat([x[:, -d_token:, :], padding.to(DEVICE)], dim=1)
            dec_in_stamp = torch.cat([x_stamp[:, -d_token:, :], y_stamp], dim=1)

            pre, scores = test_net(enc_in, enc_in_stamp, dec_in, dec_in_stamp, pred_target=pred_target)

            pre = pre.reshape(-1, pre.shape[-1])
            y = y.reshape(-1, y.shape[-1])
            pres.append(pre.detach().cpu().numpy())
            ys.append(y.detach().cpu().numpy())

        pres = np.concatenate(pres, axis=0)
        ys = np.concatenate(ys, axis=0)
        # if do_inverse:
        #     pres = pres * f_std + f_mean
        #     ys = ys * f_std + f_mean
        loss = round(np.mean(np.abs(pres - ys)).item(), 4)
        loss_rmse = round(np.sqrt(np.mean((pres - ys) ** 2).item()), 4)

        # comapre_line(sequence=dataloader.dataset.X[0, :, pred_target],
        #              pre=pres[0], true=ys[0], dataset_type=dataloader.dataset.dataset_type)

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
    from utils.visualization import draw_line, comapre_line

    feature_index = {'PM2.5': 0, 'PM10': 1, 'NO2': 2, 'CO': 3, 'O3': 4, 'SO2': 5,
                     'TEMP': 6, 'PRES': 7, 'humidity': 8, 'wind_speed': 9, 'weather': 10, 'WD': 11}
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # experiment settings
    seed_list = [30, ]
    seq_len_list = [168, ]
    pred_len_list = [120, ]
    # save_model_situation = [12, 24, 48, 72, 96, 120, 144]
    save_model_situation = [12, ]

    scalar = True
    BATCHSIZE = 32
    EPOCH = 10000
    test_interval = 1
    update_interval = 5
    target_feature = 'PM2.5'
    pred_target = feature_index[target_feature]
    # Hyper Parameters
    d_feature = 12
    d_embedding = 512
    d_hidden = 512
    q = 8
    k = 8
    v = 8
    h = 64
    N = (2, 1)
    LR = 1e-4
    LR_weight = 1e-3
    pe = True
    mask = False
    dropout = 0.05

    # Train
    def train(seed, seq_len, pred_len, save_model_flag, train_dataloader,
              val_dataloader, test_dataloader, f_std):
        d_token = seq_len

        # initialize Model
        net = OURS(seq_len=seq_len, d_feature=d_feature, d_embedding=d_embedding, d_hidden=d_hidden, pred_len=pred_len,
                   q=q,k=k, v=v, h=h, N=N, pe=pe, mask=mask, dropout=dropout).to(DEVICE)
        early_stop = Eearly_stop(patience=10, save_model_flag=save_model_flag)

        # Build optimizer and loss_function
        optimizer = torch.optim.Adam([{'params': net.input_time.parameters()},
                                      {'params': net.input_feature.parameters()},
                                      {'params': net.encoder_time.parameters()},
                                      {'params': net.encoder_TF.parameters()},
                                      {'params': net.encoder_FT.parameters()},
                                      {'params': net.output.output_time.parameters()},
                                      {'params': net.output.output_time_feature.parameters()},
                                      {'params': net.output.weight, 'lr': LR_weight}], lr=LR)
        loss_func = torch.nn.L1Loss(reduction='sum')
        # loss_func = torch.nn.MSELoss(reduction='sum')

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

                enc_in = x.to(DEVICE)
                enc_in_stamp = x_stamp
                # padding = torch.cat([torch.zeros(x.shape[0], pred_len, 10), -torch.ones(x.shape[0], pred_len, 2)], dim=-1)
                padding = torch.zeros(x.shape[0], pred_len, x.shape[-1])
                dec_in = torch.cat([x[:, -d_token:, :], padding.to(DEVICE)], dim=1)
                dec_in_stamp = torch.cat([x_stamp[:, -d_token:, :], y_stamp], dim=1)

                pre, scores = net(enc_in, enc_in_stamp, dec_in, dec_in_stamp, pred_target=pred_target)
                loss = loss_func(pre, y)
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
            train_loss = round(epoch_loss / train_dataloader.dataset.X.shape[0] / pred_len, 4)
            print(f'EPOCH:{epoch_idx}\t\tTRAIN_LOSS:{round(train_loss, 4)}')

            if epoch_idx % test_interval == 0:
                val_loss, _ = test(dataloader=val_dataloader, test_net=net,
                                                   pred_len=pred_len, d_token=d_token,
                                                   pred_target=pred_target)
                test_loss, test_loss_rmse = test(dataloader=test_dataloader, test_net=net,
                                                   pred_len=pred_len, d_token=d_token,
                                                   pred_target=pred_target)
                print(f'EPOCH:{epoch_idx}\t\tVAL_LOSS:{val_loss}'
                      f'\t\tTEST_LOSS:{test_loss}/{test_loss_rmse}||'
                      f'{round(test_loss*f_std if scalar else 0, 4)}/{round(test_loss_rmse*f_std if scalar else 0, 4)}', end='\t\t')
                if early_stop(val_loss, net, type='e'):
                    val_loss, val_loss_rmse = test(dataloader=val_dataloader, test_net=early_stop.net,
                                                   pred_len=pred_len, d_token=d_token,
                                                   pred_target=pred_target)
                    test_loss, test_loss_rmse = test(dataloader=test_dataloader, test_net=early_stop.net,
                                                     pred_len=pred_len, d_token=d_token,
                                                     pred_target=pred_target)
                    train_loss, train_loss_rmse = test(dataloader=train_dataloader, test_net=early_stop.net,
                                                       pred_len=pred_len, d_token=d_token,
                                                       pred_target=pred_target)
                    print(f'accuracy on Train:\033[0;34mMAE={train_loss}/{train_loss_rmse}||'
                          f'{round(train_loss*f_std, 4)}/{round(train_loss_rmse*f_std, 4)}\033[0m\r\n'
                          f'accuracy on Val:\033[0;34mMAE={val_loss}/{val_loss_rmse}||'
                          f'{round(val_loss*f_std, 4)}/{round(val_loss_rmse*f_std, 4)}\033[0m\r\n'
                          f'accuracy on Test:\033[0;34mMAE={test_loss}/{test_loss_rmse}||'
                          f'{round(test_loss*f_std, 4)}/{round(test_loss_rmse*f_std, 4)}\033[0m\r\n')
                    if save_model_flag:
                        result = (test_loss, test_loss_rmse, round(test_loss * f_std, 2), round(test_loss_rmse * f_std, 2))
                        early_stop.save_model(model_name='TFAN', target_feature=target_feature, seed=seed,
                                              result=result, pred_len=pred_len, seq_len=seq_len,
                                              state_dict={'d_feature': d_feature, 'seq_len': seq_len, 'd_embedding': d_embedding,
                                                          'd_hidden': d_hidden, 'pred_len': pred_len, 'q': q, 'k': k, 'v': v, 'h': h,
                                                          'N': N, 'pe': pe, 'mask': mask, 'dropout': dropout, 'pred_target': pred_target})
                    return test_loss, test_loss_rmse

            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)
            test_loss_list.append(test_loss)

            if (epoch_idx + 1) % update_interval == 0:
                draw_line(title="OURS", train_loss=train_loss_list,
                          val_loss=val_loss_list, test_loss=test_loss_list)

            # draw_heatmap(scores[0])


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
                          f'超参: Batch={BATCHSIZE} d_embedding={d_embedding} d_hidden={d_hidden} q=k=v={q} '
                          f'h={h} N={N} Dropout={dropout} LR={LR} LR_weight={LR_weight}'
                          f'\r\n Seq_len={seq_len} Pred_len={pred_len}')
                    print(f'use device:{DEVICE}\t\trandom seed:{seed}')
                    setup_seed(seed)
                    mae, rmse = train(seed=seed, seq_len=seq_len, pred_len=pred_len, f_std=f_std,
                                      save_model_flag=save_model_flag, val_dataloader=val_dataloader,
                                      train_dataloader=train_dataloader, test_dataloader=test_dataloader)
                    if scalar:
                        result[f'{target_feature} seq_len={seq_len} pred_len={pred_len} seed={seed}'] = (mae, rmse, round(mae * f_std, 4), round(rmse * f_std, 4))
                    else:
                        result[f'{target_feature} seq_len={seq_len} pred_len={pred_len} seed={seed}'] = (mae, rmse)
        for item in result.items():
            print(f'result on random seed{item}')

    run(seed_list=seed_list, seq_len_list=seq_len_list, pred_len_list=pred_len_list)
