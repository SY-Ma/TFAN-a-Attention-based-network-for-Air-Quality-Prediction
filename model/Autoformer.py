# python3.7
# -*- coding: utf-8 -*-
# @Project_Name  : Air Quality prediction Beijing
# @File          : Informer.py
# @Time          : 2022/11/10 15:49
# @Author        : SY.M
# @Software      : PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F
# from layers.Embed import DataEmbedding, DataEmbedding_wo_pos
# from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
# from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
import math
import numpy as np

class my_Layernorm(nn.Module):
    """
    Special designed layernorm for the seasonal part
    """
    def __init__(self, channels):
        super(my_Layernorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias


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


class EncoderLayer(nn.Module):
    """
    Informer encoder layer with the progressive decomposition architecture
    """
    def __init__(self, attention, d_model, d_ff=None, moving_avg=25, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        x, _ = self.decomp1(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        res, _ = self.decomp2(x + y)
        return res, attn


class Encoder(nn.Module):
    """
    Informer encoder
    """
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    """
    Informer decoder layer with the progressive decomposition architecture
    """
    def __init__(self, self_attention, cross_attention, d_model, c_out, d_ff=None,
                 moving_avg=25, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.decomp3 = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1,
                                    padding_mode='circular', bias=False)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x, trend1 = self.decomp1(x)
        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])
        x, trend2 = self.decomp2(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x, trend3 = self.decomp3(x + y)

        residual_trend = trend1 + trend2 + trend3
        residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)
        return x, residual_trend


class Decoder(nn.Module):
    """
    Informer encoder
    """
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        for layer in self.layers:
            x, residual_trend = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            trend = trend + residual_trend

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x, trend

class AutoCorrelation(nn.Module):
    """
    AutoCorrelation Mechanism with the following two phases:
    (1) period-based dependencies discovery
    (2) time delay aggregation
    This block can replace the self-attention family mechanism seamlessly.
    """

    def __init__(self, mask_flag=True, factor=1, scale=None, attention_dropout=0.1, output_attention=False):
        super(AutoCorrelation, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def time_delay_agg_training(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the training phase.
        """
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # find top k
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]
        weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            pattern = torch.roll(tmp_values, -int(index[i]), -1)
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        return delays_agg

    def time_delay_agg_inference(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the inference phase.
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index init
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0) \
            .repeat(batch, head, channel, 1).to(values.device)
        # find top k
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        weights, delay = torch.topk(mean_value, top_k, dim=-1)
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        return delays_agg

    def time_delay_agg_full(self, values, corr):
        """
        Standard version of Autocorrelation
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index init
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0) \
            .repeat(batch, head, channel, 1).to(values.device)
        # find top k
        top_k = int(self.factor * math.log(length))
        weights, delay = torch.topk(corr, top_k, dim=-1)
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[..., i].unsqueeze(-1)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * (tmp_corr[..., i].unsqueeze(-1))
        return delays_agg

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        if L > S:
            zeros = torch.zeros_like(queries[:, :(L - S), :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]

        # period-based dependencies
        q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)

        # time delay agg
        if self.training:
            V = self.time_delay_agg_training(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)
        else:
            V = self.time_delay_agg_inference(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)

        if self.output_attention:
            return (V.contiguous(), corr.permute(0, 3, 1, 2))
        else:
            return (V.contiguous(), None)


class AutoCorrelationLayer(nn.Module):
    def __init__(self, correlation, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AutoCorrelationLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_correlation = correlation
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_correlation(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()

        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class Model(nn.Module):
    """
    Informer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Decomp
        kernel_size = configs.moving_avg
        self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)
        self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.c_out,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]


# Test
def test(dataloader, test_net, pred_len, d_token, pred_target, f_std=None,
        DEVICE='cuda:0' if torch.cuda.is_available() else 'cpu'):
    test_net.eval()
    pres = []
    ys = []
    sample_num = 0
    with torch.no_grad():
        for x, x_mask, y, y_mask in dataloader:
            sample_num += x.shape[0]
            x = x.float().to(DEVICE)
            y = y.float().to(DEVICE)[:, :, pred_target]
            x_mask = x_mask.float().to(DEVICE)
            x_dec = torch.cat([x[:, -d_token:, :], torch.zeros((x.shape[0], pred_len, x.shape[2])).to(DEVICE)],
                              dim=1)
            x_dec_mask = torch.cat([x_mask[:, -d_token:, :], y_mask.to(DEVICE)], dim=1)
            pre = test_net(x, x_mask, x_dec, x_dec_mask)[:, :, pred_target].squeeze(1)
            # loss = torch.sum(torch.abs(pre-y.to(DEVICE))).item()
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


class Configs():
    def __init__(self, seq_len, pred_len):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = self.seq_len // 2
        self.enc_in = 12
        self.dec_in = 12
        self.c_out = 12
        self.d_model = 512
        self.n_heads = 8
        self.d_ff = 2048
        self.e_layers = 2
        self.d_layers = 1
        self.moving_avg = 25
        self.factor = 1
        self.dropout = 0.05
        self.embed = 'timeF'
        self.activation = 'gelu'
        self.output_attention = False
        self.freq = 'h'
        self.wavelet = 0


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
    seed_list = [31, ]
    seq_len_list = [168, ]
    pred_len_list = [72, 120, ]
    save_model_situation = [12, 24, 48, 72, 96, 120, 144]
    scalar = True
    BATCHSIZE = 32
    EPOCH = 10000
    test_interval = 1
    update_interval = 999
    target_feature = 'NO2'
    pred_target = feature_index[target_feature]

    # Train
    def train(seed, config, train_dataloader, test_dataloader, val_dataloader, f_std,
              save_model_flag, scalar=True):

        early_stop = Eearly_stop(patience=20, save_model_flag=save_model_flag)

        net = Model(config).to(DEVICE)
        # Build optimizer and loss_function
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
        # loss_func = torch.nn.MSELoss(reduction='sum')
        loss_func = torch.nn.MSELoss(reduction='sum')

        train_loss_list = []
        val_loss_list = []
        test_loss_list = []
        for epoch_idx in range(EPOCH):
            net.train()
            epoch_loss = 0.0
            for x, x_mask, y, y_mask in train_dataloader:
                optimizer.zero_grad()
                x = x.float().to(DEVICE)
                y = y.float().to(DEVICE)[:, :, pred_target]
                x_mask = x_mask.float().to(DEVICE)
                x_dec = torch.cat([x[:, -config.label_len:, :], torch.zeros((x.shape[0], config.pred_len, x.shape[2])).to(DEVICE)],
                                  dim=1)
                x_dec_mask = torch.cat([x_mask[:, -config.label_len:, :], y_mask.to(DEVICE)], dim=1)
                pre = net(x, x_mask, x_dec, x_dec_mask)[:, :, pred_target].squeeze(1)
                # print(f'x.shape:{x.shape}\tpre.shape:{pre.shape}\ty.shape:{y.shape}')
                loss = loss_func(pre, y)
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
            train_loss = round(epoch_loss / train_dataloader.dataset.__len__() / config.pred_len, 4)
            print(f'EPOCH:{epoch_idx}\t\tTRAIN_LOSS:{train_loss}')

            if epoch_idx % test_interval == 0:
                val_loss, _ = test(dataloader=val_dataloader, test_net=net, pred_len=config.pred_len, d_token=config.label_len, pred_target=pred_target)
                test_loss, test_loss_rmse = test(dataloader=test_dataloader, test_net=net, pred_len=config.pred_len, d_token=config.label_len, pred_target=pred_target)
                print(f'EPOCH:{epoch_idx}\t\tVAL_LOSS:{val_loss}'
                      f'\t\tTEST_LOSS:{test_loss}/{test_loss_rmse}||'
                      f'{round(test_loss * f_std if scalar else 0, 4)}/{round(test_loss_rmse * f_std if scalar else 0, 4)}',
                      end='\t\t')
                if early_stop(val_loss, net, type='e'):
                    val_loss, val_loss_rmse = test(dataloader=val_dataloader, test_net=early_stop.net, pred_len=config.pred_len, d_token=config.label_len, pred_target=pred_target)
                    test_loss, test_loss_rmse = test(dataloader=test_dataloader, test_net=early_stop.net, pred_len=config.pred_len, d_token=config.label_len, pred_target=pred_target)
                    train_loss, train_loss_rmse = test(dataloader=train_dataloader, test_net=early_stop.net, pred_len=config.pred_len, d_token=config.label_len, pred_target=pred_target)
                    print(f'accuracy on Train:\033[0;34mMAE={train_loss}/{train_loss_rmse}||'
                          f'{round(train_loss * f_std, 4)}/{round(train_loss_rmse * f_std, 4)}\033[0m\r\n'
                          f'accuracy on Val:\033[0;34mMAE={val_loss}/{val_loss_rmse}||'
                          f'{round(val_loss * f_std, 4)}/{round(val_loss_rmse * f_std, 4)}\033[0m\r\n'
                          f'accuracy on Test:\033[0;34mMAE={test_loss}/{test_loss_rmse}||'
                          f'{round(test_loss * f_std, 4)}/{round(test_loss_rmse * f_std, 4)}\033[0m\r\n')
                    if save_model_flag:
                        result = (test_loss, test_loss_rmse, round(test_loss * f_std, 2), round(test_loss_rmse * f_std, 2))
                        early_stop.save_model(model_name='Autoformer', target_feature=target_feature, seed=seed,
                                              result=result, pred_len=config.pred_len, seq_len=config.seq_len,
                                              state_dict={'seq_len': config.seq_len, 'pred_len': config.pred_len,
                                                          'pred_target': pred_target, 'config': config})
                    return test_loss, test_loss_rmse

            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)
            test_loss_list.append(test_loss)

            if (epoch_idx + 1) % update_interval == 0:
                draw_line(title="Informer", train_loss=train_loss_list,
                          val_loss=val_loss_list, test_loss=test_loss_list)


    def run(seed_list, seq_len_list, pred_len_list):
        result = {}
        for seq_len in seq_len_list:
            for pred_len in pred_len_list:
                # building dataset
                save_model_flag = True if pred_len in save_model_situation else False
                config = Configs(seq_len=seq_len, pred_len=pred_len)
                print(f'Predicted Target:{target_feature}\r\n')
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
                    mae, rmse = train(seed=seed, config=config, train_dataloader=train_dataloader,
                                      val_dataloader=val_dataloader, test_dataloader=test_dataloader,
                                      scalar=scalar, save_model_flag=save_model_flag, f_std=f_std)
                    if scalar:
                        result[f'{target_feature} seq_len={seq_len} pred_len={pred_len} seed={seed}'] = (mae, rmse, round(mae * f_std, 4), round(rmse * f_std, 4))
                    else:
                        result[f'{target_feature} seq_len={seq_len} pred_len={pred_len} seed={seed}'] = (mae, rmse)
        for item in result.items():
            print(f'result on random seed{item}')


    run(seed_list=seed_list, seq_len_list=seq_len_list, pred_len_list=pred_len_list)