# python3.7
# -*- coding: utf-8 -*-
# @Project_Name  : Air Quality prediction Beijing
# @File          : cosSquareFormer.py
# @Time          : 2022/11/12 16:53
# @Author        : SY.M
# @Software      : PyCharm

import torch
import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from math import sqrt
import os
import numpy as np
# from layers.SelfAttention_Family import FullAttention

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

class DataEmbedding_wo_pos_temp(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos_temp, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x)
        return self.dropout(x)

class DataEmbedding_wo_temp(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_temp, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
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

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn

class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
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
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x

class Model(nn.Module):
    """
    Vanilla Transformer with O(L^2) complexity
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Embedding
        if configs.embed_type == 0:
            self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                            configs.dropout)
            self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        elif configs.embed_type == 1:
            self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
            self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        elif configs.embed_type == 2:
            self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
            self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)

        elif configs.embed_type == 3:
            self.enc_embedding = DataEmbedding_wo_temp(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
            self.dec_embedding = DataEmbedding_wo_temp(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        elif configs.embed_type == 4:
            self.enc_embedding = DataEmbedding_wo_pos_temp(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
            self.dec_embedding = DataEmbedding_wo_pos_temp(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        MultiHeadAttentionCosSquareformerNew(D=configs.d_model, H=configs.n_heads),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        MultiHeadAttentionCosSquareformerNew(D=configs.d_model, H=configs.n_heads),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        MultiHeadAttentionCosSquareformerNew(D=configs.d_model, H=configs.n_heads),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]


class MultiHeadAttentionCosSquareformerNew(nn.Module):
    '''Multi-head self-attention module'''

    def __init__(self, D, H):
        super(MultiHeadAttentionCosSquareformerNew, self).__init__()
        self.H = H  # number of heads
        self.D = D  # dimension

        self.wq = nn.Linear(D, D * H)
        self.wk = nn.Linear(D, D * H)
        self.wv = nn.Linear(D, D * H)

        self.dense = nn.Linear(D * H, D)

    def concat_heads(self, x):
        '''(B, H, S, D) => (B, S, D*H)'''
        B, H, S, D = x.shape
        x = x.permute((0, 2, 1, 3)).contiguous()  # (B, S, H, D)
        x = x.reshape((B, S, H * D))  # (B, S, D*H)
        return x

    def split_heads(self, x):
        '''(B, S, D*H) => (B, H, S, D)'''
        B, S, D_H = x.shape
        x = x.reshape(B, S, self.H, self.D)  # (B, S, H, D)
        x = x.permute((0, 2, 1, 3))  # (B, H, S, D)
        return x

    def forward(self, query, key, value, attn_mask):
        query = query.reshape(query.shape[0], query.shape[1], -1)
        key = key.reshape(key.shape[0], key.shape[1], -1)
        value = value.reshape(value.shape[0], value.shape[1], -1)
        q = self.wq(query)  # (B, S, D*H)
        k = self.wk(key)  # (B, S, D*H)
        v = self.wv(value)  # (B, S, D*H)

        q = self.split_heads(q).permute(0, 2, 1, 3)  # (B, S, H, D)
        k = self.split_heads(k).permute(0, 2, 1, 3)  # (B, S, H, D)
        v = self.split_heads(v).permute(0, 2, 1, 3)  # (B, S, H, D)
        Q_B = q.shape[0]
        Q_S = q.shape[1]
        K_B = k.shape[0]
        K_S = k.shape[1]

        q = torch.nn.functional.elu(q) + 1  # Sigmoid torch.nn.ReLU()
        k = torch.nn.functional.elu(k) + 1  # Sigmoid torch.nn.ReLU()

        # q, k, v -> [batch_size, seq_len, n_heads, d_head]
        q_cos = (torch.cos(3.1415 * torch.arange(Q_S) / Q_S).unsqueeze(0)).repeat(Q_B, 1).cuda()
        q_sin = (torch.sin(3.1415 * torch.arange(Q_S) / Q_S).unsqueeze(0)).repeat(Q_B, 1).cuda()
        k_cos = (torch.sin(3.1415 * torch.arange(K_S) / K_S).unsqueeze(0)).repeat(K_B, 1).cuda()
        k_sin = (torch.sin(3.1415 * torch.arange(K_S) / K_S).unsqueeze(0)).repeat(K_B, 1).cuda()
        # cos, sin -> [batch_size, seq_len]
        q_cos = torch.einsum('bsnd,bs->bsnd', q, q_cos)
        q_sin = torch.einsum('bsnd,bs->bsnd', q, q_sin)
        k_cos = torch.einsum('bsnd,bs->bsnd', k, k_cos)
        k_sin = torch.einsum('bsnd,bs->bsnd', k, k_sin)
        # q_cos, q_sin, k_cos, k_sin -> [batch_size, seq_len, n_heads, d_head]

        kv_cos = torch.einsum('bsnx,bsnz->bnxz', k_cos, v)
        # kv_cos -> [batch_size, n_heads, d_head, d_head]
        qkv_cos = torch.einsum('bsnx,bnxz->bsnz', q_cos, kv_cos)
        # qkv_cos -> [batch_size, seq_len, n_heads, d_head]

        kv_sin = torch.einsum('bsnx,bsnz->bnxz', k_sin, v)
        # kv_sin -> [batch_size, n_heads, d_head, d_head]
        qkv_sin = torch.einsum('bsnx,bnxz->bsnz', q_sin, kv_sin)
        # qkv_sin -> [batch_size, seq_len, n_heads, d_head]

        kv = torch.einsum('bsnx,bsnz->bnxz', k, v)
        # kv -> [batch_size, n_heads, d_head, d_head]
        qkv = torch.einsum('bsnx,bnxz->bsnz', q, kv)
        # qkv_cos -> [batch_size, seq_len, n_heads, d_head]

        # denominator
        denominator = 1.0 / (torch.einsum('bsnd,bnd->bsn', q, k.sum(axis=1)) + torch.einsum('bsnd,bnd->bsn', q_cos,
                                                                                            k_cos.sum(axis=1))
                             + torch.einsum('bsnd,bnd->bsn',
                                            q_sin, k_sin.sum(axis=1))
                             + 1e-5)
        # denominator -> [batch_size, seq_len, n_heads]

        O = torch.einsum('bsnz,bsn->bsnz', qkv + qkv_cos +
                         qkv_sin, denominator).contiguous()
        # output -> [batch_size, seq_len, n_heads, d_head]

        concat_attention = self.concat_heads(O.permute(0, 2, 1, 3))  # (B, S, D*H)
        output = self.dense(concat_attention)  # (B, S, D)

        return output, None

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

            enc_in = x.to(DEVICE)
            enc_in_stamp = x_stamp
            dec_in = torch.cat(
                [x[:, -d_token:, :], torch.zeros((x.shape[0], pred_len, x.shape[-1])).to(DEVICE)],
                dim=1)
            dec_in_stamp = torch.cat([x_stamp[:, -d_token:, :], y_stamp], dim=1)

            pre = test_net(enc_in, enc_in_stamp, dec_in, dec_in_stamp)[:, :, pred_target]

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


class Config():
    def __init__(self, seq_len, pred_len):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_token = self.seq_len // 2
        self.output_attention = False
        self.enc_in = 12
        self.dec_in = 12
        self.d_model = 512
        self.embed = 'fixed'
        self.freq = 'h'
        self.dropout = 0.05
        self.d_ff = 2048
        self.activation = 'relu'
        self.n_heads = 8
        self.c_out = 12
        self.embed_type = 0
        self.e_layers = 2
        self.d_layers = 1
        self.LR = 1e-4


if __name__ == '__main__':
    from data_process.data_processer import DataProcesser
    import torch
    from utils.random_seed import setup_seed
    from utils.early_stop import Eearly_stop
    from torch.utils.data import DataLoader
    import numpy as np
    from utils.visualization import draw_line

    feature_index = {'PM2.5': 0, 'PM10': 1, 'NO2': 2, 'CO': 3, 'O3': 4, 'SO2': 5,
                     'TEMP': 6, 'PRES': 7, 'Humidity': 8, 'wind_speed': 9, 'weather': 10, 'WD': 11}
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    seed_list = [30, ]
    seq_len_list = [168, ]
    pred_len_list = [24, ]
    save_model_situation = [12, 24, 48, 72, 96, 120, 144]

    scalar = True
    BATCHSIZE = 8
    EPOCH = 10000
    test_interval = 1
    update_interval = 999
    target_feature = 'PM10'
    pred_target = feature_index[target_feature]

    # Train
    def train(seed, config, train_dataloader, val_dataloader, test_dataloader,
              f_std, save_model_flag, scalar=True):
        # initialize Model
        net = Model(configs=config).to(DEVICE)
        early_stop = Eearly_stop(patience=10, save_model_flag=save_model_flag)

        # Build optimizer and loss_function
        optimizer = torch.optim.Adam(net.parameters(), lr=config.LR)
        loss_func = torch.nn.MSELoss(reduction='sum')

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
                dec_in = torch.cat([x[:, -config.d_token:, :], torch.zeros((x.shape[0], config.pred_len, x.shape[-1])).to(DEVICE)],
                                   dim=1)
                dec_in_stamp = torch.cat([x_stamp[:, -config.d_token:, :], y_stamp], dim=1)

                pre = net(enc_in, enc_in_stamp, dec_in, dec_in_stamp)[:, :, pred_target]

                loss = loss_func(pre, y)
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
            train_loss = round(epoch_loss / train_dataloader.dataset.X.shape[0] / config.pred_len, 4)
            print(f'EPOCH:{epoch_idx}\t\tTRAIN_LOSS:{round(train_loss, 4)}')

            if epoch_idx % test_interval == 0:
                val_loss, _ = test(dataloader=val_dataloader, test_net=net, pred_len=config.pred_len, d_token=config.d_token, pred_target=pred_target)
                test_loss, test_loss_rmse = test(dataloader=test_dataloader, test_net=net, pred_len=config.pred_len, d_token=config.d_token, pred_target=pred_target)
                print(f'EPOCH:{epoch_idx}\t\tVAL_LOSS:{val_loss}'
                      f'\t\tTEST_LOSS:{test_loss}/{test_loss_rmse}||'
                      f'{round(test_loss * f_std if scalar else 0, 4)}/{round(test_loss_rmse * f_std if scalar else 0, 4)}',
                      end='\t\t')
                if early_stop(val_loss, net, type='e'):
                    val_loss, val_loss_rmse = test(dataloader=val_dataloader, test_net=early_stop.net, pred_len=config.pred_len, d_token=config.d_token, pred_target=pred_target)
                    test_loss, test_loss_rmse = test(dataloader=test_dataloader, test_net=early_stop.net, pred_len=config.pred_len, d_token=config.d_token, pred_target=pred_target)
                    train_loss, train_loss_rmse = test(dataloader=train_dataloader, test_net=early_stop.net, pred_len=config.pred_len, d_token=config.d_token, pred_target=pred_target)
                    print(f'accuracy on Train:\033[0;34mMAE={train_loss}/{train_loss_rmse}||'
                          f'{round(train_loss * f_std, 4)}/{round(train_loss_rmse * f_std, 4)}\033[0m\r\n'
                          f'accuracy on Val:\033[0;34mMAE={val_loss}/{val_loss_rmse}||'
                          f'{round(val_loss * f_std, 4)}/{round(val_loss_rmse * f_std, 4)}\033[0m\r\n'
                          f'accuracy on Test:\033[0;34mMAE={test_loss}/{test_loss_rmse}||'
                          f'{round(test_loss * f_std, 4)}/{round(test_loss_rmse * f_std, 4)}\033[0m\r\n')
                    if save_model_flag:
                        result = (
                        test_loss, test_loss_rmse, round(test_loss * f_std, 2), round(test_loss_rmse * f_std, 2))
                        early_stop.save_model(model_name='csoSquareFormer', target_feature=target_feature, seed=seed,
                                              result=result, pred_len=config.pred_len, seq_len=config.seq_len,
                                              state_dict={'seq_len': config.seq_len, 'pred_len': config.pred_len,
                                                          'pred_target': pred_target, 'config': config})

                    return test_loss, test_loss_rmse

            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)
            test_loss_list.append(test_loss)

            if (epoch_idx + 1) % update_interval == 0:
                draw_line(title="cosSquareFormer", train_loss=train_loss_list,
                          val_loss=val_loss_list, test_loss=test_loss_list)


    def run(seed_list, seq_len_list, pred_len_list):
        result = {}
        for seq_len in seq_len_list:
            for pred_len in pred_len_list:
                # building dataset
                save_model_flag = True if pred_len in save_model_situation else False
                config = Config(seq_len=seq_len, pred_len=pred_len)
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
