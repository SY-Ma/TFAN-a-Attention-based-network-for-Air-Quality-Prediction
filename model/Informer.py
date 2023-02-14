# python3.7
# -*- coding: utf-8 -*-
# @Project_Name  : Air Quality prediction2
# @File          : Informer.py
# @Time          : 2022/11/7 18:38
# @Author        : SY.M
# @Software      : PyCharm


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from math import sqrt


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape  # (256, 8, 12, 64)
        _, _, L_Q, _ = Q.shape  # [256, 8, 12, 64]

        # print(f'K.shape:{K.shape}\tQ.shape:{Q.shape}')

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)  # [256, 8, 12, 12, 64]
        # print(f'K_expand.shape:{K_expand.shape}')
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q  [12, 9]
        # print(f'index_sample.shape:{index_sample.shape}')
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        # K_sample = K_expand[:, :, :, index_sample, :]
        # print(f'K_sample.shape:{K_sample.shape}')
        # K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), :, :]
        # print(f'K_sample.shape:{K_sample.shape}')
        # [256, 8, 12, 64] * [256, 8, 12, 12, 64, 9] = [256, 8, 12, 12, 9]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)
        # print(f'Q_K_sample.shape:{Q_K_sample.shape}')  # [256, 8, 12, 12, 9]

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        # print(f'factor:{self.factor}\tU_part:{U_part}\tu:{u}')

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        return context.transpose(2, 1).contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads,
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

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
        if self.mix:
            out = out.transpose(2, 1).contiguous()
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
                                   kernel_size=3, padding=padding, padding_mode='circular')
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

        minute_size = 4;
        hour_size = 24
        weekday_size = 7;
        day_size = 32;
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
        hour_x = self.hour_embed(x[:, :, 2])
        # weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + day_x + month_x + minute_x #+ weekday_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model)

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
        x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)

class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
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
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))

        return self.norm3(x+y)

class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        return x

class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=padding,
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
        # x [B, L, D]
        # x = x + self.dropout(self.attention(
        #     x, x, x,
        #     attn_mask = attn_mask
        # ))
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
            x, attn = self.attn_layers[-1](x, attn_mask=attn_mask)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class EncoderStack(nn.Module):
    def __init__(self, encoders, inp_lens):
        super(EncoderStack, self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.inp_lens = inp_lens

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        x_stack = [];
        attns = []
        for i_len, encoder in zip(self.inp_lens, self.encoders):
            inp_len = x.shape[1] // (2 ** i_len)
            x_s, attn = encoder(x[:, -inp_len:, :])
            x_stack.append(x_s);
            attns.append(attn)
        x_stack = torch.cat(x_stack, -2)

        return x_stack, attns

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


class Informer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True, mix=True,
                 device=torch.device('cuda:0')):
        super(Informer, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn == 'prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers - 1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)

        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, 0], attns
        else:
            return dec_out[:, -self.pred_len:, 0]  # [B, L, D]


class InformerStack(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=[3, 2, 1], d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True, mix=True,
                 device=torch.device('cuda:0')):
        super(InformerStack, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn == 'prob' else FullAttention
        # Encoder

        inp_lens = list(range(len(e_layers)))  # [0,1,2,...] you can customize here
        encoders = [
            Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                            d_model, n_heads, mix=False),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation
                    ) for l in range(el)
                ],
                [
                    ConvLayer(
                        d_model
                    ) for l in range(el - 1)
                ] if distil else None,
                norm_layer=torch.nn.LayerNorm(d_model)
            ) for el in e_layers]
        self.encoder = EncoderStack(encoders, inp_lens)
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)

        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
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
    with torch.no_grad():
        for x, x_mask, y, y_mask in dataloader:
            x = x.float().to(DEVICE)
            y = y.float().to(DEVICE)[:, :, pred_target]
            x_mask = x_mask.float().to(DEVICE)
            x_dec = torch.cat([x[:, -d_token:, :], torch.zeros((x.shape[0], pred_len, x.shape[2])).to(DEVICE)],
                              dim=1)
            x_dec_mask = torch.cat([x_mask[:, -d_token:, :], y_mask.to(DEVICE)], dim=1)
            pre = test_net(x, x_mask, x_dec, x_dec_mask).squeeze(1)

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
    from utils.visualization import draw_line

    feature_index = {'PM2.5': 0, 'PM10': 1, 'NO2': 2, 'CO': 3, 'O3': 4, 'SO2': 5,
                     'TEMP': 6, 'PRES': 7, 'humidity': 8, 'wind_speed': 9, 'weather': 10, 'WD': 11}
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # experiment settings
    seed_list = [30, ]
    seq_len_list = [168, ]
    pred_len_list = [24, 72, 120]
    save_model_situation = [12, 24, 48, 72, 96, 120, 144]
    scalar = True
    BATCHSIZE = 32
    EPOCH = 10000
    target_feature = 'PM10'
    pred_target = feature_index[target_feature]
    test_interval = 1
    update_interval = 999

    # Train
    def train(seed, pred_len, seq_len, save_model_flag, train_dataloader, val_dataloader,
              test_dataloader, f_std, ):
        d_token = seq_len // 2
        early_stop = Eearly_stop(patience=10, save_model_flag=save_model_flag)

        net = Informer(enc_in=12, dec_in=12, c_out=12, out_len=pred_len, seq_len=seq_len, label_len=d_token,
                       attn='prob', factor=5).to(DEVICE)
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
                x_dec = torch.cat([x[:, -d_token:, :], torch.zeros((x.shape[0], pred_len, x.shape[2])).to(DEVICE)],
                                  dim=1)
                x_dec_mask = torch.cat([x_mask[:, -d_token:, :], y_mask.to(DEVICE)], dim=1)
                pre = net(x, x_mask, x_dec, x_dec_mask).squeeze(1)
                # print(f'x.shape:{x.shape}\tpre.shape:{pre.shape}\ty.shape:{y.shape}')
                loss = loss_func(pre, y)
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
            train_loss = round(epoch_loss / train_dataloader.dataset.__len__() / pred_len, 4)
            print(f'EPOCH:{epoch_idx}\t\tTRAIN_LOSS:{train_loss}')

            if epoch_idx % test_interval == 0:
                val_loss, _ = test(dataloader=val_dataloader, test_net=net, pred_len=pred_len, d_token=d_token, pred_target=pred_target)
                test_loss, test_loss_rmse = test(dataloader=test_dataloader, test_net=net, pred_len=pred_len, d_token=d_token, pred_target=pred_target)
                print(f'EPOCH:{epoch_idx}\t\tVAL_LOSS:{val_loss}'
                      f'\t\tTEST_LOSS:{test_loss}/{test_loss_rmse}||'
                      f'{round(test_loss * f_std if scalar else 0, 4)}/{round(test_loss_rmse * f_std if scalar else 0, 4)}',
                      end='\t\t')
                if early_stop(val_loss, net, type='e'):
                    val_loss, val_loss_rmse = test(dataloader=val_dataloader, test_net=early_stop.net, pred_len=pred_len, d_token=d_token, pred_target=pred_target)
                    test_loss, test_loss_rmse = test(dataloader=test_dataloader, test_net=early_stop.net, pred_len=pred_len, d_token=d_token, pred_target=pred_target)
                    train_loss, train_loss_rmse = test(dataloader=train_dataloader, test_net=early_stop.net, pred_len=pred_len, d_token=d_token, pred_target=pred_target)
                    print(f'accuracy on Train:\033[0;34mMAE={train_loss}/{train_loss_rmse}||'
                          f'{round(train_loss * f_std, 4)}/{round(train_loss_rmse * f_std, 4)}\033[0m\r\n'
                          f'accuracy on Val:\033[0;34mMAE={val_loss}/{val_loss_rmse}||'
                          f'{round(val_loss * f_std, 4)}/{round(val_loss_rmse * f_std, 4)}\033[0m\r\n'
                          f'accuracy on Test:\033[0;34mMAE={test_loss}/{test_loss_rmse}||'
                          f'{round(test_loss * f_std, 4)}/{round(test_loss_rmse * f_std, 4)}\033[0m\r\n')
                    if save_model_flag:
                        result = (test_loss, test_loss_rmse, round(test_loss * f_std, 2), round(test_loss_rmse * f_std, 2))
                        early_stop.save_model(model_name='Informer', target_feature=target_feature, seed=seed,
                                              result=result, pred_len=pred_len, seq_len=seq_len,
                                              state_dict={'enc_in': 12, 'dec_in': 12, 'c_out': 12, 'out_len': pred_len, 'pred_target': pred_target,
                                                          'seq_len': seq_len, 'label_len': d_token, 'attn': 'prob', 'factor': 5})
                    return test_loss, test_loss_rmse

            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)
            test_loss_list.append(test_loss)

            if epoch_idx % update_interval == 0:
                draw_line(title="Informer Prob-Attention", train_loss=train_loss_list,
                          val_loss=val_loss_list, test_loss=test_loss_list)


    def run(seed_list, seq_len_list, pred_len_list):
        result = {}
        for seq_len in seq_len_list:
            for pred_len in pred_len_list:
                # building dataset
                save_model_flag = True if pred_len in save_model_situation else False
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
                    mae, rmse = train(seed=seed, train_dataloader=train_dataloader, pred_len=pred_len,
                                      val_dataloader=val_dataloader, test_dataloader=test_dataloader,
                                      save_model_flag=save_model_flag, f_std=f_std, seq_len=seq_len)
                    if scalar:
                        result[f'{target_feature} seq_len={seq_len} pred_len={pred_len} seed={seed}'] = (mae, rmse, round(mae * f_std, 4), round(rmse * f_std, 4))
                    else:
                        result[f'{target_feature} seq_len={seq_len} pred_len={pred_len} seed={seed}'] = (mae, rmse)
        for item in result.items():
            print(f'result on random seed{item}')

    run(seed_list=seed_list, seq_len_list=seq_len_list, pred_len_list=pred_len_list)