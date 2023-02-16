# TFAN a Temporal Feature correlations Attention based Network using novel data fusion technology for Air Quality Prediction
***A deep learning model based on Attention machinism for air quality prediction***
## 摘要 
>空气污染问题对人类的身体健康与社会的可持续发展带来了严重的威胁，精准的空气质量预测对改善空气污染、加强城市污染防治与人类健康保护具有重要意义。大多数现有研究方法对某一种空气成分的趋势分析只依赖于此属性的时间与空间数据，从而忽略了相同时间区间中其他若干空气成分对此属性变化趋势的影响。本研究提出了一种时间特征注意力深度学习模型TFAN，其通过注意力机制关注不同时间戳之间的相关性、不同特征之间的相关性与时间-特征之间的潜在联系，结合历史数据特征，对未来空气污染物浓度进行预测。本文使用北京市的空气质量数据对所提出的模型进行了全面的评估，实验结果表明TFAN相比较于多种基线模型具有优势。\
>**关键词**：空气质量;时间序列预测;注意力机制;数据融合

## Abstract
>Air pollution has a negative impact on human health and the sustainable development of society. An accurate pollutants concentration prediction is of great significance for reducing air pollution degree, strengthening urban pollution prevention, and human health protection. Many existing methods in analyzing the variation tendency of a certain air component only rely on its temporal and spatial information, but ignores the interaction between different attributes in the same time interval. This paper proposes a Temporal-Feature correlations attention-based deep learning model TFAN which contains a novel data fusion. It focuses on the correlation between different time stamps, the correlation between different features, and the potential relationship between time-features through the attention mechanism. The TFAN also combines the historical data characteristics to predict the future air pollutant concentration. The actual air quality data of Beijing City is used to comprehensively evaluate the proposed model, and the experimental results show that TFAN has advantages over multiple baseline models. \
>**Key Words**：air quality；time series prediction；attention mechanism；data fusion

Please cite as:\
`under review`

## Introduction
The development of society and progress of civilization has  brought convenience to people's life, but it has also brought a series of destruction to the natural environment, especially the air pollution problem. The harmful gases and particulates exposed to the air pose a serious threat to human health and the sustainable development of society. Air Pollutants include PM2.5, NO2, CO, etc. They are mainly from the burning of fossil fuels, power plants, and heavy industries. PM2.5 was the fifth-ranking mortality risk factor in 2015. Moreover, particulate matter and NO2 can cause irreversible respiratory disease, longer-term exposure of both also greatly increases the risk of death from cardiovascular disease and shortens life expectancy. Air quality prediction is of great significance for reducing air pollution degree, strengthening urban pollution prevention， and human health protection.\
However, forecasting air quality data is challenging for three reasons as follows:
* The dynamic mutability of air quality data Due to the interaction of different
* Sensor-level intra-characteristics and external factors
* Complex temporal correlation and unstable spatial correlation
The Example of geo-sensory multivariate time series and the illustration of dynamic interaction is as follow:
![Error!](images/空气质量预测相关性_2.1.png)

## Data
- [See homepage of data](https://www.microsoft.com/en-us/research/publication/forecasting-fine-grained-air-quality-based-on-big-data/?from=https://research.microsoft.com/apps/pubs/?id=246398&type=exact)
- [Download raw data](https://www.microsoft.com/en-us/research/uploads/prod/2016/02/Data-1.zip)

## Architecture
Self-Attention is also used in TFAN to concentrate the timestamp dimension and the feature dimension separately. It can fully learn their respective data distributions. Moreover, we carry out the attention calculation of Temporal-Feature and Feature-Temporal after Self-Attention. \
the Architecture of TFAN is as follow:
![Error!](images/模型架构_4.png)

## Code
- [Code for data process](https://github.com/SY-Ma/TFAN-a-Temporal-Feature-correlations-Attention-based-Network-using-novel-data-fusion-technology-for/blob/main/data_process/data_processer.py)
- [Code for network, train and test](https://github.com/SY-Ma/TFAN-a-Temporal-Feature-correlations-Attention-based-Network-using-novel-data-fusion-technology-for/blob/main/model/OURS_TF_score.py)
- [Code for benchmarks](https://github.com/SY-Ma/TFAN-a-Temporal-Feature-correlations-Attention-based-Network-using-novel-data-fusion-technology-for/tree/main/model)

## Result
- The comparison of pollutants concentration prediction precision of various model please see in paper
- The general fit of TFAN on six pollutants which be part of the same sample from test dataset. is as follow:
![Error!](images/line%20result%203.png)

## Demo
```
import torch
from data_process.data_processer import DataProcesser    # Data preprocessing and build dataset
from utils.random_seed import setup_seed                 # set random seed
from utils.early_stop import Eearly_stop                 # early stop mechinism
from torch.utils.data import DataLoader
import numpy as np

feature_index = {'PM2.5': 0, 'PM10': 1, 'NO2': 2, 'CO': 3, 'O3': 4, 'SO2': 5,
                'TEMP': 6, 'PRES': 7, 'humidity': 8, 'wind_speed': 9, 'weather': 10, 'WD': 11}   # feature index
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# 0. experiment settings
seed_list = [30, ]                 # random seed list
seq_len_list = [168, ]             # sequence length list 
pred_len_list = [120, ]            # predicted length list
save_model_situation = [12, 24, 48, 72, 96, 120, 144]    # if predicted length in this list, the save the net parameters to .pt file
scalar = True                      # do zero-mean normalization or not
BATCHSIZE = 32
EPOCH = 10000
test_interval = 1                  # test interval
update_interval = 5                # interval to update loss curve
target_feature = 'PM2.5'           # target feature name
pred_target = feature_index[target_feature]
# Hyper Parameters
d_feature = 12                     # feature numbers
d_embedding = 512                  # embedding demension
d_hidden = 512                     # hidden length
q = 8
k = 8
v = 8
h = 64
N = (2, 1)                         # (self attention encoder number, TF and FT attention encoder number)
LR = 1e-4
LR_weight = 1e-3
pe = True                          # do position encoding or not
mask = False                       # attention mask
dropout = 0.05

# 1. load data and build dataset
print('loading data...')
train_dataset = DataProcesser(scalar=scalar, seq_len=seq_len, pred_len=pred_len, dataset_type='train')  # seq_len and pred_len obtain from seq_len_list and pred_len_list
val_dataset = DataProcesser(scalar=scalar, seq_len=seq_len, pred_len=pred_len, dataset_type='val')
test_dataset = DataProcesser(scalar=scalar, seq_len=seq_len, pred_len=pred_len, dataset_type='test')
train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCHSIZE, shuffle=True)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCHSIZE, shuffle=False)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCHSIZE, shuffle=False)
if scalar:
    f_mean = standard_scalar.mean_[pred_target]            # help to reverse transform the data
    f_std = np.sqrt(standard_scalar.var_[pred_target])     # help to reverse transform the data
else:
    f_mean = f_std = 1

# 2. initialize net and optimizer
net = OURS(seq_len=seq_len, d_feature=d_feature, d_embedding=d_embedding, d_hidden=d_hidden, pred_len=pred_len,
                   q=q,k=k, v=v, h=h, N=N, pe=pe, mask=mask, dropout=dropout).to(DEVICE)
early_stop = Eearly_stop(patience=10, save_model_flag=save_model_flag)
optimizer = torch.optim.Adam([{'params': net.input_time.parameters()},
                                      {'params': net.input_feature.parameters()},
                                      {'params': net.encoder_time.parameters()},
                                      {'params': net.encoder_TF.parameters()},
                                      {'params': net.encoder_FT.parameters()},
                                      {'params': net.output.output_time.parameters()},
                                      {'params': net.output.output_time_feature.parameters()},
                                      {'params': net.output.weight, 'lr': LR_weight}], lr=LR)
loss_func = torch.nn.L1Loss(reduction='sum')   

# 3. train and test
[...]
Please look the .py files for more details
```

## Reference
```bibtex
@article{2021An,
  title={An overview of air quality analysis by big data techniques: Monitoring, forecasting, and traceability},
  author={ Huang, W.  and  Li, T.  and  Liu, J.  and  Xie, P.  and  Teng, F. },
  journal={Information Fusion},
  volume={75},
  number={2},
  year={2021},
}
@article{Zhongang2018Deep,
  title={Deep Air Learning: Interpolation, Prediction, and Feature Analysis of Fine-Grained Air Quality},
  author={Zhongang and Qi and Tianchun and Wang and Guojie and Song and Weisong and Hu and Xi and Li and },
  journal={IEEE Transactions on Knowledge and Data Engineering},
  volume={30},
  number={12},
  pages={2285-2297},
  year={2018},
}
@inproceedings{2015Forecasting,
  title={Forecasting Fine-Grained Air Quality Based on Big Data},
  author={ Yu, Z.  and  Yi, X.  and  Ming, L.  and  Li, R.  and  Shan, Z. },
  booktitle={the 21th ACM SIGKDD International Conference},
  year={2015},
}
```
