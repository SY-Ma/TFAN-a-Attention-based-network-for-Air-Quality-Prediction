# python3.7
# -*- coding: utf-8 -*-
# @Project_Name  : Air Quality prediction2
# @File          : random_seed.py
# @Time          : 2022/11/7 16:20
# @Author        : SY.M
# @Software      : PyCharm


import torch
import numpy as np


def setup_seed(seed):
    """
    设置随机种子
    :param seed: 随机种子数
    :return: None
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
