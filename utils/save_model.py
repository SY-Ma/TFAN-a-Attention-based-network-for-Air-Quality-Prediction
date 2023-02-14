# python3.7
# -*- coding: utf-8 -*-
# @Project_Name  : Air Quality prediction Beijing
# @File          : save_model.py
# @Time          : 2022/11/26 14:40
# @Author        : SY.M
# @Software      : PyCharm
import torch
import os


def save_model(model_name: str, target_feature: str, result: tuple, state_dict: dict, pred_len: int, seq_len: int,
        root='G:\PyCharmWorkSpace\PyCharmProjects\Air quality\Air Quality prediction Beijing\saved_model',):
    save_path = os.path.join(root, model_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_file = os.path.join(save_path, f'{model_name} {target_feature} {seq_len}->{pred_len} {result}.pt')
    torch.save(state_dict, save_file)
    print('save model successful！')


def load_model(file_path):
    torch.load(file_path)

