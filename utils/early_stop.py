# python3.7
# -*- coding: utf-8 -*-
# @Project_Name  : Air Quality prediction2
# @File          : early_stop.py
# @Time          : 2022/11/7 16:20
# @Author        : SY.M
# @Software      : PyCharm


import copy
import os
import torch

class Eearly_stop():
    def __init__(self, patience=5, save_model_flag=True,
                 saved_path='G:\PyCharmWorkSpace\PyCharmProjects\Air quality\Air Quality prediction Beijing\saved_model'):
        self.saved_path = saved_path
        self.origin_patience = patience
        self.patience = patience
        self.max_acc = float('-inf')
        self.min_e = float('inf')
        self.net = None
        self.score = None
        self.save_model_flag = save_model_flag

    def judge_acc(self, acc, net, score=None):
        if acc > self.max_acc:
            self.max_acc = acc
            self.patience = self.origin_patience
            self.net = copy.deepcopy(net).detach()
            # self.score = copy.deepcopy(score.detach())
        else:
            self.patience -= 1
            if self.patience <= 0:
                print('\033[0;34mpatience == 0, stop training!\033[0m\t\t')
                return True  # 停止训练
        print('patience = ', self.patience, '---------------------------')

        return False  # 继续训练

    def judge_e(self, e, net, score):
        if e < self.min_e:
            self.min_e = e
            self.patience = self.origin_patience
            with torch.no_grad():
                self.net = copy.deepcopy(net)
            # self.score = copy.deepcopy(score.detach())
        else:
            self.patience -= 1
            if self.patience <= 0:
                print('\033[0;34mpatience == 0, stop training!\033[0m\t\t')
                return True  # 停止训练
        print('patience = ', self.patience, '---------------------------')

    def __call__(self, acc, net, type, score=None):
        if type == 'acc':
            return self.judge_acc(acc, net, score)
        elif type == 'e':
            return self.judge_e(acc, net, score)

    def save_model(self, model_name: str, target_feature: str, result: tuple,
                   state_dict: dict, pred_len: int, seq_len: int, seed: int):
        save_path = os.path.join(self.saved_path, model_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        state_dict['model_param'] = self.net.state_dict()
        file_name = f'{model_name} seed={seed} {target_feature} {seq_len}_{pred_len} {result}.pt'
        save_file = os.path.join(save_path, file_name)
        torch.save(state_dict, save_file)
        print('save model successful!')