# python3.7
# -*- coding: utf-8 -*-
# @Project_Name  : Air Quality prediction2
# @File          : visualization.py
# @Time          : 2022/11/7 16:43
# @Author        : SY.M
# @Software      : PyCharm


import matplotlib.pyplot as plt
import time
import os

import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def draw_line(title, train_loss, val_loss, test_loss):
    plt.figure()
    plt.clf()
    plt.plot(train_loss, label='train', color="blue")
    plt.plot(val_loss, label='val', color="green")
    plt.plot(test_loss, label='test', color="red")
    plt.text(len(train_loss) - 1, train_loss[-1], train_loss[-1],
             ha='left', va='baseline', color='blue', fontsize=12)
    plt.text(len(train_loss) - 1, val_loss[-1], val_loss[-1],
             ha='left', va='baseline', color="green", fontsize=12)
    plt.text(len(train_loss) - 1, test_loss[-1], test_loss[-1],
             ha='left', va='baseline', color="red", fontsize=12)
    plt.title(title)
    plt.legend(loc='best')
    plt.show()
    plt.close()


def comapre_line(sequence, pre, true, dataset_type):
    mean = np.mean(sequence)
    sequence = sequence[-24:]
    seq_last = sequence[-1]
    true_first = true[0]
    gap = [seq_last, true_first]
    gap_len = [sequence.shape[0] - 1, sequence.shape[0]]
    x_len = range(0, sequence.shape[0])
    y_len = range(sequence.shape[0], sequence.shape[0] + pre.shape[0])

    plt.scatter(sequence.shape[0], mean, marker='*', c='red')
    plt.axvline(sequence.shape[0], linestyle='--', c='black')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.plot(x_len, sequence, c='black', label='Ground Truth')
    plt.plot(gap_len, gap, c='black')
    plt.plot(y_len, true, c='black')
    plt.plot(y_len, pre, c='blue', label='Inference Series')
    plt.title(dataset_type)
    plt.legend(loc='best')

    plt.show()
    plt.close()


def result_visual():
    TFAN_result = {'PM2.5': [12.0712, 12.52598, 11.13036667, 11.17423333, 11.0591, 11.3443, 11.75573333],
                   'PM10': [30.179, 32.8457, 32.9758, 30.9725, 30.4392, 30.6343, 30.8294],
                   'NO2': [11.1153, 11.3015, 11.4654, 11.7932, 11.5958, 11.7671, 12.0577],
                   'CO': [0.1851, 0.1803, 0.1681, 0.1704, 0.1703, 0.169133333, 0.1731],
                   'O3': [15.9042, 15.8931, 16.1539, 16.4148, 16.8698, 16.8809, 17.1528],
                   'SO2': [3.4011, 3.4537, 3.5969, 3.4411, 3.3147, 3.1483, 3.2136]}
    cosSquareFormer_result = {'PM2.5': [15.7493, 15.8316, 14.8277, 14.4492, 14.7948, 13.9884, 11.6762],
                              'PM10': [33.1709, 37.9969, 36.4229, 40.7677, 36.3969, 35.018, 40.0652],
                              'NO2': [13.164, 13.395, 13.5067, 13.3018, 13.5253, 13.2907, 13.3465],
                              'CO': [0.2202, 0.2248, 0.2137, 0.2103, 0.2105, 0.1909, 0.1839],
                              'O3': [17.9963, 17.0085, 17.5524, 17.852, 18.5235, 17.7688, 18.0351],
                              'SO2': [3.8075, 4.0749, 4.2898, 4.1866, 3.7212, 3.7148, 3.2852]}
    Informer_result = {'PM2.5': [14.2106, 13.6922, 13.4453, 13.6757, 13.3301, 13.0339, 13.2067],
                       'PM10': [32.2213, 35.6294, 35.6294, 35.3563, 34.4327, 34.7839, 35.9546],
                       'NO2': [13.2609, 12.974, 13.3987, 13.2162, 13.5961, 13.4359, 13.5291],
                       'CO': [0.2132, 0.214, 0.2003, 0.2034, 0.204, 0.1958, 0.1965],
                       'O3': [17.4858, 18.4014, 19.6278, 20.0995, 20.3381, 20.3547, 20.9152],
                       'SO2': [3.6285, 3.858, 3.6748, 3.7064, 3.5379, 3.3147, 3.3758]}
    Autoformer_result = {'PM2.5': [23.9531, 19.4027, 21.4763, 22.4719, 22.2827, 23.7391, 25.8374],
                         'PM10': [42.0034, 43.9677, 43.0571, 42.6148, 39.649, 42.0815, 53.6198],
                         'NO2': [13.4397, 13.8494, 12.8772, 13.5477, 12.9517, 12.9368, 14.3076],
                         'CO': [0.2718, 0.2706, 0.22, 0.2407, 0.218, 0.2352, 0.237],
                         'O3': [20.8764, 21.3592, 23.0517, 22.3858, 22.1083, 23.6843, 24.8496],
                         'SO2': [4.6414, 4.9363, 4.4582, 4.4014, 3.9444, 4.5277, 4.5003], }
    Repeat_result = {'PM2.5': [24.1967, 32.8931, 40.9294, 46.5266, 48.7201, 49.4034, 49.6224],
                     'PM10': [35.9797, 45.9081, 54.5921, 56.8008, 58.4123, 57.7116, 57.5291],
                     'NO2': [19.7367, 22.1464, 23.3259, 23.6783, 24.2549, 24.2666, 24.3123],
                     'CO': [0.3059, 0.3879, 0.4531, 0.4917, 0.5006, 0.4952, 0.4874],
                     'O3': [59.537, 63.2313, 67.5041, 69.579, 69.0963, 70.2738, 71.2021],
                     'SO2': [6.4417, 7.4764, 8.0614, 8.0172, 7.618, 7.3665, 7.1968]}
    MLP_result = {'PM2.5': [15.3226, 16.85008, 17.92414, 18.54042, 19.1778, 19.9187, 19.3762],
                  'PM10': [33.89026667, 37.93773333, 40.9645, 42.46306667, 41.9705, 42.15343333, 41.94883333],
                  'NO2': [14.0099, 15.0696, 15.83155, 15.7844, 16.0529, 16.27085, 16.42665],
                  'CO': [0.2245, 0.24515, 0.2535, 0.2648, 0.26585, 0.2725, 0.2643],
                  'O3': [19.2866, 20.9151, 22.51805, 23.4387, 23.7881, 24.85275, 25.0923],
                  'SO2': [4.155566667, 4.4246, 4.488433333, 4.580466667, 4.595233333, 4.7054, 4.7264]}
    LSTM_result = {'PM2.5': [19.699, 21.6985, 20.4889, 21.1801, 24.4632, 23.6322, 26.3805],
                   'PM10': [34.7839, 39.1677, 41.7693, 42.3156, 41.301, 43.1222, 40.9238],
                   'NO2': [13.2758, 13.6632, 14.2144, 15.1792, 14.1735, 14.3076, 14.5571],
                   'CO': [0.2467, 0.2637, 0.2692, 0.2713, 0.2696, 0.2688, 0.2763],
                   'O3': [21.9141, 23.0461, 24.3391, 25.7986, 25.8208, 26.2869, 27.7075],
                   'SO2': [4.3972, 5.1237, 4.8815, 5.2901, 5.6123, 5.4206, 5.1806]}
    LSTF_Linear_result = {'PM2.5': [23.37266, 29.84368, 33.86102, 35.5403, 34.7324, 34.5843, 34.4444],
                          'PM10': [37.1774, 45.0083, 49.249, 50.5108, 50.2506, 49.8474, 49.7303],
                          'NO2': [14.3895, 15.3096, 15.7044, 15.8274, 15.8087, 15.7417, 15.7529],
                          'CO': [0.285, 0.3314, 0.3522, 0.3603, 0.3624, 0.3569, 0.3572],
                          'O3': [27.7131, 30.7152, 33.279, 34.2557, 34.966, 35.3322, 35.5209],
                          'SO2': [5.1637, 5.7871, 5.9282, 5.9176, 5.766, 5.5744, 5.3911]}

    features = ['PM2.5', 'PM10', 'NO2', 'CO', 'O3', 'SO2']
    # feature = 'PM10'
    # feature = 'NO2'
    # feature = 'CO'
    # feature = 'O3'
    # feature = 'SO2'

    fig = plt.figure(figsize=(12, 6))
    # axes = fig.subplots(nrows=2, ncols=3)
    x_label = [12, 24, 48, 72, 96, 120, 144]
    font_xy = {'family': 'Times New Roman',
               'weight': 'normal',
               'size': 8}
    font_legend = {'family': 'Times New Roman',
                   'weight': 'normal',
                   'size': 12}
    font_label = {'family': 'Times New Roman',
                  'weight': 'normal',
                  'size': 15}

    location = [(0.08, 0.5, 0.25, 0.33), (0.38, 0.5, 0.25, 0.33), (0.68, 0.5, 0.25, 0.33),
                (0.08, 0.09, 0.25, 0.33), (0.38, 0.09, 0.25, 0.33), (0.68, 0.09, 0.25, 0.33)]

    # for ax, feature in zip(fig.axes, features):
    for i, feature in enumerate(features):
        ax = plt.axes(location[i])
        # plt.plot(ground_truth, label="Ground Truth", c='black', linestyle='-')
        ax.plot(TFAN_result[feature], label="TFAN", c='r', linestyle='-')
        ax.plot(cosSquareFormer_result[feature], label="cosSquareFormer", c='g', linestyle='-')
        ax.plot(Informer_result[feature], label="Informer", c='c', linestyle='-')
        ax.plot(Autoformer_result[feature], label="Autoformer", c='m', linestyle='-')
        ax.plot(LSTM_result[feature], label="LSTM", c='y', linestyle='-')
        ax.plot(LSTF_Linear_result[feature], label="LSTF-Linear", c='gold', linestyle='--')
        ax.plot(MLP_result[feature], label="MLP", c='black', linestyle='--')
        ax.plot(Repeat_result[feature], label="Repeat", c='b', linestyle='--')
        ax.set_title(feature, font_label)
        ax.set_xticks(range(len(x_label)), labels=x_label, font=font_xy)
        # ax.set_yticks(font=font_xy)
        ax.tick_params(axis='y', labelsize=6)
        # plt.rcParams['ytick.labelsize'] = 4
        # plt.yticks(font=font_xy)
        if i in [3, ]:
            plt.ylabel('Pollution concentration', font_label, y=1.1)
        # if i in [3, 4, 5]:
        #     plt.xlabel('Pred_len', font_label)

    # plt.xticks(range(len(x_label)), labels=x_label, font=font_xy)
    # plt.yticks(font=font_xy)
    plt.xlabel('Pred_len', font_label, x=-0.7)
    # plt.ylabel('Pollution concentration', font_label, y=2.0, x=-6.0)
    # plt.legend(loc=(-1.60, 2.425), prop=font_legend, ncol=4)
    plt.legend(loc=(-1.75, 2.425), prop=font_legend, ncol=4)
    # plt.legend(loc=(-2.3, 2.45), prop=font_legend, ncol=8)
    plt.show()
    plt.close(fig)


def result_visual_test():
    pred_len_24 = {'PM2.5': [0.1795374, 0.170982, 0.1639286, 0.159266667, 0.15172],
                   'PM10': [0.23695, 0.2419, 0.245225, 0.242233333, 0.250366667],
                   'NO2': [0.33015, 0.3227, 0.31785, 0.3094, 0.3034],
                   'CO': [0.15445, 0.15095, 0.14905, 0.1467, 0.1469],
                   'O3': [0.302566667, 0.2915, 0.2897, 0.288166667, 0.2864],
                   'SO2': [0.157825, 0.16325, 0.1675, 0.162933333, 0.162733333]}
    pred_len_120 = {'PM2.5': [0.1501, 0.1438, 0.1414, 0.1395, 0.1379],
                    'PM10': [0.24, 0.2392, 0.237, 0.2332, 0.2357],
                    'NO2': [0.36, 0.3485, 0.33915, 0.32465, 0.3159],
                    'CO': [0.15425, 0.15025, 0.1412, 0.13615, 0.1378],
                    'O3': [0.336566667, 0.328, 0.321766667, 0.310866667, 0.3042],
                    'SO2': [0.164725, 0.16135, 0.164225, 0.155675, 0.149075]}

    features = ['PM2.5', 'PM10', 'NO2', 'CO', 'O3', 'SO2']

    fig = plt.figure(figsize=(8, 3.5))
    # plt.style.use('ggplot')
    # plt.style.use('seaborn')
    x_label = [24, 48, 72, 120, 144]
    font_xy = {'family': 'Times New Roman',
               'weight': 'normal',
               'size': 8}
    font_legend = {'family': 'Times New Roman',
                   'weight': 'normal',
                   'size': 9}
    font_label = {'family': 'Times New Roman',
                  'weight': 'normal',
                  'size': 12}
    font_label_2 = {'family': 'Times New Roman',
                    'weight': 'normal',
                    'size': 12}
    height, winth = 0.6, 0.4
    location = [(0.08, 0.18, winth, height), (0.55, 0.18, winth, height)]
    cs = ['r', 'g', 'c', 'm', 'y', 'gold']

    # ax1 = plt.subplot(211)
    ax1 = plt.axes(location[0])
    for idx, feature in enumerate(features):
        data = np.array(pred_len_24[feature]) / pred_len_24[feature][3]
        # data = (data - np.mean(data)) / np.std(data)
        ax1.plot(data, label=feature, c=cs[idx], linestyle='-', marker='.', markersize=5)
        # ax1.plot(pred_len_24[feature], label=feature)
    ax1.set_title('Pred_len=24', font_label_2)
    ax1.set_xticks(range(len(x_label)), labels=x_label, font=font_xy)
    ax1.tick_params(axis='y', labelsize=6)
    plt.xlabel('Seq_len', font_label, x=0.5)
    plt.ylabel('rRMSE', font_label, y=0.51)

    # ax2 = plt.subplot(212)
    ax2 = plt.axes(location[1])
    for idx, feature in enumerate(features):
        data = np.array(pred_len_120[feature]) / pred_len_120[feature][3]
        # data = (data - np.mean(data)) / np.std(data)
        ax2.plot(data, label=feature, c=cs[idx], linestyle='-', marker='.', markersize=5)
        # ax2.plot(pred_len_120[feature], label=feature)
    ax2.set_title('Pred_len=120', font_label_2)
    ax2.set_xticks(range(len(x_label)), labels=x_label, font=font_xy)
    ax2.tick_params(axis='y', labelsize=6)
    # plt.xlabel('Seq_len', font_label, x=1.1)

    plt.xlabel('Seq_len', font_label, x=0.5)
    plt.legend(loc=(-0.50, 1.125), prop=font_legend, ncol=3)

    plt.show()
    plt.close(fig)


if __name__ == '__main__':
    # result_visual()
    result_visual_test()
