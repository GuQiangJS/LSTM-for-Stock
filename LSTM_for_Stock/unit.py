from inspect import signature
import matplotlib.pyplot as plt
import numpy as np


def _get_param(cls, con_name):
    sig = signature(cls)
    return sig.parameters.get(con_name)


def get_param_default_value(cls, con_name):
    param = _get_param(cls, con_name)
    if param:
        return param.default
    return None


def plot_by_xl(X, Y, window, days, figsize=(15, 5), top=100):
    """按预测值的斜率绘图"""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.plot(X[:, 0, 0], label='True Data',color='b')

    xl = {}
    for i, data in enumerate(Y):
        xl[i] = abs((data[-1] - data[0]) / (i + window + days - i + window))
    xl_top = np.array(sorted(xl.items(),
                             key=lambda kv: kv[1],
                             reverse=True))[:top, 0]
    # Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(Y):
        # 绘制预测值从开始点到结束点的直线
        if i in xl_top:
            plt.plot([i + window, i + window + days], [data[0], data[-1]],color='r')
        # 绘制预测值从开始点到结束点的直线
        plt.legend()
    return plt


def plot(X, Y, window, figsize=(15, 5)):
    """绘图

    Args:
        figsize: 绘图大小。默认为(15,5)
        X: 真实值数组。三维数组。可以是从 `DataLoader.get_train_data`、
           `DataLoader.get_valid_data` 中获取到的X，或与之格式一致的均可。
           计算时会取 `[:,0,0]` 来作为绘图值。
        Y: 预测值或未来值的二维数组。
        days (int): `Y` 及 `other_Y` 绘制之间跳过的数量。
        other_Y: 其他与Y相同类型的二维数组。（可能 `Y` 为真实的未来值；`other_Y` 为预测值）
        window: 窗口期。绘图时会先绘制X值，然后遍历所有Y值得第一个维度，
                每次跳空 `window` 值后开始绘制。
    """
    # plt.figure(figsize=figsize)
    # plt.plot(X[:, 0, 0])
    # for i in range(len(Y)):
    #     if days and i % days != 0:
    #         continue
    #     append = [None for p in range(window + i)]
    #     plt.plot(append + list(Y[i]))
    #     if other_Y is not None:
    #         plt.plot(append + list(other_Y[i]))
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.plot(X[:, 0, 0], label='True Data')
    # Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(Y):
        # 绘制所有预测值
        padding = [None for p in range(i + window)]
        plt.plot(padding + list(data))
        # 绘制所有预测值
        plt.legend()
    return plt
