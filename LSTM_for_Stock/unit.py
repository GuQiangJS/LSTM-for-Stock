from inspect import signature

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

_DEFAULT_FONT = ['Noto Sans CJK SC', 'SimHei']

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = _DEFAULT_FONT
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

sns.set(font=_DEFAULT_FONT)


def set_font(font):
    """設置繪圖字體

    Args:
        font (list or tuple): 字體集合。
    """
    if font:
        matplotlib.rcParams['font.sans-serif'] = font
        sns.set(font=font)


def _get_param(cls, con_name):
    sig = signature(cls)
    return sig.parameters.get(con_name)


def get_param_default_value(cls, con_name):
    param = _get_param(cls, con_name)
    if param:
        return param.default
    return None


def calc_slope(arr):
    """計算斜度。
    將 二維數組 `arr` 中的每一項轉換為斜度值。

    Args:
        arr : 二維數組

    Returns: 二維數組
    """
    result = []
    for d in arr:
        result.append((d[-1] - d[0]) / len(d))
    return np.array(result)


def plot_result_by_pct_change(X, Y, window, days, figsize=(15, 5), top=100):
    """按預測值的斜率繪圖

    Args:
        figsize: 繪圖大小。默認為(15,5)。
        top (int): 繪製斜率**絕對值**前 n 位的線。默認100。
    X: 真實值數組。
           1. 三維數組。可以是從 `DataLoader.get_train_data`、
           `DataLoader.get_valid_data` 中獲取到的X，或與之格式一致的均可。
           計算時會取 `[:,0,0]` 來作為繪圖值。
           2. 一維數組。直接使用。
        Y: 預測值或未來值的數組。
           如果是一維數組，默認每一項為待繪製的斜率值，直接使用該值進行繪圖。
           如果是二維數組，會調用 `calc_slope` 將每一維的值計算為單一值，再當做斜率進行繪圖。
    days (int): `Y` 及 `other_Y` 繪製之間跳過的數量。
    other_Y: 其他與Y相同類型的二維數組。
                （可能 `Y` 為真實的未來值；`other_Y` 為預測值）
    window: 窗口期。繪圖時會先繪製X值，然後遍歷所有Y值得第一個維度，
                每次跳空 `window` 值後開始繪製。
    """
    if len(X.shape) == 3:
        X = X[:, 0, 0]
    # if len(Y.shape) == 2:
    #     sort_Y = [y[-1] - y[0] for y in Y]
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.plot(X, color='b')
    for i, data in enumerate(Y):
        try:
            # if top > 0 and (data[-1] - data[0] not in sorted(sort_Y)[:top] and
            #                 data[-1] - data[0] not in sorted(sort_Y)[-top:]):
            #     continue
            # 绘制预测值从开始点到结束点的直线
            plt.plot([i + window, i + window + days - 1],
                     [X[i + window], data[-1]],
                     color=('r' if data[-1] > X[i + window] else 'g'))
            # 绘制预测值从开始点到结束点的直线
        except Exception:
            continue
    return plt


def plot_result_by_slope(X, Y, window, days, figsize=(15, 5), top=100):
    """按預測值的斜率繪圖

    Args:
        figsize: 繪圖大小。默認為(15,5)。
        top (int): 繪製斜率**絕對值**前 n 位的線。默認100。
    X: 真實值數組。
           1. 三維數組。可以是從 `DataLoader.get_train_data`、
           `DataLoader.get_valid_data` 中獲取到的X，或與之格式一致的均可。
           計算時會取 `[:,0,0]` 來作為繪圖值。
           2. 一維數組。直接使用。
        Y: 預測值或未來值的數組。
           如果是一維數組，默認每一項為待繪製的斜率值，直接使用該值進行繪圖。
           如果是二維數組，會調用 `calc_slope` 將每一維的值計算為單一值，再當做斜率進行繪圖。
    days (int): `Y` 及 `other_Y` 繪製之間跳過的數量。
    other_Y: 其他與Y相同類型的二維數組。
                （可能 `Y` 為真實的未來值；`other_Y` 為預測值）
    window: 窗口期。繪圖時會先繪製X值，然後遍歷所有Y值得第一個維度，
                每次跳空 `window` 值後開始繪製。
    """
    if len(X.shape) == 3:
        X = X[:, 0, 0]
    if len(Y.shape) == 2:
        Y = calc_slope(Y)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.plot(X, color='b')
    for i, data in enumerate(Y):
        try:
            if top > 0 and (data not in sorted(Y)[:top] and
                            data not in sorted(Y)[-top:]):
                continue
            v = data * days + X[i + window]
            # 绘制预测值从开始点到结束点的直线
            plt.plot([i + window, i + window + days - 1],
                     [X[i + window], v],
                     color=('r' if data > 0 else 'g'))
            # 绘制预测值从开始点到结束点的直线
        except Exception:
            continue
    return plt


def plot_result(X, Y, window, days, figsize=(15, 5), top=100):
    """按預測值的繪圖

    Args:
        figsize: 繪圖大小。默認為(15,5)。
        top (int): 繪製**絕對值**前 n 位的線。默認100。
    X: 真實值數組。
           1. 三維數組。可以是從 `DataLoader.get_train_data`、
           `DataLoader.get_valid_data` 中獲取到的X，或與之格式一致的均可。
           計算時會取 `[:,0,0]` 來作為繪圖值。
           2. 一維數組。直接使用。
        Y : 預測值或未來值的數組。二維數組，每一維單獨繪製。
    days (int): `Y` 及 `other_Y` 繪製之間跳過的數量。
    other_Y: 其他與Y相同類型的二維數組。
                （可能 `Y` 為真實的未來值；`other_Y` 為預測值）
    window: 窗口期。繪圖時會先繪製X值，然後遍歷所有Y值得第一個維度，
                每次跳空 `window` 值後開始繪製。
    """
    if len(X.shape) == 3:
        X = X[:, 0, 0]
    sorted_y = sorted(calc_slope(Y)) if len(Y.shape) == 2 else None
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.plot(X, color='b')
    for i, data in enumerate(Y):
        try:
            slope = calc_slope([data])
            if slope and top > 0 and (slope not in sorted_y[:top]
                                      and slope not in sorted_y[-top:]):
                continue
            start = i + window - 1
            skip = [None for j in range(start)]
            # 绘制预测值从开始点到结束点的直线
            plt.plot(skip + [X[start]] + list(data),
                     color=('r' if slope > 0 else 'g'))
            # 绘制预测值从开始点到结束点的直线
        except Exception:
            continue
    return plt


def plot_history(history):
    # 绘制训练 & 验证的准确率值
    legend = []
    if 'acc' in history.history:
        plt.plot(history.history['acc'])
        legend.append('訓練集')
    if 'val_acc' in history.history:
        plt.plot(history.history['val_acc'])
        legend.append('測試集')
    plt.title('模型準確性')
    plt.ylabel('準確性')
    plt.xlabel('代數/輪次')
    plt.legend(legend, loc='upper left')
    plt.show()

    legend = []
    # 绘制训练 & 验证的损失值
    if 'loss' in history.history:
        plt.plot(history.history['loss'])
        legend.append('訓練集')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'])
        legend.append('測試集')
    plt.title('模型損失值')
    plt.ylabel('損失值')
    plt.xlabel('代數/輪次')
    plt.legend(legend, loc='upper left')
    plt.show()

# def plot(X, Y, window, figsize=(15, 5)):
#     """绘图
#
#     Args:
#         figsize: 绘图大小。默认为(15,5)
#         X: 真实值数组。三维数组。可以是从 `DataLoader.get_train_data`、
#            `DataLoader.get_valid_data` 中获取到的X，或与之格式一致的均可。
#            计算时会取 `[:,0,0]` 来作为绘图值。
#         Y: 预测值或未来值的二维数组。
#         days (int): `Y` 及 `other_Y` 绘制之间跳过的数量。
#         other_Y: 其他与Y相同类型的二维数组。（可能 `Y` 为真实的未来值；`other_Y` 为预测值）
#         window: 窗口期。绘图时会先绘制X值，然后遍历所有Y值得第一个维度，
#                 每次跳空 `window` 值后开始绘制。
#     """
#     # plt.figure(figsize=figsize)
#     # plt.plot(X[:, 0, 0])
#     # for i in range(len(Y)):
#     #     if days and i % days != 0:
#     #         continue
#     #     append = [None for p in range(window + i)]
#     #     plt.plot(append + list(Y[i]))
#     #     if other_Y is not None:
#     #         plt.plot(append + list(other_Y[i]))
#     fig = plt.figure(figsize=figsize)
#     ax = fig.add_subplot(111)
#     ax.plot(X[:, 0, 0], label='True Data')
#     # Pad the list of predictions to shift it in the graph to it's correct start
#     for i, data in enumerate(Y):
#         # 绘制所有预测值
#         padding = [None for p in range(i + window)]
#         plt.plot(padding + list(data))
#         # 绘制所有预测值
#         plt.legend()
#     return plt
