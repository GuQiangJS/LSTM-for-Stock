"""
测试数据拆分 DataHelper
"""

import logging

import numpy as np
import pandas as pd

from LSTM_for_Stock.data_processor import DataHelper


def test_train_test_split_1():
    """刚好够拆分一个训练集
    ZZZZZZZ
    XXXXXY_
    _XXXXXY
    """
    dic = {'close': [], 'v': []}
    for i in range(7):
        dic['close'].append(i)
        dic['v'].append(i)
    df = pd.DataFrame.from_dict(dic)
    batch_size = 6
    train, test = DataHelper.train_test_split(df, batch_size=batch_size,
                                              train_size=1)
    # logging.info(train_X)
    # logging.info(train_Y)
    assert train is not None
    assert test is not None
    assert len(train) == 2
    assert len(test) == 0
    # 判断内容是否正确
    for i in range(2):
        for k in dic.keys():
            assert train[i][k].size == batch_size


def test_train_test_split_2():
    """不够拆分一个训练集
    ZZ
    """
    dic = {'close': [], 'v': []}
    for i in range(2):
        dic['close'].append(i)
        dic['v'].append(i)
    df = pd.DataFrame.from_dict(dic)
    batch_size = 3
    train, test = DataHelper.train_test_split(
        df, batch_size=batch_size, train_size=1)
    # logging.info(train_X)
    # logging.info(train_Y)
    assert train is not None
    assert test is not None
    assert len(train) == 0
    assert len(test) == 0


def test_train_test_split_3():
    """可以拆分多个训练集
    ZZZZZZZZZZZ
    XXXXXY_____
    _XXXXXY____
    __XXXXXY___
    ___XXXXXY__
    ____XXXXXY_
    _____XXXXXY
    """
    dic = {'close': [], 'v': []}
    for i in range(11):
        dic['close'].append(i)
        dic['v'].append(i)
    df = pd.DataFrame.from_dict(dic)
    batch_size = 6
    train, test = DataHelper.train_test_split(
        df, batch_size=batch_size, train_size=1)
    assert train is not None
    assert test is not None
    assert len(train) == 6
    assert len(test) == 0
    for i in range(6):
        for k in dic.keys():
            assert train[i][k].size == batch_size


def test_train_test_split_4():
    """虽然数据多，但是根据拆分比率，训练集只有7条，只够一条训练集
    ZZZZZZZZZZZ
    XXXXXY_____
    _XXXXXY____
    __XXXXXY___
    ___XXXXXY__
    ~~~~~~~~~~训练集占比0.6,以上为训练集，以下为测试集
    ____XXXXXY_
    _____XXXXXY
    """
    dic = {'close': [], 'v': []}
    for i in range(11):
        dic['close'].append(i)
        dic['v'].append(i)
    df = pd.DataFrame.from_dict(dic)
    batch_size = 6
    train, test = DataHelper.train_test_split(df, batch_size=batch_size,
                                              train_size=0.6)
    assert train is not None
    assert test is not None
    assert len(train) == 4
    assert len(test) == 2
    for i in range(6):
        if i + 1 <= 4:
            for k in dic.keys():
                assert train[i][k].size == batch_size
        else:
            for k in dic.keys():
                assert test[i - 4][k].size == batch_size


def test_train_test_split_5():
    """
    ZZZZZZZZZZZZZ
    XXXXXY_______
    _XXXXXY______
    __XXXXXY_____
    ___XXXXXY____
    ~~~~~~~~~~~~~训练集占比0.5,以上为训练集，以下为测试集
    ____XXXXXY___
    _____XXXXXY__
    ______XXXXXY_
    _______XXXXXY
    """
    dic = {'close': [], 'v': []}
    for i in range(13):
        dic['close'].append(i)
        dic['v'].append(i)
    df = pd.DataFrame.from_dict(dic)
    batch_size = 6
    train, test = DataHelper.train_test_split(df, batch_size=batch_size,
                                              train_size=0.5, )
    # logging.info(train_X)
    # logging.info(train_Y)
    # 总共12条数据，根据训练集0.6的比率，训练集为6条，测试集为6条
    # 训练集6条，window=5，days=1，刚好取一条train_X和train_Y
    # 训练集6条，window=5，days=1，刚好取一条test_X和test_Y
    assert train is not None
    assert test is not None
    assert len(train) == 4
    assert len(test) == 4
    for i in range(6):
        if i + 1 <= 4:
            for k in dic.keys():
                assert train[i][k].size == batch_size
        else:
            for k in dic.keys():
                assert test[i - 4][k].size == batch_size


def test_train_test_split_6():
    """
    ZZZZZZZZZZZZZ
    XXXXYY_______
    _XXXXYY______
    __XXXXYY_____
    ___XXXXYY____
    ____XXXXYY___
    _____XXXXYY__
    ______XXXXYY_
    ~~~~~~~~~~~~默认拆分比率0.85
    _______XXXXYY
    """
    dic = {'close': [], 'v': []}
    for i in range(13):
        dic['close'].append(i)
        dic['v'].append(i)
    df = pd.DataFrame.from_dict(dic)
    batch_size = 6
    train, test = DataHelper.train_test_split(df, batch_size=batch_size)
    assert len(train) == 7
    assert len(test) == 1
    for i in range(7):
        if i + 1 <= 7:
            for k in dic.keys():
                assert train[i][k].size == batch_size
        else:
            for k in dic.keys():
                assert test[i - 7][k].size == batch_size


def test_dataframe_series_to_array():
    """
    ZZZZZZZ
    XXXXXY_
    _XXXXXY
    """
    dic = {'close': [], 'X': [], 'Y': [], 'Z': []}
    for i in range(7):
        dic['close'].append(i)
        dic['X'].append(i)
        dic['Y'].append(i)
        dic['Z'].append(i)
    df = pd.DataFrame.from_dict(dic)
    batch_size = 6
    train, test = DataHelper.train_test_split(
        df, batch_size=batch_size, train_size=1)
    del test
    for i in range(len(train)):
        logging.info('训练集 X DataFrame')
        logging.info(train[i])
        logging.info('训练集 X numpy.ndarray')
        logging.info(train[i].values)
        logging.info('训练集 Y DataFrame')
        logging.info(train[i])
        logging.info('训练集 Y numpy.ndarray')
        logging.info(train[i].values)


def test_xy_split_1():
    arr = [i for i in range(2, 8)]
    window = len(arr) - 2
    days = 2
    x, y = DataHelper.xy_split_1([pd.DataFrame(arr, columns=['c'])], window,
                                 days,
                                 col_name='c')
    logging.info(arr)
    logging.info(x)
    logging.info(y)
    logging.info(type(x[0]))
    logging.info(type(y[0]))
    assert np.array_equal(np.array([1, 1.5, 2, 2.5]), np.array(x[0]['c'].values))
    assert np.array_equal(np.array([1.2, 1.4]), np.array(y[0].values))


def test_xy_split_2():
    arr = [i for i in range(2, 8)]
    window = len(arr) - 2
    days = 2
    x, y = DataHelper.xy_split_2([pd.DataFrame(arr, columns=['c'])], window,
                                 days,
                                 col_name='c')
    logging.info(arr)
    logging.info(x)
    logging.info(y)
    logging.info(type(x[0]))
    logging.info(type(y[0]))
    s = pd.Series()
    s.describe()
    assert np.array_equal(np.array([1, 1.5, 2, 2.5]), np.array(x[0]['c'].values))
    assert np.array_equal(np.array([3, 3.5]), np.array(y[0].values))
