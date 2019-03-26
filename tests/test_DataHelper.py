"""
测试数据拆分 DataHelper
"""

import pytest
import pandas as pd
from LSTM_for_Stock.data_processor import DataHelper
import logging
import numpy as np
from numpy import array_equal as arr_eq
from numpy import array as arr


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
    window = 5
    days = 1
    train_X, train_Y, test_X, test_Y = DataHelper.train_test_split(
        df, train_size=1, window=window, days=days)
    # logging.info(train_X)
    # logging.info(train_Y)
    assert train_X is not None
    assert train_Y is not None
    assert test_X is not None
    assert test_Y is not None
    assert len(train_X) == 2
    assert len(train_Y) == 2
    assert len(test_X) == 0
    assert len(test_Y) == 0
    #判断内容是否正确
    for i in range(2):
        assert train_X[i].size == window
        assert train_Y[i].size == days
        assert arr_eq(train_X[i]['v'].values, arr(dic['v'][i:i + window]))
        assert arr_eq(train_Y[i].values,
                      arr(dic['close'][i + window:window + days + i]))


def test_train_test_split_2():
    """不够拆分一个训练集
    ZZ
    """
    dic = {'close': [], 'v': []}
    for i in range(2):
        dic['close'].append(i)
        dic['v'].append(i)
    df = pd.DataFrame.from_dict(dic)
    train_X, train_Y, test_X, test_Y = DataHelper.train_test_split(
        df, train_size=1, window=2, days=1)
    # logging.info(train_X)
    # logging.info(train_Y)
    assert train_X is not None
    assert train_Y is not None
    assert test_X is not None
    assert test_Y is not None
    assert len(train_X) == 0
    assert len(train_Y) == 0
    assert len(test_X) == 0
    assert len(test_Y) == 0


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
    window = 5
    days = 1
    train_X, train_Y, test_X, test_Y = DataHelper.train_test_split(
        df, train_size=1, window=window, days=days)
    # logging.info(train_X)
    # logging.info(train_Y)
    assert train_X is not None
    assert train_Y is not None
    assert test_X is not None
    assert test_Y is not None
    assert len(train_X) == 6
    assert len(train_Y) == 6
    assert len(test_X) == 0
    assert len(test_Y) == 0
    for i in range(6):
        assert train_X[i].size == window
        assert train_Y[i].size == days
        assert arr_eq(train_X[i]['v'].values, arr(dic['v'][i:i + window]))
        assert arr_eq(train_Y[i].values,
                      arr(dic['close'][i + window:window + days + i]))


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
    window = 5
    days = 1
    train_X, train_Y, test_X, test_Y = DataHelper.train_test_split(
        df, train_size=0.6, window=window, days=days)
    # logging.info(train_X)
    # logging.info(train_Y)
    assert train_X is not None
    assert train_Y is not None
    assert test_X is not None
    assert test_Y is not None
    assert len(train_X) == 4
    assert len(train_Y) == 4
    assert len(test_X) == 2
    assert len(test_Y) == 2
    for i in range(6):
        if i + 1 <= 4:
            assert train_X[i].size == window
            assert train_Y[i].size == days
            assert arr_eq(train_X[i]['v'].values, arr(dic['v'][i:i + window]))
            assert arr_eq(train_Y[i].values,
                          arr(dic['close'][i + window:window + days + i]))
        else:
            assert test_X[i - 4].size == window
            assert test_Y[i - 4].size == days
            assert arr_eq(test_X[i - 4]['v'].values,
                          arr(dic['v'][i:i + window]))
            assert arr_eq(test_Y[i - 4].values,
                          arr(dic['close'][i + window:window + days + i]))


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
    window = 5
    days = 1
    train_X, train_Y, test_X, test_Y = DataHelper.train_test_split(
        df, train_size=0.5, window=window, days=days)
    # logging.info(train_X)
    # logging.info(train_Y)
    #总共12条数据，根据训练集0.6的比率，训练集为6条，测试集为6条
    #训练集6条，window=5，days=1，刚好取一条train_X和train_Y
    #训练集6条，window=5，days=1，刚好取一条test_X和test_Y
    assert train_X is not None
    assert train_Y is not None
    assert test_X is not None
    assert test_Y is not None
    assert len(train_X) == 4
    assert len(train_Y) == 4
    assert len(test_X) == 4
    assert len(test_Y) == 4
    for i in range(6):
        if i + 1 <= 4:
            assert train_X[i].size == window
            assert train_Y[i].size == days
            assert arr_eq(train_X[i]['v'].values, arr(dic['v'][i:i + window]))
            assert arr_eq(train_Y[i].values,
                          arr(dic['close'][i + window:window + days + i]))
        else:
            assert test_X[i - 4].size == window
            assert test_Y[i - 4].size == days
            assert arr_eq(test_X[i - 4]['v'].values,
                          arr(dic['v'][i:i + window]))
            assert arr_eq(test_Y[i - 4].values,
                          arr(dic['close'][i + window:window + days + i]))


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
    window = 4
    days = 2
    train_X, train_Y, test_X, test_Y = DataHelper.train_test_split(
        df, window=window, days=days)
    assert len(train_X) == 7
    assert len(train_Y) == 7
    assert len(test_X) == 1
    assert len(test_Y) == 1
    for i in range(7):
        if i + 1 <= 7:
            assert train_X[i].size == window
            assert train_Y[i].size == days
            assert arr_eq(train_X[i]['v'].values, arr(dic['v'][i:i + window]))
            assert arr_eq(train_Y[i].values,
                          arr(dic['close'][i + window:window + days + i]))
        else:
            assert test_X[i - 7].size == window
            assert test_Y[i - 7].size == days
            assert arr_eq(test_X[i - 7]['v'].values,
                          arr(dic['v'][i:i + window]))
            assert arr_eq(test_Y[i - 7].values,
                          arr(dic['close'][i + window:window + days + i]))


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
    window = 5
    days = 1
    train_X, train_Y, test_X, test_Y = DataHelper.train_test_split(
        df, train_size=1, window=window, days=days)
    del test_X
    del test_Y
    for i in range(len(train_X)):
        logging.info('训练集 X DataFrame')
        logging.info(train_X[i])
        logging.info('训练集 X numpy.ndarray')
        logging.info(train_X[i].values)
        logging.info('训练集 Y DataFrame')
        logging.info(train_Y[i])
        logging.info('训练集 Y numpy.ndarray')
        logging.info(train_Y[i].values)