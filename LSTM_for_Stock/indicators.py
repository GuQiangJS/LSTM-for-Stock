# 指标计算器

import pandas as pd
import talib


def talib_OBV(DataFrame):
    res = talib.OBV(DataFrame.close.values, DataFrame.volume.values)
    return pd.DataFrame({'OBV': res}, index=DataFrame.index)


def talib_DEMA(DataFrame, N=30):
    res = talib.DEMA(DataFrame.close.values, timeperiod=N)
    return pd.DataFrame({'DEMA': res}, index=DataFrame.index)


def talib_KAMA(DataFrame, N=30):
    res = talib.KAMA(DataFrame.close.values, timeperiod=N)
    return pd.DataFrame({'KAMA': res}, index=DataFrame.index)

def talib_MIDPOINT(DataFrame, N=14):
    res = talib.MIDPOINT(DataFrame.close.values, timeperiod=N)
    return pd.DataFrame({'MIDPOINT': res}, index=DataFrame.index)

def talib_MIDPRICE(DataFrame, N=14):
    res = talib.MIDPRICE(DataFrame.high.values, DataFrame.low.values, timeperiod=N)
    return pd.DataFrame({'MIDPRICE': res}, index=DataFrame.index)

def talib_T3(DataFrame, N=5, vfactor=0):
    res = talib.T3(DataFrame.close.values, timeperiod=N,vfactor=vfactor)
    return pd.DataFrame({'T3': res}, index=DataFrame.index)

def talib_TEMA(DataFrame, N=30):
    res = talib.TEMA(DataFrame.close.values, timeperiod=N)
    return pd.DataFrame({'TEMA': res}, index=DataFrame.index)

def talib_TRIMA(DataFrame, N=30):
    res = talib.TRIMA(DataFrame.close.values, timeperiod=N)
    return pd.DataFrame({'TRIMA': res}, index=DataFrame.index)

def talib_WMA(DataFrame, N=30):
    res = talib.WMA(DataFrame.close.values, timeperiod=N)
    return pd.DataFrame({'WMA': res}, index=DataFrame.index)


def talib_AVGPRICE(DataFrame):
    """AVGPRICE - Average Price 平均价格函数"""
    res = talib.AVGPRICE(DataFrame.open.values, DataFrame.high.values,
                         DataFrame.low.values, DataFrame.close.values)
    return pd.DataFrame({'AVGPRICE': res}, index=DataFrame.index)


def talib_MEDPRICE(DataFrame):
    """MEDPRICE - Median Price 中位数价格"""
    res = talib.MEDPRICE(DataFrame.high.values, DataFrame.low.values)
    return pd.DataFrame({'MEDPRICE': res}, index=DataFrame.index)


def talib_TYPPRICE(DataFrame):
    """TYPPRICE - Typical Price 代表性价格"""
    res = talib.TYPPRICE(DataFrame.high.values, DataFrame.low.values,
                         DataFrame.close.values)
    return pd.DataFrame({'TYPPRICE': res}, index=DataFrame.index)


def talib_WCLPRICE(DataFrame):
    """WCLPRICE - Weighted Close Price 加权收盘价"""
    res = talib.WCLPRICE(DataFrame.high.values, DataFrame.low.values,
                         DataFrame.close.values)
    return pd.DataFrame({'WCLPRICE': res}, index=DataFrame.index)


def talib_NATR(DataFrame, N=14):
    res = talib.NATR(DataFrame.high.values, DataFrame.low.values,
                     DataFrame.close.values, timeperiod=N)
    return pd.DataFrame({'NATR': res}, index=DataFrame.index)


def talib_TRANGE(DataFrame):
    res = talib.TRANGE(DataFrame.high.values, DataFrame.low.values,
                       DataFrame.close.values)
    return pd.DataFrame({'TRANGE': res}, index=DataFrame.index)


def talib_APO(DataFrame, fastperiod=12, slowperiod=26, matype=0):
    res = talib.APO(DataFrame.close.values, fastperiod=fastperiod,
                    slowperiod=slowperiod, matype=matype)
    return pd.DataFrame({'APO': res}, index=DataFrame.index)


def talib_DEMA(DataFrame, N=30):
    res = talib.DEMA(DataFrame.close.values, timeperiod=N)
    return pd.DataFrame({'DEMA': res}, index=DataFrame.index)
