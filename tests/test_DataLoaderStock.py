"""
训练 DataLoaderStock 及 相关的 Wrapper
"""
import datetime
import logging

import numpy as np
import pandas as pd
from QUANTAXIS.QAFetch.QAQuery_Advance import \
    QA_fetch_stock_block_adv as get_block

from LSTM_for_Stock.data_processor import DataLoader
from LSTM_for_Stock.data_processor import DataLoaderStock
from LSTM_for_Stock.data_processor import Wrapper
from LSTM_for_Stock.data_processor import Wrapper_default
from LSTM_for_Stock.data_processor import Wrapper_fillna
from LSTM_for_Stock.data_processor import get_ipo_date
from LSTM_for_Stock.data_processor import get_block_code

def test_init():
    dl = DataLoaderStock('601398')
    assert '601398' == dl.stock_code
    assert '399300' == dl.benchmark_code
    assert '1990-01-01' == dl.start
    assert DataLoader.today() == dl.end
    assert 'qfq' == dl.fq
    assert False == dl.online

    dl = DataLoaderStock('601398', '000300')
    assert '601398' == dl.stock_code
    assert '000300' == dl.benchmark_code
    assert '1990-01-01' == dl.start
    assert DataLoader.today() == dl.end
    assert 'qfq' == dl.fq
    assert False == dl.online

    dl = DataLoaderStock('601398', '000300', 'bfq')
    assert '601398' == dl.stock_code
    assert '000300' == dl.benchmark_code
    assert '1990-01-01' == dl.start
    assert DataLoader.today() == dl.end
    assert 'bfq' == dl.fq
    assert False == dl.online

    dl = DataLoaderStock('601398', '000300', 'bfq', True)
    assert '601398' == dl.stock_code
    assert '000300' == dl.benchmark_code
    assert '1990-01-01' == dl.start
    assert DataLoader.today() == dl.end
    assert 'bfq' == dl.fq
    assert True == dl.online

    dl = DataLoaderStock('601398', '000300', 'bfq', True, '2000-01-01')
    assert '601398' == dl.stock_code
    assert '000300' == dl.benchmark_code
    assert '2000-01-01' == dl.start
    assert DataLoader.today() == dl.end
    assert 'bfq' == dl.fq
    assert True == dl.online

    dl = DataLoaderStock('601398', '000300', 'bfq', True, '2000-01-01',
                         '2000-12-31')
    assert '601398' == dl.stock_code
    assert '000300' == dl.benchmark_code
    assert '2000-01-01' == dl.start
    assert '2000-12-31' == dl.end
    assert 'bfq' == dl.fq
    assert True == dl.online


def test_fetch_stock_day_online():
    dl = DataLoaderStock('601398')
    df = dl._DataLoaderStock__fetch_stock_day_online(dl.stock_code,
                                                     dl.start,
                                                     dl.end)
    logging.info(df.columns)
    logging.info(df.head())
    assert set(dl._stock_columns) == set(df.columns)
    assert not df.empty
    for col in df.columns:
        assert df[col].dtype == np.float32


def test_fetch_index_day_online():
    dl = DataLoaderStock('601398')
    df = dl._DataLoaderStock__fetch_index_day_online()
    logging.info(df.columns)
    logging.info(df.head())
    assert set(dl._index_columns) == set(df.columns)
    assert not df.empty
    for col in df.columns:
        assert df[col].dtype == np.float32


def test_fetch_stock_day():
    dl = DataLoaderStock('601398')
    df = dl._DataLoaderStock__fetch_stock_day(dl.stock_code,
                                              dl.start,
                                              dl.end)
    logging.info(df.columns)
    logging.info(df.head())
    assert set(dl._stock_columns) == set(df.columns)
    assert not df.empty
    for col in df.columns:
        assert df[col].dtype == np.float32


def test_fetch_index_day():
    dl = DataLoaderStock('601398')
    df = dl._DataLoaderStock__fetch_index_day()
    logging.info(df.columns)
    logging.info(df.head())
    assert set(dl._index_columns) == set(df.columns)
    assert not df.empty
    for col in df.columns:
        assert df[col].dtype == np.float32


def test_load():
    dl = DataLoaderStock('601398')
    df = dl.load()
    logging.info(df.columns)
    logging.info(df.head())
    assert not df.empty

    class wrapper1(Wrapper):
        def build(self, df: pd.DataFrame) -> pd.DataFrame:
            result = df.copy()
            result = result.fillna(method='ffill')
            return result.dropna()

    dl = DataLoaderStock('601398', wrapper=wrapper1())
    df = dl.load()
    assert len(df.index) != len(dl.data_raw.index)
    logging.info(df.head())
    logging.info(dl.data_raw.head())
    assert not df.empty

    dl_fillna = DataLoaderStock('601398', wrapper=Wrapper_fillna())
    df_fillna = dl_fillna.load()
    assert df.equals(df_fillna)
    logging.info(df_fillna.shape)
    for col in df.columns:
        np.array_equal(df[col].values, df_fillna[col].values)


def _test_dt(code):
    """判断股票上市时间是否晚于指定时间"""
    try:
        return datetime.datetime(2005, 1, 1) >= get_ipo_date(code)
    except:
        return False


def _test_code(code):
    return code[0] in ['0', '3', '6']

def test_append_codes():
    codes = [code for code in get_block_code('000002') if
             _test_dt(code) and _test_code(code)]
    dl = DataLoaderStock('000002', wrapper=Wrapper_default(),
                         appends=codes)
    print(codes)
    assert len(codes)>0
    df = dl.load()
    assert not df.empty
    print(df)
    print(len(df.columns))


# def test_111():
#     import time
#     import os
#     import datetime
#     f="C:/Users/GuQiang/Downloads/阿加莎--无人生还.mobi"
#     time.localtime(os.stat(f).st_mtime)
