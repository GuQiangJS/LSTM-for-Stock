# 循环训练所有上市日期<399300开始日期的股票

import os
import sys

nb_dir = os.path.split(os.getcwd())[0]
if nb_dir not in sys.path:
    sys.path.append(nb_dir)
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from keras.backend import clear_session
from LSTM_for_Stock.model import SequentialModel
from LSTM_for_Stock.data_processor import DataHelper
from LSTM_for_Stock.data_processor import DataLoaderStock
from LSTM_for_Stock.data_processor import Wrapper
import numpy as np
from LSTM_for_Stock.unit import PlotHelper
import keras
import matplotlib.pyplot as plt
import logging
import time
import QUANTAXIS as QA
import pandas as pd
from datetime import datetime
from scipy import stats
import talib

logger = logging.getLogger()
logger.setLevel('INFO')
BASIC_FORMAT = "%(asctime)s:%(levelname)s:%(message)s"
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
chlr = logging.StreamHandler()  # 输出到控制台的handler
chlr.setFormatter(formatter)
chlr.setLevel('INFO')  # 也可以不设置，不设置就默认用logger的level
fhlr = logging.FileHandler(
    os.path.join(nb_dir, '.train_result', 'train_all.log'))  # 输出到文件的handler
fhlr.setFormatter(formatter)
logger.addHandler(chlr)
logger.addHandler(fhlr)


class wrapper(Wrapper):
    def build(self, df):
        result = df.copy()
        result = result.fillna(method='ffill')
        result = result.drop(columns=['up_count', 'down_count'])
        return result.dropna()

def talib_AVGPRICE(DataFrame):
    """AVGPRICE - Average Price 平均价格函数"""
    res = talib.AVGPRICE(DataFrame.open.values,DataFrame.high.values,DataFrame.low.values,DataFrame.close.values)
    return pd.DataFrame({'AVGPRICE': res}, index=DataFrame.index)

def talib_MEDPRICE(DataFrame):
    """MEDPRICE - Median Price 中位数价格"""
    res = talib.MEDPRICE(DataFrame.high.values,DataFrame.low.values)
    return pd.DataFrame({'MEDPRICE': res}, index=DataFrame.index)

def talib_TYPPRICE(DataFrame):
    """TYPPRICE - Typical Price 代表性价格"""
    res = talib.TYPPRICE(DataFrame.high.values,DataFrame.low.values,DataFrame.close.values)
    return pd.DataFrame({'TYPPRICE': res}, index=DataFrame.index)

def talib_WCLPRICE(DataFrame):
    """WCLPRICE - Weighted Close Price 加权收盘价"""
    res = talib.WCLPRICE(DataFrame.high.values,DataFrame.low.values,DataFrame.close.values)
    return pd.DataFrame({'WCLPRICE': res}, index=DataFrame.index)

def talib_NATR(DataFrame,N=14):
    res = talib.NATR(DataFrame.high.values,DataFrame.low.values,DataFrame.close.values,timeperiod=N)
    return pd.DataFrame({'NATR': res}, index=DataFrame.index)

def talib_TRANGE(DataFrame):
    res = talib.TRANGE(DataFrame.high.values,DataFrame.low.values,DataFrame.close.values)
    return pd.DataFrame({'TRANGE': res}, index=DataFrame.index)

def talib_APO(DataFrame, fastperiod=12, slowperiod=26, matype=0):
    res = talib.APO(DataFrame.close.values, fastperiod=fastperiod, slowperiod=slowperiod, matype=matype)
    return pd.DataFrame({'APO': res}, index=DataFrame.index)

def talib_DEMA(DataFrame,N=30):
    res = talib.DEMA(DataFrame.close.values,timeperiod=N)
    return pd.DataFrame({'DEMA': res}, index=DataFrame.index)

class wrapper_CCI(Wrapper):
    def build(self, df):
        result = df.copy()

        result['ATR_14']=QA.talib_indicators.ATR(result)
        result['ATR_5']=QA.talib_indicators.ATR(result,N=5)
        result['ATR_30']=QA.talib_indicators.ATR(result,N=30)
        result['NATR_14']=talib_NATR(result)
        result['NATR_5']=talib_NATR(result,N=5)
        result['NATR_30']=talib_NATR(result,N=30)
        result['TRANGE']=talib_NATR(result)

        result['AVGPRICE']=talib_AVGPRICE(result)
        result['MEDPRICE']=talib_MEDPRICE(result)
        result['TYPPRICE']=talib_TYPPRICE(result)
        result['WCLPRICE']=talib_WCLPRICE(result)

        result['EMA_5']=QA.QA_indicator_EMA(result,5)
        result['EMA_10']=QA.QA_indicator_EMA(result,10)
        result['EMA_15']=QA.QA_indicator_EMA(result,15)
        result['EMA_30']=QA.QA_indicator_EMA(result,30)

        # result['DEMA_5']=talib_DEMA(result,5)
        # result['DEMA_10']=talib_DEMA(result,10)
        # result['DEMA_15']=talib_DEMA(result,15)
        # result['DEMA_30']=talib_DEMA(result,30)

        result = result.fillna(method='ffill')
        result = result.drop(columns=['up_count', 'down_count'])
        return result.dropna()

class normalize_CCI(object):
    """数据标准化器"""

    def __init__(self, *args, **kwargs):
        pass

    def build(self, df):
        """执行数据标准化。**数据归一化**。

        Args:
            df (pd.DataFrame 或 pd.Series): 待处理的数据。

        Returns:
            pd.DataFrame 或 pd.Series: 与传入类型一致。
        """
        tmp=df.copy()
        for col in tmp.columns:
            if col in ['ATR_14','ATR_5','ATR_30','NATR_14','NATR_5','NATR_30','TRANGE']:
                continue
            if col in ['CCI']:
                tmp[col]=sklearn.preprocessing.normalize([tmp[col]])[0]
            tmp[col]=tmp[col] / tmp.iloc[0][col]
        return tmp
    

class normalize(object):
    """数据标准化器"""

    def __init__(self, *args, **kwargs):
        pass

    def build(self, df):
        """执行数据标准化。**数据归一化**。

        Args:
            df (pd.DataFrame 或 pd.Series): 待处理的数据。

        Returns:
            pd.DataFrame 或 pd.Series: 与传入类型一致。
        """
        return np.round(df / df.iloc[0], 8)


def _get_model_file_path(code, window, days, p: str = None):
    if p is None:
        p = os.path.join(nb_dir, '.train_result')
    filename = 'model_{2}_{0:02d}_{1:02d}.h5'.format(window, days, code)
    return os.path.join(p, filename)


def save_model(m: keras.models.Model, p: str = None, *args, **kwargs):
    if p is None:
        p = os.path.join(nb_dir, '.train_result')
    os.makedirs(p, exist_ok=True)
    window = kwargs.pop('window', None)
    days = kwargs.pop('days', None)
    stockcode = kwargs.pop('stockcode', None)
    if stockcode is None or window is None or days is None:
        raise ValueError()
    p = _get_model_file_path(stockcode, window, days, p)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    m.save(p)
    return p


def save_history_img(history, p: str = None, *args, **kwargs):
    if p is None:
        p = os.path.join(nb_dir, '.train_result')
    os.makedirs(p, exist_ok=True)
    window = kwargs.pop('window', None)
    days = kwargs.pop('days', None)
    benchmark = kwargs.pop('benchmark', None)
    stockcode = kwargs.pop('stockcode', None)
    filename = 'history_{2}_{0:02d}_{1:02d}.svg'.format(
        window, days, stockcode)
    save_path = os.path.join(p, filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    PlotHelper.plot_history(
        history,
        stockcode=stockcode,
        benchmark=benchmark,
        window=window,
        days=days,
        show=False,
        save_path=save_path)
    return save_path


def do(code, window, days, *args, **kwargs):
    exists_file = _get_model_file_path(code, window, days,
                                       os.path.join(nb_dir, '.train_result'))
    if os.path.exists(exists_file) and kwargs.pop('rebuild', False) == False:
        if (datetime.now() - datetime.fromtimestamp(
                time.mktime(time.localtime(
                    os.stat(exists_file).st_mtime)))).days < window:
            logging.info(
                "{0}:{1} Last Modified < {2}".format(code, exists_file, window))
            return None

    # logging.info('{0} - {1:02d} - {2:02d} Start.'.format(code,window,days))
    dl = DataLoaderStock(code, wrapper=wrapper_CCI())
    df = dl.load()
    train, test = DataHelper.train_test_split(df, batch_size=window + days)

    X_train, Y_train = DataHelper.xy_split_2(train, window, days,
                                             norm=normalize_CCI())
    X_test, Y_test = DataHelper.xy_split_2(
        test, window, days, norm=normalize_CCI())

    X_train_arr = []
    Y_train_arr = []
    for x in X_train:
        X_train_arr.append(x.values)
    for y in Y_train:
        Y_train_arr.append(y.values)
    X_test_arr = []
    Y_test_arr = []
    for x in X_test:
        X_test_arr.append(x.values)
    for y in Y_test:
        Y_test_arr.append(y.values)

    layers = [
        {'units': 100, 'type': 'lstm', 'input_shape': X_train_arr[0].shape,
         'return_sequences': True},
        {'type': 'dropout', 'rate': 0.15},
        {'units': 200, 'type': 'lstm',
         'return_sequences': True},
        {'type': 'dropout', 'rate': 0.15},
        {'units': 100, 'type': 'lstm',
         'return_sequences': False},
        # {'units': 500, 'type': 'lstm', 'input_shape': X_train_arr[0].shape},
        {'units': days, 'type': 'dense'}]
    complie = {
        #     "optimizer":"adam",
        "loss": "mse",
        "optimizer": "rmsprop",
        #     "loss":"categorical_crossentropy",
        "metrics": [
            "mae", "acc"
        ]
    }

    model = SequentialModel()
    model.build_model(layers, complie)
    batch_size = kwargs.pop("batch_size", 128)
    history = model.train(
        np.array(X_train_arr),
        np.array(Y_train_arr),
        train={
            'epochs': kwargs.pop("train_train_epochs", 1000),
            'verbose': kwargs.pop("train_verbose", 0),
            'batch_size': batch_size,
            'validation_split': kwargs.pop("train_valid_split", 0.15)
        })

    if kwargs.pop('show_summary', False):
        model.model.summary()

    save_path = save_model(model.model, stockcode=code, window=window,
                           days=days)
    logging.info('model savmodeed:' + save_path)
    # his_image_path=save_history_img(history,stockcode=code,window=window,days=days,benchmark=dl.benchmark_code)
    # logging.info('history image:'+his_image_path)

    pred = model.predict(np.array(X_test_arr))
    pred_slope = []
    for day in range(days):
        df_result = pd.DataFrame(
            {'pred': pred[:, day], 'real': np.array(Y_test_arr)[:, day]})

        slope = stats.linregress(pred[:, day],
                                 np.array(Y_test_arr)[:, day]).slope
        print('Slope Day{0}:{1}'.format(day + 1, slope))
        pred_slope.append(slope)
        # save_path=os.path.join(os.path.join(nb_dir,'.train_result'), 'pred_{2}_{0:02d}_{1:02d}.csv'.format(window,days,code))
        # os.makedirs(os.path.dirname(save_path),exist_ok=True)
        # df_result.to_csv(save_path, encoding="utf-8")
        # logging.info('pred result dataframe:'+save_path)

        plt.figure(figsize=(15, 8))
        save_path = os.path.join(os.path.join(nb_dir, '.train_result'),
                                 'pred_{2}_{0:02d}_{1:02d}_{3:02d}.svg'.format(
                                     window, days,
                                     code, day + 1))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.title(
            '{0} Window:{1},Days:{5}/{2},BatchSize:{3},Optimizer:{4}'.format(
                code, window, days, batch_size, complie['optimizer'], day + 1
            ))
        plt.plot(df_result['pred'])
        plt.plot(df_result['real'])
        plt.savefig(save_path, format="svg")
    # RuntimeWarning: More than 20 figures have been opened. Figures created
    # through the pyplot interface (`matplotlib.pyplot.figure`) are retained
    # until explicitly closed and may consume too much memory. (To control
    # this warning, see the rcParam `figure.max_open_warning`).
    plt.close('all')
    # logging.info('pred result image:'+save_path)
    # logging.info('{0} - {1:02d} - {2:02d} Done.'.format(code,window,days))
    logging.info("".join(['-' for i in range(50)]))
    if kwargs.pop('clear', True):
        clear_session()
    return {'slope': pred_slope}


if __name__ == "__main__":
    lst = QA.QA_fetch_stock_list_adv().code.values

    index = QA.QA_fetch_index_day_adv('399300', start='1990-01-01',
                                      end='2019-03-31')
    # logging.info(index.date[0].date())
    skips=[]
    valid_lst = []
    for code in lst:

        stock = QA.QA_fetch_stock_day_adv(code, start='1990-01-01',
                                          end='2019-03-31')
        if stock and stock.date[0].date() <= index.date[
            0].date():
            v = 1
            times = 0
            last_slope=100
            while (v > 0.005):
                times = times + 1
                logging.info(
                    '{0}/{1} - {2} Times:{3}'.format(list(lst).index(code) + 1,
                                                     len(lst), code, times))
                s = do(code, 5, 1, rebuild=True if times > 1 else False,show_summary=True)
                if s:
                    v = abs(s['slope'][0] - 1)
                    if v==last_slope:
                        skips.append(code)
                        logging.warning('{0} over times.'.format(code))
                        break
                    print('Slope:{}'.format(v))
                else:
                    break
            print(
                '{0}/{1} - {2} - Done.'.format(list(lst).index(code) + 1,
                                               len(lst), code))
        else:
            logging.info("SKIP:" + code)
    
    print('skips:{0}'.format(','.join(skips)))
