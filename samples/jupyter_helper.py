import os
import sys

nb_dir = os.path.split(os.getcwd())[0]
if nb_dir not in sys.path:
    sys.path.append(nb_dir)
import traceback
import pandas as pd
from LSTM_for_Stock.model import SequentialModel
from LSTM_for_Stock.data_processor import DataHelper
from LSTM_for_Stock.data_processor import DataLoaderStock
import logging
from LSTM_for_Stock.data_processor import Wrapper
from LSTM_for_Stock.data_processor import Wrapper_fillna
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.backend import clear_session
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import time
import numpy as np
from sklearn.model_selection import cross_val_score  # K折交叉验证模块
import matplotlib
from LSTM_for_Stock.loss import root_mean_squared_error
matplotlib.rcParams["figure.figsize"] = [16, 5]
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号\n
import QUANTAXIS as QA
import sklearn
import talib
import seaborn as sns
from scipy import stats
from LSTM_for_Stock import indicators


class wrapper(Wrapper):
    def build(self, df):
        result = df.copy()
        result = result.fillna(method='ffill')
        result = result.drop(columns=['up_count', 'down_count'])
        return result.dropna()


class wrapper_CCI(Wrapper):
    def build(self, df):
        result = df.copy()
        result = result.fillna(method='ffill')
        result = result.drop(columns=['up_count', 'down_count'])

        result['AD'] = QA.talib_indicators.AD(result)
        result['ADOSC_5_10'] = QA.talib_indicators.ADOSC(result, N1=5, N2=10)
        # result['ADOSC_5_15'] = QA.talib_indicators.ADOSC(result, N1=5, N2=15)
        # result['ADOSC_5_30'] = QA.talib_indicators.ADOSC(result, N1=5, N2=30)
        # result['ADOSC_10_20'] = QA.talib_indicators.ADOSC(result, N1=10, N2=20)
        # result['ADOSC_10_30'] = QA.talib_indicators.ADOSC(result, N1=10, N2=30)
        result['OBV'] = indicators.talib_OBV(result)

        result['ATR_5'] = QA.talib_indicators.ATR(result, N=5)
        # result['ATR_10'] = QA.talib_indicators.ATR(result, N=10)
        # result['ATR_15'] = QA.talib_indicators.ATR(result, N=15)
        # result['ATR_20'] = QA.talib_indicators.ATR(result, N=20)
        # result['ATR_30'] = QA.talib_indicators.ATR(result, N=30)
        result['NATR_5'] = indicators.talib_NATR(result, N=5)
        # result['NATR_10'] = indicators.talib_NATR(result, N=10)
        # result['NATR_15'] = indicators.talib_NATR(result, N=15)
        # result['NATR_20'] = indicators.talib_NATR(result, N=20)
        # result['NATR_30'] = indicators.talib_NATR(result, N=30)
        result['TRANGE'] = indicators.talib_NATR(result)

        result['AVGPRICE'] = indicators.talib_AVGPRICE(result)
        result['MEDPRICE'] = indicators.talib_MEDPRICE(result)
        result['TYPPRICE'] = indicators.talib_TYPPRICE(result)
        result['WCLPRICE'] = indicators.talib_WCLPRICE(result)

        result['EMA_5'] = QA.QA_indicator_EMA(result, 5)
        # result['EMA_10'] = QA.QA_indicator_EMA(result, 10)
        # result['EMA_15'] = QA.QA_indicator_EMA(result, 15)
        # result['EMA_20'] = QA.QA_indicator_EMA(result, 20)
        # result['EMA_30'] = QA.QA_indicator_EMA(result, 30)

        result['DEMA_5'] = indicators.talib_DEMA(result, 5)
        # result['DEMA_10'] = indicators.talib_DEMA(result, 10)
        # result['DEMA_15'] = indicators.talib_DEMA(result, 15)
        # result['DEMA_20'] = indicators.talib_DEMA(result, 20)
        # result['DEMA_30'] = indicators.talib_DEMA(result, 30)

        result['KAMA_5'] = indicators.talib_KAMA(result, 5)
        # result['KAMA_10'] = indicators.talib_KAMA(result, 10)
        # result['KAMA_15'] = indicators.talib_KAMA(result, 15)
        # result['KAMA_20'] = indicators.talib_KAMA(result, 20)
        # result['KAMA_30'] = indicators.talib_KAMA(result, 30)

        result['MIDPOINT_5'] = indicators.talib_MIDPOINT(result, 5)
        # result['MIDPOINT_10'] = indicators.talib_MIDPOINT(result, 10)
        # result['MIDPOINT_15'] = indicators.talib_MIDPOINT(result, 15)
        # result['MIDPOINT_20'] = indicators.talib_MIDPOINT(result, 20)
        # result['MIDPOINT_30'] = indicators.talib_MIDPOINT(result, 30)

        result['MIDPRICE_5'] = indicators.talib_MIDPRICE(result, 5)
        # result['MIDPRICE_10'] = indicators.talib_MIDPRICE(result, 10)
        # result['MIDPRICE_15'] = indicators.talib_MIDPRICE(result, 15)
        # result['MIDPRICE_20'] = indicators.talib_MIDPRICE(result, 20)
        # result['MIDPRICE_30'] = indicators.talib_MIDPRICE(result, 30)

        result['T3_5'] = indicators.talib_T3(result, N=5)
        # result['T3_10'] = indicators.talib_T3(result, N=10)
        # result['T3_15'] = indicators.talib_T3(result, N=15)
        # result['T3_20'] = indicators.talib_T3(result, N=20)
        # result['T3_30'] = indicators.talib_T3(result, N=30)

        result['TEMA_5'] = indicators.talib_TEMA(result, N=5)
        # result['TEMA_10'] = indicators.talib_TEMA(result, N=10)
        # result['TEMA_15'] = indicators.talib_TEMA(result, N=15)
        # result['TEMA_20'] = indicators.talib_TEMA(result, N=20)
        # result['TEMA_30'] = indicators.talib_TEMA(result, N=30)

        result['TRIMA_5'] = indicators.talib_TRIMA(result, N=5)
        # result['TRIMA_10'] = indicators.talib_TRIMA(result, N=10)
        # result['TRIMA_15'] = indicators.talib_TRIMA(result, N=15)
        # result['TRIMA_20'] = indicators.talib_TRIMA(result, N=20)
        # result['TRIMA_30'] = indicators.talib_TRIMA(result, N=30)

        result['WMA_5'] = indicators.talib_WMA(result, N=5)
        # result['WMA_10'] = indicators.talib_WMA(result, N=10)
        # result['WMA_15'] = indicators.talib_WMA(result, N=15)
        # result['WMA_20'] = indicators.talib_WMA(result, N=20)
        # result['WMA_30'] = indicators.talib_WMA(result, N=30)

        # https://www.kaggle.com/kratisaxena/lstm-gru-models-for-stock-movement-analysis
        # result['RSI_5'] = QA.QA_indicator_RSI(result, 5, 5, 5)['RSI1']
        # result['MOM_5'] = indicators.talib_MOM(result, 5)
        # result[[
        #     'BB_SMA_LOWER_5', 'BB_SMA_MIDDLE_5', 'BB_SMA_UPPER_5']] = indicators.talib_BBANDS(
        #     result, 5)
        # result[['AROON_DOWN_5','AROON_UP_5']] = QA.talib_indicators.AROON(result, 5)

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
        tmp = df.copy()
        for col in tmp.columns:
            if col in ['ATR_14', 'ATR_5', 'ATR_30', 'NATR_14', 'NATR_5',
                       'NATR_30', 'TRANGE']:
                continue
            if col in ['CCI']:
                tmp[col] = sklearn.preprocessing.normalize([tmp[col]])[0]
            tmp[col] = tmp[col] / tmp.iloc[0][col]
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
        return df / df.iloc[0]


def do(code='000002', window=5, days=3, wrapper=wrapper(), norm=normalize(),
       *args, **kwargs):
    dl = DataLoaderStock(code, wrapper=wrapper)
    df = dl.load()
    # print(df.head(window+2))
    train, test = DataHelper.train_test_split(df, batch_size=window + days)
    # print(train[0])
    X_train, Y_train = DataHelper.xy_split_2(train, window, days, norm=norm)
    X_test, Y_test = DataHelper.xy_split_2(test, window, days, norm=norm)
    # print(X_train[0])
    # print(X_train[0])
    # print(Y_train[0])
    # print(X_test[0])
    # print(Y_test[0])
    batch_size = kwargs.pop('batch_size', 128)
    verbose = kwargs.pop('verbose', 0)

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

    clear_session()
    model = Sequential()
    # https://www.researchgate.net/publication/327967988_Predicting_Stock_Prices_Using_LSTM
    # For analyzing the efficiency of the system  we are used  the
    # Root Mean Square Error(RMSE). The error or the difference between
    # the  target  and  the  obtained  output  value  is minimized by
    # using RMSE value. RMSE is the square root of the mean/average of the
    # square of all of the error. The use of  RMSE  is  highly  common  and
    # it  makes  an  excellent general  purpose  error  metric  for
    # numerical  predictions. Compared  to  the  similar  Mean  Absolute  Error,
    # RMSE amplifies and severely punishes large errors.
    model.add(
        LSTM(128, input_shape=X_train_arr[0].shape, return_sequences=True))
    # model.add(Dropout(0.1))
    model.add(
        LSTM(64, input_shape=X_train_arr[0].shape, return_sequences=False))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(days, activation='linear'))
    # model.add(Dropout(0.2))
    # model.add(LSTM(16, return_sequences=True))
    # model.add(Dropout(0.2))
    # model.add(LSTM(64, return_sequences=True))
    # model.add(Dropout(0.1))
    # model.add(LSTM(128))
    # model.add(Dropout(0.2))
    # model.add(Dense(days))
    model.compile(loss=root_mean_squared_error,
                  optimizer="rmsprop",
                  metrics=["mae", "acc"])
    start = time.time()
    history = model.fit(
        np.array(X_train_arr),
        np.array(Y_train_arr),
        epochs=kwargs.pop('epochs', 500),
        shuffle=kwargs.pop('shuffle', False),
        verbose=verbose,
        batch_size=batch_size,
        validation_split=kwargs.pop('validation_split', 0.15),
        callbacks=[EarlyStopping(monitor="loss", patience=10, verbose=verbose,
                                 mode="auto")]
    )
    if kwargs.pop('summary',True):
        model.summary()
    end = time.time()
    return {'start': start, 'end': end, 'X_test_arr': X_test_arr,
            'Y_test_arr': Y_test_arr, 'model': model, 'code': code,
            'window': window, 'days': days, 'batch_size': batch_size,
            'history': history}

def show_history(h,*args,**kwargs):
    start = h['start']
    end = h['end']
    X_test_arr = h['X_test_arr']
    Y_test_arr = h['Y_test_arr']
    model = h['model']
    code = h['code']
    window = h['window']
    days = h['days']
    batch_size = h['batch_size']
    history = h['history']

    print('Net time using : ', end - start, ' secs.')

    score = model.evaluate(np.array(X_test_arr), np.array(Y_test_arr))

    if kwargs.pop('print_score',True):
        print("Score:")
        for i in range(len(model.metrics_names)):
            print('{0}:{1}'.format(model.metrics_names[i], score[i]))

    plt=kwargs.pop('plt',True)

    if plt:
        plt.figure(figsize=(15, 8))

        # 绘制训练 & 验证的准确率值
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

        plt.figure(figsize=(15, 8))
        # 绘制训练 & 验证的损失值
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

    pred = model.predict(np.array(X_test_arr))
    pred_slope = []
    for day in range(days):
        df_result = pd.DataFrame(
            {'pred': pred[:, day], 'real': np.array(Y_test_arr)[:, day]})

        if plt:
            plt.figure(figsize=(15, 8))
            plt.title(
                '预测。code={0},window={1},day={2}/{3},batch_size={4}'.format(code,
                                                                        window,
                                                                        day + 1,
                                                                        days,
                                                                        batch_size))
            plt.plot(df_result['pred'])
            plt.plot(df_result['real'])
            plt.show()

            sns.regplot(x=pred[:, day], y=np.array(Y_test_arr)[:, day])
            plt.show()
        slope = stats.linregress(pred[:, day],
                                 np.array(Y_test_arr)[:, day]).slope
        print('Slope Day{0}:{1}'.format(day + 1, slope))
        pred_slope.append(slope)
    plt.close('all')

    return {'score': score, 'pred': pred, 'real': np.array(Y_test_arr),
            'slope': pred_slope}
