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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM,CuDNNLSTM
from keras.backend import clear_session
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import time
import numpy as np
from sklearn.model_selection import cross_val_score # K折交叉验证模块
import matplotlib
matplotlib.rcParams["figure.figsize"]=[16,5]
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK SC','SimHei']
matplotlib.rcParams['axes.unicode_minus']=False #用来正常显示负号\n
import QUANTAXIS as QA
import sklearn
import talib
import seaborn as sns
from scipy import stats


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
        return df / df.iloc[0]

def do(code='000002',window=5,days=3,wrapper=wrapper(),norm=normalize(),*args,**kwargs):
    dl = DataLoaderStock(code, wrapper=wrapper)
    df = dl.load()
    # print(df.head(window+2))
    train, test = DataHelper.train_test_split(df, batch_size=window + days)

    X_train, Y_train = DataHelper.xy_split_2(train, window, days,norm=norm)
    X_test, Y_test = DataHelper.xy_split_2(test, window, days,norm=norm)

    # print(X_train[0])
    # print(Y_train[0])
    # print(X_test[0])
    # print(Y_test[0])
    batch_size=kwargs.pop('batch_size',128)
    verbose=kwargs.pop('verbose',0)

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
    model.add(LSTM(128, input_shape=X_train_arr[0].shape, return_sequences=True))
    model.add(Dropout(0.1))
    # model.add(LSTM(256, return_sequences=True))
    # model.add(Dropout(0.2))
    # model.add(LSTM(128, return_sequences=True))
    # model.add(Dropout(0.2))
    # model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(LSTM(128))
    model.add(Dropout(0.2))
    model.add(Dense(days))
    model.compile(loss='mse',
                optimizer="rmsprop",
                metrics=["mae", "acc"])
    start = time.time()
    history = model.fit(
        np.array(X_train_arr),
        np.array(Y_train_arr),
        epochs=kwargs.pop('epochs',500),
        shuffle=kwargs.pop('shuffle',False),
        verbose=verbose,
        batch_size=batch_size,
        validation_split=kwargs.pop('validation_split',0.15),
        callbacks=[EarlyStopping(monitor="loss", patience=10, verbose=verbose, mode="auto")]
    )
    model.summary()
    end = time.time()
    return {'start':start,'end':end,'X_test_arr':X_test_arr,'Y_test_arr':Y_test_arr,'model':model,'code':code,'window':window,'days':days,'batch_size':batch_size,'history':history}

def show_history(h):

    start=h['start']
    end=h['end']
    X_test_arr=h['X_test_arr']
    Y_test_arr=h['Y_test_arr']
    model=h['model']
    code=h['code']
    window=h['window']
    days=h['days']
    batch_size=h['batch_size']
    history=h['history']

    print('Net time using : ', end-start, ' secs.')

    score=model.evaluate(np.array(X_test_arr),np.array(Y_test_arr))

    print("Score:")
    for i in range(len(model.metrics_names)):
        print('{0}:{1}'.format(model.metrics_names[i],score[i]))
    
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
    pred_slope=[]
    for day in range(days):
        df_result = pd.DataFrame({'pred': pred[:, day], 'real': np.array(Y_test_arr)[:, day]})

        plt.figure(figsize=(15, 8))
        plt.title('预测。code={0},window={1},day={2}/{3},batch_size={4}'.format(code,window,day+1,days,batch_size))
        plt.plot(df_result['pred'])
        plt.plot(df_result['real'])
        plt.show()
        
        sns.regplot(x=pred[:, day],y=np.array(Y_test_arr)[:, day])
        plt.show()
        slope=stats.linregress(pred[:, day],np.array(Y_test_arr)[:, day]).slope
        print('Slope Day{0}:{1}'.format(day+1,slope))
        pred_slope.append(slope)
    plt.close('all')

    return {'score':score,'pred':pred,'real':np.array(Y_test_arr),'slope':pred_slope}