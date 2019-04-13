import os
import sys

nb_dir = os.path.split(os.getcwd())[0]
if nb_dir not in sys.path:
    sys.path.append(nb_dir)
import pandas as pd
from LSTM_for_Stock.data_processor import DataHelper
from LSTM_for_Stock.data_processor import DataLoaderStock
from LSTM_for_Stock.data_processor import Normalize
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.backend import clear_session
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import time
import numpy as np
import matplotlib
from LSTM_for_Stock.loss import root_mean_squared_error

matplotlib.rcParams["figure.figsize"] = [16, 5]
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号\n
import sklearn
import seaborn as sns
from scipy import stats
from LSTM_for_Stock.data_processor import Wrapper_default


def do(code='000002', window=5, days=3, wrapper=Wrapper_default(),
       norm=Normalize(),
       *args, **kwargs):
    dl = DataLoaderStock(code, wrapper=wrapper,
                         appends=kwargs.pop('appends', []))
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
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(days, activation='linear'))
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
    if kwargs.pop('summary', True):
        model.summary()
    end = time.time()
    return {'start': start, 'end': end, 'X_test_arr': X_test_arr,
            'Y_test_arr': Y_test_arr, 'model': model, 'code': code,
            'window': window, 'days': days, 'batch_size': batch_size,
            'history': history, 'data': df, 'X_train': X_train,
            'Y_train': Y_train, 'X_test': X_test, 'Y_test': Y_test}


def show_history(h, *args, **kwargs):
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

    if kwargs.pop('print_score', True):
        print("Score:")
        for i in range(len(model.metrics_names)):
            print(' {0}:{1}'.format(model.metrics_names[i], score[i]))

    show_plt = kwargs.pop('show_plt', True)

    if show_plt:
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

        if show_plt:
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
