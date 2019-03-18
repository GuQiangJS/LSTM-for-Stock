import datetime
import unittest
import random
import QUANTAXIS as QA
import numpy as np
from numpy import array_equal as arr_equal
from numpy import round
from LSTM_for_Stock.data_processor import DataLoader
import matplotlib.pyplot as plt


class DataLoaderTestCase(unittest.TestCase):
    def test_init(self):
        dataloader = DataLoader('601398', '399300')
        end = datetime.datetime.today().strftime('%Y-%m-%d')
        d = QA.QA_fetch_stock_day_adv('601398',
                                      start='1990-01-01',
                                      end=end).to_qfq().data
        for col in ['close', 'open', 'high', 'low', 'volume']:
            self.assertTrue(np.array_equal(d[col].values,
                                           dataloader.data[col].values), col)

    # def test_loaddata_online(self):
    #     dataloader=DataLoader('601398', '399300',online=True,start='2019-03-01')
    #     print(dataloader.data)

    def test_get_train_data(self):
        dataloader = DataLoader('601398', '399300')
        d = dataloader.data
        for z in [[30, 1], [30, 2], [30, 4], [60, 5]]:
            X, Y = dataloader.get_train_data(z[0], z[1], False)
            self.assertEqual(X.shape[0], Y.shape[0])
            self.assertEqual(X.shape[1], z[0])
            self.assertEqual(Y.shape[1], z[1])
            for i in range(len(Y)):
                for j in range(z[1]):
                    self.assertEqual(Y[i][j], d.iloc[j + i + z[0]]['close'],
                                     'z={0},i={1}'.format(z, i))

    def test_get_train_full(self):
        """测试完整分割数据。所有数据均为训练数据，无验证数据。"""
        dataloader = DataLoader('601398', '399300', split=0)
        d = dataloader.data
        window = 40
        days = 5
        X, Y = dataloader.get_train_data(window, days, False)

        # 维度大小测试
        size=len(d) - window - days+1
        self.assertEqual(size, X.shape[0],
                         'X的第一维大小应该是测试数据量{0}-{1}-{2}+1'.format(len(d), window,
                                                             days))
        self.assertEqual(window, X.shape[1],
                         'X的第二维大小应该是测试数据量{0}'.format(window))
        self.assertEqual(size, Y.shape[0],
                         'Y的第一维大小应该是测试数据量{0}-{1}-{2}+1'.format(len(d), window,
                                                             days))
        self.assertEqual(days, Y.shape[1],
                         'X的第二维大小应该是测试数据量{0}'.format(window))
        # 维度大小测试

        # 开头数据准确性测试
        self.assertTrue(arr_equal(np.round(X[0],4),
                                  np.round(d.iloc[:window].values,4)),
                                  'X的第一条数据应该是测试数据的前{}条数据'.format(window))
        self.assertTrue(arr_equal(np.round(Y[0],4),
                                  np.round(d.iloc[window:window+days]['close'].values,4)),
                                  'Y的第一条数据应该是测试数据跳过{0}条数据后的前{1}条数据'.format(window,days))
        # 开头数据准确性测试

        # 结尾数据准确性测试
        self.assertTrue(arr_equal(np.round(Y[-1],4),
                                  np.round(d.iloc[-days:]['close'].values,4)),
                                  'Y最后一条数据应该是测试数据最后{0}条数据之前的{0}条数据'.format(days))
        self.assertTrue(arr_equal(np.round(X[-1],4),
                                  np.round(d.iloc[-days-window:-days].values,4)),
                                  'X最后一条数据应该是测试数据最后{0}条数据之前的{0}条数据'.format(days+window,window))
        # 结尾数据准确性测试

        for i in range(len(X)):
            x_start=i
            x_end=window+i
            y_start=window+i
            y_end=i+window+days
            real_X = d.iloc[x_start:x_end]
            self.assertTrue(arr_equal(round(real_X['close'].values, 4),
                                      round(X[i][:,0], 4)),
                            '真实收盘价数据与X中第一条不符。{0}-{1}'.format(
                                real_X['close'],X[i][:,0]
                            ))
            real_Y = d.iloc[y_start:y_end]
            self.assertTrue(arr_equal(real_Y['close'].values, Y[i]),
                            '真实收盘价数据与Y不符。{0}-{1}'.format(
                                real_Y['close'],Y[i]
                            ))
            self.assertFalse(real_X.iloc[-1].equals(real_Y.iloc[0]),
                             '真实数据中X的最后一条与真实数据中Y的第一条不应该相符')

    def test_plot_train_data_full(self):
        from sklearn import linear_model
        import pandas as pd
        dataloader = DataLoader('601398', '399300', split=0)
        d = dataloader.data
        window = 40
        days = 10
        X, Y = dataloader.get_train_data(window, days, False)
        # plt.plot(X[:,0,0])
        # plt.plot([None for p in range(window)]+[Y[0][0],None,None,None,Y[0][-1]])
        # plt.show()
        fig = plt.figure(facecolor='white')
        ax = fig.add_subplot(111)
        ax.plot(X[:,0,0], label='True Data')

        xl={}
        for i, data in enumerate(Y):
            xl[i]=abs((data[-1] - data[0]) / (i + window + days - i + window))
        xl_top100=np.array(sorted(xl.items(),
                                  key=lambda kv: kv[1],
                                  reverse=True))[:100, 0]
        # Pad the list of predictions to shift it in the graph to it's correct start
        for i, data in enumerate(Y):
            # padding = [None for p in range(i+ window)]
            # plt.plot(padding + list(data), label='Prediction')
            # plt.legend()
            if i in xl_top100:
                plt.plot([i+window,i+window+days],[data[0],data[-1]])
                print(xl[i])
        plt.show()

    def plot_train_data2(self):
        """測試繪圖，做規則化的訓練價格和測試價格"""
        start = random.randint(0, 500)
        window = 60
        test = 15
        dataloader = DataLoader('601398', '399300')
        x, y = dataloader.get_train_data(window, test, True)
        print(dataloader.data.iloc[start:start + window + test]['close'])
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.lineplot(x=range(len(x[start])), y=x[start][:, 0])
        sns.lineplot(x=range(len(x[start]), len(x[start]) + len(y[start])),
                     y=y[start])
        plt.show()

    def plot_train_data1(self):
        """測試繪圖，不做規則化的訓練價格和測試價格"""
        start = random.randint(0, 500)
        window = 30
        test = 5
        dataloader = DataLoader('601398', '399300')
        x, y = dataloader.get_train_data(window, test, False)
        print(dataloader.data.iloc[start:start + window + test]['close'])
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.lineplot(x=range(len(x[start])), y=x[start][:, 0])
        sns.lineplot(x=range(len(x[start]), len(x[start]) + len(y[start])),
                     y=y[start])
        plt.show()


if __name__ == '__main__':
    unittest.main()
