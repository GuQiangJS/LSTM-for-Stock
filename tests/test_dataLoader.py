import datetime
import unittest
import random
import QUANTAXIS as QA
import numpy as np

from LSTM_for_Stock.data_processor import DataLoader


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

    def test_loaddata_online(self):
        dataloader=DataLoader('601398', '399300',online=True,start='2019-03-01')
        print(dataloader.data)

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

    def plot_train_data2(self):
        """測試繪圖，做規則化的訓練價格和測試價格"""
        start = random.randint(0,500)
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
        start = random.randint(0,500)
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
