import unittest

from LSTM_for_Stock.data_processor import DataLoader
from LSTM_for_Stock.unit import plot_result_by_slope
from LSTM_for_Stock.unit import plot_result_by_pct_change
from LSTM_for_Stock.unit import plot_result
from LSTM_for_Stock.unit import calc_slope
from LSTM_for_Stock.model import Model
from LSTM_for_Stock.data_processor import FeaturesAppender

import numpy as np


class MyTestCase(unittest.TestCase):
    def test_plot_plot_result_by_slope(self):
        """按照斜率繪圖"""
        dataloader = DataLoader('601398', '399300', split=0)
        window = 40
        days = 10
        X, Y = dataloader.get_train_data(window, days, False)
        Y_slope = calc_slope(Y)
        plot_result_by_slope(X[:, 0, 0], Y_slope, window, days).show()
        plot_result_by_slope(X[:, 0, 0], Y, window, days).show()

    def test_plot_plot_result(self):
        dataloader = DataLoader('601398', '399300', split=0)
        window = 40
        days = 4
        X, Y = dataloader.get_train_data(window, days, False)
        plot_result(X[:, 0, 0], Y, window, days).show()

    class append(FeaturesAppender):
        def appendFeautres(self, df):
            df['pct_change'] = df['close'].pct_change()
            cols = ['pct_change'] + [col for col in df.columns if
                                     col != 'pct_change']
            df = df[cols].dropna()
            return df, ['pct_change']

    def test_plot_result_by_pct_change(self):
        data = DataLoader('601398', '399300',
                          features_appender=MyTestCase.append())

        window = 10
        days = 5
        X_tra, Y_tra = data.get_train_data(window, days, False)
        plot_result_by_pct_change(X_tra, Y_tra, window, days, top=100).show()

    def test_plot_result_1(self):
        data = DataLoader('601398', '399300',
                          features_appender=MyTestCase.append())
        window = 10
        days = 5
        norm = False
        split = 0.15
        X_tra, Y_tra = data.get_train_data(window, days, norm)
        X_val, Y_val = data.get_valid_data(window, days, norm)
        model = Model()
        layers = [{'units': 5, 'type': 'lstm'},
                  {'units': days, 'type': 'dense'}]
        complie = {'loss': 'mse', 'optimizer': 'adam'}
        train = {'epochs': 2, 'verbose': 2, 'validation_split': split}
        model.build_model(layers, complie)
        model.train(X_tra, Y_tra, train)
        pred = model.predict(X_val)
        print(model.evaluate(X_val,pred))
        plot_result_by_slope(X_val, Y_val, window, days, top=-1).show()
        plot_result_by_slope(X_val, pred, window, days, top=-1).show()

    def test_plot_result(self):
        self._do('601398', train_epochs=10, window=60, days=5)

    def _do(self, code, benchmark='399300', use_slope=True, train_epochs=10,
            window=10,
            days=3, norm=True, split=0.15, features_appender=None, top=30):
        dataloader = DataLoader(code, benchmark,
                                features_appender=features_appender)
        model = Model()
        comp = {"optimizer": "adam",
                "loss": "mse",
                "metrics": [
                    "mae",
                    "acc"
                ]}
        layers = [{'units': 10, 'type': 'lstm', },
                  {'units': 1 if use_slope else days, 'type': 'dense'}]
        train = {'epochs': train_epochs, 'verbose': 2,
                 'validation_split': split, 'batch_size': 128}

        model.build_model(layers, comp)
        X_tra, Y_tra = dataloader.get_train_data(window, days, norm)
        X_tra_round = np.round(X_tra, 4)
        if use_slope:
            Y_tra_slope = calc_slope(Y_tra)
            history = model.train(X_tra_round, Y_tra_slope, train)
        else:
            Y_tra_round = Y_tra
            history = model.train(X_tra_round, Y_tra_round, train)

        X_val, Y_val = dataloader.get_valid_data(window, days, norm)
        X_val_round = np.round(X_val, 4)
        score = None
        if use_slope:
            Y_val_slope = calc_slope(Y_val)
            score = model.evaluate(X_val_round, Y_val_slope,
                                   {'batch_size': 128})
        else:
            Y_val_round = Y_val
            score = model.evaluate(X_val_round, Y_val_round,
                                   {'batch_size': 128})

        for i in range(len(model.model.metrics_names)):
            print('{0}:{1}'.format(model.model.metrics_names[i], score[i]))

        try:
            pred_slope = model.predict(X_val_round, {'batch_size': 128})
            data_online = DataLoader(code, benchmark, split=0,
                                     start=dataloader._df_valid.index[
                                         0].strftime('%Y-%m-%d'), online=True,
                                     features_appender=features_appender)
            X_online = data_online.get_train_data(window, days, norm)[0][:, 0,
                       0]
            # 验证集误差计算
            if use_slope:
                plot_result_by_slope(X_online, pred_slope, window, days,
                                     top=top).show()
            else:
                plot_result(X_online, pred_slope, window, days,
                            top=top).show()
        except Exception as e:
            print(e)
            # traceback.print_exc()


if __name__ == '__main__':
    unittest.main()
