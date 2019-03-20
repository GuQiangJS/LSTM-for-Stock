import unittest

from LSTM_for_Stock.data_processor import DataLoader
from LSTM_for_Stock.unit import plot_result_by_slope
from LSTM_for_Stock.unit import calc_slope


class MyTestCase(unittest.TestCase):
    def test_plot_train_data_full(self):
        """按照斜率繪圖"""
        dataloader = DataLoader('601398', '399300', split=0)
        window = 40
        days = 10
        X, Y = dataloader.get_train_data(window, days, False)
        Y_slope=calc_slope(Y)
        plot_result_by_slope(X[:, 0, 0], Y_slope, window, days).show()
        plot_result_by_slope(X[:, 0, 0], Y, window, days).show()


if __name__ == '__main__':
    unittest.main()
