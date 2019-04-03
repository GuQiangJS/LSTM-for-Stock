# 创建最新交易日报表

import os
import sys
nb_dir = os.path.split(os.getcwd())[0]
if nb_dir not in sys.path:
    sys.path.append(nb_dir)
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from LSTM_for_Stock.data_processor import DataLoaderStock
from app.train_all import wrapper
from app.train_all import normalize
from LSTM_for_Stock.data_processor import DataHelper
from LSTM_for_Stock.data_processor import Normalize
from LSTM_for_Stock.model import SequentialModel
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import datetime
import logging
from jinja2 import Environment
from jinja2 import FileSystemLoader
import gc

root_dir = os.path.dirname(os.getcwd())
model_path = os.path.join(root_dir, '.train_result')
import timeit


class Predict(object):
    def __init__(self, code, window, days):
        self.code = code
        self.window = window
        self.days = days
        self.batch_size = window + days
        model_filename = 'model_{2}_{0:02d}_{1:02d}.h5'.format(window, days,
                                                               code)
        m = SequentialModel()
        m.load(os.path.join(model_path, model_filename))
        self.model = m
        self.data = DataLoaderStock(code, wrapper=wrapper()).load()

    def __predict(self, X):
        return self.model.predict(X)

    def do_prediction(self):
        """预测

        Args:
            code:
            window:
            days:

        Returns:
            (pd.DataFrame,np.array)

        """

        last_X = Normalize().build(self.data[-self.window:])
        last_Y = self.__predict(np.array([last_X.values]))
        feature_price = []
        for y in last_Y[0]:
            feature_price.append(self.data.iloc[-self.window]['close'] * y)
        del last_X
        return self.data.iloc[-self.window:], last_Y[0], feature_price

    # def plot_history(self):
    #     X_test_arr = []
    #     Y_test_arr = []
    #     train, test = DataHelper.train_test_split(self.data,
    #                                               train_size=0.8,
    #                                               batch_size=self.batch_size)
    #     X_test, Y_test = DataHelper.xy_split_2(test, self.window, self.days,
    #                                            norm=normalize())
    #     for x in X_test:
    #         X_test_arr.append(x.values)
    #     for y in Y_test:
    #         Y_test_arr.append(y.values)
    #     pred_slope = self.__predict(np.array(X_test_arr))
    #     result = {}
    #     for i in range(self.days):
    #         df_result = pd.DataFrame(
    #             {'pred': pred_slope[:, i], 'real': np.array(Y_test_arr)[:, i]})
    #         plt.title(
    #             '{0}_{1}_{2}/Days:{3}'.format(self.code, self.window, self.days,
    #                                           i + 1))
    #         plt.figure(figsize=(15, 8))
    #         plt.plot(df_result['pred'])
    #         plt.plot(df_result['real'])
    #         result[i] = [df_result, plt]
    #     return result


def start_code(code, window, days):
    d = {}
    p = Predict(code, window, days)
    ds, y, feature_price = p.do_prediction()
    del p
    d['code'] = code
    d['last_date'] = ds.index[-1].date()
    d['last_price'] = ds.iloc[-1]['close']
    d['first_date'] = ds.index[0].date()
    d['first_price'] = ds.iloc[0]['close']
    # d[code] = {'last_date': ds.index[-1].date(),
    #            'last_close': ds.iloc[-1]['close']}
    d['precents'] = y
    d['feature_price'] = feature_price
    # for i in range(len(y)):
    #     d['day{}_per'.format(i + 1)] = y[i]
    #     d['day{}_pri'.format(i + 1)] = feature_price[i]
    logging.info('{0}_{1}_{2} Done.'.format(code, window, days))
    return d


RE_FILENAME = re.compile(r'model_(\d{6})_(\d{2})_(\d{2}).h5')
lst = []
for f in os.listdir(model_path):
    m = RE_FILENAME.match(f)
    if m:
        g = m.groups()
        if g:
            code = g[0]
            window = g[1]
            days = g[2]
            lst.append([code, window, days])
result = []
for l in lst:
    logging.info('{0}/{1}'.format(lst.index(l) + 1, len(lst)))
    result.append(start_code(l[0], int(l[1]), int(l[2])))

web_path = os.path.join(root_dir, 'web')
PATH = os.path.dirname(os.path.abspath(__file__))
file = os.path.join(web_path, "{0}.html".format(
    datetime.datetime.now().strftime('%Y%m%d%H%M%S')))
TEMPLATE_ENVIRONMENT = Environment(
    autoescape=False,
    loader=FileSystemLoader(os.path.join(PATH, 'templates')),
    trim_blocks=False)
template = TEMPLATE_ENVIRONMENT.get_template('daily_report.html')
with open(file, 'w', encoding='utf-8') as f:
    html = template.render(title=datetime.date.today().strftime("%Y-%m-%d"),
                           result=result)
    f.write(html)
pass
