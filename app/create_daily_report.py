# 创建最新交易日报表

import os
import sys

nb_dir = os.path.split(os.getcwd())[0]
if nb_dir not in sys.path:
    sys.path.append(nb_dir)
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
import decimal
from bson import json_util
from LSTM_for_Stock.data_processor import DataLoaderStock
from LSTM_for_Stock.data_processor import Wrapper_default
from LSTM_for_Stock.data_processor import Normalize
from LSTM_for_Stock.model import SequentialModel
from app.train_all import get_train_acc
from app.train_all import get_last_train_date
from LSTM_for_Stock.loss import root_mean_squared_error
from QUANTAXIS.QAUtil import QA_util_datetime_to_strdate
from QUANTAXIS.QAUtil import QA_util_to_datetime
import QUANTAXIS as QA
import os
import numpy as np
import re
import datetime
import logging
from jinja2 import Environment
from jinja2 import FileSystemLoader

root_dir = os.path.dirname(os.getcwd())
model_path = os.path.join(root_dir, '.train_result')


class Predict(object):
    def __init__(self, code, window, days):
        self.code = code
        self.window = window
        self.days = days
        self.batch_size = window + days
        model_filename = 'model_{2}_{0:02d}_{1:02d}.h5'.format(window, days,
                                                               code)
        m = SequentialModel()
        m.load(os.path.join(model_path, model_filename),
               custom_objects={
                   'root_mean_squared_error': root_mean_squared_error})
        self.model = m
        self.data = DataLoaderStock(code, wrapper=Wrapper_default()).load()

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
    d['name'] = QA.QA_fetch_stock_list_adv().loc[code]['name']
    d['window'] = window
    d['days'] = days
    d['last_date'] = ds.index[-1].date()
    d['last_price'] = ds.iloc[-1]['close']
    d['last_change'] = ds['close'][-1] / ds['close'][-2] - 1
    d['first_date'] = ds.index[0].date()
    d['first_price'] = ds.iloc[0]['close']
    # d[code] = {'last_date': ds.index[-1].date(),
    #            'last_close': ds.iloc[-1]['close']}
    d['precents'] = y
    d['feature_price'] = feature_price
    d['acc'] = get_train_acc(code, window, days)
    last_dt = get_last_train_date(code, window, days)
    d['last_train_date'] = QA_util_datetime_to_strdate(
        last_dt) if last_dt else ''
    # for i in range(len(y)):
    #     d['day{}_per'.format(i + 1)] = y[i]
    #     d['day{}_pri'.format(i + 1)] = feature_price[i]
    logging.info('{0}_{1}_{2} Done.'.format(code, window, days))
    return d


RE_FILENAME = re.compile(r'model_(\d{6})_(\d{2})_(\d{2}).h5')
lst_w = {}
for f in os.listdir(model_path):
    m = RE_FILENAME.match(f)
    if m:
        g = m.groups()
        if g:
            code = g[0]
            window = g[1]
            days = g[2]
            if window not in lst_w.keys():
                lst_w[window] = []
            lst_w[window].append([code, window, days])
result_full = {}  # 完全结果集
result_simple = {}  # 比率高于1.1的结果集
for window, lst in lst_w.items():
    if window not in result_full.keys():
        result_full[window] = []
    if window not in result_simple.keys():
        result_simple[window] = []
    for l in lst:
        logging.info('{0}/{1}'.format(lst.index(l) + 1, len(lst)))
        rc = start_code(l[0], int(l[1]), int(l[2]))
        result_full[window].append(rc)
        for p in rc['precents']:
            if p > 1.1 and rc['last_change'] * 100 < 9.8:
                result_simple[window].append(rc)
                break


def default(obj):
    if isinstance(obj, datetime.date):
        return {"$dt": QA_util_datetime_to_strdate(obj)}
    if isinstance(obj, decimal.Decimal):
        return json_util.default(float(obj))
    return json_util.default(obj)


def _parse_date(doc):
    return QA_util_to_datetime(doc["$dt"])


def hook(dct):
    if "$dt" in dct:
        return _parse_date(dct)
    return json_util.object_hook(dct)


# result_path = os.path.join(root_dir, '.daily_result')
# os.makedirs(result_path, exist_ok=True)
# file = os.path.join(result_path, "{0}.json".format(
#     datetime.datetime.now().strftime('%Y%m%d%H%M%S')))
# with open(file, 'w', encoding='utf-8') as f:
#     str = json.dumps(result_full, sort_keys=True, indent=1, default=default)
#     file.write(str)
# logging.info('Daily Result JSON Saved at:' + file)

web_path = os.path.join(root_dir, 'web')
os.makedirs(web_path, exist_ok=True)
PATH = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_ENVIRONMENT = Environment(
    autoescape=False,
    loader=FileSystemLoader(os.path.join(PATH, 'templates')),
    trim_blocks=False)
template = TEMPLATE_ENVIRONMENT.get_template('daily_report.html')


def write_web(path, result):
    if len(result) > 0:
        with open(path, 'w', encoding='utf-8') as f:
            html = template.render(
                title=QA_util_datetime_to_strdate(datetime.datetime.today()),
                result=result)
            f.write(html)
        logging.info('WebPage Saved at:' + file)
    else:
        logging.info('WebPage Skip.')


def get_datetime_str(dt=datetime.datetime.now(), s='%Y%m%d%H%M%S'):
    return dt.strftime(s)


for window, result in result_full.items():
    if len(result) > 0:
        path = os.path.join(web_path, window)
        os.makedirs(path, exist_ok=True)
        file = os.path.join(path, "{0}_full.html".format(get_datetime_str()))
        write_web(file, result)

for window, result in result_simple.items():
    if len(result) > 0:
        path = os.path.join(web_path, window)
        os.makedirs(path, exist_ok=True)
        file = os.path.join(path, "{0}_simple.html".format(get_datetime_str()))
        write_web(file, result)
