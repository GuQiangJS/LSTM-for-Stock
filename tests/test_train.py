from LSTM_for_Stock.model import SequentialModel
from LSTM_for_Stock.data_processor import DataHelper
from LSTM_for_Stock.data_processor import DataLoaderStock
import logging
from LSTM_for_Stock.data_processor import Wrapper
from LSTM_for_Stock.data_processor import Wrapper_fillna
import numpy as np
from keras import backend

class wrapper(Wrapper):
    def build(self, df):
        result = df.copy()
        result = result.fillna(method='ffill')
        result = result.drop(columns=['up_count', 'down_count'])
        return result.dropna()

def m(y_true, y_pred):
    pass


dl = DataLoaderStock('601398', wrapper=wrapper())
df = dl.load()
window = 10
days = 3
train, test = DataHelper.train_test_split(df, batch_size=window + days)

X_train, Y_train = DataHelper.xy_split_2(train, window, days)
X_test, Y_test = DataHelper.xy_split_2(test, window, days)

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

layers = [{'units': 5, 'type': 'lstm', 'input_shape': X_train_arr[0].shape},
          {'units': days, 'type': 'dense'}]
complie = {'loss': 'mse', 'optimizer': 'adam',"metrics": ['acc','mae',m]}

model = SequentialModel()
model.build_model(layers, complie)
history = model.train(np.array(X_train_arr), np.array(Y_train_arr),
                      train={'epochs': 50, 'verbose': 2})
logging.info(history)
logging.info(model.evaluate(np.array(X_test_arr), np.array(Y_test_arr)))
logging.info(model.predict(np.array(X_test_arr)))
