from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.models import Sequential
from LSTM_for_Stock.unit import get_param_default_value as def_val


class Model(object):
    pass


class SequentialModel(Model):
    def __init__(self):
        self.__model = Sequential()
        self.__history=None

    @property
    def model(self):
        return self.__model

    @property
    def history(self):
        return self.__history

    def build_model(self, layers, compile={}):
        """根據配置項構建 `self.model`。

        Args:
            layers ([dict]): 層定義集合。集合中每一項為一層的定義。
                             層定義包含 `type`:(dense或lstm) 用來定義層的類型。
                             層定義中其他屬性參見 `Dense`_ 和 `LSTM`_ 構造函數參數定義。
            compile (dict):  訓練配置模型定義。定義可用屬性參見 `compile`_ 函數定義。

        Returns:

        .. _Dense:
        https://keras.io/zh/layers/core/#dense
        .. _LSTM:
        https://keras.io/zh/layers/recurrent/#lstm
        .. _compile:
        https://keras.io/zh/models/model/#compile

        """
        for layer in layers:
            t = layer.pop('type')
            if t == 'dense':
                # https://keras.io/zh/layers/core/
                self.__model.add(Dense.from_config(layer))
            elif t == 'lstm':
                # https://keras.io/zh/layers/recurrent/#lstm
                self.__model.add(LSTM.from_config(layer))
            elif t == 'dropout':
                # https://keras.io/zh/layers/recurrent/#Dropout
                self.__model.add(Dropout.from_config(layer))

        # https://keras.io/zh/models/model/#compile
        self.__model.compile(**compile)

    def train(self,
              X,
              Y,
              train={},
              callbacks=[
                  EarlyStopping(
                      monitor="loss", patience=10, verbose=1, mode="auto")
              ]):
        """訓練模型

        Args:
            X: 訓練集
            Y: 測試集
            train {dict}: 訓練模型配置。定義可用屬性參見 `fit`_ 函數定義。
            callbacks:

        Returns:
            :py:class:`keras.callbacks.History`: 參見 `History`_

        .. _fit:
        https://keras.io/zh/models/model/#fit
        .. _History:
        https://keras.io/zh/callbacks/#__history
        """
        epochs = train.pop('epochs', 100)
        batch_size = train.pop('batch_size',
                               def_val(self.__model.fit, 'batch_size'))
        verbose = train.pop('verbose', def_val(self.__model.fit, 'verbose'))
        validation_split = train.pop(
            'validation_split', def_val(self.__model.fit, 'validation_split'))
        validation_data = train.pop(
            'validation_data', def_val(self.__model.fit, 'validation_data'))
        shuffle = train.pop('shuffle', def_val(self.__model.fit, 'shuffle'))
        class_weight = train.pop('class_weight',
                                 def_val(self.__model.fit, 'class_weight'))
        sample_weight = train.pop('sample_weight',
                                  def_val(self.__model.fit, 'sample_weight'))
        initial_epoch = train.pop('initial_epoch',
                                  def_val(self.__model.fit, 'initial_epoch'))
        steps_per_epoch = train.pop(
            'steps_per_epoch', def_val(self.__model.fit, 'steps_per_epoch'))
        validation_steps = train.pop(
            'validation_steps', def_val(self.__model.fit, 'validation_steps'))
        self.__history = self.__model.fit(
            X,
            Y,
            epochs=epochs,
            callbacks=callbacks,
            batch_size=batch_size,
            verbose=verbose,
            validation_data=validation_data,
            validation_split=validation_split,
            shuffle=shuffle,
            class_weight=class_weight,
            sample_weight=sample_weight,
            initial_epoch=initial_epoch,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps)

        return self.__history

    def predict(self, X, predict={}):
        """模型預測

        Args:
            predict:
            X: 待預測的數據集
            predict {dict}: 預測模型配置。定義可用屬性參見 `predict`_ 函數定義。
            callbacks:
        Returns:
            參考 `predict`_ 函數定義。

        .. _predict:
        https://keras.io/zh/models/model/#predict
        """
        steps = predict.pop('steps', def_val(self.__model.predict, 'steps'))
        batch_size = predict.pop('batch_size',
                                 def_val(self.__model.predict, 'batch_size'))
        verbose = predict.pop('verbose',
                              def_val(self.__model.predict, 'verbose'))
        return self.__model.predict(
            X, batch_size=batch_size, verbose=verbose, steps=steps)

    def evaluate(self, X, Y, evaluate={}):
        """計算模型誤差

        Args:
            X: 輸入數據
            Y: 標籤
            evaluate {dict}: 預測模型配置。定義可用屬性參見 `evaluate`_ 函數定義。
            callbacks:
        Returns:
            參考 `evaluate`_ 函數定義。

        .. _evaluate:
        https://keras.io/zh/models/model/#evaluate
        """
        steps = evaluate.pop('steps', def_val(self.__model.evaluate, 'steps'))
        sample_weight = evaluate.pop(
            'sample_weight', def_val(self.__model.evaluate, 'sample_weight'))
        batch_size = evaluate.pop('batch_size',
                                  def_val(self.__model.evaluate, 'batch_size'))
        verbose = evaluate.pop('verbose',
                               def_val(self.__model.evaluate, 'verbose'))
        return self.__model.evaluate(
            X,
            Y,
            batch_size=batch_size,
            verbose=verbose,
            steps=steps,
            sample_weight=sample_weight)
