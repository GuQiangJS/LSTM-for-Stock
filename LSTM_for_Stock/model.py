from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import Sequential

from .unit import get_param_default_value as _dv


class Model(object):
    """

    """

    def __init__(self):
        self.model = Sequential()

    def build_model(self, layers, compile):
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
                self.model.add(Dense.from_config(layer))
            elif t == 'lstm':
                # https://keras.io/zh/layers/recurrent/#lstm
                self.model.add(LSTM.from_config(layer))
            elif t == 'dropout':
                # https://keras.io/zh/layers/recurrent/#Dropout
                self.model.add(Dropout.from_config(layer))

        # https://keras.io/zh/models/model/#compile
        self.model.compile(**compile)

    def train(self, X, Y, train, callbacks=[
        EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")]):
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
        https://keras.io/zh/callbacks/#history
        """
        epochs = train.pop('epochs', 100)
        batch_size = train.pop('batch_size', _dv(self.model.fit, 'batch_size'))
        verbose = train.pop('verbose', _dv(self.model.fit, 'verbose'))
        validation_split = train.pop('validation_split',
                                     _dv(self.model.fit, 'validation_split'))
        validation_data = train.pop('validation_data',
                                    _dv(self.model.fit, 'validation_data'))
        shuffle = train.pop('shuffle', _dv(self.model.fit, 'shuffle'))
        class_weight = train.pop('class_weight',
                                 _dv(self.model.fit, 'class_weight'))
        sample_weight = train.pop('sample_weight',
                                  _dv(self.model.fit, 'sample_weight'))
        initial_epoch = train.pop('initial_epoch',
                                  _dv(self.model.fit, 'initial_epoch'))
        steps_per_epoch = train.pop('steps_per_epoch',
                                    _dv(self.model.fit, 'steps_per_epoch'))
        validation_steps = train.pop('validation_steps',
                                     _dv(self.model.fit, 'validation_steps'))
        self.history = self.model.fit(X, Y, epochs=epochs, callbacks=callbacks,
                                      batch_size=batch_size, verbose=verbose,
                                      validation_data=validation_data,
                                      validation_split=validation_split,
                                      shuffle=shuffle,
                                      class_weight=class_weight,
                                      sample_weight=sample_weight,
                                      initial_epoch=initial_epoch,
                                      steps_per_epoch=steps_per_epoch,
                                      validation_steps=validation_steps)

        self.model.summary()
        return self.history

    def predict(self, X, predict):
        """模型預測

        Args:
            X: 待預測的數據集
            predict {dict}: 預測模型配置。定義可用屬性參見 `predict`_ 函數定義。
            callbacks:
        Returns:
            參考 `predict`_ 函數定義。

        .. _fit:
        https://keras.io/zh/models/model/#predict
        """
        steps = predict.pop('steps', _dv(self.model.predict, 'steps'))
        batch_size = predict.pop('batch_size',
                                 _dv(self.model.predict, 'batch_size'))
        verbose = predict.pop('verbose', _dv(self.model.predict, 'verbose'))
        return self.model.predict(X, batch_size=batch_size, verbose=verbose,
                                  steps=steps)
