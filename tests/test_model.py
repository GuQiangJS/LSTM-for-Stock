import pytest

from LSTM_for_Stock.model import SequentialModel
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import CuDNNLSTM


def test_build_model():
    m = SequentialModel()
    layers = []
    layers.append({'units': 5, 'type': 'lstm'})
    layers.append({'units': 10, 'type': 'cudnnlstm'})
    layers.append({'units': 15, 'type': 'dense', 'activation': 'linear'})
    complie = {'loss': 'mse', 'optimizer': 'adam'}
    m.build_model(layers, complie)
    assert 3 == len(m.model.layers)
    assert isinstance(m.model.layers[0], LSTM)
    assert isinstance(m.model.layers[1], CuDNNLSTM)
    assert isinstance(m.model.layers[2], Dense)
    assert 5 == m.model.layers[0].units
    assert 10 == m.model.layers[1].units
    assert 15 == m.model.layers[2].units
    assert 'mse' == m.model.loss
