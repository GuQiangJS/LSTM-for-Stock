import pytest
from LSTM_for_Stock.model import SequentialModel
import datetime
import os
import logging
from keras.models import Sequential
from keras.models import load_model
from keras.models import model_from_json
from keras.backend import clear_session

def __load_split()->Sequential:
    json_file = open("model.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("model.h5")
    return model

def __load_complete()->Sequential:
    return load_model("model.h5")

def __save_complete(model:Sequential):
    model.save("model.h5")

def __save_split(model:Sequential):
    model.save_weights('model.h5')
    model_json = model.to_json()
    with open('model.json', "w") as json_file:
        json_file.write(model_json)
    json_file.close()

def __create_model()->SequentialModel:

    layers = [
    {'units': 100,'type': 'lstm','input_shape': (10,5)},
    {'units': 3,'type': 'dense'}]
    complie = {
    #     "optimizer":"adam",
        "loss":"mse",
        "optimizer":"rmsprop",
    #     "loss":"categorical_crossentropy",
        "metrics": [
            "mae", "acc"
        ]
    }

    model = SequentialModel()
    model.build_model(layers, complie)
    return model

def test_model_load_split():
    """此方法读取模型会时间增加"""
    __save_split(__create_model().model)
    for i in range(50):
        oldtime = datetime.datetime.now()
        __load_split()
        logging.info(u'相差：%s' % (datetime.datetime.now() - oldtime))
    if os.path.exists('model.h5'):
        os.remove('model.h5')
    if os.path.exists('model.json'):
        os.remove('model.json')

def test_model_load_complete():
    """此方法读取模型会时间增加"""
    __save_complete(__create_model().model)
    for i in range(50):
        oldtime = datetime.datetime.now()
        __load_complete()
        logging.info(u'相差：%s' % (datetime.datetime.now() - oldtime))
    if os.path.exists('model.h5'):
        os.remove('model.h5')


def test_model_load_split_clear():
    """此方法读取模型时间正常"""
    __save_split(__create_model().model)
    for i in range(10):
        oldtime = datetime.datetime.now()
        clear_session()
        __load_split()
        logging.info(u'相差：%s' % (datetime.datetime.now() - oldtime))
    if os.path.exists('model.h5'):
        os.remove('model.h5')
    if os.path.exists('model.json'):
        os.remove('model.json')

def test_model_load_complete_clear():
    """此方法读取模型时间正常"""
    __save_complete(__create_model().model)
    for i in range(10):
        oldtime = datetime.datetime.now()
        clear_session()
        __load_complete()
        logging.info(u'相差：%s' % (datetime.datetime.now() - oldtime))
    if os.path.exists('model.h5'):
        os.remove('model.h5')
