# Kears 损失值计算扩展

import keras.backend as K

def root_mean_squared_error(y_true, y_pred):
    """均方根误差"""
    # https://zh.wikipedia.org/wiki/%E5%9D%87%E6%96%B9%E6%A0%B9%E8%AF%AF%E5%B7%AE
    return K.sqrt(K.mean(K.square(y_pred - y_true)))