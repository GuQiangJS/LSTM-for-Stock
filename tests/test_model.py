import unittest
from LSTM_for_Stock.model import Model
import inspect
from keras.models import Sequential
from keras.layers import Dense
from LSTM_for_Stock import unit

class MyTestCase(unittest.TestCase):
    def test_build_model(self):
        m=Model()
        layers=[]
        layers.append({'units':5,'type':'lstm'})
        layers.append({'units':15,'type':'dense','activation':'linear'})
        complie={'loss':'mse','optimizer':'adam'}
        m.build_model(layers,complie)
        self.assertEqual(2,len(m.model.layers))
        self.assertEqual(5,m.model.layers[0].units)
        self.assertEqual(15,m.model.layers[1].units)
        self.assertEqual('mse',m.model.loss)

    def test1(self):
        Model().train()

if __name__ == '__main__':
    unittest.main()
