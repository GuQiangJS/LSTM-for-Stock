import unittest

from LSTM_for_Stock.data_processor import DataLoader
from LSTM_for_Stock.model import Model


class MyTestCase(unittest.TestCase):
    def test_build_model(self):
        m = Model()
        layers = []
        layers.append({'units': 5, 'type': 'lstm'})
        layers.append({'units': 15, 'type': 'dense', 'activation': 'linear'})
        complie = {'loss': 'mse', 'optimizer': 'adam'}
        m.build_model(layers, complie)
        self.assertEqual(2, len(m.model.layers))
        self.assertEqual(5, m.model.layers[0].units)
        self.assertEqual(15, m.model.layers[1].units)
        self.assertEqual('mse', m.model.loss)

    def test_evaluate(self):
        data = DataLoader('601398', '399300')
        window = 10
        days = 5
        norm = True
        split = 0.15
        X_tra, Y_tra = data.get_train_data(window, days, norm)
        X_val, Y_val = data.get_valid_data(window, days, norm)
        model = Model()
        layers = [{'units': 5, 'type': 'lstm'},
        {'units': days, 'type': 'dense'}]
        complie = {'loss': 'mse', 'optimizer': 'adam'}
        train = {'epochs': 2, 'verbose': 2, 'validation_split': split}
        model.build_model(layers, complie)
        model.train(X_tra, Y_tra, train)
        print(model.evaluate(X_val, Y_val,{'batch_size':X_val.shape[0]}))


if __name__ == '__main__':
    unittest.main()
