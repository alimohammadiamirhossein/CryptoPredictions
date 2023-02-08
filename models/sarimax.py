from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

class Sarimax:
    model = None
    result = None
    train_size = -1
    test_size = -1
    sc_in = MinMaxScaler(feature_range=(0, 1))
    sc_out = MinMaxScaler(feature_range=(0, 1))

    def __init__(self, args):
        print(args)
        self.train_size = -1
        self.test_size = -1
        self.order = tuple(map(int, args.order.split(', ')))
        self.seasonal_order = tuple(map(int, args.seasonal_order.split(', ')))
        self.enforce_invertibility = args.enforce_invertibility
        self.enforce_stationarity = args.enforce_stationarity

    def fit(self, data_x):
        data_x = np.array(data_x)
        train_x = data_x[:, 1:-1]
        train_y = data_x[:, -1]
        self.train_size = train_x.shape[0]
        train_x = self.sc_in.fit_transform(train_x)
        train_y = train_y.reshape(-1, 1)
        train_y = self.sc_out.fit_transform(train_y)
        train_x = np.array(train_x, dtype=float)
        train_y = np.array(train_y, dtype=float)
        self.model = SARIMAX(train_y,
                exog=train_x,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_invertibility=self.enforce_invertibility, enforce_stationarity=self.enforce_stationarity)
        self.result = self.model.fit()

    def predict(self, test_x):
        test_x = np.array(test_x, dtype=float)
        test_x = self.sc_in.transform(test_x)
        self.test_size = test_x.shape[0]
        pred_y = self.result.predict(start=self.train_size, end=self.train_size + self.test_size - 1, exog=test_x)
        # pred_y = self.result.predict(exog=test_x)
        print(pred_y.shape, test_x.shape)
        pred_y = pred_y.reshape(-1, 1)
        pred_y = self.sc_out.inverse_transform(pred_y)
        return pred_y
