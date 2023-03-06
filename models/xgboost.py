# Import the model we are using
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import pandas as pd


class MyXGboost:

    def __init__(self, args):
        self.reg = xgb.XGBRegressor()
        self.params = {
            "learning_rate": [0.10, 0.20, 0.30],
            "max_depth": [1, 3, 4, 5, 6, 7],
            "n_estimators": [int(x) for x in np.linspace(start=100, stop=1000, num=10)],
            "min_child_weight": [int(x) for x in np.arange(3, 10, 1)],
            "gamma": [0.0, 0.2, 0.4, 0.6],
            "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1],
            "colsample_bytree": [0.5, 0.7, 0.9, 1],
            "colsample_bylevel": [0.5, 0.7, 0.9, 1],
        }
        self.model_xg = RandomizedSearchCV(
                self.reg,
                param_distributions=self.params,
                n_iter=20,
                n_jobs=-1,
                cv=5,
                verbose=3,
                )
        self.response_col = args.response_col
        self.date_col = args.date_col

    def fit(self, data_x):
        self.regressors = []
        for col in data_x.columns:
            if col != self.response_col and col != self.date_col:
                self.regressors.append(col)
        train_x = pd.DataFrame()
        train_y = pd.DataFrame()
        train_x[self.regressors] = data_x[self.regressors].astype(float)
        train_y[self.response_col] = data_x[self.response_col].astype(float)
        self.model_xg.fit(train_x,train_y)

    def predict(self, test_x):
        valid_x = pd.DataFrame()
        valid_x[self.regressors] = test_x[self.regressors].astype(float)
        pred_y = self.model_xg.predict(valid_x)

        return pred_y


# Train the model on training data
