# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
import numpy as np


class RandomForest:

    def __init__(self, args):
        self.n_estimators = args.n_estimators
        self.random_state = args.random_state
        self.model = RandomForestRegressor(n_estimators=self.n_estimators, random_state=self.random_state)

    def fit(self, data_x):
        data_x = np.array(data_x)
        train_x = data_x[:, 1:-1]
        train_y = data_x[:, -1]
        # print(train_x)
        self.model.fit(train_x, train_y)

    def predict(self, test_x):
        test_x = np.array(test_x.iloc[:, 1:], dtype=float)
        pred_y = self.model.predict(test_x)
        return pred_y


# Train the model on training data
