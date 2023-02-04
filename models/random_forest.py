# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
import numpy as np

class RandomForest:
    model = None

    def __init__(self, n_estimators=1000, random_state=42):
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)

    def fit(self, data_x):
        data_x = np.array(data_x)
        train_x = data_x[:, :-1]
        train_y = data_x[:, :-1]
        self.model.fit(train_x, train_y)

    def predict(self, test_x):
        pred_y = self.model.predict(test_x)
        return pred_y



# Train the model on training data
