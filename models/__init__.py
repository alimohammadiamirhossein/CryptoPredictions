from .random_forest import RandomForest
from .sarimax import Sarimax
from .orbit import Orbit
from .LSTM import MyLSTM
from .GRU import MyGRU
from .arima import MyARIMA
from .prophet import MyProphet
from .xgboost import MyXGboost
from .neural_prophet import Neural_Prophet


MODELS = {'random_forest': RandomForest,
          'sarimax': Sarimax,
          'orbit': Orbit,
          'lstm': MyLSTM,
          'gru': MyGRU,
          'arima': MyARIMA,
          'prophet': MyProphet,
          'xgboost': MyXGboost,
          'neural_prophet': Neural_Prophet
          }
