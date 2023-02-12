from .random_forest import RandomForest
from .sarimax import Sarimax
# from .orbit import Orbit
from .LSTM import MyLSTM
from .GRU import MyGRU

MODELS = {'random_forest': RandomForest,
          'sarimax': Sarimax,
          # 'orbit': Orbit,
          'lstm': MyLSTM,
          'gru': MyGRU
          }