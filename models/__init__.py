from .random_forest import RandomForest
from .sarimax import Sarimax

MODELS = {'random_forest': RandomForest,
          'sarimax': Sarimax
          }