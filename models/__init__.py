from .random_forest import RandomForest
from .sarimax import Sarimax
from .orbit import Orbit

MODELS = {'random_forest': RandomForest,
          'sarimax': Sarimax,
          'orbit': Orbit
          }