import logging
import os
from itertools import chain
import sys

# sys.path.append('../')

import hydra
from omegaconf import DictConfig
from models import MODELS
from data_loader import get_dataset
from factory.trainer import Trainer
from factory.evaluator import Evaluator
import pandas as pd
import numpy as np

from sklearn.model_selection import TimeSeriesSplit
from path_definition import HYDRA_PATH
# from schedulers import SCHEDULERS
from utils.reporter import Reporter
from data_loader.creator import create_dataset
# from utils.save_load import load_snapshot, save_snapshot, setup_training_dir

import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Activation
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


@hydra.main(config_path=HYDRA_PATH, config_name="stacking")
def stacking(cfg: DictConfig):
    if cfg.load_path is None and cfg.model is None:
        msg = 'either specify a load_path or config a model.'
        logger.error(msg)
        raise Exception(msg)

    elif cfg.load_path is not None:
        dataset = pd.read_csv(cfg.load_path)
        dataset = dataset[
            (dataset['Date'] > cfg.dataset.train_start_date) & (dataset['Date'] < cfg.dataset.valid_end_date)]
        if cfg.dataset.features is not None:
            features = cfg.dataset.features.split(',')
            features = [s.strip() for s in features]
        else:
            features = dataset.columns
            features.remove('Date')

        dates = dataset['Date']
        df = dataset[features]
        try:
            df['Mean'] = (df['Low'] + df['High']) / 2
        except:
            logger.error('your dataset should have High and Low columns')
        df = df.drop('Date', axis=1)
        df = df.dropna()
        arr = np.array(df)
        dataset = create_dataset(arr, list(dates), look_back=cfg.dataset.window_size, features=features)

    elif cfg.model is not None:
        dataset = get_dataset(cfg.dataset.name, cfg.dataset.train_start_date, cfg.dataset.valid_end_date, cfg)

    cfg.save_dir = os.getcwd()
    reporter = Reporter(cfg)
    reporter.setup_saving_dirs(cfg.save_dir)

    train_dataset = dataset[
        (dataset['Date'] > cfg.dataset.train_start_date) & (dataset['Date'] < cfg.dataset.train_end_date)]
    valid_dataset = dataset[
        (dataset['Date'] > cfg.dataset.valid_start_date) & (dataset['Date'] < cfg.dataset.valid_end_date)]

    X_train = np.array(train_dataset.iloc[:, 1:-1])
    y_train = np.array(train_dataset.iloc[:, -1])
    X_train = tf.convert_to_tensor(X_train, np.float32)
    y_train = tf.convert_to_tensor(y_train, np.float32)
    # print(X_train.shape)
    model = Sequential([

        # reshape 28 row * 28 column data to 28*28 rows
        # Flatten(input_shape=(28, 28)),

        # dense layer 1
        Dense(60, activation='sigmoid'),

        # dense layer 2
        Dense(128, activation='sigmoid'),

        # output layer
        Dense(10, activation='sigmoid'),
    ])

    model.compile(optimizer='adam',
                  loss='MeanSquaredError',
                  metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=10,
              batch_size=2000,
              validation_split=0.2)





if __name__ == '__main__':
    stacking()