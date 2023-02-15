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
from factory.stacker import Stacker
import pandas as pd
import numpy as np

from sklearn.model_selection import TimeSeriesSplit
from path_definition import HYDRA_PATH
# from schedulers import SCHEDULERS
from utils.reporter import Reporter
from data_loader.creator import create_dataset
# from utils.save_load import load_snapshot, save_snapshot, setup_training_dir

from sklearn.metrics import mean_squared_error
from math import sqrt


import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras import Model
from keras.layers import Flatten
from keras.layers import Dense, dot, Input
from keras.layers import Activation
import matplotlib.pyplot as plt

from metrics import METRICS

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

    stacker = Stacker(cfg)
    stacker.add_new_col(np.array(train_dataset.prediction), 'prediction', is_train=True)
    stacker.add_new_col(np.array(valid_dataset.prediction), 'prediction', is_train=False)

    for key in cfg.models.keys():
        item = cfg.models[key]
        model = MODELS[item['type']](item)
        arg = cfg.copy()
        arg['model'] = item
        trainer = Trainer(arg, train_dataset, None, model)
        trainer.train()
        prediction_ = trainer.prediction()
        stacker.add_new_col(np.array(prediction_), item['type'], is_train=True)
        evaluator = Evaluator(cfg, test_dataset=valid_dataset, model=model, reporter=reporter)
        prediction_ = evaluator.prediction()
        stacker.add_new_col(np.array(prediction_), item['type'], is_train=False)

    train_ = pd.DataFrame(stacker.train_dataset)
    valid_ = pd.DataFrame(stacker.valid_dataset)

    X_train = np.array(train_.iloc[:, 1:])
    y_train = np.array(train_.iloc[:, 0])
    X_train = tf.convert_to_tensor(X_train, np.float32)
    y_train = tf.convert_to_tensor(y_train, np.float32)

    X_valid = np.array(valid_.iloc[:, 1:])
    y_valid = np.array(valid_.iloc[:, 0])
    X_valid = tf.convert_to_tensor(X_valid, np.float32)
    # print(X_train.shape)

    model_in = Input(shape=(X_train.shape[1],))
    dense_0 = Dense(30, activation='relu')(model_in)
    dense_1 = Dense(30, activation='relu')(dense_0)
    dense_2 = Dense(X_train.shape[1], activation='softmax')(dense_1)
    model_out = dot([dense_2, model_in], axes=1, normalize=False)
    model = Model(inputs=model_in, outputs=model_out)

    model.compile(optimizer='adam',
                  loss='MeanSquaredError',
                  metrics=['accuracy'])

    model.build(X_train.shape)

    model.fit(X_train, y_train, epochs=300,
              batch_size=32,
              validation_split=0)

    prediction_ = model.predict(X_valid)
    # print(prediction_)
    # print(y_valid)

    for metric_name in cfg.metrics:
        metric_func = METRICS[metric_name]
        metric_value = metric_func(prediction_, y_valid, True)
        print(metric_name, metric_value)


if __name__ == '__main__':
    stacking()