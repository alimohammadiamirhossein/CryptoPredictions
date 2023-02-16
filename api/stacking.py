import logging
import os
from itertools import chain
import sys

# sys.path.append('../')

import hydra
from omegaconf import DictConfig
from data_loader import get_dataset

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
    stacker.add_new_col(np.array(train_dataset.prediction), 'prediction', situation=0)
    stacker.add_new_col(np.array(valid_dataset.prediction), 'prediction', situation=1)

    for key in cfg.models.keys():
        stacker.model_handler(key, train_dataset, valid_dataset, reporter)

    train_ = pd.DataFrame(stacker.train_dataset)
    valid_ = pd.DataFrame(stacker.valid_dataset)

    X_train = tf.convert_to_tensor(np.array(train_.iloc[:, 1:]), np.float32)
    y_train = tf.convert_to_tensor(np.array(train_.iloc[:, 0]), np.float32)

    y_valid = np.array(valid_.iloc[:, 0])
    X_valid = tf.convert_to_tensor(np.array(valid_.iloc[:, 1:]), np.float32)
    # print(X_train.shape)

    stacker.create_model(X_train.shape[1])
    stacker.fit(X_train, y_train)

    prediction_ = stacker.model.predict(X_valid)

    dic_ = {}
    for metric_name in cfg.metrics:
        metric_func = METRICS[metric_name]
        metric_value = metric_func(prediction_, y_valid, True)
        dic_[metric_name] = metric_value
    stacker.metric_results['stacking'] = dic_
    print(stacker.metric_results)


if __name__ == '__main__':
    stacking()
