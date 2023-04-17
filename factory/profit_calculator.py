from backtesting import Strategy
import logging
import time
import gc

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
from path_definition import *
from os.path import join
from metrics import METRICS
from models import MODELS
from factory.trainer import Trainer
from factory.evaluator import Evaluator
from utils.reporter import Reporter
from backtest.strategies import Strategies
from backtest.myCandlesStrat import MyCandlesStrat
from backtesting import Backtest


class ProfitCalculator:
    def __init__(self, args, dataset, profit_calculator, mean_prediction, reporter):
        self.args = args
        self.model = MODELS[self.args.model.type](self.args.model)
        self.original = profit_calculator
        self.dataset = dataset
        self.reporter = reporter
        self.mean_prediction = mean_prediction
        # self.metrics = ['f1_score']
        self.is_regression = args.model.is_regression
        self.signal = None

    def profit_calculator(self):
        # self.low_calculator()
        # self.high_calculator()
        _, self.original = self.split_the_dataset(self.original)
        # print(self.original)
        # print(self.predicted_high.shape, self.predicted_low.shape)
        # print(self.mean_prediction.shape)
        arr = np.row_stack((self.mean_prediction, self.mean_prediction)).T
        # arr = np.row_stack((self.predicted_low, self.predicted_high, self.mean_prediction)).T
        # final = pd.DataFrame(arr, columns=['predicted_low', 'predicted_high', 'predicted_mean'])
        final = pd.DataFrame(arr, columns=['predicted_mean', 'prediction_low'])
        print(final)
        self.original.reset_index(drop=True, inplace=True)
        final = pd.concat([self.original, final], axis=1)
        signal = np.array(Strategies(final).signal1()).T
        signal = pd.DataFrame(signal, columns=['signal'])
        final = pd.concat([self.original, signal], axis=1)
        final.to_csv('C:/Users/samen/Desktop/term9/CryptoPredictions/profit_calculation.csv',
                     encoding='utf-8', index=False)

    def split_the_dataset(self, dataset):
        train_dataset = dataset[
            (dataset['Date'] > self.args.dataset_loader.train_start_date) & (
                    dataset['Date'] < self.args.dataset_loader.train_end_date)]
        valid_dataset = dataset[
            (dataset['Date'] > self.args.dataset_loader.valid_start_date) & (
                    dataset['Date'] < self.args.dataset_loader.valid_end_date)]
        return train_dataset, valid_dataset

    def low_calculator(self):
        dataset_tmp = self.dataset.drop(['predicted_high'], axis=1, inplace=False)
        logger.info("Low price training started.")
        dataset_tmp = dataset_tmp.rename({'predicted_low': 'prediction',
                                        }, axis=1)
        train_dataset, valid_dataset = self.split_the_dataset(dataset_tmp)
        Trainer(self.args, train_dataset, None, self.model).train()
        test_dataX = valid_dataset.drop(['prediction'], axis=1)
        self.predicted_low = self.model.predict(test_dataX)

    def high_calculator(self):
        logger.info("High price training started.")
        dataset_tmp = self.dataset.drop(['predicted_low'], axis=1, inplace=False)
        dataset_tmp = dataset_tmp.rename({'predicted_high': 'prediction',
                                          }, axis=1)
        train_dataset, valid_dataset = self.split_the_dataset(dataset_tmp)
        Trainer(self.args, train_dataset, None, self.model).train()
        test_dataX = valid_dataset.drop(['prediction'], axis=1)
        self.predicted_high = self.model.predict(test_dataX)




