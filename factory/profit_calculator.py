import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
from models import MODELS
from factory.trainer import Trainer
from backtest.strategies import Strategies


class ProfitCalculator:
    def __init__(self, args, dataset, profit_calculator, mean_prediction, reporter):
        self.args = args
        self.model = MODELS[self.args.model.type](self.args.model)
        self.original = profit_calculator
        self.dataset = dataset
        self.reporter = reporter
        self.mean_prediction = mean_prediction
        self.is_regression = args.model.is_regression
        self.save_dir = args.save_dir
        self.signal = None
        self.predicted_high = None
        self.predicted_low = None

    def profit_calculator(self):
        self.low_calculator()
        self.high_calculator()
        _, self.original = self.split_the_dataset(self.original)
        final = self.create_dataframe()
        address = self.setup_saving_dirs(self.save_dir)
        final.to_csv(address, encoding='utf-8', index=False)

    def create_dataframe(self):
        arr = np.row_stack((self.predicted_low, self.predicted_high, self.mean_prediction)).T
        predicteds = pd.DataFrame(arr, columns=['predicted_low', 'predicted_high', 'predicted_mean'])
        self.original.reset_index(drop=True, inplace=True)
        df = pd.concat([self.original, predicteds], axis=1)
        s1 = np.array(Strategies(df).signal1())
        s2 = np.array(Strategies(df).signal2())
        signal = np.row_stack((s1, s2)).T
        signal = pd.DataFrame(signal, columns=['signal1', 'signal2'])
        # final = pd.concat([self.original, signal], axis=1)
        final = pd.concat([df, signal], axis=1)
        return final

    def setup_saving_dirs(self, parent_dir):
        os.makedirs(os.path.join(parent_dir, 'backTest_dataset'), exist_ok=False)
        address = os.path.join(self.reporter.parent_dir, 'backTest_dataset',
                               f'{self.reporter.symbol}_{self.reporter.model}_backTest.csv')
        return address

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
        test_data_x = valid_dataset.drop(['prediction'], axis=1)
        self.predicted_low = self.model.predict(test_data_x)

    def high_calculator(self):
        logger.info("High price training started.")
        dataset_tmp = self.dataset.drop(['predicted_low'], axis=1, inplace=False)
        dataset_tmp = dataset_tmp.rename({'predicted_high': 'prediction',
                                          }, axis=1)
        train_dataset, valid_dataset = self.split_the_dataset(dataset_tmp)
        Trainer(self.args, train_dataset, None, self.model).train()
        test_data_x = valid_dataset.drop(['prediction'], axis=1)
        self.predicted_high = self.model.predict(test_data_x)
