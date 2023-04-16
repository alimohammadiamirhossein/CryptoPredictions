import logging
import time
import gc

import numpy as np

logger = logging.getLogger(__name__)
from path_definition import *
from os.path import join
from metrics import METRICS
from models import MODELS
from factory.trainer import Trainer

class ProfitCalculator:
    def __init__(self, args, dataset, model, reporter):
        self.args = args
        self.model = MODELS[self.args.model.type](self.args.model)
        self.dataset = dataset
        self.reporter = reporter
        # self.metrics = ['f1_score']
        self.is_regression = args.model.is_regression

    def split_the_dataset(self, dataset):
        train_dataset = dataset[
            (dataset['Date'] > self.args.dataset_loader.train_start_date) & (
                    dataset['Date'] < self.args.dataset_loader.train_end_date)]
        valid_dataset = dataset[
            (dataset['Date'] > self.args.dataset_loader.valid_start_date) & (
                    dataset['Date'] < self.args.dataset_loader.valid_end_date)]

    def low_calculator(self):
        logger.info("Low price training started.")
        self.dataset.drop(['predicted_high'], axis=1, inplace=True)
        print(self.dataset)
        Trainer(self.args, train_dataset, None, self.model).train()
