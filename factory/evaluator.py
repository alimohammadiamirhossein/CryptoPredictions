import logging
import time
import gc

import numpy as np

logger = logging.getLogger(__name__)
from path_definition import *
from os.path import join
from metrics import METRICS


class Evaluator:
    def __init__(self, args, test_dataset, model, reporter):
        self.args = args
        self.model = model
        self.test_dataset = test_dataset
        self.test_dataset_Y = np.array(test_dataset.prediction)
        self.dates = test_dataset.Date
        self.test_dataset_X = test_dataset.drop(['prediction'], axis=1)
        self.metrics = args.metrics
        self.reporter = reporter
        # self.metrics = ['f1_score']
        self.is_regression = args.model.is_regression

    def evaluate(self):
        logger.info("Evaluating started.")
        time0 = time.time()

        predicted_df = self.model.predict(self.test_dataset_X)

        self.reporter.new_cross_started()
        for metric_name in self.metrics:
            metric_func = METRICS[metric_name]
            metric_value = metric_func(predicted_df, self.test_dataset_Y, self.is_regression)
            self.reporter.update_metric(metric_name, metric_value)

        if self.is_regression:
            self.reporter.plot_continues_data(self.dates, self.test_dataset_Y, predicted_df)

        return predicted_df



