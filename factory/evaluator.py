import logging
import time
import gc

import numpy as np

logger = logging.getLogger(__name__)
from path_definition import *
from os.path import join
from metrics import METRICS

class Evaluator:
    def __init__(self, args, test_dataset, model):
        self.args = args
        self.model = model
        self.test_dataset = test_dataset
        self.test_dataset_X = np.array(test_dataset)[:, 1:-1]
        self.test_dataset_Y = np.array(test_dataset)[:,-1]
        # self.metrics = args.metrics
        self.metrics = ['f1_score']
        self.is_regression = args.model.is_regression

    def evaluate(self):
        logger.info("Evaluating started.")
        time0 = time.time()

        predicted_df = self.model.predict(self.test_dataset_X)

        for metric_name in self.metrics:
            metric_func = METRICS[metric_name]
            metric_value = metric_func(predicted_df, self.test_dataset_Y, self.is_regression)
            print(f'{metric_name}: {metric_value}')


        logger.info("-" * 100)
        logger.info('Training is completed in %.2f seconds.' % (time.time() - time0))




