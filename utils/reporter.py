import sys
import os
import json

import numpy as np
import pandas as pd



class Reporter:
    def __init__(self, args):
        self.metrics = None
        self.args = args
        self.setup()

    def setup(self):
        self.metrics = {}
        for item in self.args.metrics:
            self.metrics[item] = None

    def update_metric(self, metric_name, value):
        self.metrics[metric_name] = value

    def print_pretty_metrics(self, logger, metrics):
        for metric_name in self.metrics:
            metric_func = METRICS[metric_name]
            metric_value = metric_func(predicted_df, self.test_dataset_Y, self.is_regression)
            print(f'{metric_name}: {metric_value}')

