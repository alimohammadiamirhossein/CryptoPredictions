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

    def print_pretty_metrics(self, logger):
        for metric_name, metric_value in self.metrics.items():
            value = str(metric_value)
            if isinstance(metric_value, float):
                value = "{:.2f}".format(metric_value)
            logger.info(f'\n{metric_name}:\n{value}')

