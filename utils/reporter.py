import sys
import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .average_meter import AverageMeter

class Reporter:
    def __init__(self, args):
        self.metrics = None
        self.args = args
        self.parent_dir = args.save_dir
        self.plot_counter = 0
        self.setup()

    def setup(self):
        self.metrics = {}
        for item in self.args.metrics:
            self.metrics[item] = AverageMeter()

    def update_metric(self, metric_name, value):
        if isinstance(value, float):
            self.metrics[metric_name].update(value)

    def setup_saving_dirs(self, parent_dir):
        os.makedirs(os.path.join(parent_dir, 'plots'), exist_ok=False)
        os.makedirs(os.path.join(parent_dir, 'metrics_history'), exist_ok=False)

    def print_pretty_metrics(self, logger):
        if len(self.metrics) > 0:
            logger.info(f'Metrics Result:')
        for metric_name, metric_value in self.metrics.items():
            value = str(metric_value.get_average())
            if isinstance(metric_value.get_average(), float):
                value = "{:.2f}".format(metric_value.get_average())
            logger.info(f'\n{metric_name}:\n{value}')

    def save_metrics(self):
        with open(os.path.join(self.parent_dir, 'metrics_history', f'metrics.txt'), "w") as text_file:
            if len(self.metrics) > 0:
                text_file.write(f'Metrics Result:')
            for metric_name, metric_value in self.metrics.items():
                value = str(metric_value.get_average())
                if isinstance(metric_value.get_average(), float):
                    value = "{:.2f}".format(metric_value.get_average())
                text_file.write(f'\n\n{metric_name}:\n{value}')


    def plot_continues_data(self, dates, testX, predicted_df):
        plt.figure(figsize=(25, 15), dpi=80, facecolor='w', edgecolor='k')
        ax = plt.gca()
        plt.plot(dates, testX, color='red', label='Real BTC Price')
        plt.plot(dates, predicted_df, color='blue', label='Predicted BTC Price')
        plt.title('BTC Price Prediction', fontsize=40)
        # x = .reset_index().index
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(18)
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(18)
        plt.xlabel('Time', fontsize=40)
        plt.ylabel('BTC Price(USD) [Closed]', fontsize=40)
        plt.legend(loc=2, prop={'size': 25})
        plt.savefig(os.path.join(self.parent_dir, 'plots', f'plot_{self.plot_counter}'))
        self.plot_counter += 1
        plt.close()
        # plt.show()
            
    

