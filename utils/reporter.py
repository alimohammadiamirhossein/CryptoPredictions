import os


import pandas as pd
import matplotlib.pyplot as plt
from .average_meter import AverageMeter


class Reporter:
    def __init__(self, args):
        self.metrics = None
        self.args = args
        self.parent_dir = args.save_dir
        self.model = args.model.type
        self.symbol = args.symbol
        self.plot_counter = 0
        self.df_of_cross_validation = None
        self.counter_cross_validation = -1
        self.setup()

    def setup(self):
        self.metrics = {}
        cols = []
        for item in self.args.metrics:
            self.metrics[item] = AverageMeter()
            cols.append(item)
        self.df_of_cross_validation = pd.DataFrame(columns=cols)
        # self.add_new_row_to_data_frame()

    def update_metric(self, metric_name, value):
        if isinstance(value, float):
            self.metrics[metric_name].update(value)
            self.df_of_cross_validation.iloc[self.counter_cross_validation][metric_name] = value

    def add_new_row_to_data_frame(self, index_=None):
        df1 = {}
        for item in self.args.metrics:
            df1[item] = None
        if index_ is None:
            index_ = self.counter_cross_validation
        df1 = pd.DataFrame(df1, index=[index_])
        self.df_of_cross_validation = pd.concat([self.df_of_cross_validation, df1])

    def new_cross_started(self):
        self.counter_cross_validation = self.counter_cross_validation + 1
        self.add_new_row_to_data_frame(f"validation-{self.counter_cross_validation}")

    def setup_saving_dirs(self, parent_dir):
        os.makedirs(os.path.join(parent_dir, 'plots'), exist_ok=False)
        os.makedirs(os.path.join(parent_dir, 'metrics_history'), exist_ok=False)

    def add_average(self):
        self.counter_cross_validation = self.counter_cross_validation + 1
        self.add_new_row_to_data_frame(f"average")
        for metric_name, metric_value in self.metrics.items():
            self.update_metric(metric_name, metric_value.get_average())

    def print_pretty_metrics(self, logger):
        result = "\n"
        str_ = '|'.join([''.ljust(15)] + [a.center(15) for a in self.metrics.keys()])
        result += (str_ + '\n')
        for index, row in self.df_of_cross_validation.iterrows():
            str_ = '|'.join([index.ljust(15)] + ["{:.2f}".format(a).center(15) if a > 1
                                                     else "{:.3f}".format(a).center(15) for a in list(row)])
            result += (str_ + '\n')
        logger.info(result)

    def save_metrics(self):
        address = os.path.join(self.parent_dir, 'metrics_history', f'{self.symbol}_{self.model}_metrics.csv')
        self.df_of_cross_validation.to_csv(address)
        with open(os.path.join(self.parent_dir, 'metrics_history', f'metrics.txt'), "w") as text_file:
            result = "\n"
            str_ = '|'.join([''.ljust(15)] + [a.center(15) for a in self.metrics.keys()])
            result += (str_ + '\n')
            for index, row in self.df_of_cross_validation.iterrows():
                str_ = '|'.join([index.ljust(15)] + ["{:.2f}".format(a).center(15) if a > 1
                                                     else "{:.3f}".format(a).center(15) for a in list(row)])
                result += (str_ + '\n')
            text_file.write(result)

    def plot_continues_data(self, dates, testX, predicted_df):
        plt.figure(figsize=(25, 15), dpi=80, facecolor='w', edgecolor='k')
        ax = plt.gca()
        plt.plot(dates, testX, color='red', label=f'Real {self.symbol} Price')
        plt.plot(dates, predicted_df, color='blue', label=f'Predicted {self.symbol} Price')
        plt.title(f'{self.symbol} Price Prediction', fontsize=40)
        # x = .reset_index().index
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(18)
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(18)
        plt.xlabel('Time', fontsize=40)
        plt.ylabel(f'{self.symbol} Price(USD) [Closed]', fontsize=40)
        plt.legend(loc=2, prop={'size': 25})
        plt.savefig(os.path.join(self.parent_dir, 'plots', f'plot_{self.plot_counter}'))
        self.plot_counter += 1
        plt.close()
        # plt.show()
