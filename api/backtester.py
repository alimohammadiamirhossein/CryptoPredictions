import logging
import os

from backtesting import Strategy, Backtest
import pandas as pd

import hydra
from omegaconf import DictConfig
from utils.reporter import Reporter
from path_definition import HYDRA_PATH

global df

@hydra.main(config_path=HYDRA_PATH, config_name="train")
def backTester(cfg: DictConfig):
    cfg.save_dir = os.getcwd()
    reporter = Reporter(cfg)
    reporter.setup_saving_dirs(cfg.save_dir)

    address = os.path.join(reporter.parent_dir,
                           f'{reporter.symbol}_{reporter.model}_backTest.csv')

    df = pd.read_csv(address)
    bt = Backtest(df, MyCandlesStrat, cash=100_000, commission=.002)
    stat = bt.run()
    logging.info(stat)
    save_report(stat, reporter)


def save_report(stat, reporter):
    address = os.path.join(reporter.parent_dir,
                           f'{reporter.symbol}_{reporter.model}_backTest_report.txt')

    with open(address, "w") as text_file:
        text_file.write(stat)


def SIGNAL():
    return df.signal


class MyCandlesStrat(Strategy):
    def init(self):
        super().init()
        self.signal1 = self.I(SIGNAL)

    def next(self):
        super().next()
        if self.signal1 == 2:
            sl1 = self.data.Close[-1] - 750e-4
            tp1 = self.data.Close[-1] + 600e-4
            # self.buy(sl=sl1, tp=tp1)
            self.buy(sl=sl1)
        elif self.signal1 == 1:
            sl1 = self.data.Close[-1] + 750e-4
            tp1 = self.data.Close[-1] - 600e-4
            # self.sell(sl=sl1, tp=tp1)
            self.sell(sl=sl1)


if __name__ == '__main__':
    backTester()

