import logging
import os

from backtesting import Strategy, Backtest
import pandas as pd

import hydra
from omegaconf import DictConfig
from utils.reporter import Reporter
from path_definition import HYDRA_PATH

df = None
address = ""


@hydra.main(config_path=HYDRA_PATH, config_name="backtest")
def backTester(cfg: DictConfig):
    global address
    global df
    table_list = []
    address = cfg.dataframe_path
    for filename in os.listdir(address):
        if filename.endswith('.csv'):
            table_list.append(filename)
    filename = table_list[0]
    file_address = os.path.join(address, filename)
    df = pd.read_csv(file_address)
    bt = Backtest(df, MyCandlesStrat, cash=100_000, commission=.002)
    stat = bt.run()
    logging.info(stat)
    save_report(stat, address, filename)


def save_report(stat, address, fname):
    fname = fname.split('.')[0]
    a = str(stat)
    new_add = os.path.join(address, f'{fname}.txt')
    with open(new_add, "w") as text_file:
        text_file.write(a)


def SIGNAL():
    global df
    return df.signal1


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

