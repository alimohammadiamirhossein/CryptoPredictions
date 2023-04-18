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
    address = os.path.join(cfg.dataframe_path)
    df = pd.read_csv(address)
    bt = Backtest(df, MyCandlesStrat, cash=100_000, commission=.002)
    stat = bt.run()
    logging.info(stat)
    save_report(stat, address)


def save_report(stat, address):
    a = str(stat)
    print(a)
    print(11)
    with open(address, "w") as text_file:
        text_file.write(a)


def SIGNAL():
    global df
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

