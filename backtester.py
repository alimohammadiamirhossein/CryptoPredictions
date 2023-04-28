import logging
import os

from backtesting import Strategy, Backtest
from backtest.strategies import Strategies
import pandas as pd

import hydra
from omegaconf import DictConfig
from utils.reporter import Reporter
from path_definition import HYDRA_PATH
from data_loader.indicators import *
import numpy as np

df = None
address = ""
strategy_signal = ""
buy_stop_loss = 0.8
buy_take_profit = 1.2
sell_stop_loss = 1.2
sell_take_profit = 0.8


@hydra.main(config_path=HYDRA_PATH, config_name="backtest")
def backTester(cfg: DictConfig):
    global address
    global df
    global strategy_signal
    global buy_stop_loss
    global buy_take_profit
    global sell_stop_loss
    global sell_take_profit
    table_list = []
    address = cfg.dataframe_path
    strategy_signal = cfg.strategy_signal
    buy_stop_loss = cfg.buy_stop_loss
    buy_take_profit = cfg.buy_take_profit
    sell_stop_loss = cfg.sell_stop_loss
    sell_take_profit = cfg.sell_take_profit
    for filename in os.listdir(address):
        if filename.endswith('.csv'):
            table_list.append(filename)
    filename = table_list[0]
    file_address = os.path.join(address, filename)
    df = pd.read_csv(file_address)
    df = add_indicators(df, cfg)
    df = add_signals(df)
    bt = Backtest(df, MyCandlesStrat, cash=100_000, commission=.002)
    stat = bt.run()
    logging.info(stat)
    save_report(stat, address, filename)


def add_signals(df):
    strategy = Strategies(df)
    sig3 = np.array(strategy.signal3())
    sig4 = np.array(strategy.signal4())
    sigs = np.row_stack((sig3, sig4)).T
    sigs = pd.DataFrame(sigs, columns=['signal3', 'signal4'])
    df = pd.concat([df, sigs], axis=1)
    return df


def add_indicators(df, cfg):
    df['sma_30'] = sma(np.array(df.Close), 30)
    df['sma_100'] = sma(np.array(df.Close), 30)
    exp1 = df.ewm(span=26, adjust=False).mean()
    exp2 = df.ewm(span=12, adjust=False).mean()
    macd = pd.DataFrame(exp1 - exp2).rename(columns={'Close': 'macd'})
    macd = macd['macd']
    signal = pd.DataFrame(macd.ewm(span=9, adjust=False).mean()).rename(columns={'macd': 'signal'})
    hist = pd.DataFrame(macd - signal['signal']).rename(columns={0: 'hist'})
    frames = [macd, signal, hist]
    df2 = pd.concat(frames, join='inner', axis=1)
    df = pd.concat([df, df2], axis=1)
    return df
    # return arr1, dates


def save_report(stat, address, fname):
    fname = fname.split('.')[0]
    a = str(stat)
    new_add = os.path.join(address, f'{fname}.txt')
    with open(new_add, "w") as text_file:
        text_file.write(a)


def SIGNAL():
    global df
    global strategy_signal
    if strategy_signal is "":
        return df.signal1
    else:
        return df[strategy_signal]


class MyCandlesStrat(Strategy):
    def init(self):
        super().init()
        self.signal1 = self.I(SIGNAL)

    def next(self):
        super().next()
        global buy_stop_loss
        global buy_take_profit
        global sell_stop_loss
        global sell_take_profit
        if self.signal1 == 2:
            sl1 = buy_stop_loss * self.data.Close[-1]
            tp1 = buy_take_profit * self.data.Close[-1]
            self.buy(sl=sl1, tp=tp1)
        elif self.signal1 == 1:
            sl1 = sell_stop_loss * self.data.Close[-1]
            tp1 = sell_take_profit * self.data.Close[-1]
            self.sell(sl=sl1, tp=tp1)


if __name__ == '__main__':
    backTester()
