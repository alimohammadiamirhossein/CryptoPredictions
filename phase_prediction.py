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
base_period = 10
test_period = 4
tresh_hold = 0.6


@hydra.main(config_path=HYDRA_PATH, config_name="phase_prediction")
def phasePrediction(cfg: DictConfig):
    global df
    global base_period
    global test_period
    global tresh_hold
    table_list = []
    address = cfg.dataframe_path
    base_period = cfg.base_period
    test_period = cfg.test_period
    tresh_hold = cfg.tresh_hold
    for filename in os.listdir(address):
        if filename.endswith('.csv'):
            table_list.append(filename)
    filename = table_list[0]
    file_address = os.path.join(address, filename)
    df = pd.read_csv(file_address)
    phase_calculator()


def phase_calculator():
    global df
    global base_period
    global test_period
    global tresh_hold
    total = 0
    correct = 0
    unpredictable = [0] * df.shape[0]
    for index, row in df.iterrows():
        if index < base_period+test_period+1:
            continue
        mean_error = 0
        for i in range(1+test_period, base_period+test_period+1):
            a = df.iloc[index-i]
            mean_tmp = (a.Low + a.High)/2
            mean_error += a.predicted_mean - mean_tmp
        mean_error = mean_error / base_period

        counter = 0
        for i in range(1, 1+test_period):
            b = df.iloc[index-i]
            mean = (b.Low + b.High) / 2
            error = b.predicted_mean - mean
            if error > mean_error:
                counter += 1
        if counter >= tresh_hold * test_period:
            unpredictable[index] = 1
            total += 1
            c = df.iloc[index]
            mean = (c.Low + c.High) / 2
            error = c.predicted_mean - mean
            if error > mean_error:
                correct += 1

    print(unpredictable)
    print(correct, total, index)
    print('accuracy', 100*correct/total)







if __name__ == '__main__':
    phasePrediction()

