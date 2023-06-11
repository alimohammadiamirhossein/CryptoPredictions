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
from scipy import stats
from statsmodels.stats.diagnostic import normal_ad


@hydra.main(config_path=HYDRA_PATH, config_name="phase_prediction")
def phasePrediction(cfg: DictConfig):
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
    # phase_calculator1(df, base_period, test_period, tresh_hold)
    # conformal_prediction(df)
    # percentage_differences(df)
    # RMSFE(df)


def get_best_distribution(data):
    dist_names = ["norm", "exponweib", "weibull_max", "weibull_min", "pareto", "genextreme"]
    dist_results = []
    params = {}
    for dist_name in dist_names:
        dist = getattr(stats, dist_name)
        param = dist.fit(data)

        params[dist_name] = param
        # Applying the Kolmogorov-Smirnov test
        D, p = stats.kstest(data, dist_name, args=param)
        print("p value for "+dist_name+" = "+str(p))
        dist_results.append((dist_name, p))

    # select the best fitted distribution
    best_dist, best_p = (max(dist_results, key=lambda item: item[1]))
    # store the name of the best fit and its p value

    print("Best fitting distribution: "+str(best_dist))
    print("Best p value: "+ str(best_p))
    print("Parameters for the best fit: "+ str(params[best_dist]))

    return best_dist, best_p, params[best_dist]


def RMSFE(df):
    residuals = []
    for index, row in df.iterrows():
        b = df.iloc[index]
        quantile_index = -1
        mean = (b.Low + b.High) / 2
        error = b.predicted_mean - mean
        error = error if error > 0 else -1*error
        residuals.append(error)
    residuals = sorted(residuals)
    get_best_distribution(residuals)


def percentage_differences(df):
    cou = [0, 0, 0, 0, 0]
    tresh = [0.06, 0.5, 1.0, 2, 4]
    for index, row in df.iterrows():
        if index == 0:
            continue
        b = df.iloc[index].Close
        c = df.iloc[index-1].Close
        percentage = (b - c)/c
        if percentage * 100 < tresh[0]:
            cou[0] += 1
        for i in range(1, len(cou)):
            if tresh[i-1] <= percentage*100 < tresh[i]:
                cou[i] += 1
    sum_ = 0
    for i in range(len(cou)):
        sum_ += cou[i]
    for i in range(len(cou)):
        print(tresh[i], end=', ')
    print()
    for i in range(len(cou)):
        print(str(100*cou[i]/sum_)[:5], end=', ')


def conformal_prediction(df):
    calibration_set = {}
    counter = 0
    low_confidences = []
    for index, row in df.iterrows():
        b = df.iloc[index]
        quantile_index = -1
        mean = (b.Low + b.High) / 2
        error = b.predicted_mean - mean
        error = error if error > 0 else -1*error
        keys_ = list(calibration_set.keys())
        if len(keys_) > 0 and error < calibration_set[keys_[0]]:
            quantile_index = 0
        for i in range(len(keys_)-1):
            next_key = keys_[i+1]
            current_key = keys_[i]
            if calibration_set[next_key] > error > calibration_set[current_key]:
                   quantile_index = i+1

        if len(keys_) > 0:
            prob = (1+quantile_index)/len(keys_)
            confidence_rate = (1 - prob)*100
            if confidence_rate < 20:
                counter += 1
                low_confidences.append(index)
            #     print(str(confidence_rate) + "  ###################################")
            # else:
            #     print(confidence_rate)
        #update calibration set
        calibration_set[index] = error
        calibration_set = dict(sorted(calibration_set.items(), key=lambda item: item[1]))

    print(counter , len(keys_))
    hyper_params = list(range(3, 8))
    for hyper_param in hyper_params:
        test_low_confidence_counter = 0
        for i in range(1, len(low_confidences)):
            if low_confidences[i] - low_confidences[i-1] <= hyper_param:
                test_low_confidence_counter += 1
        print(hyper_param, str(100*test_low_confidence_counter/ len(low_confidences))[:5])




def phase_calculator1(df, base_period, test_period, tresh_hold):
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

