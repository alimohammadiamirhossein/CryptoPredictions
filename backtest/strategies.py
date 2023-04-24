from backtesting import Strategy
import pandas as pd
import numpy as np

class Strategies:
    def __init__(self, df):
        self.df = df

    def signal1(self):
        position = False
        signal = [0] * self.df.shape[0]
        for i in range(1, len(signal)):
            if self.df['predicted_mean'][i] > self.df['Close'][i-1]:
                if position is False:
                    signal[i] = 2
                    position = True
                else:
                    signal[i] = 0
            else:
                if position is True:
                    signal[i] = 1
                    position = False
                else:
                    signal[i] = 0
        return signal

    def signal2(self):
        signal = [0] * self.df.shape[0]
        for i in range(10, len(signal)):
            buy_bool = True
            for j in range(10):
                if self.df['predicted_high'][i] < self.df['High'][i-j]:
                    buy_bool = False
            if buy_bool is True:
                signal[i] = 2
            sell_bool = True
            for j in range(10):
                if self.df['predicted_low'][i] > self.df['Low'][i - j]:
                    sell_bool = False
            if sell_bool is True:
                signal[i] = 1
        return signal

    def signal3(self):
        buy_price = []
        sell_price = []
        macd_signal = []
        signal = 0

        for i in range(len(self.df)):
            if self.df['macd'][i] > self.df['signal'][i]:
                if signal != 2:
                    signal = 2
                    macd_signal.append(signal)
                else:
                    macd_signal.append(0)
            elif self.df['macd'][i] < self.df['signal'][i]:
                if signal != 1:
                    signal = 1
                    macd_signal.append(signal)
                else:
                    macd_signal.append(0)
            else:
                macd_signal.append(0)

        return macd_signal

    def signal4(self):
        position = False
        signal = []
        for i in range(len(self.df)):
            if self.df['sma_30'][i] > self.df['sma_100'][i]:
                if position == False:
                    signal.append(2)
                    position = True
                else:
                    signal.append(0)
            elif self.df['sma_30'][i] < self.df['sma_100'][i]:
                if position == True:
                    signal.append(1)
                    position = False
                else:
                    signal.append(0)
            else:
                signal.append(0)
        return signal