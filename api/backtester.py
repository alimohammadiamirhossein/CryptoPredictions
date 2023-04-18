from backtesting import Strategy, Backtest
import pandas as pd


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
    df = pd.read_csv('C:/Users/samen/Desktop/term9/CryptoPredictions/profit_calculation.csv')
    bt = Backtest(df, MyCandlesStrat, cash=100_000, commission=.002)
    stat = bt.run()
    print(stat)
