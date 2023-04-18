from backtesting import Strategy


class Strategies:
    def __init__(self, df):
        self.df = df

    def signal1(self):
        signal = [0] * self.df.shape[0]
        for i in range(1, len(signal)):
            if self.df['predicted_mean'][i] > self.df['Close'][i-1]:
                signal[i] = 2
            else:
                signal[i] = 1
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
