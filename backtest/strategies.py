from backtesting import Strategy

    
class Strategies:
    def __init__(self, df):
        self.df = df

    def signal1(self):
        print(self.df)
        signal = [0] * self.df.shape[0]
        for i in range(1, len(signal)):
            if self.df['predicted_mean'][i] > self.df['Close'][i-1]:
                signal[i] = 2
            else:
                signal[i] = 1
        return signal
