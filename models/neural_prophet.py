from neuralprophet import NeuralProphet


class Neural_Prophet:

    def __init__(self, args):
        self.response_col = args.response_col
        self.date_col = args.date_col
        self.is_daily = args.is_daily
        self.is_hourly = args.is_hourly
        self.quantile_list = [round(((1 - args.confidence_level) / 2), 2),
                              round((args.confidence_level + (1 - args.confidence_level) / 2), 2)]
        self.model = None
        self.regressors = []

    def fit(self, data_x):
        yearly_seasonality = False
        weekly_seasonality = False
        daily_seasonality = False
        n_lags = 30

        if self.is_daily:
            n_lags = 2 * 30
            yearly_seasonality = True
            weekly_seasonality = True
        elif self.is_hourly:
            n_lags = 3 * 24
            daily_seasonality = True

        self.model = NeuralProphet(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
            n_lags=n_lags,
            learning_rate=0.003,
            quantiles=self.quantile_list,
        )

        self.regressors = []
        for col in data_x.columns:
            if col != self.response_col and col != self.date_col:
                self.regressors.append(col)
        for feature in self.regressors:
          self.model.add_regressor(feature)

        data_x[self.regressors] = data_x[self.regressors].astype(float)
        data_x[self.response_col] = data_x[self.response_col].astype(float)
        ml_df1 = data_x.reset_index().rename(columns={self.date_col: 'ds', self.response_col: 'y'})
        self.model.fit(ml_df1)

    def predict(self, test_x):
        test_x[self.regressors] = test_x[self.regressors].astype(float)
        test_x = test_x.reset_index().rename(columns={self.date_col: 'ds', self.response_col: 'y'})
        pred_y = self.model.predict(test_x)
        return pred_y.yhat

