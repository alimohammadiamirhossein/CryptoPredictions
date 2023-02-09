from orbit.models import DLT

class Orbit:
    model = None

    def __init__(self, args):
        self.response_col = args.response_col
        self.date_col = args.date_col
        self.estimator = args.estimator
        self.seasonality = args.seasonality
        self.seed = args.seed
        self.global_trend_option = args.global_trend_option
        self.n_bootstrap_draws = args.n_bootstrap_draws

    def fit(self, data_x):
        regressors = []
        for col in data_x.columns:
            if col != self.response_col and col != self.date_col:
                regressors.append(col)

        self.model = DLT(
            response_col=self.response_col,
            date_col=self.date_col,
            regressor_col=regressors,
            estimator=self.estimator,
            seasonality=self.seasonality,
            seed=self.seed,
            global_trend_option=self.global_trend_option,
            # for prediction uncertainty
            n_bootstrap_draws=self.n_bootstrap_draws,
        )

        self.model.fit(data_x, point_method="mean")

    def predict(self, test_x):
        predicted_df = self.model.predict(df=test_x)
        return predicted_df