# # Import the model we are using
# from fbprophet import Prophet
# import numpy as np
#
#
# class MyProphet:
#
#     def __init__(self, args):
#         self.response_col = args.response_col
#         self.date_col = args.date_col
#
#     def fit(self, data_x):
#         self.model_fbp = Prophet()
#         self.regressors = []
#         for col in data_x.columns:
#             if col != self.response_col and col != self.date_col:
#                 self.regressors.append(col)
#         for feature in self.regressors:
#           self.model_fbp.add_regressor(feature)
#         data_x[self.regressors] = data_x[self.regressors].astype(float)
#         data_x[self.response_col] = data_x[self.response_col].astype(float)
#         ml_df1 = data_x.reset_index().rename(columns={self.date_col: 'ds', self.response_col: 'y'})
#         # print(train_x)
#         self.model_fbp.fit(ml_df1)
#
#     def predict(self, test_x):
#         test_x[self.regressors] = test_x[self.regressors].astype(float)
#         test_x = test_x.reset_index().rename(columns={self.date_col: 'ds', self.response_col: 'y'})
#         pred_y = self.model_fbp.predict(test_x)
#         return pred_y.yhat
#
#
