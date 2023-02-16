import logging
import time
import gc

import numpy as np

logger = logging.getLogger(__name__)
from path_definition import *
from os.path import join
from metrics import METRICS
from metrics.metrics import preprocess
from models import MODELS
from factory.trainer import Trainer
from factory.evaluator import Evaluator

from keras.models import Sequential
from keras import Model
from keras.layers import Flatten
from keras.layers import Dense, dot, Input
from keras.layers import Activation


class Stacker:
    def __init__(self, args):
        self.args = args
        self.train_dataset = {}
        self.valid_dataset = {}
        self.confidence_rate = {}
        self.metric_results = {}

    def create_model(self, shape_):
        model_in = Input(shape=(shape_,))
        dense_0 = Dense(64, activation='relu')(model_in)
        dense_1 = Dense(128, activation='relu')(dense_0)
        dense_2 = Dense(64, activation='relu')(dense_1)
        dense_3 = Dense(shape_, activation='softmax')(dense_2)
        model_out = dot([dense_3, model_in], axes=1, normalize=False)
        model = Model(inputs=model_in, outputs=model_out)

        model.compile(optimizer='adam',
                      loss='MeanSquaredError',
                      metrics=['accuracy'])

        self.model = model

    def fit(self, X_train, y_train):
        self.model.build(X_train.shape)
        self.model.fit(X_train, y_train, epochs=self.args.epochs,
                  batch_size=self.args.batch_size,
                  validation_split=0)

    def model_handler(self, key, train_dataset, valid_dataset, reporter):
        item = self.args.models[key]
        model = MODELS[item['type']](item)
        arg = self.args.copy()
        arg['model'] = item
        if key != 'orbit':
            df3 = train_dataset.sample(frac=1).reset_index(drop=True).iloc[-2000:]
        else:
            df3 = train_dataset.iloc[-4000:]
        trainer = Trainer(arg, df3, None, model)
        trainer.train()
        evaluator = Evaluator(arg, test_dataset=train_dataset, model=model, reporter=reporter)
        prediction_ = evaluator.prediction()
        self.add_new_col(np.array(prediction_), item['type'], situation=0)
        evaluator = Evaluator(arg, test_dataset=valid_dataset, model=model, reporter=reporter)
        prediction_ = evaluator.prediction()
        self.add_new_col(np.array(prediction_), item['type'], situation=1)
        y_valid = np.array(valid_dataset.iloc[:, -1])
        dic = {}
        for metric_name in arg.metrics:
            metric_func = METRICS[metric_name]
            metric_value = metric_func(prediction_, y_valid, True)
            dic[metric_name] = metric_value
        conf_ = preprocess(prediction_, y_valid, True)
        self.add_new_col(np.array(conf_), item['type'], situation=2)
        self.metric_results[key] = dic

    def add_new_col(self, col, col_name, situation=0):
        if situation == 0: # train
            self.train_dataset[col_name] = col
        elif situation == 1: # valid
            self.valid_dataset[col_name] = col
        elif situation == 2: # confidence
            self.confidence_rate[col_name] = col





