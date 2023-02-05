import logging

import math
import numpy as np
import sklearn.metrics
# import precision_score, \
#     recall_score, confusion_matrix, classification_report, \
#     accuracy_score, f1_score

logger = logging.getLogger(__name__)


def preprocess(pred, target, is_regression=False):
    if is_regression:
        y_test = []
        prediction = []
        for i in range(len(target) - 1):
            m1 = target[i + 1] - target[i]
            m2 = pred[i + 1] - pred[i]
            y_test.append(m1 > 0)
            prediction.append(m2 > 0)
        return y_test, prediction

    return target, pred


def accuracy_score(pred, target, is_regression=False):
    y_test, prediction = preprocess(pred, target, is_regression)
    ac = sklearn.metrics.accuracy_score(y_test, prediction)
    return ac


def f1_score(pred, target, is_regression=False):
    y_test, prediction = preprocess(pred, target, is_regression)
    f1 = sklearn.metrics.f1_score(y_test, prediction)
    return f1


def recall_score(pred, target, is_regression=False):
    y_test, prediction = preprocess(pred, target, is_regression)
    rec = sklearn.metrics.recall_score(y_test, prediction)
    return rec


def precision_score(pred, target, is_regression=False):
    y_test, prediction = preprocess(pred, target, is_regression)
    prec = sklearn.metrics.precision_score(y_test, prediction)
    return prec


def classification_report(pred, target, is_regression=False):
    y_test, prediction = preprocess(pred, target, is_regression)
    cls = sklearn.metrics.classification_report(y_test, prediction)
    return cls


def confusion_matrix(pred, target, is_regression=False):
    y_test, prediction = preprocess(pred, target, is_regression)
    conf = sklearn.metrics.confusion_matrix(y_test, prediction)
    return conf


def rmse(pred, target, is_regression=False):
    rmse = math.sqrt(np.square(np.subtract(pred, target)).mean())
    # print('\n RMSE:', "{:.2f}".format(rmse))
    return rmse

