import logging
import time
import gc

import numpy as np

logger = logging.getLogger(__name__)
from path_definition import *
from os.path import join
from metrics import METRICS


class Stacker:
    def __init__(self, args):
        self.args = args
        self.train_dataset = {}
        self.valid_dataset = {}

    def add_new_col(self, col, col_name, is_train=True):
        if is_train:
            self.train_dataset[col_name] = col
        else:
            self.valid_dataset[col_name] = col




