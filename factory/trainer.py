import logging
import time
import gc

logger = logging.getLogger(__name__)
from path_definition import *
from os.path import join


class Trainer:
    def __init__(self, args, train_dataset, valid_dataset, model):
        self.args = args
        self.model = model
        self.use_validation = False if valid_dataset is None else True


    def train(self):
        logger.info("Training started.")
        time0 = time.time()

        self.model.fit(train_dataset)

        logger.info("-" * 100)
        logger.info('Training is completed in %.2f seconds.' % (time.time() - time0))




