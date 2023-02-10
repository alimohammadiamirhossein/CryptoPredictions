import logging
import os
from itertools import chain
import sys

# sys.path.append('../')

import hydra
from omegaconf import DictConfig
from models import MODELS
from data_loader import get_dataset
from factory.trainer import Trainer
from factory.evaluator import Evaluator

from sklearn.model_selection import TimeSeriesSplit
from path_definition import HYDRA_PATH
# from schedulers import SCHEDULERS
from utils.reporter import Reporter
# from utils.save_load import load_snapshot, save_snapshot, setup_training_dir

logger = logging.getLogger(__name__)


@hydra.main(config_path=HYDRA_PATH, config_name="train")
def train(cfg: DictConfig):
    if cfg.load_path is None and cfg.model is None:
        msg = 'either specify a load_path or config a model.'
        logger.error(msg)
        raise Exception(msg)

    cfg.save_dir = os.getcwd()
    reporter = Reporter(cfg)
    reporter.setup_saving_dirs(cfg.save_dir)
    model = MODELS[cfg.model.type](cfg.model)

    dataset = get_dataset(cfg.dataset.name, cfg.dataset.train_start_date, cfg.dataset.valid_end_date, cfg)
    print(dataset)



    if cfg.validation_method == 'simple':
        train_dataset = dataset[
            (dataset['Date'] > cfg.dataset.train_start_date) & (dataset['Date'] < cfg.dataset.train_end_date)]
        valid_dataset = dataset[
            (dataset['Date'] > cfg.dataset.valid_start_date) & (dataset['Date'] < cfg.dataset.valid_end_date)]
        
        Trainer(cfg, train_dataset, None, model).train()
        Evaluator(cfg, test_dataset=valid_dataset, model=model, reporter=reporter).evaluate()
        reporter.print_pretty_metrics(logger)
        reporter.save_metrics()

    elif cfg.validation_method == 'cross_validation':
        n_split = dataset.shape[0]//365
        tscv = TimeSeriesSplit(n_splits=n_split)

        for train_index, test_index in tscv.split(dataset):
            train_dataset, valid_dataset = dataset.iloc[train_index], dataset.iloc[test_index]
            Trainer(cfg, train_dataset, None, model).train()
            Evaluator(cfg, test_dataset=valid_dataset, model=model, reporter=reporter).evaluate()

        reporter.add_average()
        reporter.print_pretty_metrics(logger)
        reporter.save_metrics()



        # Trainer(cfg, train_dataset, None, model).train()
        # Evaluator(cfg, test_dataset=valid_dataset, model=model, reporter=reporter).evaluate()


if __name__ == '__main__':
    train()
