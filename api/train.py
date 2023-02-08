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

    train_dataset = get_dataset(cfg.train_dataset, cfg.train_start_date, cfg.train_end_date, cfg)
    valid_dataset = get_dataset(cfg.valid_dataset, cfg.valid_start_date, cfg.valid_end_date, cfg)

    model = MODELS[cfg.model.type](cfg.model)
    reporter = Reporter(cfg)

    reporter.setup_saving_dirs(cfg.save_dir)

    cfg.save_dir = os.getcwd()
    Trainer(cfg, train_dataset, None, model).train()
    Evaluator(cfg, test_dataset=valid_dataset, model=model, reporter=reporter).evaluate()


if __name__ == '__main__':
    train()
