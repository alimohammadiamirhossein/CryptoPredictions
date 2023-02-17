from .CoinMarketDataset import CoinMarketDataset
from .Bitmex import BitmexDataset
from datetime import datetime
import pandas as pd
import numpy as np

DATASETS = ['CoinMarket', 'Bitmex']
DATA_TYPES = ['train', 'validation', 'test']


def get_dataset(dataset_name, start_date, end_date, args):
    assert dataset_name in DATASETS, \
        f"Dataset {args.dataset_name} is not available."
    if dataset_name == 'CoinMarket':
        main_features = ['High', 'Volume', 'Low', 'Close', 'Open', 'Mean']
        # main_features = ['High', 'Low', 'Mean']

        if start_date == "-1":
            start_date = None
        else:
            start_date = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')

        if end_date == "-1":
            start_date = None
        else:
            end_date = datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')

        btc = CoinMarketDataset(main_features=main_features, start_date=start_date,
                                end_date=end_date, window_size=args.dataset_loader.window_size)
        dataset = btc.get_dataset()

    elif dataset_name == 'Bitmex':
        btc = BitmexDataset(args)
        dataset = btc.get_dataset()

    return dataset


