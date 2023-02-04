from .BTCDataset import BTCDataset
from datetime import datetime

DATASETS = ['Bitcoin']
DATA_TYPES = ['train', 'validation', 'test']


def get_dataset(dataset_name, start_date, end_date, args):
    # if dataset_path is None:
    #     return None
    assert dataset_name in DATASETS, \
        f"Dataset {args.dataset_name} is not available."
    if dataset_name == 'Bitcoin':
        main_features = ['High', 'Low', 'Close', 'Open', 'Mean']
        main_features = ['High', 'Low', 'Mean']

        if start_date == "-1":
            start_date = None
        else:
            start_date = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')

        if end_date == "-1":
            start_date = None
        else:
            end_date = datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')

        btc = BTCDataset(main_features=main_features, start_date=start_date, end_date=end_date)
        dataset = btc.get_dataset()

    return dataset
