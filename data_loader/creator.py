import numpy as np
import pandas as pd
from datetime import datetime


def preprocess(dataset, cfg, logger=None):
    dataset = dataset[
        (dataset['Date'] > cfg.dataset_loader.train_start_date) & (dataset['Date'] < cfg.dataset_loader.valid_end_date)]
    if cfg.dataset_loader.features is not None:
        features = cfg.dataset_loader.features.split(',')
        features = [s.strip() for s in features]
    else:
        features = dataset.columns
        features.remove('Date')

    dates = dataset['Date']
    df = dataset[features]

    if 'low' in df.columns:
        df = df.rename({'low': 'Low'}, axis=1)

    if 'high' in df.columns:
        df = df.rename({'high': 'High'}, axis=1)

    try:
        df['Mean'] = (df['Low'] + df['High']) / 2
    except:
        if logger is not None:
            logger.error('your dataset_loader should have High and Low columns')

    df = df.drop('Date', axis=1)
    df = df.dropna()
    arr = np.array(df)
    dataset = create_dataset(arr, list(dates), look_back=cfg.dataset_loader.window_size, features=features)
    return dataset


def create_dataset(dataset, dates, look_back, features):
    data_x = []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), :]
        a = a.reshape(-1)
        d = datetime.strptime(
            str(dates[i]).split('+')[0].split('.')[0], '%Y-%m-%d %H:%M:%S')
        b = [d]
        b = b + a.tolist()
        b.append(dataset[(i + look_back), :][-1])
        data_x.append(b)

    data_x = np.array(data_x)
    y = data_x[:, 1:].astype(np.float)
    cols = ['Date']
    counter = 0
    counter_date = 0
    for i in range(data_x.shape[1] - 2):
        name = features[counter]
        cols.append(f'{name}_day{counter_date}')
        counter += 1
        if counter >= len(features):
            counter = 0
            counter_date += 1

    cols.append('prediction')

    data_frame = pd.DataFrame(data_x, columns=cols)
    return data_frame

