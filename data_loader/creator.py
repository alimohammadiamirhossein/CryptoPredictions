import numpy as np
import pandas as pd
from datetime import datetime
from .indicators import calculate_indicators, add_indicators_to_dataset


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
    df1 = df.drop('Mean', axis=1)
    arr = np.array(df1)

    indicators = calculate_indicators(mean_=np.array(df.Mean), low_=np.array(df.Low),
                                      high_=np.array(df.High), open_=np.array(df.open),
                                      close_=np.array(df.close), volume_=np.array(df.volume))

    indicators_names = list(cfg.dataset_loader.indicators_names.split(' '))

    arr1, dates = add_indicators_to_dataset(indicators, indicators_names, dates, mean_=np.array(df.Mean))
    arr = np.concatenate((arr[100:], arr1), axis=1)
    features.remove('Date')
    features = features + indicators_names
    dataset, profit_calculator = create_dataset(arr, list(dates), look_back=cfg.dataset_loader.window_size, features=features)
    return dataset, profit_calculator


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
    # y = data_x[:, 1:].astype(np.float)
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
    last_col = []
    for i in range(len(features)):
        name = features[i]
        last_col.append(f'{name}_day{counter_date-1}')
    last_col.append('prediction')
    last_col.remove(f'High_day{counter_date-1}')
    last_col.remove(f'Low_day{counter_date - 1}')
    last_col.remove(f'mean_day{counter_date - 1}')
    profit_calculator = data_frame.copy()[['Date',f'Low_day{counter_date-1}', f'High_day{counter_date-1}',
                                       f'close_day{counter_date-1}', f'open_day{counter_date-1}',
                                       f'volume_day{counter_date-1}']]
    data_frame.drop(last_col, axis=1, inplace=True)
    data_frame = data_frame.rename({f'High_day{counter_date-1}': 'predicted_high',
                       f'Low_day{counter_date-1}': 'predicted_low',
                       f'mean_day{counter_date-1}': 'prediction'
                       }, axis=1)

    profit_calculator = profit_calculator.rename({f'High_day{counter_date - 1}': 'High',
                                    f'Low_day{counter_date - 1}': 'Low',
                                    f'open_day{counter_date - 1}': 'Open',
                                    f'close_day{counter_date - 1}': 'Close',
                                    f'volume_day{counter_date - 1}': 'Volume',
                                    }, axis=1)

    return data_frame, profit_calculator

