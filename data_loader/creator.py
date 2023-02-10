import numpy as np
import pandas as pd
from datetime import datetime


def create_dataset(dataset, dates, look_back, features):
    data_x = []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), :]
        a = a.reshape(-1)
        d = datetime.strptime(str(dates[i]).split('+')[0], '%Y-%m-%d %H:%M:%S')
        b = [d]
        b = b + a.tolist()
        b.append(dataset[(i + look_back), :][-1])
        data_x.append(b)

    print(13, data_x[0])
    data_x = np.array(data_x)
    y = data_x[:, 1:].astype(np.float)
    print(12,data_x[0])
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

