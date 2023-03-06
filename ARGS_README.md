# Arguments
This is a description to provide details about arguments of CryptoPredictions API.
Cryptopredictions is an open-source toolbox for pose prediction in Python. It is implemented by Amirhossein Alimohammadi and Ahmad Nosratbakhsh.  

# Hydra
```
CryptoPredictions
├── configs
│   ├── hydra                     
|      ├── data
|         └── main.yaml                 -- main config file for data module (Essentially responsible for creating dataloader)             
|      ├── model
|         ├── pv_lstml.py                     
│         ├── neareset_neighbor.py            
|         ├── zero_vel.py  
|         ├── ...              
|      ├── optimizer
|         ├── adam.yaml                 -- config file for adam optimizer
|         ├── sgd.yaml                  -- config file for stochastic gradient descent optimizer
|         ├── ...   
|      ├── scheduler
|         ├── reduce_lr_on_plateau.yaml -- config file for reducing learning_rate on plateau technique arguments
|         ├── step_lr.yaml              -- config file for step of scheduler arguments                               
|         ├── ...   
|      ├── visualize.yaml               -- config file for visualizer API arguments
|      ├── evaluate.yaml                -- config file for evaluate API arguments 
|      ├── preprocess.yaml              -- config file for preprocess API arguments
|      ├── train.yaml                   -- config file for train API arguments
|      ├── generate_output.yaml         -- config file for generate_output API arguments       
|      └── metrics.yaml                 -- config file for metrics
|                    
└─── logging.conf                 -- logging configurations
```
Now we will precisely explain each module.
#### dataset_loader
Directory Location: 'configs/hydra/dataset_loader'

`common.yaml`:
```
window_size:                Number of previous days to be included in observation
train_start_date:           Start time of the train dataset
train_end_date:             End time of the train dataset
valid_start_date:           Start time of the validation dataset
valid_end_date:             End time of the validation dataset
features:                   Regressors you want to keep from the original dataset(it depends on the dataset)
indicators_names:           Name of the indicators you want to use
```

`Bitmex.yaml`:
```
name:                       Name of the dataloader (should not be changed)
binsize:                    You could select it from ["1m", "5m", "1h", "1d"] (based on our best knowledge 1h and 1d are the best practice)
batch_size:                 It doesn not really matter but by increasing it you can improve the fetching time  
symbol:                     For filling the symbol please visit https://github.com/alimohammadiamirhossein/CryptoPredictions/blob/main/README.md#Dataset.
```

#### model
Directory Location: 'configs/hydra/model'

`arima.yaml`:
```
type:                       Name of the model (should not be changed)
order:                      The (p,d,q) order of the model for the number of AR parameters, differences, and MA parameters to use.
is_regression:              If the model is classifier set it to be True otherwise set it to be False.
```

`gru.yaml`:
```
type:                       Name of the model (should not be changed)
hidden_dim:                 Hidden dimension of the architecture of the model
epochs:                     Number of epochs to be trained. (if you use bigger dataset e.g. hourly, it is better to set epochs and hidden_dim bigger)
is_regression:              If the model is classifier set it to be True otherwise set it to be False.
```

`lstm.yaml`:
```
type:                       Name of the model (should not be changed)
hidden_dim:                 Hidden dimension of the architecture of the model
epochs:                     Number of epochs to be trained. (if you use bigger dataset e.g. hourly, it is better to set epochs and hidden_dim bigger)
is_regression:              If the model is classifier set it to be True otherwise set it to be False.
```

`orbit.yaml`:
```
type:                       Name of the model (should not be changed)
response_col:               Name of the column you want to predict(it depends on the dataset but its default is prediction)
date_col:                   Name of the date column(it depends on the dataset but its default is Date)
estimator:                  Name of the estimator {'stan-mcmc', 'stan-map'} – default to be ‘stan-mcmc’
seasonality:                Length of seasonality
seed:                       Random seed 
global_trend_option:        Transformation function for the shape of the forecasted global trend. { 'linear', 'loglinear', 'logistic', 'flat'}
n_bootstrap_draws:          Number of samples to bootstrap in order to generate the prediction interval.
is_regression:              If the model is classifier set it to be True otherwise set it to be False.
```

`prophet.yaml`:
```
type:                       Name of the model (should not be changed)
response_col:               Name of the column you want to predict(it depends on the dataset but its default is prediction)
date_col:                   Name of the date column(it depends on the dataset but its default is Date)
is_regression:              If the model is classifier set it to be True otherwise set it to be False.
```

`random_forest.yaml`:
```
type:                       Name of the model (should not be changed)
n_estimators:               The number of trees in the forest.
random_state:               Controls both the randomness of the bootstrapping of the samples used when building trees.
is_regression:              If the model is classifier set it to be True otherwise set it to be False.
```

`sarimax.yaml`:
```
type:                       Name of the model (should not be changed)
order:                      The (p,d,q) order of the model for the number of AR parameters, differences, and MA parameters.
seasonal_order:             The (P,D,Q,s) order of the seasonal component of the model for the AR parameters, differences, MA parameters, and periodicity.
enforce_invertibility:      Whether or not to transform the MA parameters to enforce invertibility in the moving average component of the model.
enforce_stationarity:       Whether or not to transform the AR parameters to enforce stationarity in the autoregressive component of the model.
is_regression:              If the model is classifier set it to be True otherwise set it to be False.
```

`xgboost.yaml`:
```
type:                       Name of the model (should not be changed)
************************************************************
is_regression:              If the model is classifier set it to be True otherwise set it to be False.
```


### Train
Check preprocessing config file: "configs/hydra/train.yaml" for more details.

You can change trainer via commandline like below:

```
model:                      Name of the model
dataset_loader:             Name of the dataset_loader
validation_method:          You could select the method of validation from {simple, cross_validation}
load_path:                  Path to load a dataset 
save_dir:                   Path to save the model 
```

### Metrics
File Location: 'configs/hydra/metrics.yaml'

`metrics.yaml`:
```
metrics:                    You can add metrics to you evaluation.{accuracy_score, f1_score, recall_score, precision_score, rmse}
```


