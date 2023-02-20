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
