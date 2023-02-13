# CryptoPredictions

CryptoPredictions is an open-source toolbox for price prediction/forecasting a sequence of prices of cryptocurrencies given an observed sequence.

![Observation](https://user-images.githubusercontent.com/45646480/218426854-e6cf9f39-8424-4f56-bc89-7d618e0bb384.png)

![Prediction](https://user-images.githubusercontent.com/45646480/218425368-6761a215-04f8-43c3-96eb-d2df0468455f.png)

# Overview

The main parts of the library are as follows:

```
CryptoPredictions
├── api
|   ├── preprocess.py                   -- script to run the preprocessor module
|   ├── train.py                        -- script to train the models, runs factory.trainer.py
│   ├── evaluate.py                     -- script to evaluate the models, runs factory.evaluator.py
├── models                    
│   ├── orbit.py
|   ├── prophet.py
|   ├── LSTM.py
│   ├── sarimax.py
|   ├── random_forest.py
|   ├── GRU.py
|   ├── ...
├── data_loader
|   ├── BTCDataset.py        
|   ├── ...
```
