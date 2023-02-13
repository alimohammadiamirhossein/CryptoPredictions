# CryptoPredictions

CryptoPredictions is an open-source toolbox for price prediction/forecasting a sequence of prices of cryptocurrencies given an observed sequence.

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
