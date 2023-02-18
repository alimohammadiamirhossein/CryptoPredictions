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
|   ├── CoinMarketDataset.py        
|   ├── Bitmex.py        
|   ├── ...
```

# Getting Started  
To get started as quickly as possible, follow the instructions in this section. This should allow you train a model from scratch, evaluate your pretrained models, and produce basic visualizations.  

### Dependencies  
Make sure you have the following dependencies installed before proceeding:  
- Python 3.7+ distribution
- pip >= 21.3.1 

### Virtualenv  
You can create and activate virtual environment like below:  
```bash  

pip install --upgrade virtualenv

virtualenv -p python3.7 <venvname>  

source <venvname>/bin/activate  

pip install --upgrade pip
```  
### Requirements  
Furthermore, you just have to install all the packages you need:  
  
```bash  
pip install -r requirements.txt  
```  
Before moving forward, you need to install Hydra and know its basic functions to run different modules and APIs.  
hydra is A framework for elegantly configuring complex applications with hierarchical structure.
For more information about Hydra, read their official page [documentation](https://hydra.cc/).

## Hydra
In order to have a better structure and understanding of our arguments, we use Hydra to  dynamically create a hierarchical configuration by composition and override it through config files and the command line.
If you have any issues and errors install hydra like below:
```bash
pip install hydra-core --upgrade
```

# Dataset

You can use more than 15 cryptocurrencies data by giving the symbol of the selected cryptocurrency to the config files. Moreover, the csv files of these cryptocurrencies could be found in ./data .

<div align="center">

|  Name	     | Symbol |  Name	      | Symbol |  Name	 | Symbol |
| :---:      |  :---: |  :---:      | :---:  | :---:   | :---:  |
| Bitcoin    | XBTUSD | Ethereum    | ETHUSD | BNB     | BNBUSD |
| Cardano    | ADAUSD | Dogecoin    | DOGEUSD| Solana  | SOLUSD |
| Polkadot   | DOTUSD | Litecoin    | LTCUSD | TRON    | TRXUSD |
|Avalanche   | AVAXUSD|Chainlink    | LINKUSD| Aptos   | APTUSD |
|Bitcoin Cash| BCHUSD |NEAR Protocol| NEARUSD| ApeCoin | APEUSD |
|Cronos      | CROUSD |Axie Infinity| AXSUSD | EOS     | EOSUSD |

</div>

# Indicators

In order to have a richer dataset, library provides you with more than 30 indicators. You could select the indicators you want to have in your dataset and the library will calculate them and add them to the dataset.

The list of of available indicators supported by the library is as follow:

<div align="center">

|  Name	                                | Symbol          |  Name	                            | Symbol          |  Name	                            | Symbol   |
| :---:                                 |  :---:          |  :---:                            | :---:           | :---:                             | :---:    |
| Simple Moving Average                 | sma             |  Weighted Moving Average          | wma             | Cumulative Moving Average         | cma      |
| Exponential Moving Average            | ema             |  Double Exponential Moving Average| dema            | Triple Exponential Moving Average | trix     |
| Moving Average Convergence Divergence | macd            |  Stochastic                       | stoch           | KDJ                               | kdj      |
| William %R                            | wpr             |  Relative Strengh Index           | rsi             | Stochastic Relative Strengh Index | srsi     |
|  Chande Momentum Oscillator           | cmo             |  Bollinger Bands                  | bollinger       | Keltner Channel                   | kc       |
| Donchian Channel                      | dc              |  Heiken Ashi                      | heiken          | Ichimoku                          | ichi     |
| Volume Profile                        | vp              |  True Range                       | tr              | Average True Range                | atr      |
| Average Directionnal Index            | adx             |  On Balance Volume                | obv             | Momentum                          | mmt      |
| Rate Of Change                        | roc             |  Aroon                            | aroon           | Chaikin Money Flow                | cmf      |
| Volatility Index                      | vix             | Chopiness Index                   | chop            | Center Of Gravity                 | cog      |


</div>



