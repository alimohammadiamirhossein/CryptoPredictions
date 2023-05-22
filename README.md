# CryptoPredictions

CryptoPredictions is an open-source toolbox for price prediction/forecasting a sequence of prices of cryptocurrencies given an observed sequence.

![Observation](https://user-images.githubusercontent.com/45646480/218426854-e6cf9f39-8424-4f56-bc89-7d618e0bb384.png)

![Prediction](https://user-images.githubusercontent.com/45646480/218425368-6761a215-04f8-43c3-96eb-d2df0468455f.png)

# Why CryptoPredictions?

This library offers you a wide range of services that you may not find anywhere else.
The exclusive benefits of CryptoPredictions are:

* At the outset of our work, we faced a serious challenge of dataset scarcity. Many papers and repos fetched the data through different websites, such as Yahoo Finance. However, we have overcome this obstacle by using platforms such as Bitmex, which offer a common structure for different currencies.

* Before the advent of our library, users had to run different codes for different models, making it difficult to compare them fairly. Fortunately, CryptoPredictions has made it possible to conduct a unified and equitable evaluation of different models. With Hydra, users can easily structure and understand arguments, making it easier to run codes on different settings and check results.

* By using Hydra, users have a better understanding of the arguments. Furthermore, it is far easier to run a code on different settings and  check the result.

* While some models may perform exceptionally well in terms of accuracy, they often require a well-defined strategy for successful trading. Our backtester can help you determine the effectiveness of your model in real-world scenarios.

* We understand that evaluating models can be challenging, which is why we offer a variety of metrics to help you measure progress towards accomplishing your tasks. By analyzing multiple metrics, you can identify areas for improvement and correct what is not working. To learn more about the pros and cons of each metric, please refer to the metrics section [here](https://github.com/alimohammadiamirhossein/CryptoPredictions/blob/main/Report.pdf).

* At CryptoPredictions, we do not fetch indicators from different websites, because it leads to problems such as null rows and the lack of information on indicators for all cryptocurrencies. Instead, CryptoPredictions calculates them in a way that doesn't carry the mentioned problems and could be generalized to other datasets.

* We hope that it will inspire you to develop even better projects, and we look forward to your feedback. If you find CryptoPredictions useful and valuable, we would greatly appreciate it if you could take a moment to give it a star on Github. Your support would mean a lot to us and help us to continue improving the library for the community :)
 

# Overview

The main parts of the library are as follows:

```
CryptoPredictions
├── train.py                            -- script to train the models, runs factory.trainer.py
├── backtester.py                       -- script to calculate the profit by selecting a strategy to buy and sell based on the prediction
├── models                    
│   ├── orbit.py
|   ├── prophet.py
|   ├── LSTM.py
│   ├── sarimax.py
|   ├── random_forest.py
|   ├── xgboost.py
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
for more information about Hydra and config files please visit [here](https://github.com/alimohammadiamirhossein/CryptoPredictions/blob/main/Documents/ARGS_README.md#Hydra).

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

# Metrics

The essential step in any machine learning model is to evaluate the accuracy of the model. The list of of available metrics supported by the library is as follow:


* accuracy_score: Number of correct predictions/Total number of predictions
* precision_score:  the proportion of positively predicted labels that are actually correct
* recall_score: the model's ability to correctly predict the positives out of actual positives
* f1_score: 2.Precision.Recall/(Precision+Recall)
* MAE: Mean Absolute Error
* MAPE: Mean Absolute Percentage Error
* MASE: Mean Absolute Scaled Error
* RMSE: Root Mean Square Error
* SMAPE: Symmetric Mean Absolute Percentage Error
* Stochastic: the possibility that the outcome is not that expected, given that both the model and parameters are correct 


# Results 

As a conclusion, we tested models using different cryptocurrencies on the various available metrics and the result is as follows:

![image](https://user-images.githubusercontent.com/45646480/233838187-1d80c7d9-46d1-4072-a67e-1ba69c4f4268.png)

You can see other graphs [here](https://github.com/alimohammadiamirhossein/CryptoPredictions/blob/main/Documents/results.md).


# Report

To gain a better understanding of the models, metrics, and validation method used in this library, we prepared a report and you can read it from [here](https://github.com/alimohammadiamirhossein/CryptoPredictions/blob/main/CryptoPredictions.pdf).

# Our Team

* Amirhossein Alimohammadi
* Ahmad Nosratbakhsh

<img src="https://user-images.githubusercontent.com/45646480/233836824-13a268e7-9464-46df-95bd-1ee8631519a8.jpg" data-canonical-src="https://user-images.githubusercontent.com/45646480/233836824-13a268e7-9464-46df-95bd-1ee8631519a8.jpg" width="540" />




