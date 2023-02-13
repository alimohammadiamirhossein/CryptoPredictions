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
