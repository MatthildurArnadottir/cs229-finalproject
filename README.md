### About
This project uses Liquid Time-Constant neural networks to train models that predict interest rates and industrial energy consumption. The code also includes the possibility to train LSTM, CT-RNN and linear regression models on the same datasets for comparison. 

### Data preparation
The original interest rate dataset, along with each feature in a separate csv file, is in `data/interest_rate_data`. The original energy consumption dataset  has a few separate csv files that are combined in training, and stored in `data/energy_data`. The function `prepare_data.py` in `experiments_with_ltcs` was used to combine the features of the interest rate dataset. 

The python module `feature_extraction.py` performs PCA on both datasets and saves the transformed data in separate files.

- `interest_rate.py`:  Trains an LTC, LSTM, or CT-RNN model on either the original interest rate dataset, or the PCA transformed dataset. To specify the data, the desired datapath is set before training, and the variable `pca` is set to either `True` or `False`.
- `energy.py`: Trains an LTC, LSTM, or CT-RNN model on either the original energy consumption dataset, or the PCA transformed dataset. To specify the data, the function that loads the desired dataset is called within the `EnergyData` class, and the corresponding input size is set in the `EnergyModel` class.
- `baseline_model.py`: Trains a linear regression model on any one of the available datasets. To specify the data to use, the path to the desired data is set before calling the corresponding training function.
