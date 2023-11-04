# Stock Price Prediction with LSTM

This program leverages the Long Short-Term Memory (LSTM) architecture, a type of recurrent neural network (RNN), to predict stock prices based on historical data. LSTMs are particularly suitable for time series forecasting due to their capability to remember and utilize patterns from sequential data. This solution provides a way to capture intricate patterns and relationships in historical stock prices to predict future prices.

## Features

- **LSTM Model**: Uses LSTM layers to handle time series data, capturing patterns over different time horizons.
- **Scalability**: Designed to predict stock prices for various stock symbols.
- **Data Normalization**: Employs MinMaxScaler to normalize stock prices between -1 and 1, ensuring efficient training and prediction.

## Prerequisites

- Python 3.8 or newer
- PyTorch
- pandas
- NumPy
- scikit-learn

To install the necessary libraries, use:

\```
pip install torch pandas numpy scikit-learn
\```

## Usage

### Preparing Data

The program expects stock data in a CSV format with the following structure:

\```
Symbol,Date,Open,High,Low,Close,Volume
AADI,11-14-2018,28.95,31.2,28.2,29.1,2253
...
\```

Set the `data_str` variable in your script to point to your desired CSV file.

### Training

1. Run your training script:

\```
python your_training_script.py
\```

This will train the LSTM model and save its parameters to `model1.pth`.

### Predicting

1. Ensure you have the model saved as `model1.pth`.
2. Run the prediction script:

\```
python predict.py
\```

The script will use the trained LSTM model to forecast future stock prices based on the provided historical data.

## Notes

- The LSTM model works best when fed with sequential data with inherent patterns. It might not capture sudden, unpredictable changes in stock prices caused by external factors.
- Adjust the sequence length based on your data. If your dataset's length is shorter than the sequence length, modify it accordingly.
- When using the predictions, remember to inverse transform the outputs if you wish to view prices in their original scale.
