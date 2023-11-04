import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error  
from model import LSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Define the device

def predict_prices(normalized_prices: np.ndarray, 
                   model: LSTM, 
                   scaler: MinMaxScaler, 
                   days: int) -> np.ndarray:
    model.eval()
    last_known_sequence = normalized_prices[-128:]  # Assuming window size of 128
    predictions = []

    for i in range(days):
        with torch.no_grad():
            # Reset the hidden state
                # model.hidden_cell1 = (torch.zeros(1, 1, model.hidden_layer_size).to(device),
                #                       torch.zeros(1, 1, model.hidden_layer_size).to(device))
                # model.hidden_cell2 = (torch.zeros(1, 1, model.hidden_layer_size).to(device),
                #                       torch.zeros(1, 1, model.hidden_layer_size).to(device))
            
            seq = torch.tensor(last_known_sequence).float().to(device)
            next_pred = model(seq).item()
            predictions.append(next_pred)
            last_known_sequence = np.append(last_known_sequence[1:], next_pred)
            print(last_known_sequence[-1])
    
    # Revert the normalization
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).ravel()
    return predictions


# Load the model
model_path = 'model1.pth'

model = LSTM().to(device)
model.load_state_dict(torch.load(model_path))
# model.eval()

data_str = 'data/stock_min.csv'
df = pd.read_csv(data_str)
agl_rows = df[df['Symbol'] == 'SQ']

prices = agl_rows['Close'].values.astype(np.float32)
print(prices)

scaler = MinMaxScaler(feature_range=(-1, 1))
normalized_prices = scaler.fit_transform(prices.reshape(-1, 1)).ravel()
predictions = predict_prices(normalized_prices, model, scaler, days=100)

# input_tensor = torch.tensor(normalized_prices, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

# # Make predictions
# with torch.no_grad():
#     predictions = model(input_tensor)

print(predictions)
