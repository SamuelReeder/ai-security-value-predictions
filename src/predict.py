import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error  
from model import LSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

def predict_prices(normalized_prices: np.ndarray, 
                   model: LSTM, 
                   scaler: MinMaxScaler, 
                   days: int) -> np.ndarray:
    model.eval()
    last_known_sequence = normalized_prices[-128:] 
    predictions = []

    for i in range(days):
        with torch.no_grad():
        
            seq = torch.tensor(last_known_sequence, dtype=torch.float32).to(device)  
            next_pred = model(seq).squeeze()
            predictions.append(next_pred.item())
            last_known_sequence = np.append(last_known_sequence[1:], next_pred)
    
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).ravel()
    return predictions
