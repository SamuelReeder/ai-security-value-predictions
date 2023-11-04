import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error  
from typing import List, Tuple, Dict, Union


data_str = 'data/stock_min.csv'
df = pd.read_csv(data_str)
df['Close'] = df.groupby('Symbol')['Close'].transform(lambda x: np.ravel(MinMaxScaler(feature_range=(-1, 1)).fit_transform(x.values.reshape(-1,1))))
     
min_length = min(df.groupby('Symbol').size())
sequence_length = 128
if sequence_length >= min_length:
    sequence_length = min_length - 1
    print(f"Adjusted sequence length to {sequence_length} due to shorter data series.")

def create_inout_sequences(data: np.ndarray, seq_length: int = 128) -> List[Tuple[np.ndarray, np.ndarray]]:
    inout_seq = []
    for i in range(seq_length, len(data)):
        train_seq = data[i-seq_length:i]
        train_label = data[i:i+1]
        inout_seq.append((train_seq, train_label))
    return inout_seq


scaler = MinMaxScaler(feature_range=(-1, 1))
    normalized_train_data = np.ravel(scaler.fit_transform(train_group['Close'].values.reshape(-1,1))).astype(np.float32)