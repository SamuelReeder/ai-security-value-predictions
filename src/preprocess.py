import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error  
from typing import List, Tuple, Dict, Union, Any

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Data:
    
    def __init__(self, path: str, multiple: bool, sequence_len: int = 32, train_frac: float = 0.8) -> None:
        self.path = path
        self.multiple = multiple
        self.sequence_len = sequence_len
        self.train_frac = train_frac
        
    def preprocess_close(self) -> (pd.DataFrame, MinMaxScaler):
        df = pd.read_csv(self.path)
        scalar = MinMaxScaler(feature_range=(-1, 1))
        if not self.multiple:
            df['Close'] = np.ravel(scalar.fit_transform(df['Close'].values.reshape(-1,1))).flatten().astype(np.float32)
        else: 
            df['Close'] = df.groupby('Symbol')['Close'].apply(lambda x: scalar.fit_transform(x.values.reshape(-1, 1)).flatten().astype(np.float32))
        return (df, scalar)

            
        # min_length = min(df.groupby('Symbol').size()) if multiple else len(df)
        # self.sequence_length = 128
        # if self.sequence_length >= min_length:
        #     self.sequence_length = min_length - 1
        #     print(f"Adjusted sequence length to {sequence_length} due to shorter data series.")

    def create_inout_sequences(self, data: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        inout_seq = []
        for i in range(self.sequence_len, len(data)):
            train_seq = data[i-self.sequence_len:i]
            train_label = data[i:i+1]
            inout_seq.append((train_seq, train_label))
        return inout_seq

    # def create_datasets() -> None:
    #     for symbol, group in df.groupby('Symbol'):

    #         train_size = int(len(group) * train_frac)
    #         train_group = group.iloc[:train_size]
    #         test_group = group.iloc[train_size:]

    #         if train_group.empty or test_group.empty or len(train_group) < 2:
    #             continue

    #         scalar = MinMaxScaler(feature_range=(-1, 1))
    #         normalized_train_data = np.ravel(scalar.fit_transform(train_group['Close'].values.reshape(-1,1))).flatten().astype(np.float32)
    #         stock_data[symbol] = {
    #             'train': create_inout_sequences(normalized_train_data),
    #             'test': test_group['Close'].values  # Keep test data in original scale for evaluation
    #         }
    #         scalers[symbol] = scaler
            
    def create_dataset(self, data: pd.DataFrame, scalar: MinMaxScaler) -> Dict[str, Any]:
        train_size = int(len(data) * self.train_frac)
        train_group = data.iloc[:train_size]
        test_group = data.iloc[train_size:]

        if train_group.empty or test_group.empty or len(train_group) < 2:
            raise ValueError("Invalid data")

        stock_data = {
            'train': self.create_inout_sequences(train_group['Close'].values),
            'test': test_group['Close'].values
        }
        
        return stock_data

# read csv
# normalize close
# 