import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error  
from typing import List, Tuple, Dict, Union


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

train_frac = 0.8
stock_data = {}
scalers = {}

for symbol, group in df.groupby('Symbol'):

    train_size = int(len(group) * train_frac)
    train_group = group.iloc[:train_size]
    test_group = group.iloc[train_size:]

    if train_group.empty or test_group.empty or len(train_group) < 2:
        continue

    scaler = MinMaxScaler(feature_range=(-1, 1))
    normalized_train_data = np.ravel(scaler.fit_transform(train_group['Close'].values.reshape(-1,1))).astype(np.float32)
    stock_data[symbol] = {
        'train': create_inout_sequences(normalized_train_data),
        'test': test_group['Close'].values  # Keep test data in original scale for evaluation
    }
    scalers[symbol] = scaler
# could either train model with all data and put stop chars between symbols or do batch wise training with one symbol at a time

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=1, dropout=0.5, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm1 = nn.LSTM(input_size, hidden_size)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)
        self.hidden_cell1 = (torch.zeros(1, 1, self.hidden_size).to(device),
                             torch.zeros(1, 1, self.hidden_size).to(device))
        self.hidden_cell2 = (torch.zeros(1, 1, self.hidden_size).to(device),
                             torch.zeros(1, 1, self.hidden_size).to(device))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell1 = self.lstm1(input_seq.view(len(input_seq), 1, -1), self.hidden_cell1)
        # lstm_out = self.dropout(lstm_out)
        lstm_out, self.hidden_cell2 = self.lstm2(lstm_out, self.hidden_cell2)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

def train(sequences: Dict[str, Dict[str, Union[np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]]], 
          model: LSTM, 
          optimizer: torch.optim.Optimizer, 
          loss_function: nn.Module, 
          epochs: int = 1) -> None:
    model.train()
    for i in range(epochs):
        for stock in sequences:
            for seq, labels in sequences[stock]['train']:
                model.hidden_cell1 = tuple(hc.detach() for hc in model.hidden_cell1)
                model.hidden_cell2 = tuple(hc.detach() for hc in model.hidden_cell2)
                optimizer.zero_grad()
                seq, labels = torch.tensor(seq).to(device), torch.tensor(labels).to(device)
                y_pred = model(seq)
                single_loss = loss_function(y_pred, labels)
                single_loss.backward()
                optimizer.step()

            print(f'{stock} epoch: {i:3} loss: {single_loss.item():10.8f}')
        print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

    model.eval()
    total_mse = 0
    for stock in stock_data:
        test_data = stock_data[stock]['test']
        
        # You will need the last N values from the training data to make the first prediction for the test data
        # where N is the size of the window you are using for predictions.
        last_known_sequence = stock_data[stock]['train'][-1][0]
        
        # Generate predictions for the test set
        predictions = []
        for i in range(len(test_data)):
            with torch.no_grad():
                model.hidden_cell = (torch.zeros(1, 1, model.hidden_size).to(device),
                                     torch.zeros(1, 1, model.hidden_size).to(device))
                seq = torch.tensor(last_known_sequence).float().to(device)
                next_pred = model(seq).item()
                predictions.append(next_pred)
                last_known_sequence = np.append(last_known_sequence[1:], next_pred)
        
        # Convert predictions back to original scale using the saved scaler
        predictions = scalers[stock].inverse_transform(np.array(predictions).reshape(-1, 1)).ravel()
        
        # Compute the Mean Squared Error between the predictions and the actual values
        mse = mean_squared_error(test_data, predictions)
        total_mse += mse
        print(f'{stock} Test MSE: {mse:.8f}')
    
    print(f'Average Test MSE: {total_mse / len(stock_data):.8f}')

model = LSTM().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_function = nn.MSELoss()
train(stock_data, model, optimizer, loss_function) jovian.eatfordinner()


# Save model state dict
torch.save(model.state_dict(), "model1.pth")

