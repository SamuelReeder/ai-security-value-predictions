import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error  
from typing import List, Tuple, Dict, Union
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)



class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=1, dropout=0.5, num_layers=2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, dropout=dropout)
        self.linear = nn.Linear(hidden_size, output_size)
        self.num_layers = num_layers

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq.view(len(input_seq), -1, self.input_size))
        last_time_step_output = lstm_out[-1].view(-1, self.hidden_size)
        predictions = self.linear(last_time_step_output)
        return predictions
    

def train_single(data: Dict[str, Union[np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]], 
          model: LSTM, 
          optimizer: torch.optim.Optimizer, 
          loss_function: nn.Module, 
          scalar: MinMaxScaler,
          batch_size: int = 1,
          epochs: int = 15) -> None:
    model.train()
    
    loss_values = []
    
    np.random.shuffle(data['train'])


    for i in range(epochs):
        # for i, (seq, labels) in enumerate(train_loader):
        for seq, labels in data['train']:
            
            optimizer.zero_grad()
            seq = torch.tensor(seq, dtype=torch.float32).to(device)  # Convert to torch.float tensor
            labels = torch.tensor(labels, dtype=torch.float32).to(device)  # Convert to torch.float tensor
            
            y_pred = model(seq).squeeze()  # Squeeze to match target's shape
            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()
            
            loss_values.append(single_loss.item())

        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

    model.eval()
    last_known_sequence = data['train'][-1][0]
    
    predictions = []

    for i in range(len(data['test'])):
        with torch.no_grad():
            seq = torch.tensor(last_known_sequence, dtype=torch.float32).to(device)  
            next_pred = model(seq).squeeze()
            predictions.append(next_pred.item())  # Extract the scalar value
            last_known_sequence = np.append(last_known_sequence[1:], next_pred.item())  # Update the sequence
            # print(last_known_sequence)
    
    # Convert predictions to the correct scale
    norm_predictions = scalar.inverse_transform(np.array(predictions).reshape(-1, 1)).ravel()
    test = scalar.inverse_transform(data['test'].reshape(-1, 1)).ravel()

    mse = mean_squared_error(test, norm_predictions)
    print(f'Test MSE: {mse:.8f}')
    
    mse = mean_squared_error(data['test'], predictions)
    print(f'Test MSE: {mse:.8f}')
    
    plot_test(test, norm_predictions)
    plot_test(data['test'], predictions)
    
    
    
def plot_loss(loss_values: List[float]) -> None:
    plt.figure(figsize=(10,5))
    plt.plot(loss_values, label='Training loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show() 
    

def plot_test(test, norm_predictions):
    plt.figure(figsize=(10, 5))
    plt.plot(test, label='Actual Data')
    plt.plot(norm_predictions, label='Predicted Data')
    plt.title('Comparison of Actual and Predicted Values')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()
    

    