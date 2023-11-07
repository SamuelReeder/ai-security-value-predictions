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
    
    
    # def forward(self, input_seq):
    #     lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
    #     predictions = self.linear(lstm_out.view(len(input_seq), -1))  
    #     return predictions[-1]
    


def train_single(data: Dict[str, Union[np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]], 
          model: LSTM, 
          optimizer: torch.optim.Optimizer, 
          loss_function: nn.Module, 
          scalar: MinMaxScaler,
          batch_size: int = 1,
          epochs: int = 1) -> None:
    model.train()
    
    loss_values = []
    # train_loader = DataLoader(data['train'], shuffle=False, batch_size=batch_size)


    for i in range(epochs):
        
        # hidden_state = model.init_hidden(batch_size)
        for seq, labels in data['train']:
            
            

        # for seq, labels in data['train']:
            optimizer.zero_grad()
            
            # batch_size_current = seq.size(0)  # Get the current batch size (might be smaller for the last batch)
            # model.init_hidden(batch_size_current)  # Reinitialize hidden state for the current batch size

            # if seq.size(0) != batch_size:
            #     continue
                
            # print(seq.shape, labels.shape)
            seq, labels = torch.tensor(seq).to(device), torch.tensor(labels).to(device)
            # y_pred = model(seq)
            

            y_pred = model(seq)
            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()
            
            loss_values.append(single_loss.item())
            
            # hidden_state = tuple(h.detach() for h in hidden_state)

            
            # model.hidden_cell = tuple([each.data for each in model.hidden_cell])



        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')
        
    # plt.figure(figsize=(10,5))
    # plt.plot(loss_values, label='Training loss')
    # plt.title('Loss over epochs')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.show() 
    
    batch_size = 1

    model.eval()
    last_known_sequence = data['train'][-1][0]
    
    predictions = []
    # hidden_state = model.init_hidden(batch_size)

    for i in range(len(data['test'])):
        with torch.no_grad():
            seq = torch.tensor(last_known_sequence).float().to(device).unsqueeze(0)  # Add batch dimension
            next_pred = model(seq)  # This should now return a single value or a tensor with a single value
            next_pred = next_pred.squeeze(-1)  # Flatten the tensor to ensure it's one-dimensional
            predictions.append(next_pred.item())  # Extract the scalar value
            last_known_sequence = np.append(last_known_sequence[1:], next_preds.item())  # Update the sequence
    
    # Convert predictions to the correct scale
    norm_predictions = scalar.inverse_transform(np.array(predictions).reshape(-1, 1)).ravel()
    test = scalar.inverse_transform(data['test'].reshape(-1, 1)).ravel()

    # Calculate mean squared error
    mse = mean_squared_error(test, norm_predictions)
    print(f'Test MSE: {mse:.8f}')
    
    # predictions = []
    # for _ in range(len(data['test'])):
    #     with torch.no_grad():
    #         model.init_hidden(batch_size)
    #         seq = torch.tensor(last_known_sequence).float().to(device)
    #         next_pred = model(seq).item()
    #         predictions.append(next_pred)
    #         last_known_sequence = np.append(last_known_sequence[1:], next_pred)
            

    
    # norm_predictions = scalar.inverse_transform(np.array(predictions).reshape(-1, 1)).ravel()
    # test = scalar.inverse_transform(data['test'].reshape(-1, 1)).ravel()
    
    # mse = mean_squared_error(test, norm_predictions)
    # print(f'Test MSE: {mse:.8f}')
    
    mse = mean_squared_error(data['test'], predictions)
    print(f'Test MSE: {mse:.8f}')
    
    print(predictions)
    print(norm_predictions)
    plt.figure(figsize=(10, 5))
    plt.plot(test, label='Actual Data')
    plt.plot(norm_predictions, label='Predicted Data', alpha=0.7)
    plt.title('Comparison of Actual and Predicted Values')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()
    
    
    
    
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
        
        last_known_sequence = stock_data[stock]['train'][-1][0]
        
        predictions = []
        for i in range(len(test_data)):
            with torch.no_grad():
                model.hidden_cell = (torch.zeros(1, 1, model.hidden_size).to(device),
                                     torch.zeros(1, 1, model.hidden_size).to(device))
                seq = torch.tensor(last_known_sequence).float().to(device)
                next_pred = model(seq).item()
                predictions.append(next_pred)
                last_known_sequence = np.append(last_known_sequence[1:], next_pred)
        
        predictions = scalers[stock].inverse_transform(np.array(predictions).reshape(-1, 1)).ravel()
        
        mse = mean_squared_error(test_data, predictions)
        total_mse += mse
        print(f'{stock} Test MSE: {mse:.8f}')
    
    print(f'Average Test MSE: {total_mse / len(stock_data):.8f}')
