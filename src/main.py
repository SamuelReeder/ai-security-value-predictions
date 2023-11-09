import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error  
from typing import List, Tuple, Dict, Union
from preprocess import Data
from model import LSTM, train_single
from predict import predict_prices

def main():
    
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model_path = "../models/new_model.pth"
	
	# # Preprocess
	csv_path = "../data/BTC-USD.csv"
	data = Data(csv_path, False)
	df, scalar = data.preprocess_close()
	dataset = data.create_dataset(df, scalar)

	# Define model
	model = LSTM().to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
	loss_function = nn.MSELoss()

	# Train model
	train_single(dataset, model, optimizer, loss_function, scalar)

	# Save model
	torch.save(model.state_dict(), "../models/new_model2.pth")


	# model = LSTM().to(device)
	# model.load_state_dict(torch.load(model_path))	
 
	# predictions = predict_prices(df['Close'].values, model, scalar, days=100)
	# print(predictions)


if __name__ == "__main__":
    main()