import torch
import torch.nn as nn
import numpy as np
import math
from early_stopping import EarlyStopping
from dynamic_dropout import DynamicDropout



# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = DynamicDropout()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.05, weight_decay=1e-6)
        self.criterion = nn.SmoothL1Loss()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

    def train_model(self, data, val_data, epochs, seq_len):
        #early_stopping = EarlyStopping(patience=10, verbose=False)
        for epoch in range(epochs):
            self.train()
            loss = 0
            for i in range(0, data.size(1) - seq_len, seq_len):
                x = data[:, i:i + seq_len, :]
                y = data[:, i + seq_len:i + seq_len + 1, 0]
                y_pred = self.predict(x, seq_len)
                y_pred = torch.tensor(y_pred, requires_grad=True)
                loss += self.criterion(y_pred, y)
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            self.eval()
            '''with torch.no_grad():
                val_loss = 0
                for i in range(0, val_data.size(1) - seq_len, seq_len):
                    x = val_data[:, i:i + seq_len, :]
                    y = val_data[:, i + seq_len:i + seq_len + 1, 0]
                    y_pred = self.predict(x, seq_len)
                    val_loss += self.criterion(y_pred, y)
                val_loss /= (val_data.size(1) // seq_len)
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}, Val Loss: {val_loss.item()}")
                early_stopping(val_loss, self)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break'''

    def predict(self, x, seq_len):
        self.eval()
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        predictions = []
        for t in range(seq_len):
            out, _ = self.lstm(x[:, t].unsqueeze(1), (h0, c0))
            predictions.append(out.detach().numpy().ravel()[0])
        self.train()
        return predictions