import torch
import torch.nn as nn
import numpy as np
import math
from early_stopping import EarlyStopping
from dynamic_dropout import DynamicDropout
import matplotlib.pyplot as plt

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, mem_dim):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.mem_dim = mem_dim
        self.dropout = DynamicDropout()

        self.wc = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.rc = nn.Parameter(torch.randn(hidden_size, input_size))
        self.bc = nn.Parameter(torch.randn(1))
        self.wi = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.ri = nn.Parameter(torch.randn(hidden_size, input_size))
        self.bi = nn.Parameter(torch.randn(1))
        self.wf = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.rf = nn.Parameter(torch.randn(hidden_size, input_size))
        self.bf = nn.Parameter(torch.randn(1))
        self.wo = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.ro = nn.Parameter(torch.randn(hidden_size, input_size))
        self.bo = nn.Parameter(torch.randn(hidden_size, 1))
        self.fc = nn.Linear(hidden_size, 1)
        self.reset_parameters()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.05, weight_decay=1e-6)
        # self.criterion = nn.MSELoss()
        # self.criterion = nn.L1Loss()
        self.criterion = nn.SmoothL1Loss()

    def reset_parameters(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, xt, states):
        (ht_1, Ct_1) = states
        xt = xt.view(-1, self.mem_dim, 1)
        combined = torch.cat((ht_1, xt), 1)

        ft = torch.sigmoid(torch.mm(combined, self.wf) + self.bf)
        it = torch.sigmoid(torch.mm(combined, self.wi) + self.bi)
        Ct_tilde = torch.tanh(torch.mm(combined, self.wc) + self.bc)

        Ct = ft * Ct_1 + it * Ct_tilde
        ot = torch.sigmoid(torch.mm(combined, self.wo) + self.bo)
        ht = ot * torch.tanh(Ct)

        return ht, (ht, Ct)

    def init_hidden(self, seq_len):
        return (torch.zeros(seq_len, self.mem_dim, 1),
                torch.zeros(seq_len, self.mem_dim, 1))

    def train_model(self, data, val_data, epochs, seq_len):
        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=50, verbose=False)
        for epoch in range(epochs):
            states = self.init_hidden(seq_len)
            self.optimizer.zero_grad()
            loss = 0
            for t in range(0, data.size(1) - seq_len, seq_len):
                x = data[:, t:t+seq_len]
                y_true = data[:, t + 1::t+seq_len+1, 0]
                y_pred, states = self(x, states)
                loss += self.criterion(y_pred, y_true)

            val_loss = self.validate(val_data, seq_len)
            print(f'Epoch {epoch} Validation Loss {val_loss.item()}')
            early_stopping(val_loss, self)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            loss.backward()
            self.optimizer.step()

            if epoch % 10 == 0:
                print(f'Epoch {epoch} Loss {loss.item()}')

        self.load_state_dict(torch.load('checkpoint.pt'))  # <-- loads model

    def predict(self, data, seq_len):
        self.eval()
        states = self.init_hidden(seq_len)
        predictions = []
        for t in range(0, data.size(1) - seq_len, seq_len):
            x = data[:, t:t+seq_len]
            y_pred, states = self(x, states)
            predictions.append(y_pred.detach().numpy().ravel()[0])
        self.train()
        return predictions

    def validate(self, data, seq_len):
        self.eval()
        states = self.init_hidden(seq_len)
        loss = 0
        for t in range(0, data.size(1) - seq_len, seq_len):
            x = data[:, t:t+seq_len]
            y_true = data[:, t + 1::t+seq_len+1, 0]
            y_pred, states = self(x, states)
            loss += self.criterion(y_pred, y_true)
        self.train()
        return loss