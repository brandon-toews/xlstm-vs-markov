import torch
import torch.nn as nn
import numpy as np
import math
from early_stopping import EarlyStopping
from dynamic_dropout import DynamicDropout
import matplotlib.pyplot as plt


class mLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, mem_dim, layers=1, output_size=1):
        super(mLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.mem_dim = mem_dim
        self.layers = layers
        self.dropout = DynamicDropout()
        self.Wq = nn.Parameter(torch.randn(hidden_size, input_size))
        self.bq = nn.Parameter(torch.randn(hidden_size, 1))
        self.Wk = nn.Parameter(torch.randn(mem_dim, input_size))
        self.bk = nn.Parameter(torch.randn(mem_dim, 1))
        self.Wv = nn.Parameter(torch.randn(mem_dim, input_size))
        self.bv = nn.Parameter(torch.randn(mem_dim, 1))
        self.wi = nn.Parameter(torch.randn(1, input_size))
        self.bi = nn.Parameter(torch.randn(1))
        self.wf = nn.Parameter(torch.randn(1, input_size))
        self.bf = nn.Parameter(torch.randn(1))
        self.Wo = nn.Parameter(torch.randn(hidden_size, input_size))
        self.bo = nn.Parameter(torch.randn(hidden_size, 1))
        self.fc = nn.Linear(hidden_size, output_size)
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

    def forward(self, x, states):
        final_output = None
        for i in range(self.layers):
            final_output, states = self.layer_forward(x, states)
        return final_output, states

    def layer_forward(self, x, states):
        (C_prev, n_prev) = states
        x = x.view(self.input_size, -1)
        qt = torch.matmul(self.Wq, x) + self.bq
        kt = (1 / math.sqrt(self.mem_dim)) * (torch.matmul(self.Wk, x) + self.bk)
        vt = torch.matmul(self.Wv, x) + self.bv

        it = torch.exp(torch.matmul(self.wi, x) + self.bi)
        ft = torch.sigmoid(torch.matmul(self.wf, x) + self.bf)

        vt = vt.squeeze()
        kt = kt.squeeze()

        C = ft * C_prev + it * torch.ger(vt, kt)
        n = ft * n_prev + it * kt.unsqueeze(1)

        max_nqt = torch.max(torch.abs(torch.matmul(n.T, qt)), torch.tensor(1.0))
        h_tilde = torch.matmul(C, qt) / max_nqt
        ot = torch.sigmoid(torch.matmul(self.Wo, x) + self.bo)

        ht = ot * h_tilde
        ht = ht.view(-1, self.hidden_size)  # reshape ht
        ht = self.dropout(ht)
        out = self.fc(ht)

        return out, (C, n)

    def init_hidden(self):
        return (torch.zeros(self.mem_dim, self.mem_dim),
                torch.zeros(self.mem_dim, 1))

    def train_model(self, data, val_data, epochs, seq_len):
        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=150, verbose=False)
        for epoch in range(epochs):
            states = self.init_hidden()
            self.optimizer.zero_grad()
            loss = 0
            for t in range(seq_len - 1):
                x = data[:, t]
                y_true = data[:, t + 1, 0]
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
        states = self.init_hidden()
        predictions = []
        for t in range(data.shape[1] - 1):
            x = data[:, t]
            y_pred, states = self(x, states)
            predictions.append(y_pred.detach().numpy().ravel()[0])
        self.train()
        return predictions

    def validate(self, data, seq_len):
        self.eval()
        states = self.init_hidden()
        loss = 0
        for t in range(data.shape[1] - 1):
            x = data[:, t]
            y_true = data[:, t + 1, 0]
            y_pred, states = self(x, states)
            loss += self.criterion(y_pred, y_true)
        self.train()
        return loss
