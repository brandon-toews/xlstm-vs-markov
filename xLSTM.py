import torch
import torch.nn as nn
import math
from early_stopping import EarlyStopping
from dynamic_dropout import DynamicDropout


# xLSTM model with dynamic dropout and early stopping
class mLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, mem_dim, layers=1, output_size=1):
        super(mLSTM, self).__init__()
        # input_size is the number of features in the input
        self.input_size = input_size
        # hidden_size is the number of features in the hidden state h
        self.hidden_size = hidden_size
        # mem_dim is the number of features in the memory state C
        self.mem_dim = mem_dim
        # layers is the number of layers in the model
        self.layers = layers
        # dropout layer
        self.dropout = DynamicDropout()
        # query weights
        self.Wq = nn.Parameter(torch.randn(hidden_size, input_size))
        # query bias
        self.bq = nn.Parameter(torch.randn(hidden_size, 1))
        # key weights
        self.Wk = nn.Parameter(torch.randn(mem_dim, input_size))
        # key bias
        self.bk = nn.Parameter(torch.randn(mem_dim, 1))
        # value weights
        self.Wv = nn.Parameter(torch.randn(mem_dim, input_size))
        # value bias
        self.bv = nn.Parameter(torch.randn(mem_dim, 1))
        # input gate weights
        self.wi = nn.Parameter(torch.randn(1, input_size))
        # input gate bias
        self.bi = nn.Parameter(torch.randn(1))
        # forget gate weights
        self.wf = nn.Parameter(torch.randn(1, input_size))
        # forget gate bias
        self.bf = nn.Parameter(torch.randn(1))
        # output gate weights
        self.Wo = nn.Parameter(torch.randn(hidden_size, input_size))
        # output gate bias
        self.bo = nn.Parameter(torch.randn(hidden_size, 1))
        # fully connected layer to output the prediction to single value
        self.fc = nn.Linear(hidden_size, output_size)
        # initialize the parameters
        self.reset_parameters()
        # optimizer Adam with weight decay for regularization to prevent overfitting
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.05, weight_decay=1e-6)

        # self.criterion = nn.MSELoss()
        # self.criterion = nn.L1Loss()
        # L1 loss is more robust to outliers and seams to work best for this problem
        self.criterion = nn.SmoothL1Loss()

    # initialize the parameters
    def reset_parameters(self):
        # initialize the parameters using xavier uniform initialization
        for p in self.parameters():
            # if the parameter is a matrix
            if p.data.ndimension() >= 2:
                # initialize the matrix using xavier uniform initialization
                nn.init.xavier_uniform_(p.data)
            else:
                # initialize the bias to zeros
                nn.init.zeros_(p.data)

    # forward pass
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
        early_stopping = EarlyStopping(patience=150, verbose=True)
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
