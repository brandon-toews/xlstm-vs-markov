import torch
import torch.nn as nn
import math
from early_stopping import EarlyStopping
from dynamic_dropout import DynamicDropout


# Define the mLSTM module
class mLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, mem_dim):
        super(mLSTM, self).__init__()
        # input_size is the number of features in the input
        self.input_size = input_size
        # hidden_size is the number of features in the hidden state h
        self.hidden_size = hidden_size
        # mem_dim is the number of features in the memory state C
        self.mem_dim = mem_dim
        # query weights
        self.w_q = nn.Parameter(torch.randn(hidden_size, input_size))
        # query bias
        self.b_q = nn.Parameter(torch.randn(hidden_size, 1))
        # key weights
        self.w_k = nn.Parameter(torch.randn(mem_dim, input_size))
        # key bias
        self.b_k = nn.Parameter(torch.randn(mem_dim, 1))
        # value weights
        self.w_v = nn.Parameter(torch.randn(mem_dim, input_size))
        # value bias
        self.b_v = nn.Parameter(torch.randn(mem_dim, 1))
        # input gate weights
        self.w_i = nn.Parameter(torch.randn(1, input_size))
        # input gate bias
        self.b_i = nn.Parameter(torch.randn(1))
        # forget gate weights
        self.w_f = nn.Parameter(torch.randn(1, input_size))
        # forget gate bias
        self.b_f = nn.Parameter(torch.randn(1))
        # output gate weights
        self.w_o = nn.Parameter(torch.randn(hidden_size, input_size))
        # output gate bias
        self.b_o = nn.Parameter(torch.randn(hidden_size, 1))
        # initialize the parameters
        self.reset_parameters()

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
        # unpack the previous states
        (c_prev, n_prev) = states
        # reshape x
        x = x.view(self.input_size, -1)
        # query, key, value calculations
        q_t = torch.matmul(self.w_q, x) + self.b_q
        k_t = (1 / math.sqrt(self.mem_dim)) * (torch.matmul(self.w_k, x) + self.b_k)
        v_t = torch.matmul(self.w_v, x) + self.b_v

        # input and forget gate calculations
        i_t = torch.exp(torch.matmul(self.w_i, x) + self.b_i)
        f_t = torch.sigmoid(torch.matmul(self.w_f, x) + self.b_f)

        # value and key squeeze to remove the extra dimension
        v_t = v_t.squeeze()
        k_t = k_t.squeeze()

        # update the memory state and the normalization factor
        c_t = f_t * c_prev + i_t * torch.ger(v_t, k_t)
        n_t = f_t * n_prev + i_t * k_t.unsqueeze(1)

        # normalization factor
        max_nqt = torch.max(torch.abs(torch.matmul(n_t.T, q_t)), torch.tensor(1.0))
        # calculate the hidden state, part 1
        h_tilde = torch.matmul(c_t, q_t) / max_nqt
        # output gate calculation,
        o_t = torch.sigmoid(torch.matmul(self.w_o, x) + self.b_o)
        # hidden state calculation, part 2
        h_t = o_t * h_tilde
        # reshape the hidden state
        h_t = h_t.view(-1, self.hidden_size)
        # return the hidden state and the new states
        return h_t, (c_t, n_t)

    # initialize the hidden states
    def init_hidden(self):
        return (torch.zeros(self.mem_dim, self.mem_dim),
                torch.zeros(self.mem_dim, 1))


# Define the sLSTM module
class sLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(sLSTM, self).__init__()
        # input_size is the number of features in the input
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Weights for input, forget, output gates and cell state
        self.w_i = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.w_f = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.w_o = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.w_c = nn.Parameter(torch.Tensor(hidden_size, input_size))

        # recurrent weights for input, forget, output gates and cell state
        self.u_i = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.u_f = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.u_o = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.u_c = nn.Parameter(torch.Tensor(hidden_size, hidden_size))

        # biases for input, forget, output gates and cell state
        self.b_i = nn.Parameter(torch.Tensor(hidden_size))
        self.b_f = nn.Parameter(torch.Tensor(hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(hidden_size))
        self.b_c = nn.Parameter(torch.Tensor(hidden_size))

        # initialize the parameters
        self.reset_parameters()

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
        # Unpack the previous states
        h_t, c_t, m_t, n_t = states

        # Exponential gating for the input and forget gates
        i_t = torch.exp(x @ self.w_i + h_t.T @ self.u_i + self.b_i)
        f_t = torch.exp(x @ self.w_f + h_t.T @ self.u_f + self.b_f)

        # Stabilizer state
        m_t = torch.max(torch.log(f_t) + m_t, torch.log(i_t))

        # Stabilized input and forget gates
        i_t_prime = torch.exp(torch.log(i_t) - m_t)
        f_t_prime = torch.exp(torch.log(f_t) + m_t - m_t)

        # Output gate calculation
        o_t = torch.sigmoid(x @ self.w_o + h_t.T @ self.u_o + self.b_o)

        # Cell input
        z_t = torch.tanh(x @ self.w_c + h_t.T @ self.u_c + self.b_c)

        # Cell state calculations
        c_t = f_t_prime * c_t + i_t_prime * z_t

        # Normalizer state
        n_t = f_t_prime * n_t + i_t_prime

        # Hidden state output
        h_t = o_t * (c_t / n_t)

        # Return the hidden state and the new states
        return h_t, (h_t, c_t, m_t, n_t)

    # initialize the hidden states
    def init_hidden(self):
        return (torch.zeros(self.hidden_size, 1),
                torch.zeros(self.hidden_size, 1),
                torch.zeros(self.hidden_size, 1),
                torch.zeros(self.hidden_size, 1))


# Define the xLSTM block
class xLSTM_Block(nn.Module):
    def __init__(self, block_type, input_size, hidden_size, layers=2, mem_size=None):
        super(xLSTM_Block, self).__init__()
        # initialize the dropout layer
        self.dropout = DynamicDropout()
        # Create multiple mLSTM and sLSTM layers depending on the block type
        if block_type == 'mLSTM':
            # Create multiple mLSTM layers
            self.layers = nn.ModuleList([mLSTM(input_size if i == 0 else hidden_size, hidden_size, mem_size)
                                         for i in range(layers)])
        elif block_type == 'sLSTM':
            # Create multiple sLSTM layers
            self.layers = nn.ModuleList([sLSTM(hidden_size, hidden_size) for _ in range(layers)])

    # forward pass
    def forward(self, x, initial_states):
        # Initial hidden states
        hidden_states = self.layers[0].init_hidden()
        # Loop through the layers
        for i in range(len(self.layers)):
            # Forward pass through each layer
            x, hidden_states = self.layers[i](x, hidden_states)
            # Apply dropout
            x = self.dropout(x)
        # Return the hidden state and the new states
        return x, hidden_states


# Define the xLSTM model
class xLSTM_model(nn.Module):
    def __init__(self, input_size, hidden_size, mem_size, output_size=1, layers=2):
        super(xLSTM_model, self).__init__()
        # input_size is the number of features in the input
        self.input_size = input_size
        # number of layers in block
        self.layers = layers
        # Initialize xLSTM blocks
        self.blocks = nn.ModuleList()
        # Create xLSTM block for mLSTM
        self.blocks.append(xLSTM_Block('mLSTM', input_size, hidden_size, layers, mem_size))
        # Create xLSTM block for sLSTM
        self.blocks.append(xLSTM_Block('sLSTM', hidden_size, hidden_size, layers))
        # fully connected layer to output the prediction to single value
        self.fc = nn.Linear(hidden_size, output_size)
        # dropout layer
        self.dropout = DynamicDropout()
        # get all the parameters to optimize from all the layers in each block
        self.optimizing_parameters = []
        for i in range(len(self.blocks)):
            for n in range(len(self.blocks[i].layers)):
                self.optimizing_parameters += list(self.blocks[i].layers[n].parameters())

        # optimizer Adam with weight decay for regularization to prevent overfitting
        self.optimizer = torch.optim.Adam(self.optimizing_parameters, lr=0.02, weight_decay=1e-6)

        # self.criterion = nn.MSELoss()
        # self.criterion = nn.L1Loss()
        # L1 loss is more robust to outliers and seams to work best for this problem
        self.criterion = nn.SmoothL1Loss()

    # forward pass
    def forward(self, x):
        # Initial hidden states
        hidden_states = self.blocks[0].layers[0].init_hidden()
        # Forward pass through the blocks
        for i in range(len(self.blocks)):
            # Forward pass through each block
            x, hidden_states = self.blocks[i](x, hidden_states)
            # Apply dropout
            # x = self.dropout(x)

        # Fully connected layer
        x = self.fc(x)
        return x

    # Train the model
    def train_model(self, data, val_data, epochs, seq_len):
        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=50, verbose=True)
        # loop through the epochs
        for epoch in range(epochs):
            # zero the gradients
            self.optimizer.zero_grad()
            # initialize the loss
            loss = 0
            # loop through the sequence length
            for t in range(seq_len - 1):
                # get the input at time t
                x = data[:, t]
                # get the target at time t+1
                y_true = data[:, t + 1, 0]
                # get the prediction
                y_pred = self(x)
                # calculate the loss from the training data
                loss += self.criterion(y_pred, y_true)

            # validate the model on the validation data
            val_loss = self.validate(val_data)
            # print the validation loss
            print(f'Epoch {epoch} Validation Loss {val_loss.item()}')
            # call the early stopping object
            early_stopping(val_loss, self)
            # if early stopping is triggered
            if early_stopping.early_stop:
                # print message
                print("Early stopping")
                # stop the training
                break
            # calculate the average loss
            loss.backward()
            # clip the gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.optimizing_parameters, max_norm=1)
            # update the weights
            self.optimizer.step()
            # print the training loss every 10 epochs
            if epoch % 10 == 0:
                print(f'Epoch {epoch} Loss {loss.item()}')

        # load the best model before early stopping
        self.load_state_dict(torch.load('checkpoint.pt'))

    # predict the model
    def predict(self, data):
        # set the model to evaluation mode
        self.eval()
        # initialize the predictions
        predictions = []
        # loop through the sequence length
        for t in range(data.shape[1] - 1):
            # get the input at time t
            x = data[:, t]
            # get the prediction
            y_pred = self(x)
            # append the prediction to the list
            predictions.append(y_pred.detach().numpy().ravel()[0])
        # set the model back to training mode
        self.train()
        # return the predictions
        return predictions

    # validate the model with the validation data
    def validate(self, data):
        # set the model to evaluation mode
        self.eval()
        # initialize the loss
        loss = 0
        # loop through the sequence length
        for t in range(data.shape[1] - 1):
            # get the input at time t
            x = data[:, t]
            # get the target at time t+1
            y_true = data[:, t + 1, 0]
            # get the prediction
            y_pred = self(x)
            # calculate the loss from the validation data
            loss += self.criterion(y_pred, y_true)
        # set the model back to training mode
        self.train()
        # return the validation loss
        return loss
