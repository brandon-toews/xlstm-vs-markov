import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
import numpy as np
import math
import matplotlib.pyplot as plt


def generate_sine_wave(seq_len, num_sequences):
    x = np.linspace(0, 2 * np.pi, seq_len)
    y = np.sin(x)
    return tf.convert_to_tensor(y).float().view(-1, 1).repeat(1, num_sequences).unsqueeze(0)


class mLSTM(Model):
    def __init__(self, input_size, hidden_size, mem_dim):
        super(mLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.mem_dim = mem_dim
        self.Wq = tf.Variable(tf.random.normal([hidden_size, input_size]))
        self.bq = tf.Variable(tf.random.normal([hidden_size, 1]))
        self.Wk = tf.Variable(tf.random.normal([mem_dim, input_size]))
        self.bk = tf.Variable(tf.random.normal([mem_dim, 1]))
        self.Wv = tf.Variable(tf.random.normal([mem_dim, input_size]))
        self.bv = tf.Variable(tf.random.normal([mem_dim, 1]))
        self.wi = tf.Variable(tf.random.normal([1, input_size]))
        self.bi = tf.Variable(tf.random.normal([1]))
        self.wf = tf.Variable(tf.random.normal([1, input_size]))
        self.bf = tf.Variable(tf.random.normal([1]))
        self.Wo = tf.Variable(tf.random.normal([hidden_size, input_size]))
        self.bo = tf.Variable(tf.random.normal([hidden_size, 1]))

    def call(self, x, states):
        (C_prev, n_prev) = states
        qt = tf.matmul(self.Wq, x) + self.bq
        kt = (1 / tf.sqrt(float(self.mem_dim))) * (tf.matmul(self.Wk, x) + self.bk)
        vt = tf.matmul(self.Wv, x) + self.bv

        it = tf.exp(tf.matmul(self.wi, x) + self.bi)
        ft = tf.sigmoid(tf.matmul(self.wf, x) + self.bf)

        vt = tf.squeeze(vt)
        kt = tf.squeeze(kt)

        C = ft * C_prev + it * tf.linalg.einsum('i,j->ij', vt, kt)
        n = ft * n_prev + it * tf.expand_dims(kt, 1)

        max_nqt = tf.maximum(tf.abs(tf.tensordot(tf.transpose(n), qt, axes=1)), tf.constant(1.0))
        h_tilde = tf.tensordot(C, qt, axes=1) / max_nqt
        ot = tf.sigmoid(tf.matmul(self.Wo, x) + self.bo)
        ht = ot * h_tilde

        return ht, (C, n)

input_size = 1
hidden_size = 10
mem_dim = 10
seq_len = 100
num_sequences = 1

model = mLSTM(input_size, hidden_size, mem_dim)
optimizer = tf.keras.optimizers.Adam(model.trainable_variables, learning_rate=0.01)
criterion = tf.keras.losses.MeanSquaredError()

data = generate_sine_wave(seq_len, num_sequences)

for epoch in range(200):
    states = model.init_hidden()
    optimizer.zero_grad()
    loss = 0
    for t in range(seq_len - 1):
        x = data[:, t]
        y_true = data[:, t + 1]
        y_pred, states = model(x, states)
        loss += criterion(y_pred, y_true)

    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch} Loss {loss.item()}')

test_output = []
states = model.init_hidden()
for t in range(seq_len - 1):
    x = data[:, t]
    y_pred, states = model(x, states)
    test_output.append(y_pred.detach().numpy().ravel()[0])

plt.figure(figsize=(10, 4))
plt.title('Original vs. Predicted Sine Wave')
plt.plot(data.numpy().ravel(), label='Original')
plt.plot(test_output, label='Predicted')
plt.legend()
plt.show()