import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from math import sqrt


# Define the Markov chain class
class MarkovChain:
    def __init__(self, order):
        # Order of the Markov chain
        self.order = order
        # Transition matrix
        self.transition_matrix = None
        # Unique states in the data
        self.states = None

    # Fit the Markov chain to the data
    def fit(self, data):
        # Get the unique states in the data, and convert them to strings
        self.states = np.unique(data).astype(str)
        # Initialize the transition matrix, and set all values to zero
        # with the states as the row and column indices
        self.transition_matrix = pd.DataFrame(np.zeros((len(self.states), len(self.states))), index=self.states,
                                              columns=self.states)

        # Iterate through the data, and update the transition matrix
        for i in range(len(data) - self.order):
            # Get the current and next states
            current_state = data[i].astype(str)
            # Next state is order steps ahead
            next_state = data[i + self.order].astype(str)
            # Update the transition matrix at the current and next states
            self.transition_matrix.loc[current_state, next_state] += 1

        # Normalize the transition matrix to values between 0 and 1
        # Sum up the rows of the transition matrix
        # Add a small value to the denominator to avoid division by zero
        row_sums = self.transition_matrix.sum(axis=1) + 1e-10
        # Divide each row by the row sum to get the transition probabilities
        self.transition_matrix = self.transition_matrix.div(row_sums, axis=0)

    # Predict the next state in the sequence
    def predict(self, sequence):
        # Get the current state
        state = sequence[0].astype(str)
        # If the state is not in the transition matrix
        if state not in self.transition_matrix.index:
            # Return the first state in the sequence as the prediction (fallback)
            return sequence[0]
        # Return the state with the highest probability in the transition matrix
        return self.states[np.argmax(self.transition_matrix.loc[state])].astype(sequence.dtype)


# Define a function to find the best order for the Markov Chain model
def find_best_order(train_data, test_data, min_order, max_order):
    # Initialize a list to store the results
    results = []

    # Loop through the range of orders
    for order in range(min_order, max_order + 1):
        # Create and train the Markov Chain model
        model = MarkovChain(order)
        model.fit(train_data)

        # Add the last 'order' elements of the training data to the test data
        # to ensure that the test data is long enough for the predictions
        # for the test data in the LSTM models
        test_data = np.concatenate((train_data[-order:], test_data))

        # Make predictions on the test set, feeding the model order elements at a time
        markov_pred = [model.predict(test_data[i - order:i]) for i in range(order, len(test_data))]

        # Calculate the metrics
        mse = mean_squared_error(test_data[order:], markov_pred)
        mae = mean_absolute_error(test_data[order:], markov_pred)
        corr, _ = pearsonr(test_data[order:], markov_pred)
        r2 = r2_score(test_data[order:], markov_pred)

        # Store the results
        results.append((order, sqrt(mse), mae, corr, r2))

    # Convert the results to a DataFrame
    results_df = pd.DataFrame(results, columns=['order', 'rmse', 'mae', 'corr', 'r2'])

    # Rank the results
    results_df['rmse_rank'] = results_df['rmse'].rank()
    results_df['mae_rank'] = results_df['mae'].rank()
    results_df['corr_rank'] = results_df['corr'].rank(ascending=False)
    results_df['r2_rank'] = results_df['r2'].rank(ascending=False)

    # Calculate the average rank
    results_df['avg_rank'] = results_df[['rmse_rank', 'mae_rank', 'corr_rank', 'r2_rank']].mean(axis=1)

    # Return the order with the lowest average rank
    best_order = results_df.loc[results_df['avg_rank'].idxmin(), 'order']

    # Return the best order for the Markov Chain model
    return best_order
