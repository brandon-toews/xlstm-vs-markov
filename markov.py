import numpy as np
import pandas as pd


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
