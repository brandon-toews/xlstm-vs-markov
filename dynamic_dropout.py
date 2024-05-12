import torch.nn as nn


# Class for a dynamic dropout layer
class DynamicDropout(nn.Module):
    # Constructor, takes initial dropout rate, adjustment rate, and maximum dropout rate
    def __init__(self, initial_p=0, adjustment_rate=0.02, max_p=0.1):
        super(DynamicDropout, self).__init__()
        # Dropout layer with initial dropout rate
        self.dropout = nn.Dropout(p=initial_p)
        # Dropout rate adjustment rate
        self.adjustment_rate = adjustment_rate
        # Maximum dropout rate
        self.max_p = max_p

    # Forward pass, returns the dropout layer
    def forward(self, x):
        return self.dropout(x)

    # Increase dropout rate
    def increase_dropout(self):
        # If the dropout rate is not already at the maximum
        if not self.dropout.p == self.max_p:
            # Calculate the new dropout rate
            check = self.dropout.p + self.adjustment_rate
            # If the new dropout rate is greater than the maximum
            if check > self.max_p:
                # Set the dropout rate to the maximum
                self.dropout.p = self.max_p
            else:
                # Else set the dropout rate to the new rate
                self.dropout.p = check
            # Print the new dropout rate
            print(f"Dropout increased to {self.dropout.p}")

    # Decrease dropout rate
    def decrease_dropout(self):
        # If the dropout rate is not already at 0
        if not self.dropout.p == 0:
            # Calculate the new dropout rate
            # Lower the dropout rate by twice the adjustment rate
            # Faster decrease to avoid underfitting
            check = self.dropout.p - (self.adjustment_rate * 1)
            # If the new dropout rate is less than 0
            if check < 0:
                # Set the dropout rate to 0
                self.dropout.p = 0
            else:
                # Else set the dropout rate to the new rate
                self.dropout.p = check
            # Print the new dropout rate
            print(f"Dropout decreased to {self.dropout.p}")

    # Reset dropout rate
    def reset_dropout(self):
        # Set the dropout rate to 0
        self.dropout.p = 0
