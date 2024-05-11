import numpy as np
import torch


# Class for early stopping of model training
class EarlyStopping:
    # Constructor, takes patience, verbose, and delta
    def __init__(self, patience=7, verbose=False, delta=0):
        # Patience for early stopping
        self.patience = patience
        # Verbose mode
        self.verbose = verbose
        # Counter for early stopping
        self.counter = 0
        # Best score
        self.best_score = None
        # Early stopping flag
        self.early_stop = False
        # Minimum validation loss
        self.val_loss_min = np.Inf
        # Delta for minimum change in validation loss
        self.delta = delta

    # Call function, takes validation loss and model
    def __call__(self, val_loss, model):
        # Score is negative validation loss
        score = -val_loss
        # If best score is None
        if self.best_score is None:
            # Set best score to score
            self.best_score = score
            # Save model checkpoint
            self.save_checkpoint(val_loss, model)
        # If score is less than best score plus delta
        elif score < self.best_score + self.delta:
            # Increase counter
            self.counter += 1
            # Increase dropout rate of model to help prevent overfitting
            model.dropout.increase_dropout()

            # Print verbose message
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            # If counter is greater than or equal to patience
            if self.counter >= self.patience:
                # Set early stopping flag
                self.early_stop = True
        # Else if score is greater than best score plus delta
        else:
            # Decrease dropout rate of model to help prevent underfitting
            model.dropout.decrease_dropout()
            # model.dropout.reset_dropout()

            # Update best score
            self.best_score = score
            # Reset counter
            self.counter = 0
            # Save model checkpoint
            self.save_checkpoint(val_loss, model)

    # Save model checkpoint
    def save_checkpoint(self, val_loss, model):
        # If verbose mode
        if self.verbose:
            # Print message
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # Save model state dictionary
        torch.save(model.state_dict(), 'checkpoint.pt')
        # Update minimum validation loss
        self.val_loss_min = val_loss
