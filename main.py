# Import Meteostat library and dependencies
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from math import sqrt
import matplotlib.pyplot as plt
from meteostat import Point, Daily
import numpy as np
import torch
import xLSTM
import markov as mk


# Define a function to extract features from the data
def extract_features(data, features):
    # Get the specified features
    data = data[features]

    # Find the minimum and maximum values of the data
    data_min = data.min()
    data_max = data.max()

    # Normalize the data
    data = (data - data_min) / (data_max - data_min)

    # Split data into train (60%), val (20%), and test (20%) sets
    train, val, test = (data[:int(0.6 * len(data))].values,
                        data[int(0.6 * len(data)):int(0.8 * len(data))].values, data[int(0.8 * len(data)):].values)

    # Convert to PyTorch tensors and reshape
    # The shape of the data should be (batch_size, sequence_length, num_features)
    train_tensor = torch.tensor(train).float().view(1, -1, len(features))
    val_tensor = torch.tensor(val).float().view(1, -1, len(features))
    test_tensor = torch.tensor(test).float().view(1, -1, len(features))
    # Return the tensors and the min/max values
    return train_tensor, val_tensor, test_tensor, data_min, data_max


# Define a function to denormalize the predictions
def denormalize(predictions, data_min, data_max):
    # Convert the predictions to a numpy array
    predictions = np.array(predictions)
    # Denormalize the predictions using the min/max values
    return predictions * (data_max[0] - data_min[0]) + data_min[0]


# Define a function to create xLSTM models
def create_model(train, val, test, hidden, mem, layers, seq):
    # Define the input size, hidden size, memory dimension, and sequence length
    input_size = train.shape[2]
    hidden_size = hidden
    mem_dim = mem
    seq_len = seq

    # Create the xLSTM model
    model = xLSTM.xLSTM_model(input_size, hidden_size, mem_dim, 1, layers)
    # Train the model on the training set and validate on the validation set
    # for 200 epochs with the specified sequence length
    model.train_model(train, val, 200, seq_len)

    # Training might have been stopped early if the validation loss did not improve
    # Load the best model from the checkpoint
    model.load_state_dict(torch.load('checkpoint.pt'))

    # Make predictions on the test set
    test_output = model.predict(test)

    # Return the predictions
    return test_output


# Define the main function
def main():
    # Set time period
    start = datetime(2018, 1, 1)
    end = datetime(2018, 12, 31)

    # Create Point for Vancouver, BC
    vancouver = Point(49.2497, -123.1193, 70)

    # Get daily data for 2018
    data = Daily(vancouver, start, end)
    data = data.fetch()

    data = data.interpolate(method='linear')

    print(data.head())

    # Create evaluation data for the models
    x = data['tavg'].values
    # split a univariate sequence into samples, train (60%), val (20%), test (20%)
    test = x[int(0.8 * len(data)):]

    # univariate data preparation
    uni_features = ['tavg']
    # Extract the features from the data and normalize them
    uni_train, uni_val, uni_test, uni_min, uni_max = extract_features(data, uni_features)

    # multivariate data preparation
    multi_features = ['tavg', 'tmin', 'tmax']
    # Extract the features from the data and normalize them
    multi_train, multi_val, multi_test, multi_min, multi_max = extract_features(data, multi_features)

    # Evaluate the models on the test set
    error_test = np.array(test)
    # Cut the error_test array to match the length of the predictions
    error_test = error_test[:len(uni_test[0])]

    # xLSTM model parameters
    hidden_size = 30
    mem_dim = 30
    layers = 1
    seq_len = 80

    # Create the xLSTM models
    # Univariate model, create_model function returns the predictions
    uni_model_pred = create_model(uni_train, uni_val, uni_test, hidden_size, mem_dim, layers, seq_len)
    # Multivariate model, create_model function returns the predictions
    multi_model_pred = create_model(multi_train, multi_val, multi_test, hidden_size, mem_dim, layers, seq_len)

    # Denormalize the predictions
    denorm_uni_pred = denormalize(uni_model_pred, uni_min, uni_max)
    denorm_multi_pred = denormalize(multi_model_pred, multi_min, multi_max)

    # Split the data into training (80%) and test sets (20%) for the Markov Chain model
    markov_train, markov_test = train_test_split(x, test_size=0.2, shuffle=False)

    # Convert the data to integers for the Markov Chain model
    # This narrows down the state space and makes the model more accurate
    markov_train = markov_train.round().astype(int)
    markov_test = markov_test.round().astype(int)

    # Find the best order for the Markov Chain model between 2 and 30
    best_order = mk.find_best_order(markov_train, markov_test, 2, 30)

    # Create and train the Markov Chain model with the best order
    model = mk.MarkovChain(best_order)
    model.fit(markov_train)

    # Add the last 'order' elements of the training data to the test data
    # to ensure that the test data is long enough for the predictions
    # to start of the test data in the LSTM models
    markov_test = np.concatenate((markov_train[-best_order:], markov_test))

    # Make predictions on the test set, feeding the model order elements at a time
    markov_pred = [model.predict(markov_test[i - best_order:i]) for i in range(best_order, len(markov_test))]

    # Calculate the metrics for the models
    mae_uni = mean_absolute_error(error_test[:len(uni_model_pred)], denorm_uni_pred)
    mae_multi = mean_absolute_error(error_test[:len(uni_model_pred)], denorm_multi_pred)
    mae_markov = mean_absolute_error(error_test[:len(uni_model_pred) - best_order+1], markov_pred[best_order:])

    rmse_uni = sqrt(mean_squared_error(error_test[:len(uni_model_pred)], denorm_uni_pred))
    rmse_multi = sqrt(mean_squared_error(error_test[:len(uni_model_pred)], denorm_multi_pred))
    rmse_markov = sqrt(mean_squared_error(error_test[:len(uni_model_pred) - best_order + 1], markov_pred[best_order:]))

    # Calculate correlation
    corr_uni, _ = pearsonr(error_test[:len(uni_model_pred)], denorm_uni_pred)
    corr_multi, _ = pearsonr(error_test[:len(uni_model_pred)], denorm_multi_pred)
    corr_markov, _ = pearsonr(error_test[:len(uni_model_pred) - best_order + 1], markov_pred[best_order:])

    # Calculate R-squared
    r2_uni = r2_score(error_test[:len(uni_model_pred)], denorm_uni_pred)
    r2_multi = r2_score(error_test[:len(uni_model_pred)], denorm_multi_pred)
    r2_markov = r2_score(error_test[:len(uni_model_pred) - best_order + 1], markov_pred[best_order:])

    # Print the best order for the Markov Chain model
    print('Best Order:', best_order)
    print(' ')

    # Print the metrics for the models
    print('MAE for xLSTM Univariate Model: %.3f' % mae_uni)
    print('MAE for xLSTM Multivariate Model: %.3f' % mae_multi)
    print('MAE for Markov Chain Model: %.3f' % mae_markov)
    print(' ')
    print('RMSE for xLSTM Univariate Model: %.3f' % rmse_uni)
    print('RMSE for xLSTM Multivariate Model: %.3f' % rmse_multi)
    print('RMSE for Markov Chain Model: %.3f' % rmse_markov)
    print(' ')
    print('Correlation for xLSTM Univariate Model: %.3f' % corr_uni)
    print('Correlation for xLSTM Multivariate Model: %.3f' % corr_multi)
    print('Correlation for Markov Chain Model: %.3f' % corr_markov)
    print(' ')
    print('R-squared for xLSTM Univariate Model: %.3f' % r2_uni)
    print('R-squared for xLSTM Multivariate Model: %.3f' % r2_multi)
    print('R-squared for Markov Chain Model: %.3f' % r2_markov)

    # Plot the predictions
    plt.title('Model Comparison')
    plt.plot(error_test, label='Test Data')
    plt.plot(denorm_uni_pred, label='xLSTM Univariate Model')
    plt.plot(denorm_multi_pred, label='xLSTM Multivariate Model')
    plt.plot(markov_pred[best_order:], label='Markov Chain Model')
    plt.legend()
    plt.show()


# Run the main function
if __name__ == "__main__":
    main()
