# Import Meteostat library and dependencies
from datetime import datetime
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from math import sqrt
import matplotlib.pyplot as plt
from meteostat import Point, Daily
import numpy as np
import torch

import custom_lstm
import xLSTM
from markov import MarkovChain
import old_custom_lstm


def extract_features(data, features):
    data = data[features]

    data_min = data.min()
    data_max = data.max()

    data = (data - data_min) / (data_max - data_min)


    # Split data into train and test
    train, val, test = data[:int(0.6 * len(data))].values, data[int(0.6 * len(data)):int(0.8 * len(data))].values, data[int(0.8 * len(data)):].values

    # split data into train (60%), validation (20%) and test (20%)
    # train, val, test = np.split(data, [int(.6 * len(data)), int(.8 * len(data))])

    # Convert to PyTorch tensors and reshape
    # The shape of the data should be (batch_size, sequence_length, num_features)
    train_tensor = torch.tensor(train).float().view(1, -1, len(features))
    val_tensor = torch.tensor(val).float().view(1, -1, len(features))
    test_tensor = torch.tensor(test).float().view(1, -1, len(features))
    return train_tensor, val_tensor, test_tensor, data_min, data_max


def denormalize(predictions, data_min, data_max):
    predictions = np.array(predictions)  # convert predictions to a numpy array
    return predictions * (data_max[0] - data_min[0]) + data_min[0]


def create_model(type, train, val, test, hidden, mem, layers, seq):
    input_size = train.shape[2]
    hidden_size = hidden
    mem_dim = mem
    seq_len = seq

    if type == 'xLSTM':
        model = xLSTM.mLSTM(input_size, hidden_size, mem_dim, layers)
        model.train_model(train, val, 5, seq_len)
        # print(model.state_dict())

        # Load the best model state
        model.load_state_dict(torch.load('checkpoint.pt'))

        test_output = model.predict(test, seq_len)

        return test_output

    '''elif type == 'LSTM':
        model = custom_lstm.LSTMModel(input_size, hidden_size, mem_dim)
        model.train_model(train, val, 200, 20)
        # print(model.state_dict())

        # Load the best model state
        model.load_state_dict(torch.load('checkpoint.pt'))

        test_output = model.predict(test, seq_len)

        return test_output'''

def find_best_order(train_data, test_data, min_order, max_order):
    results = []

    for order in range(min_order, max_order + 1):
        # Create and train the Markov Chain model
        model = MarkovChain(order)
        model.fit(train_data)

        test_data = np.concatenate((train_data[-order:], test_data))

        # Make predictions on the test set
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

    return best_order


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

    # Plot line chart including average, minimum and maximum temperature
    # data.plot(y=['tavg', 'tmin', 'tmax'])
    # plt.show()



    # univariate data preparation
    X = data['tavg'].values
    train, val, test = X[0:int(0.6 * len(data))], X[int(0.6 * len(data)):int(0.8 * len(data))], X[int(0.8 * len(data)):]
    # print('Train Dataset:', train)
    # print('Test Dataset:', test)

    # walk-forward validation
    history = [x for x in train]
    predictions = list()
    for i in range(len(test)):
        # make prediction
        predictions.append(history[-1])
        # observation
        history.append(test[i])
    # report performance
    rmse = sqrt(mean_squared_error(test, predictions))
    print('RMSE: %.3f' % rmse)
    # line plot of observed vs predicted
    # plt.plot(test)
    # plt.plot(predictions)
    # plt.show()

    multi_features = ['tavg', 'tmin', 'tmax']  # replace with your actual columns
    uni_features = ['tavg']

    multi_train, multi_val, multi_test, multi_min, multi_max = extract_features(data, multi_features)
    uni_train, uni_val, uni_test, uni_min, uni_max = extract_features(data, uni_features)

    hidden_size = 20
    mem_dim = 20
    layers = 3
    seq_len = 80


    error_test = np.array(test)
    error_test = error_test[:len(uni_test[0])]

    # normal_lstm_pred = create_model('LSTM', uni_train, uni_val, uni_test, hidden_size, mem_dim, seq_len)

    uni_model_pred = create_model('xLSTM', uni_train, uni_val, uni_test, hidden_size, mem_dim, layers, seq_len)
    multi_model_pred = create_model('xLSTM', multi_train, multi_val, multi_test, hidden_size, mem_dim, layers, seq_len)

    denorm_multi_pred = denormalize(multi_model_pred, multi_min, multi_max)
    denorm_uni_pred = denormalize(uni_model_pred, uni_min, uni_max)
    # denorm_normal_pred = denormalize(normal_lstm_pred, uni_min, uni_max)

    # Define the order of the Markov Chain
    order = 23

    # Split the data into training and test sets
    markov_train, markov_test = train_test_split(X, test_size=0.2, shuffle=False)

    # Convert the data to integers
    markov_train = markov_train.round().astype(int)
    markov_test = markov_test.round().astype(int)

    # Find the best order
    best_order = find_best_order(markov_train, markov_test, 2, 30)

    #best_order = 4

    # Create and train the Markov Chain model with the best order
    model = MarkovChain(best_order)
    model.fit(markov_train)

    # Make predictions on the test set
    # markov_pred = [model.predict(markov_test[i - best_order:i]) for i in range(best_order, len(markov_test))]

    # Create and train the Markov Chain model
    '''model = MarkovChain(order)
    model.fit(markov_train)'''

    markov_test = np.concatenate((markov_train[-best_order:], markov_test))

    # Initialize a list of None values of length equal to the order of the model
    # initial_predictions = [0] * order

    # Make predictions on the test set
    print('Best Order:', best_order)
    print(' ')

    markov_pred = [model.predict(markov_test[i - best_order:i]) for i in range(best_order, len(markov_test))]

    # Combine the initial predictions with the Markov Chain predictions
    # markov_pred = initial_predictions + markov_pred
    #markov_test = markov_test[best_order:]

    # rmse_normal = sqrt(mean_squared_error(error_test[:len(uni_model_pred)], denorm_normal_pred))
    rmse_uni = sqrt(mean_squared_error(error_test[:len(uni_model_pred)], denorm_uni_pred))
    rmse_multi = sqrt(mean_squared_error(error_test[:len(uni_model_pred)], denorm_multi_pred))
    rmse_markov = sqrt(mean_squared_error(error_test[:len(uni_model_pred) - best_order+1], markov_pred[best_order:]))

    # mae_normal = mean_absolute_error(error_test[:len(uni_model_pred)], denorm_normal_pred)
    mae_uni = mean_absolute_error(error_test[:len(uni_model_pred)], denorm_uni_pred)
    mae_multi = mean_absolute_error(error_test[:len(uni_model_pred)], denorm_multi_pred)
    mae_markov = mean_absolute_error(error_test[:len(uni_model_pred) - best_order+1], markov_pred[best_order:])

    # print('MAE for Normal LSTM Model: %.3f' % mae_normal)
    '''print('MAE for xLSTM Univariate Model: %.3f' % mae_uni)
    print('MAE for xLSTM Multivariate Model: %.3f' % mae_multi)'''
    print('MAE for Markov Chain Model: %.3f' % mae_markov)

    print(' ')

    # print('RMSE for Normal LSTM Model: %.3f' % rmse_normal)
    '''print('RMSE for xLSTM Univariate Model: %.3f' % rmse_uni)
    print('RMSE for xLSTM Multivariate Model: %.3f' % rmse_multi)'''
    print('RMSE for Markov Chain Model: %.3f' % rmse_markov)

    print(' ')

    # Calculate correlation
    # corr_normal, _ = pearsonr(error_test[:len(uni_model_pred)], denorm_normal_pred)
    corr_uni, _ = pearsonr(error_test[:len(uni_model_pred)], denorm_uni_pred)
    corr_multi, _ = pearsonr(error_test[:len(uni_model_pred)], denorm_multi_pred)
    corr_markov, _ = pearsonr(error_test[:len(uni_model_pred) - best_order+1], markov_pred[best_order:])

    # print('Correlation for Normal LSTM Model: %.3f' % corr_normal)
    '''print('Correlation for xLSTM Univariate Model: %.3f' % corr_uni)
    print('Correlation for xLSTM Multivariate Model: %.3f' % corr_multi)'''
    print('Correlation for Markov Chain Model: %.3f' % corr_markov)

    print(' ')

    # Calculate R-squared
    # r2_normal = r2_score(error_test[:len(uni_model_pred)], denorm_normal_pred)
    r2_uni = r2_score(error_test[:len(uni_model_pred)], denorm_uni_pred)
    r2_multi = r2_score(error_test[:len(uni_model_pred)], denorm_multi_pred)
    r2_markov = r2_score(error_test[:len(uni_model_pred) - best_order+1], markov_pred[best_order:])

    # print('R-squared for Normal LSTM Model: %.3f' % r2_normal)
    '''print('R-squared for xLSTM Univariate Model: %.3f' % r2_uni)
    print('R-squared for xLSTM Multivariate Model: %.3f' % r2_multi)'''
    print('R-squared for Markov Chain Model: %.3f' % r2_markov)

    # Set the figure size
    #plt.figure(figsize=(8, 8))
    plt.title('Markov Chain Model - FLOAT Values')
    plt.plot(error_test, label='Test Data')
    #plt.plot(markov_test, label='Markov Chain Model test data')
    # plt.plot(denorm_normal_pred, label='Normal LSTM Model')
    #plt.plot(denorm_uni_pred, label='xLSTM Univariate Model')
    #plt.plot(denorm_multi_pred, label='xLSTM Multivariate Model')
    plt.plot(markov_pred[best_order:], label='Markov Chain Model', color='tab:olive')

    plt.legend()
    plt.show()

# Run the main function
if __name__ == "__main__":
    main()