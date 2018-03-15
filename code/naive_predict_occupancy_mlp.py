import os
import sys
import argparse
import numpy as  np
import pandas as pd

from datetime               import date
from timeit                 import default_timer
from sklearn.neural_network import MLPRegressor
from sklearn.metrics        import mean_squared_error

parser = argparse.ArgumentParser()
parser.add_argument('--distance', default=25, type=int)
parser.add_argument('--trip_type', default='pickups', type=str)
parser.add_argument('--start_date', type=int, nargs=3, default=[2013, 1, 1])
parser.add_argument('--end_date', type=int, nargs=3, default=[2016, 6, 30])
parser.add_argument('--metric', type=str, default='rel_diffs')
parser.add_argument('--hidden_layer_sizes', nargs='+', type=int, default=[100])

locals().update(vars(parser.parse_args()))

fname = '_'.join(map(str, [distance, start_date[0], start_date[1], start_date[2], end_date[0], end_date[1], end_date[2], metric, hidden_layer_sizes]))

start_date, end_date = date(*start_date), date(*end_date)

predictions_path = os.path.join('..', 'data', 'naive_mlp_predictions', fname)

for path in [predictions_path]:
	if not os.path.isdir(path):
		os.makedirs(path)

# Load daily capacity data.
print('\nLoading daily per-hotel capacity data.'); start = default_timer()

occupancy = pd.read_csv(os.path.join('..', 'data', 'Unmasked Daily Capacity.csv'), index_col=False)
occupancy['Date'] = pd.to_datetime(occupancy['Date'], format='%Y-%m-%d')
occupancy = occupancy.loc[(occupancy['Date'] >= start_date) & (occupancy['Date'] <= end_date)]

print('Time: %.4f' % (default_timer() - start))

hotels = np.array(occupancy['Share ID'])
weekdays = np.array(occupancy['Date'].apply(date.weekday)).reshape([-1, 1])
dates = np.array(occupancy['Date'].apply(str)).reshape([-1, 1])
targets = np.array(occupancy['Room Demand'])

# Randomly permute the data to remove sequence biasing.
p = np.random.permutation(targets.shape[0])
hotels, weekdays, dates, targets = hotels[p], weekdays[p], dates[p], targets[p]

_, hotels = np.unique(hotels, return_inverse=True)
hotels = hotels.reshape([-1, 1])

_, dates = np.unique(dates, return_inverse=True)
dates = dates.reshape([-1, 1])

# Split the data into (training, test) subsets.
split = int(0.8 * len(targets))

train_features = [hotels[:split], dates[:split], weekdays[:split]]
train_features = np.concatenate(train_features, axis=1)

test_features = [hotels[split:], dates[split:], weekdays[split:]]
test_features = np.concatenate(test_features, axis=1)

train_targets = targets[:split]
test_targets = targets[split:]

print('\nCreating and training multi-layer perceptron regression model.\n')

model = MLPRegressor(verbose=True, hidden_layer_sizes=hidden_layer_sizes).fit(train_features, train_targets)

print('\nTraining complete. Getting predictions and calculating R^2, MSE.')

train_score = model.score(train_features, train_targets)
test_score = model.score(test_features, test_targets)

train_predictions = model.predict(train_features)
train_mse = mean_squared_error(train_targets, train_predictions)

test_predictions = model.predict(test_features)
test_mse = mean_squared_error(test_targets, test_predictions)

np.save(os.path.join(predictions_path, 'train_targets.npy'), train_targets)
np.save(os.path.join(predictions_path, 'train_predictions.npy'), train_predictions)

np.save(os.path.join(predictions_path, 'test_targets.npy'), test_targets)
np.save(os.path.join(predictions_path, 'test_predictions.npy'), test_predictions)

print('\n')
print('Training R^2:', train_score)
print('Training MSE:', train_mse)
print('\n')
print('Test R^2:', test_score)
print('Test MSE:', test_mse)
print('\n')