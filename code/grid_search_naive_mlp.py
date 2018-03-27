from __future__ import print_function

import os
import sys
import argparse
import numpy as  np
import pandas as pd

from datetime                import date
from timeit                  import default_timer
from sklearn.neural_network  import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics         import mean_squared_error

parser = argparse.ArgumentParser()
parser.add_argument('--start_date', type=int, nargs=3, default=[2014, 1, 1])
parser.add_argument('--end_date', type=int, nargs=3, default=[2016, 6, 30])
parser.add_argument('--trials', type=int, default=5)

locals().update(vars(parser.parse_args()))

fname = '_'.join(map(str, [start_date[0], start_date[1], start_date[2], end_date[0], end_date[1], end_date[2]]))

start_date, end_date = date(*start_date), date(*end_date)

print()
print(start_date, end_date, trials)

predictions_path = os.path.join('..', 'data', 'grid_search_taxi_mlp_predictions', fname)

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
weekdays = np.array(occupancy['Date'].dt.weekday).reshape([-1, 1])
months = np.array(occupancy['Date'].dt.month).reshape([-1, 1])
years = np.array(occupancy['Date'].dt.year).reshape([-1, 1])
targets = np.array(occupancy['Room Demand'])

print('\nRunning random search over hyper-parameters.')

# Randomly permute the data to remove sequence biasing.
p = np.random.permutation(targets.shape[0])
hotels, weekdays, months, years, targets = hotels[p], weekdays[p], months[p], years[p], targets[p]

_, hotels = np.unique(hotels, return_inverse=True)
hotels = hotels.reshape([-1, 1])

# Split the data into (training, test) subsets.
split = int(0.8 * len(targets))

train_features = [hotels[:split], years[:split], months[:split], weekdays[:split]]
train_features = np.concatenate(train_features, axis=1)

test_features = [hotels[split:], years[split:], months[split:], weekdays[split:]]
test_features = np.concatenate(test_features, axis=1)

train_targets = targets[:split]
test_targets = targets[split:]

print('Creating and training multi-layer perceptron regression model.')

param_grid = {'hidden_layer_sizes' : [[512, 256, 128], [512, 512, 256], [512, 256, 256], 
									  [1024, 512, 256], [512, 512, 512], [1024, 512, 512],
									  [512, 256, 128, 64], [1024, 512, 256, 128], [1024, 512, 512, 256]],
			  'alpha' : [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]}

model = GridSearchCV(MLPRegressor(), param_grid=param_grid, verbose=5, n_jobs=-1)
model.fit(train_features, train_targets)

print(); print('Best model hyper-parameters:', model.best_params_); print()

model = model.best_estimator_

print('Training complete. Running multiple train / test iterations with best hyper-parameters.')

train_scores = []
test_scores = []
train_mses = []
test_mses = []

for i in range(trials):  # Run 5 independent realizations of training / test.
	print('\nTraining, testing model %d / %d' % (i + 1, trials))
	
	# Randomly permute the data to remove sequence biasing.
	p = np.random.permutation(targets.shape[0])
	hotels, weekdays, months, years, targets = hotels[p], weekdays[p], months[p], years[p], targets[p]

	_, hotels = np.unique(hotels, return_inverse=True)
	hotels = hotels.reshape([-1, 1])

	# Split the data into (training, test) subsets.
	split = int(0.8 * len(targets))

	train_features = [hotels[:split], years[:split], months[:split], weekdays[:split]]
	train_features = np.concatenate(train_features, axis=1)

	test_features = [hotels[split:], years[split:], months[split:], weekdays[split:]]
	test_features = np.concatenate(test_features, axis=1)

	train_targets = targets[:split]
	test_targets = targets[split:]

	print('Re-training multi-layer perceptron regression model.')

	model.fit(train_features, train_targets)

	print('Training complete. Getting predictions and calculating R^2, MSE.')

	train_scores.append(model.score(train_features, train_targets))
	test_scores.append(model.score(test_features, test_targets))

	train_predictions = model.predict(train_features)
	test_predictions = model.predict(test_features)

	train_mses.append(mean_squared_error(train_targets, train_predictions))
	test_mses.append(mean_squared_error(test_targets, test_predictions))

	np.save(os.path.join(predictions_path, 'train_targets_%d.npy' % i), train_targets)
	np.save(os.path.join(predictions_path, 'train_predictions_%d.npy' % i), train_predictions)

	np.save(os.path.join(predictions_path, 'test_targets_%d.npy' % i), test_targets)
	np.save(os.path.join(predictions_path, 'test_predictions_%d.npy' % i), test_predictions)

print()
print('Mean, standard deviation of training MSE: %.0f $\pm$ %.0f' % (np.mean(train_mses), np.std(train_mses)))
print('Mean, standard deviation of training R^2: %.4f' % np.mean(train_scores))
print()
print('Mean, standard deviation of test MSE: %.0f $\pm$ %.0f' % (np.mean(test_mses), np.std(test_mses)))
print('Mean, standard deviation of test R^2: %.4f' % np.mean(test_scores))
print()