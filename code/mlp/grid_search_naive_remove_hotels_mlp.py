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
parser.add_argument('--removals', type=int, default=25)

locals().update(vars(parser.parse_args()))

fname = '_'.join(map(str, [start_date[0], start_date[1], start_date[2], end_date[0], end_date[1], end_date[2]]))

start_date, end_date = date(*start_date), date(*end_date)

data_path = os.path.join('..', '..', 'data')
predictions_path = os.path.join(data_path, 'grid_search_naive_remove_hotels_mlp_predictions', fname)
removals_path = os.path.join(data_path, 'grid_search_naive_mlp_removals', fname)

for path in [predictions_path, removals_path]:
	if not os.path.isdir(path):
		os.makedirs(path)

# Load daily capacity data.
print('\nLoading daily per-hotel capacity data.'); start = default_timer()

df = pd.read_csv(os.path.join(data_path, 'Unmasked Daily Capacity.csv'), index_col=False)
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
df = df.loc[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

print('Time: %.4f' % (default_timer() - start))

hotels = np.array(df['Share ID'])
weekdays = np.array(df['Date'].dt.weekday).reshape([-1, 1])
months = np.array(df['Date'].dt.month).reshape([-1, 1])
years = np.array(df['Date'].dt.year).reshape([-1, 1])
targets = np.array(df['Room Demand'])

hotel_names, hotels = np.unique(hotels, return_inverse=True)
hotels = hotels.reshape([-1, 1])

report = []

for i in range(removals):
	print('\nTraining, testing model (removal %d / %d)' % (i + 1, removals))
	
	# Randomly permute the data to remove sequence biasing.
	p = np.random.permutation(targets.shape[0])
	hotels, weekdays, months, years, targets = hotels[p], weekdays[p], months[p], years[p], targets[p]

	# Split the data into (training, test) subsets.
	split = int(0.8 * len(targets))

	train_features = [hotels[:split], years[:split], months[:split], weekdays[:split]]
	train_features = np.concatenate(train_features, axis=1)

	test_features = [hotels[split:], years[split:], months[split:], weekdays[split:]]
	test_features = np.concatenate(test_features, axis=1)

	train_targets = targets[:split]
	test_targets = targets[split:]

	print('Creating and training multi-layer perceptron regression model.')

	param_grid = {'hidden_layer_sizes' : [[512, 256, 128], [1024, 512, 256], [1024, 512, 256, 128]],
				  'alpha' : [1e-5, 5e-5]}

	model = GridSearchCV(MLPRegressor(), param_grid=param_grid, verbose=5, n_jobs=-1)
	model.fit(train_features, train_targets)

	print(); print('Best model hyper-parameters:', model.best_params_); print()

	model = model.best_estimator_

	train_score = model.score(train_features, train_targets)
	test_score = model.score(test_features, test_targets)

	train_predictions = model.predict(train_features)
	test_predictions = model.predict(test_features)

	train_mse = mean_squared_error(train_targets, train_predictions)
	test_mse = mean_squared_error(test_targets, test_predictions)

	np.save(os.path.join(predictions_path, 'train_targets_removals_%d.npy' % i), train_targets)
	np.save(os.path.join(predictions_path, 'train_predictions_removals_%d.npy' % i), train_predictions)

	np.save(os.path.join(predictions_path, 'test_targets_removals_%d.npy' % i), test_targets)
	np.save(os.path.join(predictions_path, 'test_predictions_removals_%d.npy' % i), test_predictions)
	
	print()
	print('*** Results after %d / %d removals ***' % (i, removals))
	print()
	print('Training MSE: %.0f' % train_mse)
	print('Training R^2: %.4f' % train_score)
	print()
	print('Test MSE: %.0f' % test_mse)
	print('Test R^2: %.4f' % test_score)
	print()
	
	# Calculate per-hotel test MSEs.
	per_hotel_test_mses = {}
	for idx, name in zip(hotels.ravel(), hotel_names):
		hotel_test_targs = test_targets[(hotels == idx)[split:].ravel()]
		hotel_test_preds = test_predictions[(hotels == idx)[split:].ravel()]
		per_hotel_test_mses[(name, idx)] = mean_squared_error(hotel_test_targs, hotel_test_preds)
	
	worst = max(list(per_hotel_test_mses.items()), key=lambda x : x[1])
	worst_mse, worst_hotel, worst_idx = worst[1], worst[0][0], worst[0][1]
	
	print('Worst MSE: %.2f' % worst_mse)
	print('Hotel with worst test MSE: %s' % worst_hotel)
	print('Index of hotel with worst test MSE: %d' % worst_idx)
	
	report.append([worst_hotel, worst_mse, train_mse, train_score, test_mse, test_score])

	# Remove offending hotel from data.
	hotel_names = hotel_names[hotel_names != worst_hotel]
	hotels, weekdays, months, years, targets = hotels[(hotels != worst_idx).ravel()], \
		weekdays[(hotels != worst_idx).ravel()], months[(hotels != worst_idx).ravel()], \
			years[(hotels != worst_idx).ravel()], targets[(hotels != worst_idx).ravel()]

# Save hotel removal report to disk.
df = pd.DataFrame(report, columns=['Hotel', 'Hotel Test MSE', 'Train MSE', 'Train R^2', 'Test MSE', 'Test R^2'])
df.to_csv(os.path.join(removals_path, 'removals.csv'))

print()

train_scores = []
test_scores = []
train_mses = []
test_mses = []

for i in range(trials):  # Run 5 independent realizations of training / test.
	print('\nTraining, testing model %d / %d' % (i + 1, trials))
	
	# Randomly permute the data to remove sequence biasing.
	p = np.random.permutation(targets.shape[0])
	hotels,  weekdays, months, years, targets = hotels[p], weekdays[p], months[p], years[p], targets[p]

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
print('*** Multiple averaged results after %d / %d removals ***' % (removals, removals))
print()
print('Mean, standard deviation of training MSE: %.0f $\pm$ %.0f' % (np.mean(train_mses), np.std(train_mses)))
print('Mean, standard deviation of training R^2: %.4f' % np.mean(train_scores))
print()
print('Mean, standard deviation of test MSE: %.0f $\pm$ %.0f' % (np.mean(test_mses), np.std(test_mses)))
print('Mean, standard deviation of test R^2: %.4f' % np.mean(test_scores))
print()
