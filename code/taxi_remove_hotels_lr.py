from __future__ import print_function

import os
import sys
import argparse
import numpy as  np
import pandas as pd

from datetime             import date
from timeit               import default_timer
from sklearn.linear_model import LinearRegression
from sklearn.metrics      import mean_squared_error

parser = argparse.ArgumentParser()
parser.add_argument('--distance', default=25, type=int)
parser.add_argument('--trip_type', default='pickups', type=str)
parser.add_argument('--start_date', type=int, nargs=3, default=[2013, 1, 1])
parser.add_argument('--end_date', type=int, nargs=3, default=[2015, 1, 1])
parser.add_argument('--trials', type=int, default=5)
parser.add_argument('--removals', type=int, default=25)

locals().update(vars(parser.parse_args()))

fname = '_'.join(map(str, [distance, start_date[0], start_date[1], start_date[2], end_date[0], end_date[1], end_date[2]]))

start_date, end_date = date(*start_date), date(*end_date)

data_path = os.path.join('..', 'data', 'all_preprocessed_%d' % distance)
taxi_occupancy_path = os.path.join('..', 'data', 'taxi_occupancy', fname)
predictions_path = os.path.join('..', 'data', 'taxi_lr_removal_predictions', fname)
removals_path = os.path.join('..', 'data', 'taxi_lr_removals', fname)

for path in [data_path, taxi_occupancy_path, predictions_path, removals_path]:
	if not os.path.isdir(path):
		os.makedirs(path)

is_data_file = os.path.isfile(os.path.join(taxi_occupancy_path, 'Taxi and occupancy data.csv'))
is_counts_file = os.path.isfile(os.path.join(taxi_occupancy_path, 'Taxi and occupancy counts.csv'))
		
if not is_counts_file and not is_data_file:
	# Load daily capacity data.
	print('\nLoading daily per-hotel capacity data.'); start = default_timer()

	occupancy = pd.read_csv(os.path.join('..', 'data', 'Unmasked Daily Capacity.csv'), index_col=False)
	occupancy['Date'] = pd.to_datetime(occupancy['Date'], format='%Y-%m-%d')
	occupancy = occupancy.loc[(occupancy['Date'] >= start_date) & (occupancy['Date'] <= end_date)]
	occupancy['Date'] = occupancy['Date'].dt.date
	occupancy = occupancy.rename(index=str, columns={'Share ID': 'Hotel Name'})
	occupancy = occupancy.drop('Unnamed: 0', axis=1)

	print('Time: %.4f' % (default_timer() - start))

	# Load preprocessed data according to command-line "distance" parameter.
	print('\nReading in the pre-processed taxicab data.'); start = default_timer()

	usecols = ['Hotel Name', 'Pick-up Time', 'Drop-off Time', 'Distance From Hotel']
	if trip_type == 'pickups':
		filename = os.path.join(data_path, 'destinations.csv')
	elif trip_type == 'dropoffs':
		filename = os.path.join(data_path, 'starting_points.csv')
	else:
		raise Exception('Expecting one of "pickups" or "dropoffs" for command-line argument "trip_type".')

	taxi_rides = pd.read_csv(filename, header=0, usecols=usecols)

	taxi_rides['Hotel Name'] = taxi_rides['Hotel Name'].apply(str.strip)
	taxi_rides['Pick-up Time'] = pd.to_datetime(taxi_rides['Pick-up Time'], format='%Y-%m-%d')
	taxi_rides['Drop-off Time'] = pd.to_datetime(taxi_rides['Drop-off Time'], format='%Y-%m-%d')
	taxi_rides = taxi_rides.loc[(taxi_rides['Pick-up Time'] >= start_date) & \
									(taxi_rides['Drop-off Time'] <= end_date)]
	taxi_rides['Date'] = taxi_rides['Pick-up Time'].dt.date
	taxi_rides = taxi_rides.drop(['Pick-up Time', 'Drop-off Time'], axis=1)

	print('Time: %.4f' % (default_timer() - start))

	# Build the dataset of ((hotel, taxi density), occupancy) input, output pairs.
	print('\nMerging dataframes on Date and Hotel Name attributes.'); start = default_timer()

	df = pd.merge(occupancy, taxi_rides, on=['Date', 'Hotel Name'])

	print('Time: %.4f' % (default_timer() - start))

	# Save merged occupancy and taxi data to disk.
	print('\nSaving merged dataframes to disk.'); start = default_timer()

	df.to_csv(os.path.join(taxi_occupancy_path, 'Taxi and occupancy data.csv'))
	
	print('Time: %.4f' % (default_timer() - start))
	
	# Count number of rides per hotel and date.
	df = df.groupby(['Hotel Name', 'Date', 'Room Demand']).count().reset_index()
	df = df.rename(index=str, columns={'Distance From Hotel': 'No. Nearby Trips'})
	df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

	# Save occupancy and taxi counts to disk.
	print('\nSaving counts to disk.'); start = default_timer()

	df.to_csv(os.path.join(taxi_occupancy_path, 'Taxi and occupancy counts.csv'))
    
	print('Time: %.4f' % (default_timer() - start))

elif is_data_file and not is_counts_file:
	# Load merged occupancy and taxi data to disk.
	print('\nLoading merged taxi and occupancy dataframes from disk.'); start = default_timer()

	df = pd.read_csv(os.path.join(taxi_occupancy_path, 'Taxi and occupancy data.csv'))
	
	print('Time: %.4f' % (default_timer() - start))
	
	# Count number of rides per hotel and date.
	df = df.groupby(['Hotel Name', 'Date', 'Room Demand']).count().reset_index()
	df = df.rename(index=str, columns={'Distance From Hotel': 'No. Nearby Trips'})
	df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

	# Save occupancy and taxi counts to disk.
	print('\nSaving counts to disk.'); start = default_timer()

	df.to_csv(os.path.join(taxi_occupancy_path, 'Taxi and occupancy counts.csv'))
    
	print('Time: %.4f' % (default_timer() - start))

else:
	# Load merged occupancy and taxi data from disk.
	print('\nLoading occupancy and taxi counts from disk.'); start = default_timer()
	
	df = pd.read_csv(os.path.join(taxi_occupancy_path, 'Taxi and occupancy counts.csv'))
	df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
	
	print('Time: %.4f' % (default_timer() - start))

hotels = np.array(df['Hotel Name'])
trips = np.array(df['No. Nearby Trips']).reshape([-1, 1])
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
	hotels, trips, weekdays, months, years, targets = hotels[p], trips[p], weekdays[p], months[p], years[p], targets[p]

	# Split the data into (training, test) subsets.
	split = int(0.8 * len(targets))

	train_features = [hotels[:split], trips[:split], years[:split], months[:split], weekdays[:split]]
	train_features = np.concatenate(train_features, axis=1)

	test_features = [hotels[split:], trips[split:], years[split:], months[split:], weekdays[split:]]
	test_features = np.concatenate(test_features, axis=1)

	train_targets = targets[:split]
	test_targets = targets[split:]

	print('Creating and training OLS regression model.')

	model = LinearRegression().fit(train_features, train_targets)

	print('Training complete. Getting predictions and calculating R^2, MSE.')

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
	hotels, trips, weekdays, months, years, targets = hotels[(hotels != worst_idx).ravel()], \
			trips[(hotels != worst_idx).ravel()], weekdays[(hotels != worst_idx).ravel()], \
			months[(hotels != worst_idx).ravel()], years[(hotels != worst_idx).ravel()], \
												targets[(hotels != worst_idx).ravel()]
				
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
	hotels, trips, weekdays, months, years, targets = hotels[p], trips[p], weekdays[p], months[p], years[p], targets[p]

	# Split the data into (training, test) subsets.
	split = int(0.8 * len(targets))

	train_features = [hotels[:split], trips[:split], years[:split], months[:split], weekdays[:split]]
	train_features = np.concatenate(train_features, axis=1)

	test_features = [hotels[split:], trips[split:], years[split:], months[split:], weekdays[split:]]
	test_features = np.concatenate(test_features, axis=1)

	train_targets = targets[:split]
	test_targets = targets[split:]

	print('Creating and training OLS regression model.')

	model = LinearRegression().fit(train_features, train_targets)

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
print('*** Multiple averaged results after %d / %d removals ***' % (trials, trials))
print()
print('Mean, standard deviation of training MSE: %.0f $\pm$ %.0f' % (np.mean(train_mses), np.std(train_mses)))
print('Mean, standard deviation of training R^2: %.4f' % np.mean(train_scores))
print()
print('Mean, standard deviation of test MSE: %.0f $\pm$ %.0f' % (np.mean(test_mses), np.std(test_mses)))
print('Mean, standard deviation of test R^2: %.4f' % np.mean(test_scores))
print()