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
parser.add_argument('--end_date', type=int, nargs=3, default=[2016, 6, 30])
parser.add_argument('--metric', type=str, default='rel_diffs')
parser.add_argument('--nrows', type=int, default=None)

locals().update(vars(parser.parse_args()))

fname = '_'.join(map(str, [distance, start_date[0], start_date[1], start_date[2], end_date[0], end_date[1], end_date[2], metric]))

start_date, end_date = date(*start_date), date(*end_date)

data_path = os.path.join('..', 'data', 'all_preprocessed_%d' % distance)
taxi_occupancy_path = os.path.join('..', 'data', 'taxi_occupancy', fname)
predictions_path = os.path.join('..', 'data', 'taxi_lr_predictions', fname)

for path in [data_path, taxi_occupancy_path, predictions_path]:
	if not os.path.isdir(path):
		os.makedirs(path)

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

taxi_rides = pd.read_csv(filename, header=0, usecols=usecols, nrows=nrows)

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

hotels = np.array(df['Hotel Name'])
trips = np.array(df['No. Nearby Trips']).reshape([-1, 1])
weekdays = np.array(df['Date'].apply(date.weekday)).reshape([-1, 1])
dates = np.array(df['Date'].apply(str)).reshape([-1, 1])
targets = np.array(df['Room Demand'])

_, hotels = np.unique(hotels, return_inverse=True)
hotels = hotels.reshape([-1, 1])

_, dates = np.unique(dates, return_inverse=True)
dates = dates.reshape([-1, 1])

features = np.concatenate([hotels, trips, hotels * trips, trips ** 2 * hotels,
						trips ** 2, dates, dates * trips, trips ** 3 * hotels,
						trips ** 3, weekdays, trips * weekdays], axis=1)

model = LinearRegression().fit(features, targets)
score = model.score(features, targets)

predictions = model.predict(features)
mse = mean_squared_error(targets, predictions)

np.save(os.path.join(predictions_path, 'targets.npy'), targets)
np.save(os.path.join(predictions_path, 'predictions.npy'), predictions)

print('\n'); print('R^2:', score)
print('MSE:', mse); print('\n')