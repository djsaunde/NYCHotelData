from __future__ import print_function, division

import os
import sys
import timeit
import argparse
import numpy as np
import pandas as pd

from util        import *
from datetime    import date
from scipy.stats import entropy
from timeit      import default_timer


def divergence(occupancy, taxi, metric):
	'''
	Compute some divergence function to capture how "close" we are to a perfect
	match between the distribution of hotel occupancy and the distribution of
	taxicab coordinates (nearby pickups, dropoffs, or both).
	'''
	occupancy = np.array(list(occupancy.values()))
	taxi = np.array(list(taxi.values()))

	if metric == 'relative_entropy':
		return entropy(occupancy, taxi)
	elif metric == 'abs_diffs':
		return np.abs(occupancy - taxi)
	elif metric == 'rel_diffs':
		return taxi / occupancy
	elif metric == 'inverse_weighted_abs_diffs':
		return np.abs(occupancy - taxi) / occupancy
	elif metric == 'weighted_abs_diffs':
		return np.abs(occupancy - taxi) * occupancy


parser = argparse.ArgumentParser()
parser.add_argument('--distance', default=25, type=int)
parser.add_argument('--trip_type', default='pickups', type=str)
parser.add_argument('--start_date', type=int, nargs=3, default=[2014, 1, 1])
parser.add_argument('--end_date', type=int, nargs=3, default=[2016, 6, 30])
parser.add_argument('--metric', type=str, default='rel_diffs')
parser.add_argument('--nrows', type=int, default=None)

locals().update(vars(parser.parse_args()))

fname = '_'.join(map(str, [distance, start_date[0], start_date[1], start_date[2], end_date[0], end_date[1], end_date[2], metric]))

start_date, end_date = date(*start_date), date(*end_date)

# Load daily capacity data.
print('\nLoading daily per-hotel capacity data.'); start = default_timer()

occupancy = pd.read_csv(os.path.join('..', 'data', 'Unmasked Daily Capacity.csv'), index_col=False)
occupancy['Date'] = pd.to_datetime(occupancy['Date'], format='%Y-%m-%d')
occupancy = occupancy.loc[(occupancy['Date'] >= start_date) & (occupancy['Date'] <= end_date)]

print('Time: %.4f' % (default_timer() - start))

# Load preprocessed data according to command-line "distance" parameter.
print('\nReading in the pre-processed taxicab data.'); start = default_timer()

usecols = ['Share ID', 'Hotel Name', 'Distance From Hotel', 'Latitude', 'Longitude', 'Pick-up Time',
							'Drop-off Time', 'Passenger Count', 'Trip Distance', 'Fare Amount']
if trip_type == 'pickups':
	filename = os.path.join('..', 'data', 'all_preprocessed_%d' % distance, 'destinations.csv')
elif trip_type == 'dropoffs':
	filename = os.path.join('..', 'data', 'all_preprocessed_%d' % distance, 'starting_points.csv')
else:
	raise Exception('Expecting one of "pickups" or "dropoffs" for command-line argument "trip_type".')

taxi_rides = pd.read_csv(filename, header=0, usecols=usecols, nrows=nrows)

taxi_rides['Hotel Name'] = taxi_rides['Hotel Name'].apply(str.strip)
taxi_rides['Pick-up Time'] = pd.to_datetime(taxi_rides['Pick-up Time'], format='%Y-%m-%d')
taxi_rides['Drop-off Time'] = pd.to_datetime(taxi_rides['Drop-off Time'], format='%Y-%m-%d')
taxi_rides = taxi_rides.loc[(taxi_rides['Pick-up Time'] >= start_date) & \
								(taxi_rides['Drop-off Time'] <= end_date)]

print('Time: %.4f' % (default_timer() - start))

hotels = set(occupancy['Share ID'].unique()) & set(taxi_rides['Hotel Name'].unique())

# Calculate number of taxi rides per hotel.
print('\nComputing number of rides per hotel.'); start = default_timer()

rides_by_hotel = {}
for hotel in hotels:
	hotel_rows = taxi_rides.loc[taxi_rides['Hotel Name'] == hotel]
	rides = [row['Distance From Hotel'] for (_, row) in hotel_rows.iterrows()]
	rides_by_hotel[hotel] = len(rides)

print('Time: %.4f' % (default_timer() - start))

# Create a dictionary which contains per-hotel daily occupancy data.
print('\nOrganizing data into per-hotel, per-day dictionary structure.'); start = default_timer()

occupancies = {}
for hotel in hotels:
	occupancies[hotel] = sum([row['Room Demand'] for (_, row) in occupancy.loc[occupancy['Share ID'] == hotel].iterrows()])

print('Time: %.4f\n' % (default_timer() - start))

occupancy_distribution = {}
for (hotel, capacity) in sorted(occupancies.items()):
	occupancy_distribution[hotel] = capacity / sum(occupancies.values())

total_trips = sum(rides_by_hotel.values())
for (hotel, n_trips) in sorted(rides_by_hotel.items()):
	taxi_distribution = {hotel : n_trips / total_trips for (hotel, n_trips) in sorted(rides_by_hotel.items())}

# Evaluate objective.
d_evaluation = divergence(occupancy_distribution, taxi_distribution, metric)
d_evaluation, hotels = zip(*sorted(zip(d_evaluation, hotels), key=lambda x : x[0]))  # Get order statistics

d_evaluation = np.array(d_evaluation)
hotels = np.array(hotels)

# Calculate median and interquartile range (IQR).
median = np.median(d_evaluation); print('Objective evaluation median:', median)
iqr = np.subtract(*np.percentile(d_evaluation, [75, 25])); print('IQR:', iqr)
low, high = median - 1.5 * iqr, median + 1.5 * iqr; print('Low:', low, '; High:', high)

# Remove all hotels with divergence outside of
# the mean plus or minus 1.5 times the IQR.
keep_idxs = (d_evaluation <= high) & (d_evaluation >= low) 
d_evaluation = d_evaluation[keep_idxs]
hotels = hotels[keep_idxs]

# Save the dataset with outlier hotels
# removed to disk for further processing.
taxi_rides = taxi_rides[taxi_rides['Hotel Name'].isin(hotels)]

data_path = os.path.join('..', 'data', 'removed_outliers')
if not os.path.isdir(data_path):
	os.makedirs(data_path)

taxi_rides.to_csv(os.path.join(data_path, fname + '.csv'))

print()