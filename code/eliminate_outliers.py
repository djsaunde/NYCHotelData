from __future__ import print_function, division

import os
import sys
import timeit
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from util                 import *
from timeit               import default_timer
from datetime             import date
from scipy.stats          import entropy
from mpl_toolkits.mplot3d import Axes3D


def objective(capacity_distros, taxi_distribution, metric):
	'''
	The objective function captures how "close" we are to a perfect match between the
	distribution of hotel capacities (occupancies?) and the distribution of taxicab
	coordinates (nearby pickups / dropoffs / both).
	'''
	capacity_distros = np.array(list(capacity_distros.values()))
	taxi_distribution = np.array(list(taxi_distribution.values()))

	if metric == 'relative_entropy':
		return entropy(capacity_distros, taxi_distribution)
	elif metric == 'abs_diffs':
		return np.sum(np.abs(capacity_distros - taxi_distribution))
	elif metric == 'rel_diffs':
		return np.sum(taxi_distribution / capacity_distros)
	elif metric == 'inverse_weighted_abs_diffs':
		return np.sum(np.abs(capacity_distros - taxi_distribution) / capacity_distros)
	elif metric == 'weighted_abs_diffs':
		return np.sum(np.abs(capacity_distros - taxi_distribution) * capacity_distros)


parser = argparse.ArgumentParser()
parser.add_argument('--distance', default=25, type=int)
parser.add_argument('--trip_type', default='pickups', type=str)
parser.add_argument('--start_date', type=int, nargs=3, default=[2013, 1, 1])
parser.add_argument('--end_date', type=int, nargs=3, default=[2016, 1, 1])
parser.add_argument('--metric', type=str, default='rel_diffs')
parser.add_argument('--plot', dest='plot', action='store_true')
parser.add_argument('--no_plot', dest='plot', action='store_false')
parser.set_defaults(plot=False)

locals().update(vars(parser.parse_args()))

fname = '_'.join(map(str, [distance, start_date[0], start_date[1], start_date[2], end_date[0], end_date[1], end_date[2], metric]))

plots_path = os.path.join('..', 'plots', 'outlier_elimination', fname)
reports_path = os.path.join('..', 'data', 'outlier_reports')

for path in [reports_path, plots_path]:
	if not os.path.isdir(path):
		os.makedirs(path)

start_date, end_date = date(*start_date), date(*end_date)

# Load daily capacity data.
print('\nLoading daily per-hotel capacity data.'); start = default_timer()

capacities = pd.read_csv(os.path.join('..', 'data', 'Unmasked Daily Capacity.csv'), index_col=False)
capacities['Date'] = pd.to_datetime(capacities['Date'], format='%Y-%m-%d')
capacities = capacities.loc[(capacities['Date'] >= start_date) & (capacities['Date'] <= end_date)]

print('Time: %.4f' % (default_timer() - start))

# Create a dictionary which contains per-hotel daily capacity data.
print('\nOrganizing data into per-hotel, per-day dictionary structure.'); start = default_timer()

hotel_capacities = {}
for hotel in capacities['Share ID'].unique():
	hotel_capacities[hotel] = sum([row['Room Demand'] for (_, row) in capacities.loc[capacities['Share ID'] == hotel].iterrows()])

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

taxi_rides = pd.read_csv(filename, header=0, usecols=usecols)

taxi_rides['Hotel Name'] = taxi_rides['Hotel Name'].apply(str.strip)
taxi_rides['Pick-up Time'] = pd.to_datetime(taxi_rides['Pick-up Time'], format='%Y-%m-%d')
taxi_rides['Drop-off Time'] = pd.to_datetime(taxi_rides['Drop-off Time'], format='%Y-%m-%d')
taxi_rides = taxi_rides.loc[(taxi_rides['Pick-up Time'] >= start_date) & \
								(taxi_rides['Drop-off Time'] <= end_date)]

rides_by_hotel = {}
for hotel in taxi_rides['Hotel Name'].unique():
	hotel_rows = taxi_rides.loc[taxi_rides['Hotel Name'] == hotel]
	rides_by_hotel[hotel] = np.array([row['Distance From Hotel'] for (_, row) in hotel_rows.iterrows()])

print('Time: %.4f' % (default_timer() - start))

capacity_distribution = {}
for (hotel, capacity) in sorted(hotel_capacities.items()):
	capacity_distribution[hotel] = capacity / sum(hotel_capacities.values())

taxi_distribution = {hotel : n_trips / sum(rides_by_hotel.values()) for (hotel, n_trips) in sorted(rides_by_hotel.items())}

objective_evals = objective(capacity_distribution, taxi_distribution, metric)

print(objective_evals)