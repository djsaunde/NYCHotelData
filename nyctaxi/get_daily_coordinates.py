from __future__ import print_function

import os
import csv
import dask
import timeit
import argparse
import xlsxwriter
import pandas as pd
import multiprocessing as mp

from util import *
from contextlib import closing
from datetime import timedelta, date, datetime

output_path = os.path.join('..', 'data', 'daily_distributions')

for directory in [ output_path ]:
	if not os.path.isdir(directory):
		os.makedirs(directory)


def get_daily(taxi_data, day):
	# Columns to read from pre-processed data.
	columns = ['Hotel Name', 'Distance From Hotel', 'Latitude', 'Longitude', 'Pick-up Time', 'Drop-off Time']

	print('\n*** Date:', day, '***\n')

	# get coordinates of new distance and time-constraint satisfying taxicab trips with nearby pick-ups
	for idx, key in enumerate(taxi_data):
		data = taxi_data[key][columns]

		# Get the pick-up (drop-off) coordinates of the trip which ended (began) near this each hotel
		current_coords = get_nearby_window(data, distance, datetime.combine(day, \
				datetime.min.time()), datetime.combine(day, datetime.max.time()))

		if idx == 0:
			coords = current_coords
		else:
			coords = all_coords.append(current_coords)

	# Convert dask dataframe into a xlsxwriter writeable data structure.
	print('Converting coordinate format for writing out to disk.')
	start = timeit.default_timer()
	coords = list(coords.compute())
	print('\n...It took', timeit.default_timer() - start, 'seconds to format the data.\n')

	print('There were', len(coords), 'satisfying rides on day', str(day))

	# Writing this day's satisfying coordinates to the .xlsx file
	print('Writing coordinates to a .csv file.')
	start = timeit.default_timer()

	with open(os.path.join(output_path, '_'.join([ '_'.join(taxi_data.keys()), str(distance), str(day) ])) + '.csv', 'w') as f:
		writer = csv.writer(f)
		writer.writerow(coords)

	print('\n...It took', timeit.default_timer() - start, 'seconds to write the coordinates to disk.\n')


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Recording satisfying coordinates in an Excel \
																file for a given window of time.')

	parser.add_argument('--day', type=int, nargs=3, default=[2013, 1, 1], help='The day on \
													which to look for satisfying coordinates.')
	parser.add_argument('--coord_type', type=str, default='pickups', help='The type of coordinates \
											to look for (one of "pickups", "dropoffs", or "both").')
	parser.add_argument('--distance', type=int, default=100, help='The distance (in feet) from hotels \
													for which to look for satisfying taxicab trips.')

	args = parser.parse_args()
	args = vars(args)

	# parse arguments and place them in local scope
	locals().update(args)

	if coord_type == 'pickups':
		data_files = [ 'destinations.csv' ]
	elif coord_type == 'dropoffs':
		data_files = [ 'starting_points.csv' ]
	elif coord_type == 'both':
		data_files = [ 'destinations.csv', 'starting_points.csv' ]

	day = date(*day)
	data_path = os.path.join('..', 'data', '_'.join(['all_preprocessed', str(distance)]))

	# Get dictionary of taxicab trip data based on `coord_type` argument.
	taxi_data = load_data(coord_type, data_files, data_path)

	# Record coordinates for each day (from start_date to end_date) for all hotels combined.
	get_daily(taxi_data, day)

	print()
