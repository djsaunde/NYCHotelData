import os
import csv
import dask
import timeit
import argparse
import xlsxwriter
import pandas as pd
import dask.bag as db
import multiprocessing as mp

from util import *
from glob import glob
from tqdm import tqdm
from contextlib import closing
from datetime import timedelta, date, datetime

output_path = os.path.join('..', 'data', 'daily_distributions')

for directory in [ output_path ]:
	if not os.path.isdir(directory):
		os.makedirs(directory)


def get_daily(taxi_data, start_date, end_date, frac_samples):
	workbook = xlsxwriter.Workbook(os.path.join(output_path, '_'.join([ '_'.join(taxi_data.keys()), \
				str(distance), str(start_date), str(end_date) ]) + '.xlsx'), {'constant_memory': True})
	worksheet = workbook.add_worksheet()

	columns = ['Hotel Name', 'Distance From Hotel', 'Latitude', 'Longitude', 'Pick-up Time', 'Drop-off Time']

	# set up main loop: loop through each day from start_date to end_date
	daily_coordinates = []
	for day_idx, date in enumerate(daterange(start_date, end_date)):

		print '\n*** Date:', date, '***\n'

		# get coordinates of new distance and time-constraint satisfying taxicab trips with nearby pick-ups
		for idx, key in enumerate(taxi_data):
			data = taxi_data[key][columns]

			# Get the pick-up (drop-off) coordinates of the trip which ended (began) near this each hotel
			current_coords = get_nearby_window(data, distance, datetime.combine(date, \
						datetime.min.time()), datetime.combine(date, datetime.max.time()))

			if idx == 0:
				coords = current_coords
			else:
				coords = all_coords.append(current_coords)

		daily_coordinates.append(coords)

	print '\n'
	for coords in tqdm(daily_coordinates):
		fpath = os.path.join('..', 'data', 'daily_distributions', 'daily_coords_%d.*.csv' % distance)
		coords.to_csv(fpath); fnames = glob(fpath)
		
		with open(os.path.join(output_path, '_'.join([ '_'.join(taxi_data.keys()), \
				str(distance), str(start_date), str(end_date) ]) + '.csv'), 'w') as out_file:
		    for fname in fnames:
		        with open(fname) as file:
		            out_file.write(file.read())

	# Create a dask Bag (parallel Python objects), then make them
	# a list of delayed objects, to be written out to disk.
	# to_write = db.from_sequence(daily_coordinates)

	# print '\n'
	# for obj in to_write:
	# 	print obj.compute()

		# # Convert dask dataframe into a xlsxwriter writeable data structure.
		# print 'Converting coordinate format for writing out to disk.'
		# start = timeit.default_timer()
		# coords = coords.compute()
		# print '\n...It took', timeit.default_timer() - start, 'seconds to format the data.\n'

		# print 'There were', len(coords), 'satisfying rides on day', str(date)

		# # Writing this day's satisfying coordinates to the .xlsx file
		# print 'Writing coordinates to a .xlsx file.'
		# start = timeit.default_timer()
		# worksheet.write(day_idx, 0, str(date))

		# idx = 1
		# for data_idx in coords.index.values:
		# 	if not type(coords[data_idx]) == pd.core.series.Series:
		# 		worksheet.write(day_idx, idx, coords[data_idx])
		# 		idx += 1

		# print '\n...It took', timeit.default_timer() - start, 'seconds to write the coordinates to disk.\n'


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Recording satisfying coordinates in an Excel \
																file for a given window of time.')

	parser.add_argument('--start_date', type=int, nargs=3, default=[2013, 1, 1], help='The day on \
												which to start looking for satisfying coordinates.')
	parser.add_argument('--end_date', type=int, nargs=3, default=[2013, 1, 7], help='The day on \
												which to stop looking for satisfying coordinates.')
	parser.add_argument('--coord_type', type=str, default='pickups', help='The type of coordinates \
											to look for (one of "pickups", "dropoffs", or "both").')
	parser.add_argument('--distance', type=int, default=100, help='The distance (in feet) from hotels \
													for which to look for satisfying taxicab trips.')
	parser.add_argument('--n_jobs', type=int, default=4, help='The number of CPU cores to use in \
																	processing the taxicab data.')
	parser.add_argument('--frac_samples', type=int, default=0.1, help='The fraction of samples to take per day.')

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

	start_date, end_date = date(*start_date), date(*end_date)
	data_path = os.path.join('..', 'data', '_'.join(['all_preprocessed', str(distance)]))

	# Get dictionary of taxicab trip data based on `coord_type` argument.
	taxi_data = load_data(coord_type, data_files, data_path)

	# Record coordinates for each day (from start_date to end_date) for all hotels combined.
	get_daily(taxi_data, start_date, end_date, frac_samples)

	print '\n'
