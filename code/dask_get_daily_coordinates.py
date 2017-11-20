import os
import csv
import dask
import timeit
import argparse
import xlsxwriter
import pandas as pd
import multiprocessing as mp

from dask_util import *
from contextlib import closing
from multiprocessing import cpu_count
from datetime import timedelta, date, datetime

output_path = os.path.join('..', 'data', 'daily_distributions')
plots_path = os.path.join('..', 'plots', 'daily')

for directory in [ output_path, plots_path ]:
	if not os.path.isdir(directory):
		os.makedirs(directory)

print 'Number of CPUs:', cpu_count()


def plot_and_record_daily(taxi_data, start_date, end_date):
	workbook = xlsxwriter.Workbook(os.path.join(output_path, '_'.join([ '_'.join(taxi_data.keys()), \
				str(distance), str(start_date), str(end_date) ]) + '.xlsx'), {'constant_memory': True})
	worksheet = workbook.add_worksheet()

	# set up main loop: loop through each day from start_date to end_date
	for day_idx, date in enumerate(daterange(start_date, end_date)):

		print '\n*** Date:', date, '***\n'

		# get coordinates of new distance and time-constraint satisfying taxicab trips with nearby pick-ups
		for idx, key in enumerate(taxi_data):
			data = taxi_data[key]

			# Get the pick-up (drop-off) coordinates of the trip which ended (began) near this each hotel
			current_coords = get_nearby_window(data, distance, datetime.combine(date, \
					datetime.min.time()), datetime.combine(date, datetime.max.time()))

			if idx == 0:
				coords = current_coords
			else:
				coords = all_coords.append(current_coords)

		# Convert dask dataframe into a xlsxwriter writeable data structure.
		print 'Converting coordinate format for writing out to disk.'
		start = timeit.default_timer()
		coords = coords.compute()
		print '\n...It took', timeit.default_timer() - start, 'seconds to format the data.\n'

		# Writing this day's satisfying coordinates to the .xlsx file
		print 'Writing coordinates to a .xlsx file.'
		start = timeit.default_timer()
		worksheet.write(day_idx, 0, str(date))
		for data_idx in coords.index.values:
			worksheet.write(day_idx, data_idx + 1, coords[data_idx])
		
		print '\n...It took', timeit.default_timer() - start, 'seconds to write the coordinates to disk.\n'


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Making scatterplots and recording satisfying coordinates in a .csv file for a given window of time.')

	parser.add_argument('--start_date', type=int, nargs=3, default=[2013, 1, 1], help='The day on which to start looking for satisfying coordinates.')
	parser.add_argument('--end_date', type=int, nargs=3, default=[2013, 1, 7], help='The day on which to stop looking for satisfying coordinates.')
	parser.add_argument('--coord_type', type=str, default='pickups', help='The type of coordinates to look for (one of "pickups", "dropoffs", or "both").')
	parser.add_argument('--distance', type=int, default=100, help='The distance (in feet) from hotels for which to look for satisfying taxicab trips.')
	parser.add_argument('--n_jobs', type=int, default=4, help='The number of CPU cores to use in processing the taxicab data.')
	parser.add_argument('--make_scatter_plots', dest='plot', action='store_true')
	parser.add_argument('--no_make_scatter_plots', dest='plot', action='store_false')
	parser.set_defaults(plot=False)

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
	# data_path = os.path.join('..', 'data', '_'.join(['all_preprocessed', str(distance)]))
	data_path = os.path.join('..', 'data', '_'.join(['all_preprocessed', str(distance)]))

	# get dictionary of taxicab trip data based on `coord_type` argument
	taxi_data = load_data(coord_type, data_files, data_path)

	# create scatterplots and record coordinates for each day (from start_date to end_date) for all hotels combined
	plot_and_record_daily(taxi_data, start_date, end_date)

	print '\n'
