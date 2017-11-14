import os
import csv
import timeit
import argparse
import xlsxwriter
import numpy as np
import pandas as pd
import multiprocessing as mp

from util import *
from contextlib import closing
from datetime import timedelta, date, datetime

output_path = os.path.join('..', 'data', 'daily_distributions')
plots_path = os.path.join('..', 'plots', 'daily')

for directory in [ output_path, plots_path ]:
	if not os.path.isdir(directory):
		os.makedirs(directory)


def plot_and_record_daily(start_date, end_date):
	workbook = xlsxwriter.Workbook(os.path.join(output_path, '_'.join([ coord_type, str(distance), \
							str(start_date), str(end_date) ]) + '.xlsx'), {'constant_memory': True})
	worksheet = workbook.add_worksheet()

	total_days = int((end_date - start_date).days)
	indices = np.zeros(total_days)

	for day_idx, date in enumerate(daterange(start_date, end_date)):
		worksheet.write(day_idx, 0, str(date))

	start = timeit.default_timer()
	for file in data_files:
		for idx, chunk in enumerate(pd.read_csv(os.path.join(data_path, file), \
									chunksize=100000, parse_dates=['Pick-up Time'])):
			
			print '- Chunk', idx, '| row', idx * 100000

			for idx, row in chunk.iterrows():
				day_idx = int((row['Pick-up Time'].date() - start_date).days)
				if day_idx >= 0 and day_idx <= total_days:
					worksheet.write(day_idx, indices[day_idx], ' '.join([row['Latitude'], row['Longitude']]))
					indices[day_idx] += 1

			print '(', 'time:', timeit.default_timer() - start, ')' 
			start = timeit.default_timer()


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
	data_path = os.path.join('..', 'data', '_'.join(['all_preprocessed', str(distance)]))

	# create scatterplots and record coordinates for each day (from start_date to end_date) for all hotels combined
	plot_and_record_daily(start_date, end_date)

	print '\n'
