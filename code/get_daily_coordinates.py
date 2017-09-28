import os
import csv
import argparse
import xlsxwriter
import pandas as pd
import multiprocessing as mp

from util import *
from contextlib import closing
from datetime import timedelta, date, datetime

data_path = os.path.join('..', 'data', 'preprocessed_100')
output_path = os.path.join('..', 'data', 'daily_distributions')
plots_path = os.path.join('..', 'plots', 'daily')

for directory in [ output_path, plots_path ]:
	if not os.path.isdir(directory):
		os.makedirs(directory)


def plot_and_record_daily(taxi_data, start_date, end_date):
	workbook = xlsxwriter.Workbook(os.path.join(output_path, '_'.join([ '_'.join(taxi_data.keys()), \
						str(distance), str(start_date), str(end_date) ]) + '.xlsx'), {'constant_memory': True})
	worksheet = workbook.add_worksheet()

	# set up main loop: loop through each day from start_date to end_date
	for day_idx, date in enumerate(daterange(start_date, end_date)):

		print date

		# get coordinates of new distance and time-constraint satisfying taxicab trips with nearby pick-ups
		all_coords = {}
		for key in taxi_data.keys():
			data = taxi_data[key]

			# Get a list of the names of the hotels we aim to plot distributions for (removing NaNs)
			hotel_names = [ hotel_name for hotel_name in data['Hotel Name'].unique() if not pd.isnull(hotel_name) ]

			# Get the pick-up (drop-off) coordinates of the trip which ended (began) near this each hotel
			if key == 'pickups':
				with closing(mp.Pool(n_jobs)) as pool:
					coords = pool.map(get_nearby_pickups_window, [ (data.loc[data['Hotel Name'] == hotel_name], distance, 
						datetime.combine(date, datetime.min.time()), datetime.combine(date, datetime.max.time())) for hotel_name in hotel_names ])
			elif key == 'dropoffs':
				with closing(mp.Pool(n_jobs)) as pool:
					coords = pool.map(get_nearby_dropoffs_window, [ (data.loc[data['Hotel Name'] == hotel_name], distance, 
						datetime.combine(date, datetime.min.time()), datetime.combine(date, datetime.max.time())) for hotel_name in hotel_names ])
			
			coords = { hotel_name : coord for (hotel_name, coord) in zip(hotel_names, coords) }

			print 'Total satisfying nearby', key, ':', sum([single_hotel_coords.shape[1] \
												for single_hotel_coords in coords.values()]), '/', len(data), '\n'
			
			print 'Satisfying nearby', key, 'by hotel:'
			for name in coords:
				print '-', name, ':', coords[name].shape[1], 'satisfying taxicab rides'

			print '\n'

			all_coords.update(coords)

		coords = all_coords

		directory = '_'.join([ '_'.join(taxi_data.keys()), str(distance), str(datetime.combine(date, datetime.min.time())), str(datetime.combine(date, datetime.max.time())) ])

		combined_coords = (np.concatenate([ coords[hotel_name][0] for hotel_name in coords.keys() ]), \
									np.concatenate([ coords[hotel_name][1] for hotel_name in coords.keys() ]))

		# Plot a scatterplot for the satisfying coordinates of all hotels combined.
		if plot:
			plot_arcgis_nyc_scatter_plot(combined_coords, '_'.join([ '_'.join(taxi_data.keys()), str(distance), str(datetime.combine(date, datetime.min.time())), 
					str(datetime.combine(date, datetime.max.time())) ]), os.path.join(plots_path, directory), title='All Hotels on ' + str(date))

		combined_coords = [ (' '.join([str(lat), str(lon)])) for (lat, lon) in \
				zip(list(combined_coords[0]), list(combined_coords[1])) if (lat, lon) != (0.0, 0.0) ]

		# Writing this day's satisfying coordinates to a .csv file
		worksheet.write(day_idx, 0, str(date))
		for data_idx in xrange(len(combined_coords)):
			worksheet.write(day_idx, data_idx + 1, combined_coords[data_idx])

		del combined_coords, coords, all_coords


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

	# get dictionary of taxicab trip data based on `coord_type` argument
	taxi_data = load_data(coord_type, data_files, data_path)

	# create scatterplots and record coordinates for each day (from start_date to end_date) for all hotels combined
	plot_and_record_daily(taxi_data, start_date, end_date)

	print '\n'
