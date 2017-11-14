import os
import argparse
import numpy as np
import pandas as pd

from util import *

records_path = os.path.join('..', 'records')
data_path = os.path.join('..', 'data', 'preprocessed_100')

if not os.path.exists(records_path):
	os.makedirs(records_path)


def record_daily(taxi_data, start_date, end_date):
	for idx, date in enumerate(daterange(start_date, end_date)):
		start_time = timeit.default_timer()

		# get coordinates of new distance and time-constraint satisfying taxicab trips with nearby pick-ups
		all_coords = {}
		for key in taxi_data.keys():
			data = taxi_data[key]

			# Create a number of CPU workers equal to the number of hotels in the data
			pool = mp.Pool(n_jobs)

			# Get a list of the names of the hotels we aim to plot distributions for (removing NaNs)
			hotel_names = [ hotel_name for hotel_name in data['Hotel Name'].unique() if not pd.isnull(hotel_name) ]

			# Get the pick-up (drop-off) coordinates of the trip which ended (began) near this each hotel
			if key == 'pickups':
				coords = pool.map(get_nearby_pickups_window, [ (data.loc[data['Hotel Name'] == hotel_name], distance, 
					datetime.combine(date, datetime.min.time()), datetime.combine(date, datetime.max.time())) for hotel_name in hotel_names ])
			elif key == 'dropoffs':
				coords = pool.map(get_nearby_dropoffs_window, [ (data.loc[data['Hotel Name'] == hotel_name], distance, 
					datetime.combine(date, datetime.min.time()), datetime.combine(date, datetime.max.time())) for hotel_name in hotel_names ])
			
			coords = { hotel_name : coord for (hotel_name, coord) in zip(hotel_names, coords) }

			print 'Total satisfying nearby', key, ':', sum([single_hotel_coords.shape[1] \
												for single_hotel_coords in coords.values()]), '/', len(data), '\n'
			
			print 'Satisfying nearby', key, 'by hotel:'
			for name in coords:
				print '-', name, ':', coords[name].shape[1], 'satisfying taxicab rides'

			all_coords.update(coords)

		coords = all_coords

		print '\nIt took', timeit.default_timer() - start_time, 'seconds to find all criteria-satifying taxicab trips (for day ' + str(date) + ').\n'

		directory = '_'.join([ '_'.join(taxi_data.keys()), str(distance), str(start_date), str(end_date) ])

		combined_coords = (np.concatenate([ coords[hotel_name][0] for hotel_name in coords.keys() ]), \
									np.concatenate([ coords[hotel_name][1] for hotel_name in coords.keys() ]))

		# Plot a scatterplot for the satisfying coordinates of all hotels combined.
		plot_arcgis_nyc_scatter_plot(combined_coords, '_'.join([ '_'.join(taxi_data.keys()), str(distance), str(datetime.combine(date, datetime.min.time())), 
				str(datetime.combine(date, datetime.max.time())) ]), os.path.join(plots_path, directory), title='All Hotels')

		combined_coords = [ (lat, lon) for (lat, lon) in zip(list(combined_coords[0]), list(combined_coords[1])) ]


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--distance', type=int, default=100)
	parser.add_argument('--start_date', type=int, nargs=3, default=[2013, 1, 1])
	parser.add_argument('--end_date', type=int, nargs=3, default=[2013, 2, 1])
	parser.add_argument('--coord_type', type=str, default='pickups')

	args = parser.parse_args()
	args = vars(args)

	locals().update(args)

	if coord_type == 'pickups':
		data_files = [ 'destinations.csv' ]
	elif coord_type == 'dropoffs':
		data_files = [ 'starting_points.csv' ]
	elif coord_type == 'both':
		data_files = [ 'destinations.csv', 'starting_points.csv' ]

	# Load daily capacity data
	daily_capacity_data = pd.read_csv(os.path.join('..', 'data', 'Unmasked Daily Capacity.xlsx'))

	# Get dictionary of taxicab trip data based on `coord_type` argument
	taxi_data = load_data(coord_type, data_files, data_path)

	# Save per-day, per hotel satisfying pickups to file
	record_daily(taxi_data, start_date, end_date)