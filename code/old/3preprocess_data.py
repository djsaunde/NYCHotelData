from __future__ import print_function

import os
import csv
import imp
import sys
import time
import gzip
import geopy
import gmplot
import timeit
import argparse
import webbrowser
import numpy as np
import multiprocess
import pandas as pd
import dask.dataframe as dd
import matplotlib.pyplot as plt

from glob import glob
from geopy.geocoders import GoogleV3
from joblib import Parallel, delayed
from IPython.core.display import HTML
from multiprocessing import cpu_count
from IPython.display import Image, display

from util import *

print('\nRunning data pre-processing.')


def preprocess(taxi_file, distance):
	'''
    Parse a single taxi data file.
	'''
	start_time = timeit.default_timer()
		
	print('\nLoading taxi data file: %s' % taxi_file)
	
	year = int(taxi_file.split('_')[2].split('-')[0])
	fpath = os.path.join('..', 'data', 'taxi_data', taxi_file)

	try:
		if 'green' in taxi_file:
			# Check for months in which (latitude, longitude) coordinates weren't recorded.
			for year_month in ['2016-07', '2016-08', '2016-09', '2016-10', '2016-11', '2016-12']:
				if year_month in taxi_file:
					return

			# Load a single .csv file of taxicab records.
			taxi_data = dd.read_csv(fpath, usecols=[1,2,5,6,7,8,9,10,11])
			taxi_data = taxi_data.rename(index=None, columns={'lpep_pickup_datetime' : 'Pick-up Time',
						'Lpep_dropoff_datetime' : 'Drop-off Time', 'Pickup_longitude' : 'Pick-up Longitude',
						'Pickup_latitude' : 'Pick-up Latitude', 'Dropoff_longitude' : 'Drop-off Longitude',
						'Dropoff_latitude' : 'Drop-off Latitude', 'Passenger_count' : 'Passenger Count',
						'Trip_distance' : 'Trip Distance', 'Fare_amount' : 'Fare Amount'})

		elif 'yellow' in taxi_file:
			# Check for months in which (latitude, longitude) coordinates weren't recorded.
			for year_month in ['2016-07', '2016-08', '2016-09', '2016-10', '2016-11', '2016-12']:
				if year_month in taxi_file:
					return

			if '2014' in taxi_file:
				# Load a single .csv file of taxicab records.
				taxi_data = dd.read_csv(fpath, usecols=[' pickup_latitude', ' pickup_longitude',
									' dropoff_latitude', ' dropoff_longitude', ' pickup_datetime',
							' dropoff_datetime', ' passenger_count', ' trip_distance', ' fare_amount'])

				taxi_data = taxi_data.rename(index=None, columns={' pickup_datetime' : 'Pick-up Time',
						' dropoff_datetime' : 'Drop-off Time', ' pickup_longitude' : 'Pick-up Longitude',
						' pickup_latitude' : 'Pick-up Latitude', ' dropoff_longitude' : 'Drop-off Longitude',
						' dropoff_latitude' : 'Drop-off Latitude', ' passenger_count' : 'Passenger Count',
						' trip_distance' : 'Trip Distance', ' fare_amount' : 'Fare Amount'})
				
			elif '2009' in taxi_file:
				# Load a single .csv file of taxicab records.
				taxi_data = dd.read_csv(fpath, usecols=['Start_Lat', 'Start_Lon', 'End_Lat', 'End_Lon',
														'Trip_Pickup_DateTime', 'Trip_Dropoff_DateTime', 
														'Passenger_Count', 'Trip_Distance', 'Fare_Amt'])

				taxi_data = taxi_data.rename(index=None, columns={'Trip_Pickup_DateTime' : 'Pick-up Time',
						'Trip_Dropoff_DateTime' : 'Drop-off Time', 'Start_Lon' : 'Pick-up Longitude',
						'Start_Lat' : 'Pick-up Latitude', 'End_Lon' : 'Drop-off Longitude',
						'End_Lat' : 'Drop-off Latitude', 'Passenger_Count' : 'Passenger Count',
						'Trip_Distance' : 'Trip Distance', 'Fare_Amt' : 'Fare Amount'})

			elif '2015' in taxi_file or '2016' in taxi_file:
				# Load a single .csv file of taxicab records.
				taxi_data = dd.read_csv(fpath, usecols=['pickup_latitude', 'pickup_longitude',
									'dropoff_latitude', 'dropoff_longitude', 'tpep_pickup_datetime', 
							'tpep_dropoff_datetime', 'passenger_count', 'trip_distance', 'fare_amount'])

				taxi_data = taxi_data.rename(index=None, columns={'tpep_pickup_datetime' : 'Pick-up Time',
						'tpep_dropoff_datetime' : 'Drop-off Time', 'pickup_longitude' : 'Pick-up Longitude',
						'pickup_latitude' : 'Pick-up Latitude', 'dropoff_longitude' : 'Drop-off Longitude',
						'dropoff_latitude' : 'Drop-off Latitude', 'passenger_count' : 'Passenger Count',
						'trip_distance' : 'Trip Distance', 'fare_amount' : 'Fare Amount'})

			else:
				# let's load a single .csv file of taxicab records (say, January 2016)
				taxi_data = dd.read_csv(fpath, usecols=['pickup_latitude', 'pickup_longitude',
									'dropoff_latitude', 'dropoff_longitude', 'pickup_datetime',
							'dropoff_datetime', 'passenger_count', 'trip_distance', 'fare_amount'])

				taxi_data = taxi_data.rename(index=None, columns={'pickup_datetime' : 'Pick-up Time',
						'dropoff_datetime' : 'Drop-off Time', 'pickup_longitude' : 'Pick-up Longitude',
						'pickup_latitude' : 'Pick-up Latitude', 'dropoff_longitude' : 'Drop-off Longitude',
						'dropoff_latitude' : 'Drop-off Latitude', 'passenger_count' : 'Passenger Count',
						'trip_distance' : 'Trip Distance', 'fare_amount' : 'Fare Amount'})
			
		else:
			# this shouldn't happen
			raise NotImplementedError

	except ValueError:
		print('\nPickup or dropoff IDs rather than (lat, lon) coordinates.')
		return

	print('\nIt took %.4f seconds to load the above taxi data files' % (timeit.default_timer() - start_time))

	# Load file containing hotel names and addresses.
	hotel_file = dd.read_csv(os.path.join('..', 'data', 'Final hotel Identification (with coordinates).csv'))
	n_hotels = len(hotel_file)

	print('\nFinding distance criterion-satisfying taxicab pick-ups.')

	# Find all end points of taxi trips ending near 
	# hotels which satisfy distance criterion, per hotel.
	start_time = timeit.default_timer()
	for idx, row in hotel_file.iterrows():
		print('Progress: (%d / %d)' % (idx + 1, n_hotels))

		# Get the taxi trips which satisfy the distance criterion.
		destinations = dask_get_satisfying_indices(taxi_data, row, distance)

		# Add columns for hotel names and IDs.
		destinations['Hotel Name'] = row['Name']
		destinations['Share ID'] = row['Share ID']

		if idx == 0:
			all_destinations = destinations
		else:
			all_destinations = all_destinations.append(destinations)

	# Write dask.DataFrame to partitioned .csv files.
	print('Re-partitioning data.')
	all_destinations = all_destinations.repartition(npartitions=10)

	print('Writing out partitioned dask.DataFrame.')
	outpath = os.path.join(processed_path, 'NPD_destinations_' + taxi_file.split('.')[0] + '.csv')
	glob_name = os.path.join(processed_path, 'NPD_destinations_' + taxi_file.split('.')[0] + '.*.csv')
	all_destinations.to_csv(glob_name, index=False)

	# print('Re-partitioning data.')
	# all_destinations = all_destinations.repartition(npartitions=10)
	# print('Writing out partitioned dask.DataFrame.')
	# outpath = os.path.join(processed_path, 'NPD_destinations_' + taxi_file.split('.')[0])
	# all_destinations.to_parquet(outpath, write_index=False)
	
	# Combine fragmented data into a single .csv file.
	print('Combining partitioned .csv files into one file.')
	fnames = glob(glob_name)
	with open(outpath, 'w') as out:
		for idx, fname in enumerate(fnames):
			with open(fname) as f:
				if idx > 0:
					next(f)

				out.write(f.read())

	end_time = timeit.default_timer() - start_time
	print('Time elapsed while finding destinations: %.4f\n' % end_time)
	
	# Find all end points of taxi trips ending near 
	# hotels which satisfy distance criterion, per hotel.
	start_time = timeit.default_timer()

	for idx, row in hotel_file.iterrows():
		print('Progress: (%d / %d)' % (idx + 1, n_hotels))

		# Get the taxi trips which satisfy the distance criterion.
		starting_points = dask_get_satisfying_indices(taxi_data, row, distance)

		# Add columns for hotel names and IDs.
		starting_points['Hotel Name'] = row['Name']
		starting_points['Share ID'] = row['Share ID']

		if idx == 0:
			all_starting_points = starting_points
		else:
			all_starting_points = all_starting_points.append(starting_points)

	# Write dask.DataFrame to partitioned .csv files.
	print('Re-partitioning data.')
	all_starting_points = all_starting_points.repartition(npartitions=10)
	
	print('Writing out partitioned dask.DataFrame.')
	outpath = os.path.join(processed_path, 'NPD_starting_points_' + taxi_file.split('.')[0] + '.csv')
	glob_name = os.path.join(processed_path, 'NPD_starting_points_' + taxi_file.split('.')[0] + '.*.csv')
	all_starting_points.to_csv(glob_name, index=False)

	# print('Re-partitioning data.')
	# all_starting_points = all_starting_points.repartition(npartitions=10)
	# print('Writing out partitioned dask.DataFrame.')
	# outpath = os.path.join(processed_path, 'NPD_destinations_' + taxi_file.split('.')[0])
	# all_starting_points.to_parquet(outpath, write_index=False)
	
	# Combine fragmented data into a single .csv file.
	print('Combining partitioned .csv files into one file.')
	fnames = glob(glob_name)
	with open(outpath, 'w') as out:
		for idx, fname in enumerate(fnames):
			with open(fname) as f:
				if idx > 0:
					next(f)

				out.write(f.read())
	
	end_time = timeit.default_timer() - start_time
	print('Time elapsed while finding starting points: %.4f\n' % end_time)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--distance', type=int, default=300, \
		help='Distance criterion (in feet) from hotels considered.')
	parser.add_argument('--file_idx', type=int, default=0, \
		help='Index of taxi file in ordered file list to preprocess.')
	parser.add_argument('--file_name', type=str, default='', \
							help='Name of taxi file to preprocess.')
	args = parser.parse_args()

	# Parse command line arguments.
	distance = args.distance
	file_idx = args.file_idx
	file_name = args.file_name.replace(',', '')

	processed_path = os.path.join('..', 'data', 'all_preprocessed_' + str(distance))
	if not os.path.isdir(processed_path):
		os.makedirs(processed_path)

	if file_name == '':
		# taxi data files to preprocess
		taxi_files = [ filename for filename in os.listdir(os.path.join('..', 'data', \
						'taxi_data', '')) if 'yellow' in filename or 'green' in filename ]
		
		# preprocess our particular taxi data file
		preprocess(taxi_files[file_idx], distance)
	else:
		# preprocess passed in taxi data file
		preprocess(file_name, distance)
