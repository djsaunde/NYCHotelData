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
import matplotlib.pyplot as plt

from geopy.geocoders import GoogleV3
from joblib import Parallel, delayed
from IPython.core.display import HTML
from multiprocessing import cpu_count
from IPython.display import Image, display

from util import *

print('\nRunning data pre-processing.')


def preprocess(taxi_file, hotel_index, distance, n_jobs):
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
			taxi_data = pd.read_csv(fpath, usecols=[1,2,5,6,7,8,9,10,11], index_col=False)

			# Get relevant rows of the data.
			pickup_lats, pickup_longs = np.array(taxi_data['Pickup_latitude']), np.array(taxi_data['Pickup_longitude'])
			dropoff_lats, dropoff_longs = np.array(taxi_data['Dropoff_latitude']), np.array(taxi_data['Dropoff_longitude'])
			pickup_time = np.array(taxi_data['lpep_pickup_datetime'])
			dropoff_time = np.array(taxi_data['Lpep_dropoff_datetime'])
			passenger_count = np.array(taxi_data['Passenger_count'])
			trip_distance = np.array(taxi_data['Trip_distance'])
			fare_amount = np.array(taxi_data['Fare_amount'])
			
		elif 'yellow' in taxi_file:
			# Check for months in which (latitude, longitude) coordinates weren't recorded.
			for year_month in ['2016-07', '2016-08', '2016-09', '2016-10', '2016-11', '2016-12']:
				if year_month in taxi_file:
					return

			if '2014' in taxi_file:
				# Load a single .csv file of taxicab records.
				taxi_data = pd.read_csv(fpath, usecols=[' pickup_latitude', ' pickup_longitude', ' dropoff_latitude', \
														' dropoff_longitude', ' pickup_datetime', ' dropoff_datetime', \
														' passenger_count', ' trip_distance', ' fare_amount'])
				
				# Get relevant rows of the data.
				pickup_lats, pickup_longs = np.array(taxi_data[' pickup_latitude']), np.array(taxi_data[' pickup_longitude'])
				dropoff_lats, dropoff_longs = np.array(taxi_data[' dropoff_latitude']), np.array(taxi_data[' dropoff_longitude'])
				pickup_time = np.array(taxi_data[' pickup_datetime'])
				dropoff_time = np.array(taxi_data[' dropoff_datetime'])
				passenger_count = np.array(taxi_data[' passenger_count'])
				trip_distance = np.array(taxi_data[' trip_distance'])
				fare_amount = np.array(taxi_data[' fare_amount'])

			elif '2009' in taxi_file:
				# Load a single .csv file of taxicab records.
				taxi_data = pd.read_csv(fpath, usecols=['Start_Lat', 'Start_Lon', 'End_Lat', 'End_Lon', \
					'Trip_Pickup_DateTime', 'Trip_Dropoff_DateTime', 'Passenger_Count', 'Trip_Distance', 'Fare_Amt'])
				
				# Get relevant rows of the data.
				pickup_lats, pickup_longs = np.array(taxi_data['Start_Lat']), np.array(taxi_data['Start_Lon'])
				dropoff_lats, dropoff_longs = np.array(taxi_data['End_Lat']), np.array(taxi_data['End_Lon'])
				pickup_time = np.array(taxi_data['Trip_Pickup_DateTime'])
				dropoff_time = np.array(taxi_data['Trip_Dropoff_DateTime'])
				passenger_count = np.array(taxi_data['Passenger_Count'])
				trip_distance = np.array(taxi_data['Trip_Distance'])
				fare_amount = np.array(taxi_data['Fare_Amt'])

			elif '2015' in taxi_file or '2016' in taxi_file:
				# Load a single .csv file of taxicab records.
				taxi_data = pd.read_csv(fpath, usecols=['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', \
					'dropoff_longitude', 'tpep_pickup_datetime', 'tpep_dropoff_datetime', 'passenger_count', 'trip_distance', 'fare_amount'])

				# Get relevant rows of the data.
				pickup_lats, pickup_longs = np.array(taxi_data['pickup_latitude']), np.array(taxi_data['pickup_longitude'])
				dropoff_lats, dropoff_longs = np.array(taxi_data['dropoff_latitude']), np.array(taxi_data['dropoff_longitude'])
				pickup_time = np.array(taxi_data['tpep_pickup_datetime'])
				dropoff_time = np.array(taxi_data['tpep_dropoff_datetime'])
				passenger_count = np.array(taxi_data['passenger_count'])
				trip_distance = np.array(taxi_data['trip_distance'])
				fare_amount = np.array(taxi_data['fare_amount'])

			else:
				# let's load a single .csv file of taxicab records (say, January 2016)
				taxi_data = pd.read_csv(fpath, usecols=['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', \
					'dropoff_longitude', 'pickup_datetime', 'dropoff_datetime', 'passenger_count', 'trip_distance', 'fare_amount'])
				
				# get relevant rows of the data and store them as numpy arrays
				pickup_lats, pickup_longs = np.array(taxi_data['pickup_latitude']), np.array(taxi_data['pickup_longitude'])
				dropoff_lats, dropoff_longs = np.array(taxi_data['dropoff_latitude']), np.array(taxi_data['dropoff_longitude'])
				pickup_time = np.array(taxi_data['pickup_datetime'])
				dropoff_time = np.array(taxi_data['dropoff_datetime'])
				passenger_count = np.array(taxi_data['passenger_count'])
				trip_distance = np.array(taxi_data['trip_distance'])
				fare_amount = np.array(taxi_data['fare_amount'])
			
		else:
			# this shouldn't happen
			raise NotImplementedError

	except ValueError:
		print('\nPickup or dropoff IDs rather than (lat, lon) coordinates.')
		return

	# remove the taxicab data from memory
	del taxi_data

	# zip together lats, longs for coordinates and append them to the lists
	pickup_coords = zip(pickup_lats, pickup_longs)
	dropoff_coords = zip(dropoff_lats, dropoff_longs)

	print('\n...loaded %d taxicab trips' % len(pickup_coords))
	print('\nIt took %.4f seconds to load the above taxi data files' % (timeit.default_timer() - start_time))

	# store data as numpy arrays (transposing to have in a more work-friendly shape)    
	pickup_coords, dropoff_coords = np.array(pickup_coords).T, np.array(dropoff_coords).T
	pickup_times, dropoff_times = np.array(pickup_time).T, np.array(dropoff_time).T
	passenger_counts, trip_distances, fare_amounts = np.array(passenger_count), \
									np.array(trip_distance), np.array(fare_amount)

	# Load file containing hotel names and addresses.
	hotel_file = pd.read_excel(os.path.join('..', 'data', \
		'Final hotel Identification (with coordinates).xlsx'), \
						sheetname='final match with coordinates')

	# split the file into lists of names and addresses
	hotel_IDs = hotel_file['Share ID']
	hotel_names = hotel_file['Name']
	hotel_addresses = hotel_file['Address']
	hotel_coords = [ (latitude, longitude) for latitude, longitude in \
					zip(hotel_file['Latitude'], hotel_file['Longitude']) ]


	print('\nFinding distance criterion-satisfying taxicab pick-ups.')

	# Find all end points of taxi trips ending near 
	# hotels which satisfy distance criterion, per hotel.
	prev_len = 0
	start_time = timeit.default_timer()
	for idx, hotel_coord in enumerate(hotel_coords):
		print('\nFinding satisfying taxicab rides for %s' % hotel_names[idx])
		
		# Get the indices of the taxi trips which satisfy the distance criterion.
		satisfying_indices, dists = get_satisfying_indices(pickup_coords.T, \
												hotel_coord, distance, n_jobs)

		destinations = np.array([dists] + [item[satisfying_indices] for item in \
					pickup_coords[0, :], pickup_coords[1, :], pickup_times, \
				dropoff_times, passenger_counts, trip_distances, fare_amounts]).T

		n_trips = destinations.shape[0]

		try:
			index = [ i for i in xrange(prev_len + 1, prev_len + n_trips + 1) ]
			destinations = pd.DataFrame(destinations, index=index, 
				columns=['Distance From Hotel', 'Latitude', 'Longitude', 
				'Pick-up Time', 'Drop-off Time', 'Passenger Count', 
				'Trip Distance', 'Fare Amount'])
		except ValueError:
			continue

		# Add columns for hotel names and IDs.
		names = pd.DataFrame([hotel_names[idx]] * n_trips, index=destinations.index, columns=['Hotel Name'])
		IDs = pd.DataFrame([hotel_IDs[idx]] * n_trips, index=destinations.index, columns=['Share ID'])
		to_write = pd.concat([IDs, names, destinations], axis=1)
		
		# Write DataFrame to .csv file.
		fname = os.path.join(processed_path, 'NPD_destinations_' + taxi_file.split('.')[0] + '.csv')
		if idx == 0:
			to_write.to_csv(fname, compression='gzip')
			
		else:
			with gzip.open(fname, 'a') as f:
				to_write.to_csv(f, header=False, compression='gzip')
		
		prev_len += len(to_write)

	end_time = timeit.default_timer() - start_time
	print('Time elapsed while finding destinations: %.4f\n' % end_time)
	
	# Find all starting locations of taxi trips ending near 
	# hotels which satisfy distance criterion, per hotel.
	prev_len = 0
	start_time = timeit.default_timer()
	for idx, hotel_coord in enumerate(hotel_coords):
		print('\nFinding satisfying taxicab rides for %s' % hotel_names[idx])
		
		# Get the indices of the taxi trips which satisfy the distance criterion.
		satisfying_indices, dists = get_satisfying_indices(dropoff_coords.T, \
												hotel_coord, distance, n_jobs)

		starting_points = np.array([dists] + [item[satisfying_indices] for item in \
					pickup_coords[0, :], pickup_coords[1, :], pickup_times, \
				dropoff_times, passenger_counts, trip_distances, fare_amounts]).T

		n_trips = starting_points.shape[0]

		try:
			index = [ i for i in range(prev_len + 1, prev_len + n_trips + 1) ]
			starting_points = pd.DataFrame(starting_points, index=index, 
				columns=['Distance From Hotel', 'Latitude', 'Longitude', 
				'Pick-up Time', 'Drop-off Time', 'Passenger Count', 
				'Trip Distance', 'Fare Amount'])
		except ValueError:
			continue		

		n_trips = starting_points.shape[0]

		# Add columns for hotel names and IDs.
		names = pd.DataFrame([hotel_names[idx]] * n_trips, index=starting_points.index, columns=['Hotel Name'])
		IDs = pd.DataFrame([hotel_IDs[idx]] * n_trips, index=starting_points.index, columns=['Share ID'])
		to_write = pd.concat([ID_frame, names, starting_points], axis=1)
		
		# Write DataFrame to .csv file.
		fname = os.path.join(processed_path, 'NPD_starting_points_' + taxi_file.split('.')[0] + '.csv')
		if idx == 0:
			to_write.to_csv(fname, compression='gzip')
			
		else:
			with gzip.open(fname, 'a') as f:
				to_write.to_csv(f, header=False, compression='gzip')
			
		prev_len += len(to_write)

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
	parser.add_argument('--hotel_idx', type=int, default=0, \
			help='Integer index of hotel to preprocess data for.')
	parser.add_argument('--n_jobs', type=int, default=8, \
		help='Number of CPU cores to use for parallel computation.')
	args = parser.parse_args()

	# Parse command line arguments.
	distance = args.distance
	file_idx = args.file_idx
	file_name = args.file_name.replace(',', '')
	hotel_idx = args.hotel_idx
	n_jobs = args.n_jobs

	processed_path = os.path.join('..', 'data', 'all_preprocessed_' + str(distance))
	if not os.path.isdir(processed_path):
		os.makedirs(processed_path)

	processed_path = os.path.join('..', 'data', 'all_preprocessed_' + str(distance))
	if not os.path.isdir(processed_path):
		os.makedirs(processed_path)

	if file_name == '':
		# taxi data files to preprocess
		taxi_files = [ filename for filename in os.listdir(os.path.join('..', 'data', \
						'taxi_data', '')) if 'yellow' in filename or 'green' in filename ]
		
		# preprocess our particular taxi data file
		preprocess(taxi_files[file_idx], hotel_idx, distance, n_jobs)
	else:
		# preprocess passed in taxi data file
		preprocess(file_name, hotel_idx, distance, n_jobs)
