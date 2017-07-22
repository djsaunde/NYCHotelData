'''
Data ingestion and preprocessing script.

@author: Dan Saunders (djsaunde.github.io)
'''

import os
import csv
import imp
import sys
import gmplot
import timeit
import argparse
import webbrowser
import numpy as np
import multiprocess
import pandas as pd
import matplotlib.pyplot as plt

from mpi4py import MPI
from geopy.geocoders import GoogleV3
from joblib import Parallel, delayed
from IPython.core.display import HTML
from multiprocessing import cpu_count
from IPython.display import Image, display

from util import *

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def preprocess(taxi_file, distance=300, api_key='AIzaSyAWV7aBLcawx2WyMO7fM4oOL9ayZ_qGz-Y'):
	'''
    	Main logic for parsing taxi datafiles.
	'''
	print '\n'

	start_time = timeit.default_timer()
		
	print '...loading taxicab data file:', taxi_file
	
	year = int(taxi_file.split('_')[2].split('-')[0])
	
	fpath = os.path.join('..', 'data', taxi_file)

	try:
		if 'green' in taxi_file:
			if taxi_file in ['green_tripdata_2016-11.csv', 'green_tripdata_2016-12.csv', 'green_tripdata_2016-07.csv', 'green_tripdata_2016-08.csv', 'green_tripdata_2016-09.csv']:
				sys.exit()
			else:
				# let's load a single .csv file of taxicab records (say, January 2016)
				taxi_data = pd.read_csv(fpath, usecols=['Pickup_latitude', 'Pickup_longitude', 'Dropoff_latitude', 'Dropoff_longitude', 'lpep_pickup_datetime', 'Lpep_dropoff_datetime', 'Passenger_count', 'Trip_distance', 'Fare_amount'])
				
				# get relevant rows of the data and store them as numpy arrays
				pickup_lats, pickup_longs = np.array(taxi_data['Pickup_latitude']), np.array(taxi_data['Pickup_longitude'])
				dropoff_lats, dropoff_longs = np.array(taxi_data['Dropoff_latitude']), np.array(taxi_data['Dropoff_longitude']),
				pickup_time = np.array(taxi_data['lpep_pickup_datetime'])
				dropoff_time = np.array(taxi_data['Lpep_dropoff_datetime'])
				passenger_count = np.array(taxi_data['Passenger_count'])
				trip_distance = np.array(taxi_data['Trip_distance'])
				fare_amount = np.array(taxi_data['Fare_amount'])

			
		elif 'yellow' in taxi_file:
			if '2014-09' in taxi_file or '2014-01' in taxi_file or '2014-07' in taxi_file or '2014-06' in taxi_file:
				# let's load a single .csv file of taxicab records (say, January 2016)
				taxi_data = pd.read_csv(fpath, usecols=[' pickup_latitude', ' pickup_longitude', ' dropoff_latitude', ' dropoff_longitude', ' pickup_datetime', ' dropoff_datetime', ' passenger_count', ' trip_distance', ' fare_amount'])
				
				# get relevant rows of the data and store them as numpy arrays
				pickup_lats, pickup_longs = np.array(taxi_data[' pickup_latitude']), np.array(taxi_data[' pickup_longitude'])
				dropoff_lats, dropoff_longs = np.array(taxi_data[' dropoff_latitude']), np.array(taxi_data[' dropoff_longitude']),
				pickup_time = np.array(taxi_data[' pickup_datetime'])
				dropoff_time = np.array(taxi_data[' dropoff_datetime'])
				passenger_count = np.array(taxi_data[' passenger_count'])
				trip_distance = np.array(taxi_data[' trip_distance'])
				fare_amount = np.array(taxi_data[' fare_amount'])

			elif '2009-11' in taxi_file or '2009-10' in taxi_file or '2009-01' in taxi_file:
				# let's load a single .csv file of taxicab records (say, January 2016)
				taxi_data = pd.read_csv(fpath, usecols=['Start_Lat', 'Start_Lon', 'End_Lat', 'End_Lon', 'Trip_Pickup_DateTime', 'Trip_Dropoff_DateTime', 'Passenger_Count', 'Trip_Distance', 'Fare_Amt'])
				
				# get relevant rows of the data and store them as numpy arrays
				pickup_lats, pickup_longs = np.array(taxi_data['Start_Lat']), np.array(taxi_data['Start_Lon'])
				dropoff_lats, dropoff_longs = np.array(taxi_data['End_Lat']), np.array(taxi_data['End_Lon']),
				pickup_time = np.array(taxi_data['Trip_Pickup_DateTime'])
				dropoff_time = np.array(taxi_data['Trip_Dropoff_DateTime'])
				passenger_count = np.array(taxi_data['Passenger_Count'])
				trip_distance = np.array(taxi_data['Trip_Distance'])
				fare_amount = np.array(taxi_data['Fare_Amt'])

			else:
				# let's load a single .csv file of taxicab records (say, January 2016)
				taxi_data = pd.read_csv(fpath, usecols=['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude', 'pickup_datetime', 'dropoff_datetime', 'passenger_count', 'trip_distance', 'fare_amount'])
				
				# get relevant rows of the data and store them as numpy arrays
				pickup_lats, pickup_longs = np.array(taxi_data['pickup_latitude']), np.array(taxi_data['pickup_longitude'])
				dropoff_lats, dropoff_longs = np.array(taxi_data['dropoff_latitude']), np.array(taxi_data['dropoff_longitude']),
				pickup_time = np.array(taxi_data['pickup_datetime'])
				dropoff_time = np.array(taxi_data['dropoff_datetime'])
				passenger_count = np.array(taxi_data['passenger_count'])
				trip_distance = np.array(taxi_data['trip_distance'])
				fare_amount = np.array(taxi_data['fare_amount'])
			
		else:
			# this shouldn't happen
			raise NotImplementedError
	except ValueError:
		taxi_data = pd.read_csv(fpath, error_bad_lines=False)
		print taxi_file, taxi_data.columns.values

	# remove the taxicab data from memory
	del taxi_data

	sys.exit()

	# zip together lats, longs for coordinates and append them to the lists
	pickup_coords = zip(pickup_lats, pickup_longs)
	dropoff_coords = zip(dropoff_lats, dropoff_longs)

	# report stats about taxi data as we go
	print '\n...loaded', len(pickup_coords), 'taxicab trips'

	print '\nIt took', timeit.default_timer() - start_time, 'seconds to load the above taxi data files'

	# store data as numpy arrays (transposing to have in a more work-friendly shape)    
	pickup_coords, dropoff_coords = np.array(pickup_coords).T, np.array(dropoff_coords).T
	pickup_times, dropoff_times = np.array(pickup_times).T, np.array(dropoff_times).T
	passenger_counts, trip_distances, fare_amounts = np.array(passenger_counts), np.array(trip_distances), np.array(fare_amounts)

	# get file containing hotel names and addresses
	hotel_file = pd.read_excel('../data/Pilot Set of Hotels.xlsx', sheetname='set 2')

	# split the file into lists of names and addresses
	hotel_IDs = hotel_file['Share ID']
	hotel_names = hotel_file['Name']
	hotel_addresses = hotel_file['Address']

	# setting up geolocator object
	geolocator = GoogleV3(api_key, timeout=10)

	# storing the geocode of the above addresses
	hotel_coords = []

	print '\n...getting hotel coordinates'

	start_time = timeit.default_timer()

	hotel_locations = multiprocess.Pool(cpu_count()).map_async(geolocator.geocode, [ hotel_address for hotel_address in hotel_addresses ])
	hotel_coords = [ (location.latitude, location.longitude) for location in hotel_locations.get() ]

	print '\nIt took', timeit.default_timer() - start_time, 'seconds to geolocate all hotels'
	print '\n...finding distance criterion-satisfying taxicab pick-ups'

	# create and open spreadsheet for nearby pick-ups and drop-offs for each hotel
	writer = pd.ExcelWriter('../data/Nearby Pickups and Dropoffs.xlsx')

	# keep track of total time elapsed for all hotels
	start_time = timeit.default_timer()

	# keep track of how much we written into the current Excel worksheet
	prev_len = 0

	# loop through each hotel and find all satisfying taxicab rides
	for idx, hotel_coord in enumerate(hotel_coords):
		
		# print progress to console
		print '\n...finding satisfying taxicab rides for', hotel_names[idx]
		
		# call the 'get_destinations' function from the 'util.py' script on all trips stored
		destinations = get_destinations(pickup_coords.T, dropoff_coords.T, pickup_times, dropoff_times, passenger_counts, trip_distances, fare_amounts, hotel_coord, distance, unit='feet').T
		
		# create pandas DataFrame from output from destinations (distance from hotel, latitude, longitude)
		index = [ i for i in range(prev_len + 1, prev_len + destinations.shape[0] + 1) ]
		destinations = pd.DataFrame(destinations, index=index, columns=['Distance From Hotel', 'Latitude', 'Longitude', 'Pick-up Time', 'Drop-off Time', 'Passenger Count', 'Trip Distance', 'Fare Amount'])
			
		# add column for hotel name
		name_frame = pd.DataFrame([hotel_names[idx]] * destinations.shape[0], index=destinations.index, columns=['Hotel Name'])
		to_write = pd.concat([name_frame, destinations], axis=1)
			
		# add column for hotel ID
		ID_frame = pd.DataFrame([hotel_IDs[idx]] * destinations.shape[0], index=destinations.index, columns=['Share ID'])
		to_write = pd.concat([ID_frame, name_frame, destinations], axis=1)
		
		# write sheet to Excel file
		if idx == 0:
			to_write.to_excel(writer, 'Nearby Pick-ups (rank ' + str(rank) + ')', index=False)
		elif idx != 0:
			to_write.to_excel(writer, 'Nearby Pick-ups (rank ' + str(rank) + ')', startrow=prev_len + 1, header=None, index=False)
		
		# keep track of where we left off in the previous workbook
		prev_len += len(to_write)

	# get and report total elapsed time for all hotels
	end_time = timeit.default_timer() - start_time
	print '( total time elapsed for all hotels:', end_time, ') \n'


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--distance', type=int, default=300, help='Distance criterion (in feet) from hotels considered.')
	parser.add_argument('--num_ranks', type=int, default=50, help='Number of MPI ranks used for preprocessing script.')
	args = parser.parse_args()

	# get requested distance criterion
	distance = args.distance
	num_ranks = args.num_ranks

	# taxi data files to preprocess
	taxi_files = [ filename for filename in os.listdir(os.path.join('..', 'data')) if 'yellow' in filename or 'green' in filename ]

	# preprocess each taxi data file
	for idx in xrange(rank, num_ranks, len(taxi_files)):
		preprocess(taxi_files[idx], distance)
