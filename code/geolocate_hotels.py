from __future__ import print_function

import os
import sys
import timeit
import pandas as pd
import multiprocess

from geopy import GoogleV3

# Get API key from text file.
try:
	key_file = open(os.path.join('..', 'key.txt'), 'r')
except IOError:
	print('\nNo API key found. Place Google Geocoding-enabled API key in "key.txt" in top-level directory.')
	print('Exiting...\n')
	sys.exit()

# Read API key and create Google Geolocation API geolocator object.
api_key = key_file.read().rstrip()
geolocator = GoogleV3(api_key, timeout=10)

# Get file containing hotel names and addresses.
print('\nReading hotel data from "Final hotel Identification.xlsx".')
hotel_file = pd.read_excel(os.path.join('..', 'data', 'Final hotel Identification.xlsx'),
												sheetname='final match', skip_footer=28)

# Split the file into pandas.Series of IDs, names, and addresses.
hotel_IDs = hotel_file['Share ID']
hotel_names = hotel_file['Name']
hotel_addresses = hotel_file['Address'].apply(lambda x : x + ', New York, New York')

start_time = timeit.default_timer()

# Store the geospatial coordinates of the above addresses.
print('Calculating hotel (latitude, longitude) coordinates.')

hotel_locations = multiprocess.Pool(8).map_async(geolocator.geocode, hotel_addresses)
hotel_coords = [ (location.latitude, location.longitude) for location in hotel_locations.get() ]
hotel_latitudes = pd.Series([ coord[0] for coord in hotel_coords ], name='Latitude')
hotel_longitudes = pd.Series([ coord[1] for coord in hotel_coords ], name='Longitude')

print('\nTime:', timeit.default_timer() - start_time)
print('\nWriting file with coordinates to disk.')

to_write = pd.concat([hotel_IDs, hotel_names, hotel_addresses, hotel_latitudes, hotel_longitudes], axis=1)

# Create .csv file for hotel data with added (latitude, longitude) coordinates columns.
to_write.to_csv(os.path.join('..', 'data', 'Final hotel Identification (with coordinates).csv'), index=False)