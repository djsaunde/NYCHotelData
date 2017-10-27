import sys
import timeit
import pandas as pd
import multiprocess

from geopy import GoogleV3


api_key='AIzaSyDjt-6hfVNeGM1C4MQEAWVLEfry0lGLgT4'
geolocator = GoogleV3(api_key, timeout=10)

# get file containing hotel names and addresses
hotel_file = pd.read_excel('../data/Final hotel Identification.xlsx', sheetname='final match', skip_footer=28)

# split the file into lists of names and addresses
hotel_IDs = hotel_file['Share ID']
hotel_names = hotel_file['Name']
hotel_addresses = hotel_file['Address']

# storing the geocode of the above addresses
hotel_coords = []

print '\n...getting hotel coordinates'

start_time = timeit.default_timer()

hotel_locations = multiprocess.Pool(8).map_async(geolocator.geocode, 
	[ hotel_address + ', New York, New York' for hotel_address in hotel_addresses ])
hotel_coords = [ (location.latitude, location.longitude) for location in hotel_locations.get() ]
hotel_latitudes = pd.Series([ coord[0] for coord in hotel_coords ], name='Latitude')
hotel_longitudes = pd.Series([ coord[1] for coord in hotel_coords ], name='Longitude')

print '\nIt took', timeit.default_timer() - start_time, 'seconds to geolocate all hotels'
print '\n...finding distance criterion-satisfying taxicab pick-ups'

file_with_coords = pd.concat([hotel_IDs, hotel_names, hotel_addresses, hotel_latitudes, hotel_longitudes], axis=1)

# create and open spreadsheet for nearby pick-ups and drop-offs for each hotel
writer = pd.ExcelWriter('../data/Final hotel Identification (with coordinates).xlsx')
file_with_coords.to_excel(writer, 'final match with coordinates', index=False)