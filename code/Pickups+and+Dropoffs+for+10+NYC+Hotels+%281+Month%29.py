
# coding: utf-8

# # Pickups and Dropoffs for 10 NYC Hotels (1 Month) 
# 
# This exercise will involve using more data (Yellow and Green taxicab data over a single month) and 10 NYC hotels (chosen by professor Rojas), finding dropoff (pickup) locations sufficiently close to all 10 of the chosen hotels and store their corresponding pickup (dropoff) locations, and store these records in an Excel workbook along with hotel name, times of day, and other relevant fields.
# 
# This is a proof-of-concept notebook for what will eventually be deployed over all data available from the [NYC Taxi and Limousine Commission Trip Record Data page](http://www.nyc.gov/html/tlc/html/about/trip_record_data.shtml).

# In[ ]:

# imports...
import csv, imp, os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from geopy.geocoders import GoogleV3
import gmplot, webbrowser, timeit
from IPython.display import Image, display
from IPython.core.display import HTML

# importing helper methods
from util import *

# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

# matplotlib setup
get_ipython().magic('matplotlib inline')
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


# ### Importing and Cleaning Data
# 
# Let's try reading in Yellow and Green taxicab data files from January 2016 to experiment with. These files together are approximately 2Gb, so it should take some time to read in and process, but after we've done so, the rest of the analysis is less memory-intensive.

# In[ ]:

# change this if you want to try different dataset(s)
taxi_files = ['../data/yellow_tripdata_2016-01.csv', '../data/green_tripdata_2016-01.csv']

# variables to store pick-up and drop-off coordinates and other relevant fields
pickup_coords, dropoff_coords, pickup_times, dropoff_times = [], [], [], []
passenger_counts, trip_distances, fare_amounts = [], [], []

for taxi_file in log_progress(taxi_files, every=1):
    
    print '...loading taxicab data file:', taxi_file[3:]
    
    if 'green' in taxi_file:
        # let's load a single .csv file of taxicab records (say, January 2016)
        taxi_data = pd.read_csv(taxi_file, usecols=['Pickup_latitude', 'Pickup_longitude', 'Dropoff_latitude', 'Dropoff_longitude', 'lpep_pickup_datetime', 'Lpep_dropoff_datetime', 'Passenger_count', 'Trip_distance', 'Fare_amount'])
        
        # get relevant rows of the data and store them as numpy arrays
        pickup_lats, pickup_longs = np.array(taxi_data['Pickup_latitude']), np.array(taxi_data['Pickup_longitude'])
        dropoff_lats, dropoff_longs = np.array(taxi_data['Dropoff_latitude']), np.array(taxi_data['Dropoff_longitude']),
        pickup_time = np.array(taxi_data['lpep_pickup_datetime'])
        dropoff_time = np.array(taxi_data['Lpep_dropoff_datetime'])
        passenger_count = np.array(taxi_data['Passenger_count'])
        trip_distance = np.array(taxi_data['Trip_distance'])
        fare_amount = np.array(taxi_data['Fare_amount'])
        
    elif 'yellow' in taxi_file:
        # let's load a single .csv file of taxicab records (say, January 2016)
        taxi_data = pd.read_csv(taxi_file, usecols=['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude', 'tpep_pickup_datetime', 'tpep_dropoff_datetime', 'passenger_count', 'trip_distance', 'fare_amount'])
        
        # get relevant rows of the data and store them as numpy arrays
        pickup_lats, pickup_longs = np.array(taxi_data['pickup_latitude']), np.array(taxi_data['pickup_longitude'])
        dropoff_lats, dropoff_longs = np.array(taxi_data['dropoff_latitude']), np.array(taxi_data['dropoff_longitude']),
        pickup_time = np.array(taxi_data['tpep_pickup_datetime'])
        dropoff_time = np.array(taxi_data['tpep_dropoff_datetime'])
        passenger_count = np.array(taxi_data['passenger_count'])
        trip_distance = np.array(taxi_data['trip_distance'])
        fare_amount = np.array(taxi_data['fare_amount'])
        
    else:
        # this shouldn't happen
        raise NotImplementedError

    # remove the taxicab data from memory
    del taxi_data

    # zip together lats, longs for coordinates and append them to the lists
    pickup_coords.extend(zip(pickup_lats, pickup_longs))
    dropoff_coords.extend(zip(dropoff_lats, dropoff_longs))
    pickup_times.extend(pickup_time)
    dropoff_times.extend(dropoff_time)
    passenger_counts.extend(passenger_count)
    trip_distances.extend(trip_distance)
    fare_amounts.extend(fare_amount)
    
pickup_coords, dropoff_coords = np.array(pickup_coords).T, np.array(dropoff_coords).T
pickup_times, dropoff_times = np.array(pickup_times).T, np.array(dropoff_times).T
passenger_counts, trip_distances, fare_amounts = np.array(passenger_counts), np.array(trip_distances), np.array(fare_amounts)


# ### Geolocating Hotels
# 
# We use the geopy client for popular geolocation packages. We use Google's geolocation service (https://developers.google.com/maps/documentation/geolocation/intro), since it appears to be the most accurate, user friendly, and requires little money to operate, even with many requests.

# In[ ]:

# get file containing hotel names and addresses
hotel_file = pd.read_excel('../data/Pilot Set of Hotels.xlsx', sheetname='Sheet1')

# split the file into lists of names and addresses
hotel_IDs = hotel_file['Share ID']
hotel_names = hotel_file['Name']
hotel_addresses = hotel_file['Address']

# setting up geolocator object
geolocator = GoogleV3(api_key='AIzaSyAWV7aBLcawx2WyMO7fM4oOL9ayZ_qGz-Y', timeout=10)


# In[ ]:

# storing the geocode of the above addresses
hotel_coords = []

print '...getting hotel coordinates'

# get and store hotel coordinates
for hotel_address in log_progress(hotel_addresses, every=1):
    
    # get the hotel's geolocation
    location = geolocator.geocode(hotel_address)
    if location == None:
        continue
    
    # get the coordinates of the hotel from the geolocation
    hotel_coord = (location.latitude, location.longitude)
    
    # add it to our list
    hotel_coords.append(hotel_coord)


# ## Finding Nearby Taxicab Pick-ups and Corresponding Drop-offs
# 
# For each hotel, we want to find all taxicab rides which begin within a certain distance of the hotel (say, 500 feet). We'll store the hotel name, address, ID, time of day (pickup and dropoff), corresponding (latitude, longitude) coordinates of the hotel, and a few other potentially useful fields.

# In[ ]:

print '...finding distance criterion-satisfying taxicab pick-ups', '\n'

# distance (in feet) criterion
distance = 300

# create and open spreadsheet for nearby pick-ups and drop-offs for each hotel
writer = pd.ExcelWriter('../data/Nearby Pickups and Dropoffs.xlsx')

# keep track of total time elapsed for all hotels
start_time = timeit.default_timer()

# keep track of how much we written into the current Excel worksheet
prev_len = 0

# loop through each hotel and find all satisfying taxicab rides
for idx, hotel_coord in log_progress(enumerate(hotel_coords), every=1, size=10):
    
    # print progress to console
    print '...finding satisfying taxicab rides for', hotel_names[idx], '\n'
    
    # call the 'get_destinations' function from the 'util.py' script on all trips stored
    destinations = get_destinations(pickup_coords.T, dropoff_coords.T, pickup_times, dropoff_times, passenger_counts, trip_distances, fare_amounts, hotel_coord, distance).T
    
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
        to_write.to_excel(writer, 'Nearby Pick-ups', index=False)
    
    if idx != 0:
        to_write.to_excel(writer, 'Nearby Pick-ups', startrow=prev_len+1, header=None, index=False)
    
    # keep track of where we left off in the previous workbook
    prev_len += len(to_write)

# close the ExcelWriter object    
writer.close()

# get and report total elapsed time for all hotels
end_time = timeit.default_timer() - start_time
print '( total time elapsed for all hotels:', end_time, ') \n'


# In[ ]:

writer.close()


# ## Finding Nearby Taxicab Drop-offs and Corresponding Pick-ups
# 
# Now, for each hotel, we want to find all taxicab rides which end within a certain distance of the hotel (again, 100 meters).

# In[ ]:

print '...finding distance criterion-satisfying taxicab drop-offs', '\n'

# keep track of total time elapsed for all hotels
start_time = timeit.default_timer()

# keep track of how much we written into the current Excel worksheet
prev_len = 0

print len(pickup_coords[0])

# loop through each hotel and find all satisfying taxicab rides
for idx, hotel_coord in log_progress(enumerate(hotel_coords), every=1, size=178):
    
    # print progress to console
    print '...finding satisfying taxicab rides for', hotel_names[idx], '\n'
    
    # call the 'get_destinations' function from the 'util.py' script on all trips stored
    destinations = get_starting_points(pickup_coords.T, dropoff_coords.T, pickup_times, dropoff_times, passenger_counts, trip_distances, fare_amounts, hotel_coord, distance).T
    
    # create pandas DataFrame from output from destinations (distance from hotel, latitude, longitude)
    index = [ i for i in range(1, destinations.shape[0] + 1) ]
    destinations = pd.DataFrame(destinations, index=index, columns=['Distance From Hotel', 'Latitude', 'Longitude', 'Pick-up Time', 'Drop-off Time', 'Passenger Count', 'Trip Distance', 'Fare Amount'])
    
    # add column for hotel name
    name_frame = pd.DataFrame([hotel_names[idx]] * destinations.shape[0], index=destinations.index, columns=['Hotel Name'])
    to_write = pd.concat([name_frame, destinations], axis=1)
    
    # add column for hotel ID
    ID_frame = pd.DataFrame([hotel_IDs[idx]] * destinations.shape[0], index=destinations.index, columns=['Share ID'])
    to_write = pd.concat([ID_frame, name_frame, destinations], axis=1)
    
    # write sheet to Excel file
    if idx == 0:
        to_write.to_excel(writer, 'Nearby Drop-offs', index=False)
    
    if idx != 0:
        to_write.to_excel(writer, 'Nearby Drop-offs', startrow=prev_len+1, header=None, index=False)
    
    # keep track of where we left off in the previous workbook
    prev_len += len(to_write)
    
# close the ExcelWriter object    
writer.close()

# get and report total elapsed time for all hotels
end_time = timeit.default_timer() - start_time
print '( total time elapsed for all hotels:', end_time, ') \n'

