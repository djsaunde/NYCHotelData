import os
import numpy  as np
import pandas as pd

from time                 import time
from sklearn.linear_model import LinearRegression


class LogitRegression(LinearRegression):
	
	def fit(self, x, p):
		p = np.asarray(p)
		p[p == 0.0] = 1e-8
		p[p == 1.0] = 1 - 1e-2
		y = np.log(p / (1 - p))
		return super().fit(x, y)

	def predict(self, x):
		y = super().predict(x)
		return 1 / (np.exp(-y) + 1)


def load_occupancy_data(data_path, start_date, end_date):
	print('\nLoading daily per-hotel capacity data.'); start = time()

	# Load daily capacity data.
	df = pd.read_csv(os.path.join(data_path, 'Unmasked Capacity and Price Data.csv'), index_col=False)
	df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
	df['ADR'] = df['ADR'].astype(str).str.replace(',', '')
	df['Room Demand'] = df['Room Demand'].astype(str).str.replace(',', '')
	df['Occ'] = df['Occ'] / 100
	df = df.loc[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
	df = df.rename(index=str, columns={'Share ID' : 'Hotel Name'})
	
	print('Time: %.4f' % (time() - start))
	
	return df

def load_merged_data(data_path, taxi_occupancy_path, preproc_data_path, start_date, end_date, trip_type):
	data_fname = 'Taxi occupancy price data.csv'
	counts_fname = 'Taxi occupancy price counts.csv'
	
	is_data_file = os.path.isfile(os.path.join(taxi_occupancy_path, data_fname))
	is_counts_file = os.path.isfile(os.path.join(taxi_occupancy_path, counts_fname))
	
	if not is_counts_file and not is_data_file:
		# Load daily capacity data.
		print('\nLoading daily per-hotel capacity data.'); start = time()

		occupancy = pd.read_csv(os.path.join(data_path, 'Unmasked Capacity and Price Data.csv'), index_col=False)
		occupancy['Date'] = pd.to_datetime(occupancy['Date'], format='%Y-%m-%d')
		occupancy = occupancy.loc[(occupancy['Date'] >= start_date) & (occupancy['Date'] <= end_date)]
		occupancy['Date'] = occupancy['Date'].dt.date
		occupancy['ADR'] = occupancy['ADR'].astype(str).str.replace(',', '')
		occupancy['Room Demand'] = occupancy['Room Demand'].astype(str).str.replace(',', '')
		occupancy = occupancy.rename(index=str, columns={'Share ID': 'Hotel Name'})
		occupancy = occupancy.drop('Unnamed: 0', axis=1)

		print('Time: %.4f' % (time() - start))

		# Load preprocessed data according to command-line "distance" parameter.
		print('\nReading in the pre-processed taxicab data.'); start = time()

		usecols = ['Hotel Name', 'Pick-up Time', 'Drop-off Time', 'Distance From Hotel']
		if trip_type == 'pickups':
			filename = os.path.join(preproc_data_path, 'destinations.csv')
		elif trip_type == 'dropoffs':
			filename = os.path.join(preproc_data_path, 'starting_points.csv')
		else:
			raise Exception('Expecting one of "pickups" or "dropoffs" for command-line argument "trip_type".')

		taxi_rides = pd.read_csv(filename, header=0, usecols=usecols)

		taxi_rides['Hotel Name'] = taxi_rides['Hotel Name'].apply(str.strip)
		taxi_rides['Pick-up Time'] = pd.to_datetime(taxi_rides['Pick-up Time'], format='%Y-%m-%d')
		taxi_rides['Drop-off Time'] = pd.to_datetime(taxi_rides['Drop-off Time'], format='%Y-%m-%d')
		taxi_rides = taxi_rides.loc[(taxi_rides['Pick-up Time'] >= start_date) & \
										(taxi_rides['Drop-off Time'] <= end_date)]
		taxi_rides['Date'] = taxi_rides['Pick-up Time'].dt.date
		taxi_rides = taxi_rides.drop(['Pick-up Time', 'Drop-off Time'], axis=1)

		print('Time: %.4f' % (time() - start))

		# Build the dataset of ((hotel, taxi density), occupancy) input, output pairs.
		print('\nMerging dataframes on Date and Hotel Name attributes.'); start = time()

		df = pd.merge(occupancy, taxi_rides, on=['Date', 'Hotel Name'])

		print('Time: %.4f' % (time() - start))

		# Save merged occupancy and taxi data to disk.
		print('\nSaving merged dataframes to disk.'); start = time()

		df.to_csv(os.path.join(taxi_occupancy_path, data_fname))

		print('Time: %.4f' % (time() - start))

		# Count number of rides per hotel and date.
		df = df.groupby(['Hotel Name', 'Date', 'Room Demand', 'ADR']).count().reset_index()
		df = df.rename(index=str, columns={'Distance From Hotel': 'No. Nearby Trips'})
		df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

		# Save occupancy and taxi counts to disk.
		print('\nSaving counts to disk.'); start = time()

		df.to_csv(os.path.join(taxi_occupancy_path, counts_fname))

		print('Time: %.4f' % (time() - start))

	elif is_data_file and not is_counts_file:
		# Load merged occupancy and taxi data to disk.
		print('\nLoading merged taxi and occupancy dataframes from disk.'); start = time()

		df = pd.read_csv(os.path.join(taxi_occupancy_path, data_fname))

		print('Time: %.4f' % (time() - start))

		# Count number of rides per hotel and date.
		df = df.groupby(['Hotel Name', 'Date', 'Room Demand', 'ADR']).count().reset_index()
		df = df.rename(index=str, columns={'Distance From Hotel': 'No. Nearby Trips'})
		df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

		# Save occupancy and taxi counts to disk.
		print('\nSaving counts to disk.'); start = time()

		df.to_csv(os.path.join(taxi_occupancy_path, counts_fname))

		print('Time: %.4f' % (time() - start))

	else:
		# Load merged occupancy and taxi data from disk.
		print('\nLoading occupancy and taxi counts from disk.'); start = time()

		df = pd.read_csv(os.path.join(taxi_occupancy_path, counts_fname))
		df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

		print('Time: %.4f' % (time() - start))
	
	return df

def encode_data(df, targets=['Occ']):
	# One-hot encoding the hotel IDs.
	hotels = np.array(df['Hotel Name'])
	hotel_names, hotels = np.unique(hotels, return_inverse=True)
	hotels = [hotels == i for i in range(max(hotels) + 1)]

	# Get the trip counts.
	if 'No. Nearby Trips' in df:
		print('*** Utilizing nearby taxi trip counts. ***')
		trips = list(df['No. Nearby Trips'])

	# One-hot encoding the day of the week.
	weekdays = np.array(df['Date'].dt.weekday)
	weekdays = [weekdays == i for i in range(7)]

	# One-hot encoding the month.
	months = np.array(df['Date'].dt.month)
	months = [months == i for i in range(1, 13)]

	# One-hot encoding the year.
	years = np.array(df['Date'].dt.year)
	years = [years == i for i in range(2014, 2017)]
	
	# Combining all observations into a design matrix.
	if 'No. Nearby Trips' in df:
		observations = np.array(hotels + weekdays + months + years + [trips]).T
	else:
		observations = np.array(hotels + weekdays + months + years).T

	# Get the target outputs (occupancy and room pricing).
	targets = np.array(df[targets], dtype=np.float32)
	
	return observations, targets