from __future import print_function

import os
import sys
import argparse
import numpy as np
import multiprocess
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from geopy.distance import vincenty
from geopy.geocoders import GoogleV3
from joblib import Parallel, delayed
from datetime import date, timedelta, datetime

input_path = os.path.join('..', 'data', 'daily_distributions')
output_path = os.path.join('..', 'data', 'daily_sampled_pairwise_distances')
hotel_list_path = os.path.join('..', 'data', 'Final hotel Identification (with coordinates).xlsx')

api_key='AIzaSyAWV7aBLcawx2WyMO7fM4oOL9ayZ_qGz-Y'

if not os.path.isdir(output_path):
	os.makedirs(output_path)


def sample_data(distro_mapping, n_samples=100):
	samples_daily_distribution = {}
	for day, coordinates in sorted(distro_mapping.items()):		
		if len(coordinates) < n_samples:
			continue

		samples_daily_distribution[day] = coordinates[np.random.choice(len(coordinates), n_samples, replace=False)]

	return samples_daily_distribution


def get_hotel_coordinates():
	df = pd.read_excel(hotel_list_path, sheetname='final match with coordinates')
	hotel_coords = { df['Name'][idx] : (df['Latitude'][idx], df['Longitude'][idx]) for idx in xrange(len(df)) }

	return hotel_coords


def get_pairwise_differences(sampled_taxi_data, hotel_coordinates):
	pairwise_differences = pd.DataFrame()

	for day, taxi_coords in tqdm(sorted(sampled_taxi_data.items())):
		for hotel, hotel_coord in sorted(hotel_coordinates.items()):
			pairwise_difference = []
			for taxi_coord in taxi_coords:
				try:
					pairwise_difference.append(vincenty(hotel_coord, taxi_coord).feet)
				except UnboundLocalError:
					continue

			pairwise_differences = pairwise_differences.append(pd.Series(pairwise_difference), ignore_index=True)

	tuples = list(zip(sorted(sampled_taxi_data.keys() * len(hotel_coordinates.keys())), \
						sorted(hotel_coordinates.keys()) * len(sampled_taxi_data.keys())))

	pairwise_differences.index = pd.MultiIndex.from_tuples(tuples, names=['Day', 'Hotel Name'])
	pairwise_differences.columns = xrange(1, 101, 1)

	return pairwise_differences


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--start_date', type=int, nargs=3, default=[2013, 1, 1], \
				help='The day on which to start looking for satisfying coordinates.')
	parser.add_argument('--end_date', type=int, nargs=3, default=[2013, 1, 7], \
				help='The day on which to stop looking for satisfying coordinates.')
	parser.add_argument('--coord_type', type=str, default='pickups', help='The type of \
						coordinates to look for (one of "pickups", "dropoffs", or "both").')
	parser.add_argument('--distance', type=int, default=100, help='The distance \
					(in feet) from hotels for which to look for satisfying taxicab trips.')
	parser.add_argument('--n_jobs', type=int, default=4, help='The number of \
								CPU cores to use in processing the taxicab data.')
	parser.add_argument('--n_samples', type=int, default=100, help='The number of samples of \
									geospatial coordinates to draw for each daily distribution.')

	args = parser.parse_args()
	args = vars(args)

	# place command-line arguments into local scope
	locals().update(args)

	if coord_type == 'both':
		coord_type = 'pickups_dropoffs'

	start_date, end_date = date(*start_date), date(*end_date)

	filename = os.path.join(input_path, '_'.join([ coord_type, str(distance), str(start_date), str(end_date) ]) + '.xlsx')

	print('\n...Reading in daily distributions workbook.')

	try:
		daily_distributions = pd.read_excel(filename, header=None)
	except Exception:
		raise Exception('Reading in the daily distributions workbook caused an error.')

	print('\n...Mapping dates to taxicab coordinates in a dictionary data structure.')

	distro_mapping = { date(*[ int(item) for item in row[0].encode('ascii', 'ignore').split('-') ]) : \
				[ datum for datum in row[1:][~row[1:].isnull()] ] for idx, row in daily_distributions.iterrows() }
	distro_mapping = { key : [ item.encode('ascii', 'ignore') for item in value ] for key, value in distro_mapping.items() }
	distro_mapping = { key : np.array([ (float(item.split()[0]), float(item.split()[1])) \
								for item in value ]) for key, value in distro_mapping.items() }

	print('\n...Sampling daily taxicab coordinates.')

	sampled_data = sample_data(distro_mapping, n_samples=n_samples)

	print('\n...Geolocating hotels.')

	hotel_coordinates = get_hotel_coordinates()

	print('\n...Calculating pairwise (hotel, coordinate) distances by day.')

	pairwise_differences = get_pairwise_differences(sampled_data, hotel_coordinates)

	print('\n...Writing calculated distances to file.')

	writer = pd.ExcelWriter(os.path.join(output_path, '_'.join([coord_type, str(distance), str(start_date), \
				str(end_date), str(n_samples)]) + '.xlsx'), engine='xlsxwriter', date_format='mmmm dd yyyy')

	pairwise_differences.to_excel(writer, float_format='%02d')

	workbook = writer.book
	worksheet = writer.sheets['Sheet1']
	worksheet.set_column('A:A', 20)
	worksheet.set_column('B:B', 65)

	writer.save()
