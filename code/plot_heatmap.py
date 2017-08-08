import os
import sys
import csv
import imp
import gmplot
import argparse
import webbrowser
import numpy as np
import pandas as pd
import cPickle as p
import multiprocessing as mp
import matplotlib.pyplot as plt

from datetime import datetime
from scipy.stats import entropy
from timeit import default_timer
from joblib import Parallel, delayed
from IPython.core.display import HTML
from IPython.display import Image, display

import warnings
warnings.filterwarnings('ignore')

from util import *

data_path = os.path.join('..', 'data', 'preprocessed')
plots_path = os.path.join('..', 'plots')


def plot_heatmap(taxi_data, distance, days, times, map_type):
	'''
	taxi_data: Dictionary of (data type, pandas DataFrame of taxicab coordinates and metadata)
	distance: Distance in feet from hotel criterion.
	days: Day of week to include.
	times: Times of days to include.
	map_type: 'static' or 'gmap', for either ARCGIS NYC map queried from their website or Google Maps overlay.
	'''
	# get coordinates of new distance and time-constraint satisfying taxicab trips with nearby pick-ups
	all_coords = {}
	for key in taxi_data.keys():
		data = taxi_data[key]

		# Create a number of CPU workers equal to the number of hotels in the data
		pool = mp.Pool(len(data['Hotel Name'].unique()))

		# Get the pick-up (drop-off) coordinates of the trip which ended (began) near this each hotel
		if key == 'pickups':
			coords = pool.map(get_nearby_pickups, [ (data.loc[data['Hotel Name'] == hotel_name], \
							distance, days, times[0], times[1] - 1) for hotel_name in data['Hotel Name'].unique() ])
		elif key == 'dropoffs':
			coords = pool.map(get_nearby_dropoffs, [ (data.loc[data['Hotel Name'] == hotel_name], \
							distance, days, times[0], times[1] - 1) for hotel_name in data['Hotel Name'].unique() ])
		coords = { hotel_name : args for (hotel_name, args) in zip(data['Hotel Name'].unique(), coords) }

		print 'Total satisfying nearby', key, ':', sum([single_hotel_coords.shape[1] \
											for single_hotel_coords in coords.values()]), '/', len(data), '\n'
		
		print 'Satisfying nearby', key, 'by hotel :'
		for name in coords:
			print '-', name, ':', coords[name].shape[1], 'satisfying taxicab rides'

		all_coords = all_coords.update(coords)

		print '\n'

	coords = all_coords

	start_time = default_timer()

	if map_type == 'static':
		# a one-liner to get all the ARCGIS maps plotted
		directory = str(distance) + 'ft_' + '_'.join([ key for key in taxi_data.keys() ]) + '_'.join([ str(day) for day in days ]) \
				+ '_weekdays_' + str(times[0]) + '_' + str(times[1]) + '_start_end_hours_heatmap.png'
		empirical_dists = Parallel(n_jobs=len(coords.keys())) (delayed(plot_arcgis_nyc_map)((coords[hotel_name][0], coords[hotel_name][1]),
							hotel_name, os.path.join(plots_path, directory)) for hotel_name in coords.keys())
	elif map_type == 'gmap':
		for hotel_name in coords.keys():
			# some map parameters
			map_name = hotel_name + '_Jan2016_' + str(distance) + 'ft_pickups_' + ','.join([ str(day) for day in days ]) + \
														'_weekdays_' + str(times[0]) + '_' + str(times[1]) + '_start_end_hours_heatmap.html'
			filepath = plots_path + map_name[:-5] + '.png'

			# get the Google maps area we wish to plot at
			gmap = gmplot.GoogleMapPlotter(np.median(coords[hotel_name][0]), np.median(coords[hotel_name][1]), 13)

			# plot the map
			gmap.heatmap(coords[hotel_name][0], coords[hotel_name][1], threshold=10, radius=1, gradient=None, opacity=0.6, dissipating=False)

			# draw the map
			gmap.draw(plots_path + map_name)

			# display it in the web browser
			webbrowser.open(plots_path + map_name)

	else:
		raise Exception('Expecting map type of "static" or "gmap".')

	print '\nIt took', default_timer() - start_time, 'seconds to plot the heatmaps\n'

	return empirical_dists, coords.keys()


def plot_KL_divergences(empirical_distributions, keys)
	'''
	To Do.
	'''
	kl_diverges = []
	for dist1 in empirical_dists:
		cur_diverges = []
		for dist2 in empirical_dists:
			cur_diverges.append(entropy(dist1, dist2))
		kl_diverges.append(cur_diverges)

	kl_diverges = np.array(kl_diverges)

	width = 0.9 / float(kl_diverges.shape[0])
	idxs = np.arange(kl_diverges.shape[0])

	plt.figure(figsize=(18, 9.5))

	for idx in xrange(len(coords.keys())):
		plt.bar(idxs + width * idx, kl_diverges[idx, :], width)

	plt.title('Kullbeck-Liebler divergence between empirical distributions of hotel trips')
	plt.xticks(idxs + width / float(len(coords.keys())), coords.keys(), rotation=15)
	plt.legend(coords.keys(), loc=1, fontsize='xx-small')
	plt.show()

	return kl_diverges


def load_data(to_plot, data_files):
	'''
	Load the pre-processed taxi data file(s) needed for 
	'''
	if to_plot == 'pickups':
		picklenames = [ 'nearby_pickups.p' ]
		dictnames = [ 'pickups' ]
	elif to_plot == 'dropoffs':
		picklenames = [ 'nearby_dropoffs.p' ]
		dictnames = [ 'dropoffs' ]
	elif to_plot == 'both':
		picklenames = [ 'nearby_pickups.p', 'nearby_dropoffs.p' ]
		dictnames = [ 'pickups', 'dropoffs' ]

	print '\n... Loading taxicab trip data (pilot set of 24 hotels)'

	start_time = timeit.default_timer()

	taxi_data = {}
	for pname, dname, data_file in zip(picklenames, dictnames, data_files):
		if pname not in os.listdir(data_path):
			print '... Loading data from disk.'
			taxi_data[dname] = pd.read_csv(os.path.join(data_path, data_file))
			p.dump(taxi_data[dname], open(os.path.join(data_path, pname), 'wb'))
		else:
			print '... Loading data from pickled object file.'
			taxi_data[dname] = p.load(open(os.path.join(data_path, pname), 'rb'))

	print '... It took', default_timer() - start_time, 'seconds to load the taxicab trip data\n'

	return taxi_data


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('--to_plot', type=str, default='both', \
						help='Whether to plot data from nearby "pickups", "dropoffs", or "both"')
	parser.add_argument('--distance', type=int, default=100, 'Distance from hotel criterion (in feet).')
	parser.add_argument('--map_type', type=str, default='static', \
						help='Plot a heatmap in Google Maps or using an ARCGIS map of NYC.')

	subparsers = parser.add_subparsers(title='subcommands', dest='subcommand')

	times_subparser = subparsers.add_parser('times', help='Parser for heatmap plotting \
													by days of the week and times of day.')

	times_subparser.add_argument('--days', type=list, default=[0, 1, 2, 3, 4, 5, 6], \
				help='Days of the week indexed by the integers 0, 1, ..., 6 for Sunday, Monday, ..., Saturday.')
	times_subparser.add_argument('--times', type=tuple, default=(0, 24), \
											help='Start and ending time (in hours) to look for data.')

	window_subparser = subparsers.add_parser('window', help='Parser for heatmap plotting a specific window of time.')

	window_subparser.add_argument('--start_datetime', type=tuple, default=(2012, 6, 26, 9), 'Tuple giving year, \
										month, day, and hour of the time from which to start looking for data.')
	window_subparser.add_argument('--end_datetime', type=tuple, default=(2012, 6, 26, 17), 'Tuple giving year, \
										month, day, and hour of the time at which to stop looking for data.')


	# parse arguments and place them in local scope
	locals().update(args)

	if subcommand not in [ 'times', 'window' ]:
        raise Exception('Specify either "times" or "window".')

    if subcommand == 'window':
    	start_datetime = datetime(start_datetime[0], start_datetime[1], start_datetime[2], start_datetime[3])
		end_datetime = datetime(end_datetime[0], end_datetime[1], end_datetime[2], end_datetime[3])

	if to_plot == 'pickups':
		data_files = [ 'destinations.csv' ]
	elif to_plot == 'dropoffs':
		data_files = [ 'starting_points.csv' ]
	elif to_plot == 'both':
		data_files = [ 'destinations.csv', 'starting_points.csv' ]		

	# get dictionary of taxicab trip data based on `to_plot` argument
	taxi_data = load_data(to_plot, data_files)

	empirical_distributions, keys = plot_heatmap(taxi_data, distance, days, times, map_type)

	kl_divergences = plot_KL_divergences(empirical_distributions, keys)