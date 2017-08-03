import os
import numpy as np
import cPickle as p
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt
import csv, imp, os, gmplot, webbrowser, os, argparse

from IPython.display import Image, display
from IPython.core.display import HTML

from datetime import datetime
from timeit import default_timer
from scipy.stats import entropy
from joblib import Parallel, delayed

import warnings
warnings.filterwarnings('ignore')

from util import *

data_path = os.path.join('..', 'data', 'preprocessed')
plots_path = os.path.join('..', 'plots')


def plot_heatmap(taxi_data, prefix, distance, days, times, map_type):
	'''
	distance: Distance in feet from hotel criterion.
	days: Day of week to include.
	times: Times of days to include.
	map_type: 'static' or 'gmap', for either ARCGIS NYC map queried from their website or Google
		Maps overlay.
	'''

	# get coordinates of new distance and time-constraint satisfying taxicab trips with nearby pick-ups
	for key in taxi_data.keys():
		data = taxi_data[key]

		pool = mp.Pool(len(data['Hotel Name'].unique()))
		coords = pool.map(get_nearby_pickups, [ (data.loc[data['Hotel Name'] == hotel_name], \
								distance, days, times[0], times[1] - 1) for hotel_name in data['Hotel Name'].unique() ])
		coords = dict([ (hotel_name, args) for (hotel_name, args) in zip(data['Hotel Name'].unique(), coords) ])

		print 'Total satisfying nearby', key, ':', sum([single_hotel_coords.shape[1] \
											for single_hotel_coords in coords.values()]), '/', len(data), '\n'
		
		print 'Satisfying nearby', key, 'by hotel :'
		for name in coords:
			print '-', name, ':', coords[name].shape[1], 'satisfying taxicab rides'

		print '\n'

	start_time = default_timer()

	if map_type == 'static':
		# a one-liner to get all the ARCGIS maps plotted
		directory = prefix + str(distance) + 'ft_' + '_'.join([ key for key in taxi_data.keys() ]) + '_'.join([ str(day) for day in days ]) \
				+ '_weekdays_' + str(times[0]) + '_' + str(times[1]) + '_start_end_hours_heatmap.png'
		empirical_dists = Parallel(n_jobs=len(coords.keys())) (delayed(plot_arcgis_nyc_map)((coords[hotel_name][0], coords[hotel_name][1]),
							hotel_name, os.path.join(plots_path, directory)) for hotel_name in coords.keys())
	else:
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

	print '\nIt took', default_timer() - start_time, 'seconds to plot the heatmaps\n'

	kl_diverges = []
	for dist1 in empirical_dists:
		cur_diverges = []
		for dist2 in empirical_dists:
			cur_diverges.append(entropy(dist1, dist2))
		kl_diverges.append(cur_diverges)

	kl_diverges = np.array(kl_diverges)

	width = 0.9 / float(len(coords.keys()))
	idxs = np.arange(len(coords.keys()))

	plt.figure(figsize=(18, 9.5))

	for idx in xrange(len(coords.keys())):
		plt.bar(idxs + width * idx, kl_diverges[idx, :], width)

	plt.title('Kullbeck-Liebler divergence between empirical distributions of hotel trips')
	plt.xticks(idxs + width / float(len(coords.keys())), coords.keys(), rotation=15)
	plt.legend(coords.keys(), loc=1, fontsize='xx-small')
	plt.show()


def load_data(to_plot, data_file):

	if data_file != 'Nearby Pickups and Dropoffs.xlsx':
		prefix = data_file.split('.')[0] + '_'
	else:
		prefix = ''

	if to_plot == 'pickups':
		picklenames = [ prefix + 'nearby_pickups.p' ]
		sheetnames = [ 'Destinations' ]
		dictnames = [ 'pickups' ]
	elif to_plot == 'dropoffs':
		picklenames = [ prefix + 'nearby_dropoffs.p' ]
		sheetnames = [ 'Starting Points' ]
		dictnames = [ 'dropoffs' ]
	elif to_plot == 'both':
		picklenames = [ prefix + 'nearby_pickups.p', prefix + 'nearby_dropoffs.p' ]
		sheetnames = [ 'Destinations', 'Starting Points' ]
		dictnames = [ 'pickups', 'dropoffs' ]

	print '\n... Loading taxicab trip data (pilot set of 24 hotels)'

	start_time = timeit.default_timer()

	taxi_data = {}
	for pname, sname, dname in zip(picklenames, sheetnames, dictnames):
		if pname not in os.listdir(data_path):
			print '... Loading data from disk.'
			taxi_data[dname] = pd.read_excel(os.path.join(data_path, data_file), sheetname=sname)
			p.dump(taxi_data[dname], open(os.path.join(data_path, pname), 'wb'))
		else:
			print '... Loading data from pickled object file.'
			taxi_data[dname] = p.load(open(os.path.join(data_path, pname), 'rb'))

	print '... It took', default_timer() - start_time, 'seconds to load the taxicab trip data\n'

	return taxi_data, prefix


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('--to_plot', type=str, default='both')
	parser.add_argument('--data_file', type=str, default='Nearby Pickups and Dropoffs.xlsx')
	parser.add_argument('--distance', type=int, default=300)
	parser.add_argument('--days', type=list, default=[0, 1, 2, 3, 4, 5, 6])
	parser.add_argument('--times', type=tuple, default=(0, 24))
	parser.add_argument('--map_type', type=str, default='static')

	# parse arguments and place them in local scope
	args = vars(parser.parse_args())
	locals().update(args)

	# get dictionary of taxicab trip data based on `to_plot` argument
	taxi_data, prefix = load_data(to_plot, data_file)

	plot_heatmap(taxi_data, prefix, distance, days, times, map_type)
