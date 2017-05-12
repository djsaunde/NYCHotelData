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

data_path = '../data/'
plots_path = '../plots/'


def plot_pickups(distance, days, times, year, month, day, map_type):
	'''
	Given a distance criterion and a specific time range or day in the historical taxicab data,
	plot the distribution of rides on a map of NYC. (days, times) is mutually exclusive with
	(year, month, day).

	distance: Distance in feet from hotel criterion.
	days: Day of week to include.
	times: Times of days to include.
	year: Year to consider.
	month: Month to consider.
	day: Day to consider.
	map_type: 'static' or 'gmap', for either ARCGIS NYC map queried from their website or Google
		Maps overlay.
	'''
	# get coordinates of new distance and time-constraint satisfying taxicab trips with nearby pick-ups
	pool = mp.Pool(len(taxicab_data['Hotel Name'].unique()))
	coords = pool.map(get_nearby_pickups, [ (taxicab_data.loc[taxicab_data['Hotel Name'] == hotel_name], \
								distance, days, times[0], times[1] - 1) for hotel_name in taxicab_data['Hotel Name'].unique() ])
	coords = dict([ (hotel_name, args) for (hotel_name, args) in zip(taxicab_data['Hotel Name'].unique(), coords) ])

	print 'Total satisfying nearby pick-up taxicab rides:', sum([single_hotel_coords.shape[1] \
											for single_hotel_coords in coords.values()]), '/', len(taxicab_data), '\n'
	print 'Satisfying nearby pick-up taxicab rides by hotel:'

	for name in coords:
		print '-', name, ':', coords[name].shape[1], 'satisfying taxicab rides'

	print '\n'

	start_time = default_timer()

	if map_type == 'static':
		# a one-liner to get all the ARCGIS maps plotted
		empirical_dists = Parallel(n_jobs=len(coords.keys())) (delayed(plot_arcgis_nyc_map)((coords[hotel_name][0], coords[hotel_name][1]), \
																hotel_name, plots_path + hotel_name + '_Jan2016_' + str(distance) + 'ft_pickups_' + \
																','.join([ str(day) for day in days ]) + '_weekdays_' + str(times[0]) + '_' + \
																str(times[1]) + '_start_end_hours_heatmap.png') for hotel_name in coords.keys())
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

	print 'It took', default_timer() - start_time, 'seconds to plot the heatmaps\n'

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

	plt.title('Kullbeck-Liebler divergence between empirical distributions of hotel taxicab pickups from nearby pickups')
	plt.xticks(idxs + width / float(len(coords.keys())), coords.keys(), rotation=15)
	plt.legend(coords.keys(), loc=1, fontsize='xx-small')
	plt.show()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('--distance', type=int, default=100)
	parser.add_argument('--days', type=list, default=[0, 1, 2, 3, 4, 5, 6])
	parser.add_argument('--times', type=tuple, default=(0, 24))
	parser.add_argument('--year', type=int, default=None)
	parser.add_argument('--month', type=int, default=None)
	parser.add_argument('--day', type=int, default=None)
	parser.add_argument('--map_type', type=str, default='static')

	args = parser.parse_args()
	distance, days, times, year, month, day, map_type = args.distance, args.days, args.times, args.year, args.month, args.day, args.map_type

	print '\n- loading nearby pick-ups taxicab ride data (pilot set of 10 hotels)'

	# load up the workbook and worksheet for working with trips that have nearby pick-up locations
	start_time = default_timer()
	if 'nearby_pickups.p' not in os.listdir(data_path):
		print '- loading data from disk'
		taxicab_data = pd.read_excel(data_path + 'Nearby Pickups and Dropoffs.xlsx', sheetname='Nearby Pick-ups')
		p.dump(nearby_pickups, open(data_path + 'nearby_pickups.p', 'wb'))
	else:
		print '- loading data from pickled object file'
		taxicab_data = p.load(open(data_path + 'nearby_pickups.p', 'rb'))

	print '- it took', default_timer() - start_time, 'seconds to load the nearby pick-ups data\n'

	plot_pickups(distance, days, times, year, month, day, map_type)