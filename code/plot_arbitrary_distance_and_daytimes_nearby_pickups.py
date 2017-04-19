# imports...
import numpy as np
import cPickle as p
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt
import csv, imp, os, gmplot, webbrowser, os

from IPython.display import Image, display
from IPython.core.display import HTML

from datetime import datetime
from timeit import default_timer
from scipy.stats import entropy
from joblib import Parallel, delayed

import warnings
warnings.filterwarnings('ignore')

# importing helper methods
from util import *

#################################################
# Pick-ups: Arbitrary Day of Week / Time of Day #
#################################################

print '\n'
print '-----------------------------------------------------'
print '--- Pick-ups: Arbitrary Day of Week / Time of Day ---'
print '-----------------------------------------------------'
print '\n'

print '- loading nearby pick-ups taxicab ride data (pilot set of 10 hotels)'

# load up the workbook and worksheet for working with trips that have nearby pick-up locations
start_time = default_timer()
if 'nearby_pickups.p' not in os.listdir('../data/'):
	print '- loading data from disk'
	nearby_pickups = pd.read_excel('../data/Nearby Pickups and Dropoffs.xlsx', sheetname='Nearby Pick-ups')
	p.dump(nearby_pickups, open('../data/nearby_pickups.p', 'wb'))
else:
	print '- loading data from pickled Python object file'
	nearby_pickups = p.load(open('../data/nearby_pickups.p', 'rb'))

print '- it took', default_timer() - start_time, 'seconds to load the nearby pick-ups data\n'

# distance criterion in feet
distance = raw_input('Enter distance (in feet) from hotel criterion (default 100): ')
if distance == '':
	distance = 100
else:
	distance = int(distance)

# days of week criterion
days = raw_input('Enter comma-separated list of days as integers (0-6) for days of week criterion (default all): ')
if days == '':
	days = [0, 1, 2, 3, 4, 5, 6]
else:
	days = [ int(day) for day in days.split(',') ]

# hours of day criterion (start hour)
start_hour = raw_input('Enter hour to begin searching for taxicab trips (default 0 -> 12:00AM): ')
if start_hour == '':
	start_hour = 0
else:
	start_hour = int(start_hour)

# hours of day criterion (end hour)
end_hour = raw_input('Enter hour to end searching for taxicab trips (default 24 -> 11:59PM): ')
if end_hour == '':
	end_hour = 24
else:
	end_hour = int(end_hour)

# choose type of map to draw
map_type = raw_input('Enter "gmap" or "static" to choose which to plot (default "static"): ')
if map_type == '':
	map_type = 'static'

print '\n'

# get coordinates of new distance and time-constraint satisfying taxicab trips with nearby pick-ups
pool = mp.Pool(len(nearby_pickups['Hotel Name'].unique()))
coords = pool.map(pickups_arbitrary_times, [ (nearby_pickups.loc[nearby_pickups['Hotel Name'] == hotel_name], distance, days, start_hour, end_hour - 1) for hotel_name in nearby_pickups['Hotel Name'].unique() ])
coords = dict([ (hotel_name, args) for (hotel_name, args) in zip(nearby_pickups['Hotel Name'].unique(), coords) ])

print 'Total satisfying nearby pick-up taxicab rides:', sum([single_hotel_coords.shape[1] for single_hotel_coords in coords.values()]), '/', len(nearby_pickups), '\n'
print 'Satisfying nearby pick-up taxicab rides by hotel:'

for name in coords:
	print '-', name, ':', coords[name].shape[1], 'satisfying taxicab rides'

print '\n'

###################################
# Drawing Plots of Nearby Pickups #
###################################

start_time = default_timer()

if map_type == 'static':
	# a one-liner to get all the ARCGIS maps plotted
	empirical_dists = Parallel(n_jobs=len(coords.keys())) (delayed(plot_arcgis_nyc_map)((coords[hotel_name][0], coords[hotel_name][1]), hotel_name, '../img/' + hotel_name + '_Jan2016_' + str(distance) + 'ft_pickups_' + ','.join([ str(day) for day in days ]) + '_weekdays_' + str(start_hour) + '_' + str(end_hour) + '_start_end_hours_heatmap.png') for hotel_name in coords.keys())
else:
	for hotel_name in coords.keys():
		# some map parameters
		map_name = hotel_name + '_Jan2016_' + str(distance) + 'ft_pickups_' + ','.join([ str(day) for day in days ]) + '_weekdays_' + str(start_hour) + '_' + str(end_hour) + '_start_end_hours_heatmap.html'
		filepath = '../img/' + map_name[:-5] + '.png'

		# get the Google maps area we wish to plot at
		gmap = gmplot.GoogleMapPlotter(np.median(coords[hotel_name][0]), np.median(coords[hotel_name][1]), 13)

		# plot the map
		gmap.heatmap(coords[hotel_name][0], coords[hotel_name][1], threshold=10, radius=1, gradient=None, opacity=0.6, dissipating=False)

		# draw the map
		gmap.draw('../img/' + map_name)

		# display it in the web browser
		webbrowser.open('../img/' + map_name)

print 'It took', default_timer() - start_time, 'seconds to plot the heatmaps'

print '\n'
kl_diverges = []
for dist1 in empirical_dists:
	cur_diverges = []
	for dist2 in empirical_dists:
		cur_diverges.append(entropy(dist1, dist2))
	kl_diverges.append(cur_diverges)

# print kl_diverges, '\n'

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