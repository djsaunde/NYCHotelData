from __future__ import division

from vincenty import vincenty
from datetime import timedelta	
from contextlib import closing
from scipy.stats import entropy
from multiprocessing import Pool
from collections import defaultdict
from joblib import Parallel, delayed
from mpl_toolkits.basemap import Basemap
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

import sys
import dask
import numpy as np
import pandas as pd
import cPickle as p
import dask.dataframe as dd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.colors as mcolors
import math, itertools, timeit, multiprocessing, os, multiprocess


class BatchCompletionCallBack(object):
	completed = defaultdict(int)

	def __init__(self, time, index, parallel):
		self.index = index
		self.parallel = parallel

	def __call__(self, index):
		BatchCompletionCallBack.completed[self.parallel] += 1
		print("done with {}".format(BatchCompletionCallBack.completed[self.parallel]))
		if self.parallel._original_iterator is not None:
			self.parallel.dispatch_next()

import joblib.parallel
joblib.parallel.BatchCompletionCallBack = BatchCompletionCallBack

# print out entire numpy arrays
np.set_printoptions(threshold=np.nan)

# tolerance level for intersection point
EPS = 0.000001


def daterange(start_date, end_date):
	for n in xrange(int((end_date - start_date).days) + 1):
		yield start_date + timedelta(n)


def load_data(to_plot, data_files, data_path, chunksize=5000000):
	'''
	Load the pre-processed taxi data file(s) needed for plotting empirical pick-up / drop-off point distributions as heatmaps.

	to_plot: Whether to plot the distribution of pick-up coordinates of trips which end near the hotel of interest, the distribution of 
	drop-off coordinates of trips which begin near the hotel of interest, or both.
	data_files: The .csv files in which the pre-processed geospatial coordinate data is stored.
	'''
	if to_plot == 'pickups':
		dictnames = [ 'pickups' ]
	elif to_plot == 'dropoffs':
		dictnames = [ 'dropoffs' ]
	elif to_plot == 'both':
		dictnames = [ 'pickups', 'dropoffs' ]

	print '\n... Loading taxicab trip data (pilot set of 24 hotels)'

	start_time = timeit.default_timer()

	taxi_data = {}
	for dname, data_file in zip(dictnames, data_files):
		print '... Loading', dname, 'data from disk.'
		taxi_data[dname] = dd.read_csv(os.path.join(data_path, data_file), parse_dates=['Pick-up Time', \
											'Drop-off Time'], dtype={'Fare Amount' : 'object'}).dropna()

	print '... It took', timeit.default_timer() - start_time, 'seconds to load the taxicab trip data\n'

	return taxi_data


def plot_arcgis_nyc_map(coords, hotel_name, directory, service='World_Street_Map', xpixels=800, dpi=150, title=None):
	'''
	Given a set of (longitude, latitude) coordinates, plot a heatmap of them onto an ARCGIS basemap of NYC.
	'''

	if title == None:
		print '- plotting scatter plot for', hotel_name, '\n'
	else:
		print '- plotting scatter plot for', title, '\n'

	# size of figures in inches
	plt.rcParams["figure.figsize"] = (18.5, 9.75)

	# create Basemap object (bounds NYC) and draw high-resolution map of NYC on it
	basemap = Basemap(llcrnrlon=-74.025, llcrnrlat=40.63, urcrnrlon=-73.76, urcrnrlat=40.85, epsg=4269)
	basemap.arcgisimage(service=service, xpixels=xpixels, dpi=dpi)

	# set up grid coordinates and get binned coordinates from taxicab data
	x, y = np.linspace(basemap.llcrnrlon, basemap.urcrnrlon, 250), np.linspace(basemap.llcrnrlat, basemap.urcrnrlat, 250)
	bin_coords, xedges, yedges = np.histogram2d(coords[1], coords[0], bins=(x, y), normed=True)
	x, y = np.meshgrid(xedges, yedges)
	to_draw = np.ma.masked_array(bin_coords, bin_coords < 0.001) / np.sum(bin_coords)

	# plot binned coordinates onto the map, use colorbar
	plt.pcolormesh(x, y, to_draw.T, cmap='rainbow', vmin=0.001, vmax=1.0)
	plt.colorbar(norm=mcolors.NoNorm)

	# title map and save it to disk
	if title == None:
		plt.title(hotel_name + ' (satisfying trips: ' + str(len(coords[1])) + ')')
	else:
		plt.title(title + ' (satisfying trips: ' + str(len(coords[1])) + ')')

	if not os.path.isdir(directory):
		os.makedirs(directory)

	# save ARCGIS plot out to disk to inspect later
	plt.savefig(os.path.join(directory, hotel_name + '.png'))

	# close out the plot to avoid multiple colorbar bug
	plt.clf()
	plt.close()

	# setting zero-valued bins to small non-zero values (for KL divergence)
	bin_coords[np.where(bin_coords == 0)] = 1e-32

	normed_distro = bin_coords / np.sum(bin_coords)

	# return normalized, binned coordinates in one-dimensional vector
	return np.ravel(normed_distro)


def plot_arcgis_nyc_scatter_plot(coords, hotel_name, directory, service='World_Street_Map', xpixels=800, dpi=150, title=None):
	'''
	Given a set of (longitude, latitude) coordinates, plot a heatmap of them onto an ARCGIS basemap of NYC.
	'''

	if title == None:
		print '- plotting scatter plot for', hotel_name
	else:
		print '- plotting scatter plot for', title

	# size of figures in inches
	plt.rcParams["figure.figsize"] = (18.5, 9.75)

	# create Basemap object (bounds NYC) and draw high-resolution map of NYC on it
	basemap = Basemap(llcrnrlon=-74.025, llcrnrlat=40.63, urcrnrlon=-73.76, urcrnrlat=40.85, epsg=4269)
	basemap.arcgisimage(service=service, xpixels=xpixels, dpi=dpi)

	# set up grid coordinates and get binned coordinates from taxicab data
	x, y = np.linspace(basemap.llcrnrlon, basemap.urcrnrlon, 250), np.linspace(basemap.llcrnrlat, basemap.urcrnrlat, 250)
	bin_coords, xedges, yedges = np.histogram2d(coords[1], coords[0], bins=(x, y), normed=True)
	x, y = np.meshgrid(xedges, yedges)
	to_draw = np.ma.masked_array(bin_coords, bin_coords < 0.001) / np.sum(bin_coords)

	# plot binned coordinates onto the map, use colorbar
	plt.scatter(coords[1], coords[0], s=5)

	# title map and save it to disk
	if title == None:
		plt.title(hotel_name + ' (satisfying trips: ' + str(len(coords[1])) + ')')
	else:
		plt.title(title + ' (satisfying trips: ' + str(len(coords[1])) + ')')

	if not os.path.isdir(directory):
		os.makedirs(directory)

	# save ARCGIS plot out to disk to inspect later
	plt.savefig(os.path.join(directory, hotel_name + '.png'))

	# close out the plot to avoid multiple colorbar bug
	plt.clf()
	plt.close()

	# setting zero-valued bins to small non-zero values (for KL divergence)
	bin_coords[np.where(bin_coords == 0)] = 1e-32

	normed_distro = bin_coords / np.sum(bin_coords)

	# return normalized, binned coordinates in one-dimensional vector
	return np.ravel(normed_distro)


def get_nearby_pickups_times(args):
	'''
	Given the days of the week and the start and end times from which to look, return all satisfying
	taxicab rides which begin nearby.
	
	pickups: a pandas DataFrame containing all fields of data from pickups of a single hotel
	distance: a new distance used to cut out even more trips based on distance from the given hotel
	days: the days of the week for which to look (0 -> Monday, 1 -> Tuesday, ... as per pandas documentation).
	start_hour: the hour of day to begin with for which to look.
	end_time: the end hour of day to begin with for which to look.
	'''
	pickups, distance, days, start_hour, end_hour = args

	# cast date-time column to pandas Timestamp type
	pickups['Pick-up Time'] = pd.to_datetime(pickups['Pick-up Time'])
		
	# get the latitude, longitude coordinates of the corresponding pick-up locations for the trips
	hotel_matches = pickups.loc[pickups['Distance From Hotel'] <= distance]
	
	# get all time-constraint satisfying nearby pickup taxicab records for this hotel
	for day in days:
		satisfying_locations = hotel_matches[hotel_matches['Pick-up Time'].dt.weekday_name == day]
	
	satisfying_locations = hotel_matches[hotel_matches['Pick-up Time'].dt.hour >= start_hour]
	satisfying_locations = hotel_matches[hotel_matches['Pick-up Time'].dt.hour <= end_hour]
	
	# add the satisfying locations for this hotel to our dictionary data structure
	satisfying_coords = np.array(zip(satisfying_locations['Latitude'], satisfying_locations['Longitude'])).T
	
	# return the satisfying nearby pick-up coordinates
	return satisfying_coords


def get_nearby_dropoffs_times(args):
	'''
	Given the days of the week and the start and end times from which to look, return all satisfying
	taxicab rides which begin nearby.
	
	dropoffs: a pandas DataFrame containing all fields of data from dropoffs of a single hotel
	distance: a new distance used to cut out even more trips based on distance from the given hotel
	days: the days of the week for which to look (0 -> Monday, 1 -> Tuesday, ... as per pandas documentation).
	start_hour: the hour of day to begin with for which to look.
	end_time: the end hour of day to begin with for which to look.
	'''
	dropoffs, distance, days, start_hour, end_hour = args

	# cast date-time column to pandas Timestamp type
	dropoffs['Drop-off Time'] = pd.to_datetime(dropoffs['Drop-off Time'])
		
	# get the latitude, longitude coordinates of the corresponding pick-up locations for the trips
	hotel_matches = dropoffs.loc[dropoffs['Distance From Hotel'] <= distance]
	
	# get all time-constraint satisfying nearby pickup taxicab records for this hotel
	for day in days:
		satisfying_locations = hotel_matches[hotel_matches['Drop-off Time'].dt.weekday_name == day]
	
	satisfying_locations = hotel_matches[(hotel_matches['Drop-off Time'].dt.hour >= start_hour) & (hotel_matches['Drop-off Time'].dt.hour <= end_hour)]
	
	# add the satisfying locations for this hotel to our dictionary data structure
	satisfying_coords = np.array(zip(satisfying_locations['Latitude'], satisfying_locations['Longitude'])).T
	
	# return the satisfying nearby pick-up coordinates
	return satisfying_coords


def get_nearby_window(trips, distance, start_datetime, end_datetime):
	'''
	Given the days of the week and the start and end times from which to look, return all satisfying
	taxicab rides which begin nearby.
	
	trips: A pandas DataFrame containing all fields of data from taxicab trips.
	distance: A new distance used to cut out even more trips based on distance from the given hotel.
	start_datetime: The date at which to start looking for data.
	end_datetime: The date at which to stop looking for data.
	'''
	# Find the trips which first satisfy the specified distance criterion.
	print 'Getting trips which satisfy the specified distance criterion.'
	satisfying_distances = trips.loc[trips['Distance From Hotel'] <= distance]
	
	# Find the trips which also satisfy the time criterion.
	print 'Getting trips which satisfy the current time criterion.'
	satisfying_locations = satisfying_distances[(satisfying_distances['Pick-up Time'] >= \
					start_datetime) & (satisfying_distances['Pick-up Time'] <= end_datetime)]
														
	# Take only the (latitude, longitude) coordinates of these trips for downstream processing.
	satisfying_coords = satisfying_locations['Latitude'].astype(str) + ' ' + satisfying_locations['Longitude'].astype(str)
	return satisfying_coords


def get_intersection_point(centers, radii):
	'''
	Given four centers and radii, each specifying a circle, calculate their intersection point, if it exists. It should be
	unique.
	
	input:
		centers: coordinates in the (x, y) plane specifying where each of the circles' centers lie
		radii: the length of the circles' radii
		
	output:
		The unique intersection point of the four circles in the (x, y) plane, if it exists. If not, we output None.
	'''
	
	# store x and y dimensions and radius for easy-to-read code 
	x0, x1, x2, x3 = centers[0][0], centers[1][0], centers[2][0], centers[3][0]
	y0, y1, y2, y3 = centers[0][1], centers[1][1], centers[2][1], centers[3][1]
	r0, r1, r2, r3 = radii[0], radii[1], radii[2], radii[3]
	
	# store distances between circle centers (for circle 0 and 1)
	dx = x1 - x0
	dy = y1 - y0
		
	# determine the straight-line distance between the two centers
	dist = math.sqrt(dy ** 2 + dx ** 2)
		
	# check for solutions for the 2-circle case (do these circles intersect or does one contain the other?)
	if dist > r0 + r1:
		return None
	if dist < abs(r0 - r1):
		return None
	
	# calculate distance from the line through the circle intersection points and the line between the circle centers
	a = (r0 ** 2 - r1 ** 2 + dist ** 2) / (2.0 * dist)
	
	# determine coordinates of this point
	point_x = x0 + dx * (a / dist)
	point_y = y0 + dy * (a / dist)
	
	# determine distance from this point to either of the intersection points
	h = math.sqrt(r0 ** 2 - a ** 2)
	
	# determine the offsets of the intersection points from this point
	rx = -dy * (h / dist)
	ry = dx * (h / dist)
	
	# determine the absolute intersection points
	intersection1_x = point_x + rx
	intersection2_x = point_x - rx
	intersection1_y = point_y + ry
	intersection2_y = point_y - ry
	
	# determine if circle 3 intersects at either of the above intersection points
	dx = intersection1_x - x2
	dy = intersection1_y - y2
	d1 = math.sqrt(dy ** 2 + dx ** 2)
	
	dx = intersection2_x - x2
	dy = intersection2_y - y2
	d2 = math.sqrt(dy ** 2 + dx ** 2)
	
	# determine if circle 4 intersects at either of the above intersection points
	dx = intersection1_x - x3
	dy = intersection1_y - y3
	d3 = math.sqrt(dy ** 2 + dx ** 2)
	
	dx = intersection2_x - x3
	dy = intersection2_y - y3
	d4 = math.sqrt(dy ** 2 + dx ** 2)
	
	# check for intersection
	if abs(d1 - r2) < EPS and abs(d3 - r3) < EPS:
		return intersection1_x, intersection1_y
	elif abs(d2 - r2) < EPS and abs(d4 - r3) < EPS:
		return intersection2_x, intersection2_y
	return None

def other_vars_same(prop_row, manhattan_row, capacity):
	'''
	Returns true if all identifying columns are the same. This is to further ensure that we've found the best
	possible match for the hotel identity unmasking.
	
	input:
		prop_row: The first row.
		manhattan_row: The second row.
		capacity: The capacity of the hotel from row prop_row (Came from the "capacities by id" workbook).
		
	output:
		True if all constraints which should be satisfied, are.
	'''
	
	# checking that the rows match on the Operation column
	row1_op, row2_op = prop_row['Operation'], manhattan_row['Operation']
	if not (row1_op == 1 and row2_op == 'Chain Management' or row1_op == 2 and row2_op == 'Franchise' or row1_op == 3 and row2_op == 'Independent'):
		return False

	# checking that the rows match on the Scale column
	row1_scale, row2_scale = prop_row['Scale'], manhattan_row['Scale']
	if not (row1_scale == 1 and row2_scale == 'Luxury Chains' or row1_scale == 2 and row2_scale == 'Upper Upscale Chains' or row1_scale == 3 and row2_scale == 'Upscale Chains' or row1_scale == 4 and row2_scale == 'Upper Midscale Chains' or row1_scale == 5 and row2_scale == 'Midscale Chains' or row1_scale == 6 and row2_scale == 'Economy Chains' or row1_scale == 7 and row2_scale == 'Independents'):
		return False
	
	# checking that the rows match on the Class column
	row1_class, row2_class = prop_row['Class'], manhattan_row['Class']
	if not (row1_class == 1 and row2_class == 'Luxury Class' or row1_class == 2 and row2_class == 'Upper Upscale Class' or row1_class == 3 and row2_class == 'Upscale Class' or row1_class == 4 and row2_class == 'Upper Midscale Class' or row1_class == 5 and row2_class == 'Midscale Class' or row1_class == 6 and row2_class == 'Economy Class'):
		return False
	
	# checking that the rows match on number of rooms / size code columns
	row1_size, row2_size = prop_row['SizeCode'], manhattan_row['Rooms']
	if not (row1_size == 1 and row2_size < 75 or row1_size == 2 and row2_size >= 75 and row2_size <= 149 or row1_size == 3 and row2_size >= 150 and row2_size <= 299 or row1_size == 4 and row2_size >= 300 and row2_size <= 500 or row1_size == 5 and row2_size > 500):
		return False
	
	# checking that the rows match on year opened / date open
	# if not math.isnan(prop_row['OpenDate']) and manhattan_row['Open Date'] != '    -  -  ':
	#    row1_year, row2_year = int(prop_row['OpenDate']), int(manhattan_row['Open Date'][0:4])
	#    if not row1_year == row2_year:
	#        return False
	
	# checking that the capacity of the masked hotel matches the capacity of the hotel from the given row
	# if not int(capacity) == manhattan_row['Rooms']:
	#    return False
	
	# the rows match on all constraints; therefore, return True
	return True


def euclidean_miles(point1, point2):
	'''
	A function which, given two coordinates in UTM, returns the Euclidean distance between the
	two in miles.
	
	input:
		point1: The first point.
		point2: The second point.
		
	output:
		The Euclidean distance between point1 and point2.
	'''
	return math.sqrt(((point1[0] - point2[0]) ** 2) + ((point1[1] - point2[1]) ** 2)) * 0.000621371


def get_hotel_coords(attr_coords, distances):
	'''
	Function which receives an unordered list of attraction coordinates and corresponding distances, and searches through all
	permutations of these in order to find the correct correspondence (based on whether or not there is an intersection point).
	
	input:
		attr_coords: the latitude, longitude pair for each of the four attractions specified by the data provider
		distances: the distances (in miles) from each attraction to a given hotel
		
	output:
		The coordinates (latitude, longitude) of the hotel in question.
	'''
	
	# try each permutation of the distances
	for perm in itertools.permutations(distances):
		# calculate intersection point
		intersection = get_intersection_point(attr_coords, perm)
		# could come back as NoneType; we check for this here
		if intersection:
			return intersection
		
		
def get_radius(cx, cy, px, py):
	'''
	Calculates the radius of a circle given its center (location of a hotel) and a point on its perimeter.
	
	input:
		cx, cy: (x, y) coordinates of the center
		px, py: (x, y) perimeter coordinates
		
	output:
		The radius of the circle.
	'''
	
	dx = px - cx
	dy = py - cy
	return math.sqrt(dx ** 2 + dy ** 2)


def coords_to_distance_miles(start_coords, end_coords):
	'''
	Given coordinates of two points, calculate the distance between them in miles.
	
	input:
		start_coords: tuple of latitude, longitude coordinates for starting point
		end_coords: tuple of latitude, longitude coordinates for destination point
		
	output:
		The distance between the two points in miles.
	'''
	
	# variables for computing coordinates -> miles
	R = 6371e3
	phi_1, phi_2 = math.radians(start_coords[0]), math.radians(end_coords[0])
	delta_phi = abs(math.radians(start_coords[0] - end_coords[0]))
	delta_lambda = abs(math.radians(start_coords[1] - end_coords[1]))
		
	# computing distance in miles
	a = (math.sin(delta_phi / 2) ** 2) + math.sin(phi_1) * math.sin(phi_2) * (math.sin(delta_lambda / 2) ** 2)
	c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
	return R * c


def memory_usage_psutil():
    # return the memory usage in MB
    import psutil
    process = psutil.Process(os.getpid())
    mem = process.memory_info()[0] / float(2 ** 20)
    return mem


def worker(args):
	hotel_coords, trip_coords = args

	return [ vincenty(hotel_coords, coord) * 5280 for coord in trip_coords ]


def get_satisfying_indices(trip_coords, hotel_coords, distance, n_jobs):
	# Start a timer to record the length of the computation.
	start_time = timeit.default_timer()

	# Chunk up the trip coordinates according to how many cpu cores we have available.
	trip_coords = [trip_coords[idx : idx + n_jobs] for idx in xrange(0, len(trip_coords), n_jobs)]
	with closing(Pool(n_jobs)) as pool:
		# Calculate distances between trip coordinates and trip coordinates.
		dists = pool.map(worker, zip([hotel_coords] * len(trip_coords), trip_coords))

	# Get the indices of the satisfying trips.
	satisfying_indices = np.where(dists <= distance)
		
	# End the timer and report length of computation.
	end_time = timeit.default_timer() - start_time
	print '( time elapsed:', end_time, ')', '\n'

	# Return the satisfying indices to perform downstream processing of these trips.
	return satisfying_indices


def log_progress(sequence, every=None, size=None):
	'''
	This method allows me to visualize progress in loops, inline in Jupyter notebooks.
	'''
	from ipywidgets import IntProgress, HTML, VBox
	from IPython.display import display

	is_iterator = False
	if size is None:
		try:
			size = len(sequence)
		except TypeError:
			is_iterator = True
	if size is not None:
		if every is None:
			if size <= 200:
				every = 1
			else:
				every = int(size / 200)     # every 0.5%
	else:
		assert every is not None, 'sequence is iterator, set every'

	if is_iterator:
		progress = IntProgress(min=0, max=1, value=1)
		progress.bar_style = 'info'
	else:
		progress = IntProgress(min=0, max=size, value=0)
	label = HTML()
	box = VBox(children=[label, progress])
	display(box)

	index = 0
	try:
		for index, record in enumerate(sequence, 1):
			if index == 1 or index % every == 0:
				if is_iterator:
					label.value = '{index} / ?'.format(index=index)
				else:
					progress.value = index
					label.value = u'{index} / {size}'.format(
						index=index,
						size=size
					)
			yield record
	except:
		progress.bar_style = 'danger'
		raise
	else:
		progress.bar_style = 'success'
		progress.value = index
		label.value = str(index or '?')


def plot_destinations(destinations, append_to_title):
	'''
	Plot destinations latitudes and longitudes to see a "map" of destinations.
	
	input:
		destinations: A numpy array of shape (2, M), where M is equal to the number of taxicab trips which satisfy
		the above distance criterion.
		append_to_title: String to append to the tite of the plot for context.
		
	output:
		Matplotlib plot of datapoints in the range of the latitude, longitude pairs.
	'''

	plt.plot(destinations[0], destinations[1], 'o')
	plt.title('Destinations From ' + append_to_title)
	plt.xlabel('Longitude')
	plt.ylabel('Latitude')
	plt.show()
	
	
def cv_kmeans(destinations, cluster_range=range(10, 41, 2)):
	'''
	A function to perform K-Means, cross-validating on the number of clusters k.
	
	input:
		destinations: A numpy array of shape (2, M), where M is equal to the number of taxicab trips which satisfy
		the above distance criterion.
		cluster_range: the values of k to consider in the cross-validation.
		
	output:
		The best object score and correspond model, and the best silhouette score and the corresponding model.
	'''
	# create variables for best model and score
	best_score_model, best_silh_model, best_score, best_silh_score = None, None, -np.inf, -np.inf

	# range over values of k, the number of clusters to use, choose the k with the best objective score
	for k in cluster_range:
		# instantiate KMeans object
		model = MiniBatchKMeans(n_clusters=k)

		# fit the model and predict cluster indices of all coordinates
		labels = model.fit_predict(destinations.T)

		# compute score according to K-Means objective
		score = model.score(destinations.T)

		# compute silhouette score metric
		silh_score = silhouette_score(destinations.T, labels, sample_size=10000)

		# if this is the best scoring model so far
		if score > best_score:
			# update the best score and store the model
			best_score, best_score_model = score, model

		# if this is the best silhouette scoring mode so far
		if silh_score > best_silh_score:
			# update the best score and store the model
			best_silh_score, best_silh_model = silh_score, model
	
	return best_score, best_score_model, best_silh_score, best_silh_model


def visualize_clustering(destinations, n_clusters, title):
	'''
	A function which fits a clustering model and plots its visualization in color.
	
	input:
		destinations: A numpy array of shape (2, M), where M is equal to the number of taxicab trips which satisfy
		the above distance criterion.
		n_clusters: The number of clusters to consider in doing K-Means clustering.
		append_to_title: String to append to the tite of the plot for context.
		
	output:
		A visualization of the clusters found by running the K-Means algorithm with the specified number of 
		clusters.
	'''
	# create model, fit it to data, get cluster indices
	model = MiniBatchKMeans(n_clusters=n_clusters)
	labels = model.fit_predict(destinations)
	centroids = model.cluster_centers_

	# plot the data and their cluster labels
	for clstr in range(n_clusters):
		# plot just the points in cluster i
		clstr_points = destinations[ np.where(labels == clstr) ]
		plt.plot(clstr_points[:, 0], clstr_points[:, 1], 'o')
		# plot the centroids
		lines = plt.plot(centroids[clstr, 0], centroids[clstr, 1], 'kx')
		# make the centroid x's bigger
		plt.setp(lines, ms=8.0)
		plt.setp(lines, mew=2.0)

	plt.title(title)
	plt.xlabel('Longitude')
	plt.ylabel('Latitude')
	plt.show()