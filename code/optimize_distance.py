from __future__ import print_function, division

import os
import sys
import timeit
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from util import *
from datetime import date
from scipy.stats import entropy
from mpl_toolkits.mplot3d import Axes3D

reports_path = os.path.join('..', 'data', 'optimization_reports')

if not os.path.isdir(reports_path):
	os.makedirs(reports_path)


def optimize_distance(hotel_capacities, taxi_rides, minimum, maximum, step, metric, fname):
	'''
	Sweep a range of distance values and calculate the differences between the proportion
	of nearby pickups / dropoffs / both (depending on command-line argument). We want the
	distance which minimizes these differences. At the same time, we wish to remove hotels
	from the optimization of the distance which are severe outliers.
	'''
	distances = range(minimum, maximum + step, step)
	evals = np.zeros(np.shape(distances))
	
	hotels = set(hotel_capacities.keys()) & set(taxi_rides.keys())
	hotel_capacities = {key : val for (key, val) in hotel_capacities.items() if key in hotels}
	taxi_rides = {key : val for (key, val) in taxi_rides.items() if key in hotels}

	removal_data = []
	for idx in range(len(hotels)):
		s = sum([value for (hotel, value) in hotel_capacities.items() if hotel in hotels])
		
		capacity_distros = {}
		for (hotel, capacity) in sorted(hotel_capacities.items()):
			if hotel in hotels:
				capacity_distros[hotel] = capacity / s

		taxi_distros = []
		for distance in distances:
			subset = {hotel : len(data[data <= distance]) for (hotel, data) in sorted(taxi_rides.items()) if hotel in hotels}
			s = sum([value for value in subset.values()])

			if s != 0:
				taxi_distros.append({hotel : n_trips / s for (hotel, n_trips) in sorted(subset.items())})
			else:
				taxi_distros.append({hotel : 0.0 for hotel in sorted(subset.keys())})

		for i, distro in enumerate(taxi_distros):
			evals[i] = objective(capacity_distros, distro, metric)

		min_eval_idx = np.argmin(evals)

		fig1 = plt.figure(figsize=(18, 9.5))
		ax1 = fig1.add_subplot(111)

		ax1.plot(distances, evals)
		ax1.set_xlim([0, max(distances)]); ax1.set_ylim([0, max(evals[~np.isinf(evals)])])
		ax1.axhline(np.min(evals), color='r', linestyle='--')

		if metric == 'relative_entropy':
			fig1.suptitle('Relative entropy', fontsize=16)
		elif metric == 'abs_diffs':
			fig1.suptitle('Sum of absolute differences', fontsize=16)
		elif metric == 'rel_diffs':
			fig1.suptitle('Sum of relative differences', fontsize=16)
		elif metric == 'weighted_abs_diffs':
			fig1.suptitle('Sum of weighted (by magnitude) absolute differences', fontsize=16)
		elif metric == 'inverse_weighted_abs_diffs':
			fig1.suptitle('Sum of inversely weighted (by magnitude) absolute differences', fontsize=16)

		fig1.savefig(os.path.join(plots_path, '_'.join([metric, str(idx)]) + '.png'))

		abs_diffs = np.zeros([len(distances), len(hotels)])
		rel_diffs = np.zeros([len(distances), len(hotels)])
		rel_entropies = np.zeros([len(distances), len(hotels)])
		for i, distance in enumerate(distances):
			for hotel_idx, hotel in enumerate(sorted(hotels)):
				abs_diffs[i, hotel_idx] = np.abs(capacity_distros[hotel] - taxi_distros[i][hotel])
				rel_diffs[i, hotel_idx] = taxi_distros[i][hotel] / capacity_distros[hotel]
				
				if capacity_distros[hotel] == 0 or taxi_distros[i][hotel] == 0:
					rel_entropies[i, hotel_idx] = 0
				else:
					rel_entropies[i, hotel_idx] = capacity_distros[hotel] * \
						np.log(capacity_distros[hotel] / taxi_distros[i][hotel])

		cm = plt.get_cmap('gist_rainbow')

		fig2 = plt.figure(figsize=(18, 9.5))
		ax2 = fig2.add_subplot(111, projection='3d')

		if metric == 'rel_diffs':
			fig2.suptitle('Relative differences per distance, hotel', fontsize=16)
		else:
			fig2.suptitle('Absolute differences per distance, hotel', fontsize=16)

		x = np.arange(np.shape(rel_diffs)[1])
		if metric == 'rel_diffs':
			for i, distance in enumerate(distances):
				ax2.bar(x, rel_diffs[i], zs=i, zdir='y', alpha=0.8, color=cm(i / len(hotels)))
		else:
			for i, distance in enumerate(distances):
				ax2.bar(x, abs_diffs[i], zs=i, zdir='y', alpha=0.8, color=cm(i / len(hotels)))

		ax2.set_yticks(range(0, len(distances) + 1)); ax2.set_yticklabels(distances)

		fig2.savefig(os.path.join(plots_path, '_'.join(['3D_rel_diffs', str(idx)]) + '.png'))

		fig3, [ax3, ax4] = plt.subplots(1, 2, figsize=(18, 9.5))

		ax3.bar(range(len(hotels)), rel_diffs[min_eval_idx, :], label='Relative differences')
		ax3.set_title('Relative differences between distributions')

		caps = [capacity_distros[key] for key in sorted(capacity_distros)]
		ax4.bar(np.arange(len(hotels)), caps, 0.25, color='r', label='Capacity distribution')

		taxis = [taxi_distros[min_eval_idx][key] for key in sorted(taxi_distros[min_eval_idx]) ]
		ax4.bar(np.arange(len(hotels)) + 0.25, taxis, 0.25, color='y', label='Best taxi distribution (dist. = %d)' % distances[min_eval_idx])
		ax4.set_title('Capacity distribution, best taxi rides distribution')

		plt.legend()

		fig3.savefig(os.path.join(plots_path, '_'.join(['distr_diffs', str(idx)]) + '.png'))

		if plot:
			plt.show()

		plt.close('all')

		if metric == 'rel_diffs':
			divergences = []
			for x in rel_diffs[min_eval_idx]:
				if x >= 1:
					divergences.append(x)
				elif x > 0 and x < 1:
					divergences.append(1 / x)
				elif x == 0:
					divergences.append(np.inf)
					
			worst_idx = np.argmax(divergences)
		elif metric == 'abs_diffs':
			worst_idx = np.argmax(abs_diffs[min_eval_idx])
		elif metric == 'relative_entropy':
			worst_idx = np.argmax(rel_entropies[min_eval_idx])
			
		to_remove = sorted(hotels)[worst_idx]
		hotels.remove(to_remove)
		
		print('Removed hotel %s (%d remaining)' % (to_remove, len(hotels)))
		
		removal_data.append([to_remove, distances[min_eval_idx], capacity_distros[to_remove],
					taxi_distros[min_eval_idx][to_remove], abs_diffs[min_eval_idx][worst_idx],
								rel_diffs[worst_idx], rel_entropies[min_eval_idx][worst_idx]])

	df = pd.DataFrame(removal_data, columns=['Removed hotel', 'Best distance', 'Capacity share', 'Taxi share', 'Abs. difference', 'Rel. divergence', 'Rel. entropy'])
	df.to_csv(os.path.join(reports_path, fname) + '.csv')

	return distances[np.argmax(evals)]


def objective(capacity_distros, taxi_distribution, metric):
	'''
	The objective function captures how "close" we are to a perfect match between the
	distribution of hotel capacities (occupancies?) and the distribution of taxicab
	coordinates (nearby pickups / dropoffs / both).
	'''
	capacity_distros = np.array(list(capacity_distros.values()))
	taxi_distribution = np.array(list(taxi_distribution.values()))

	if metric == 'relative_entropy':
		return entropy(capacity_distros, taxi_distribution)
	elif metric == 'abs_diffs':
		return np.sum(np.abs(capacity_distros - taxi_distribution))
	elif metric == 'rel_diffs':
		rel_diffs= taxi_distribution / capacity_distros
		
		divergences = []
		for x in rel_diffs:
			if x >= 1:
				divergences.append(x)
			elif x > 0 and x < 1:
				divergences.append(1 / x)
			elif x == 0:
				divergences.append(np.inf)
		
		return np.sum(divergences)
	elif metric == 'inverse_weighted_abs_diffs':
		return np.sum(np.abs(capacity_distros - taxi_distribution) / capacity_distros)
	elif metric == 'weighted_abs_diffs':
		return np.sum(np.abs(capacity_distros - taxi_distribution) * capacity_distros)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--maximum', default=100, type=int)
	parser.add_argument('--minimum', default=25, type=int)
	parser.add_argument('--step', default=25, type=int)
	parser.add_argument('--coord_type', default='pickups', type=str)
	parser.add_argument('--start_date', type=int, nargs=3, default=[2013, 1, 1])
	parser.add_argument('--end_date', type=int, nargs=3, default=[2015, 1, 1])
	parser.add_argument('--metric', type=str, default='rel_diffs')
	parser.add_argument('--plot', dest='plot', action='store_true')
	parser.add_argument('--no_plot', dest='plot', action='store_false')
	parser.set_defaults(plot=False)

	locals().update(vars(parser.parse_args()))
	
	fname = '_'.join(map(str, [minimum, maximum, coord_type, start_date[0], start_date[1],
							start_date[2], end_date[0], end_date[1], end_date[2], metric]))
	plots_path = os.path.join('..', 'plots', 'distance_optimization', fname)

	if not os.path.exists(plots_path):
		os.makedirs(plots_path)

	start_date, end_date = date(*start_date), date(*end_date)

	# Load daily capacity data.
	print('\nLoading daily per-hotel capacity data.')
	start = timeit.default_timer()

	capacities = pd.read_csv(os.path.join('..', 'data', 'Unmasked Daily Capacity.csv'), index_col=False)
	capacities['Date'] = pd.to_datetime(capacities['Date'], format='%Y-%m-%d')
	capacities = capacities.loc[(capacities['Date'] >= start_date) & (capacities['Date'] <= end_date)]

	print('Time: %.4f' % (timeit.default_timer() - start))

	# Create a dictionary which contains per-hotel daily capacity data.
	print('\nOrganizing data into per-hotel, per-day dictionary structure.')
	start = timeit.default_timer()
	
	hotel_capacities = {}
	for hotel in capacities['Share ID'].unique():
		hotel_capacities[hotel] = sum([row['Room Demand'] for (_, row) in \
			capacities.loc[capacities['Share ID'] == hotel].iterrows()])

	print('Time: %.4f' % (timeit.default_timer() - start))

	# Load preprocessed data according to command-line "distance" parameter.
	print('\nReading in the pre-processed taxicab data.')
	start = timeit.default_timer()

	usecols = ['Share ID', 'Hotel Name', 'Distance From Hotel', 'Latitude', 'Longitude', 'Pick-up Time',
								'Drop-off Time', 'Passenger Count', 'Trip Distance', 'Fare Amount']
	if coord_type == 'pickups':
		filename = os.path.join('..', 'data', '_'.join(['all_preprocessed', str(maximum)]), 'destinations.csv')
	elif coord_type == 'dropoffs':
		filename = os.path.join('..', 'data', '_'.join(['all_preprocessed', str(maximum)]), 'starting_points.csv')
	else:
		raise Exception('Expecting one of "pickups" or "dropoffs" for command-line argument coord_type.')

	taxi_rides = pd.read_csv(filename, header=0, usecols=usecols)

	taxi_rides['Hotel Name'] = taxi_rides['Hotel Name'].apply(str.strip)
	taxi_rides['Pick-up Time'] = pd.to_datetime(taxi_rides['Pick-up Time'], format='%Y-%m-%d')
	taxi_rides['Drop-off Time'] = pd.to_datetime(taxi_rides['Drop-off Time'], format='%Y-%m-%d')
	taxi_rides = taxi_rides.loc[(taxi_rides['Pick-up Time'] >= start_date) & \
									(taxi_rides['Drop-off Time'] <= end_date)]

	rides_by_hotel = {}
	for hotel in taxi_rides['Hotel Name'].unique():
		hotel_rows = taxi_rides.loc[taxi_rides['Hotel Name'] == hotel]
		rides_by_hotel[hotel] = np.array([row['Distance From Hotel'] for (_, row) in hotel_rows.iterrows()])

	print('Time: %.4f' % (timeit.default_timer() - start))

	best_distance = optimize_distance(hotel_capacities, rides_by_hotel, minimum, maximum, step, metric, fname)
