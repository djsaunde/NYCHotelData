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


def optimize_distance(hotel_capacities, taxi_rides, minimum, maximum, step, metric):
	'''
	Sweep a range of distance values and calculate the differences between the proportion
	of nearby pickups / dropoffs / both (depending on command-line argument). We want the
	distance which minimizes these differences. At the same time, we wish to remove hotels
	from the optimization of the distance which are severe outliers.
	'''
	step = 25
	distances = range(minimum, maximum + step, step)
	evals = np.zeros(np.shape(distances))

	hotels = set(hotel_capacities.keys()) & set(taxi_rides.keys())
	hotel_capacities = {key : val for (key, val) in hotel_capacities.items() if key in hotels}
	taxi_rides = {key : val for (key, val) in taxi_rides.items() if key in hotels}

	for idx in range(len(hotels)):
		s = sum([value for (hotel, value) in hotel_capacities.items()])
		
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

		ax1.plot(distances, evals, label='Entropy as a function of distance')
		ax1.set_xlim([0, max(distances)]); ax1.set_ylim([0, max(evals)])
		ax1.axhline(np.min(evals), color='r', linestyle='--')

		if metric == 'relative_entropy':
			fig1.suptitle('Relative entropy between occupancy\n' + \
					'proportions and empirical taxicab distribution', fontsize=20)
		elif metric == 'abs_diffs':
			fig1.suptitle('Sum of abs. diffs. between occupancy\n' + \
					'proportions and empirical taxicab distribution', fontsize=20)
		elif metric == 'weighted_abs_diffs':
			fig1.suptitle('Weighted (by magnitude) sum of absolute differences between\n' + \
						'occupancy proportions and empirical taxicab distribution', fontsize=20)
		elif metric == 'inverse_weighted_abs_diffs':
			fig1.suptitle('Inversely weighted (by magnitude) sum of absolute differences between\n' + \
						'occupancy proportions and empirical taxicab distribution', fontsize=20)

		fig1.savefig(os.path.join(plots_path, '_'.join(['relative_entropy', str(idx)]) + '.png'))

		abs_diffs = np.zeros([len(distances), len(hotels)])
		for idx, distance in enumerate(distances):
			for hotel_idx, hotel in enumerate(sorted(hotels)):
				abs_diffs[idx, hotel_idx] = np.abs(capacity_distros[hotel] - taxi_distros[idx][hotel])

		cm = plt.get_cmap('gist_rainbow')

		fig2 = plt.figure(figsize=(18, 9.5))
		ax2 = fig2.add_subplot(111, projection='3d')

		fig2.suptitle('Absolute value of occupancy, taxicab distribution\n' + \
				'differences per hotel, per distance criterion', fontsize=20)

		for idx, distance in enumerate(distances):
			ax2.bar(np.arange(np.shape(abs_diffs)[1]), abs_diffs[idx, :], 
												zs=idx, zdir='y', alpha=0.8,
											color=cm(1.0 * idx / len(hotels)))

		ax2.set_yticks(range(0, len(distances) + 1))
		ax2.set_yticklabels(distances)
		ax2.set_xticks(range(len(hotels)))
		ax2.set_xticklabels(sorted(hotels))
		ax2.set_zlim([0, 1])
		ax2.set_zticks(np.linspace(0, 1, 11))

		fig2.savefig(os.path.join(plots_path, '_'.join(['rel_diffs', str(idx)]) + '.png'))

		fig3, [ax3, ax4] = plt.subplots(1, 2, sharey=True, figsize=(18, 9.5))

		ax3.bar(range(len(hotels)), abs_diffs[min_eval_idx, :], label='Absolute differences')
		ax3.set_title('Absolute differences between distributions')
		ax3.set_xticks(range(len(hotels)))
		ax3.set_yticks(np.linspace(0, 1, 11))
		ax3.set_ylim([0, 1])

		width = 0.25
		
		caps = [capacity_distros[key] for key in sorted(capacity_distros)]

		rects1 = ax4.bar(np.arange(len(hotels)), caps, width, color='r', label='Capacity distribution')

		taxis = [taxi_distros[min_eval_idx][key] for key in sorted(taxi_distros[min_eval_idx]) ]
		rects2 = ax4.bar(np.arange(len(hotels)) + width, taxis, width, color='y',
			label='Best taxi rides distr. (dist. = %d)' % distances[min_eval_idx])

		ax4.set_title('Capacity distribution, best taxi rides distribution')
		ax4.set_xticks(range(len(hotels)))

		plt.ylim([0, 1])
		plt.legend()
		
		fig3.savefig(os.path.join(plots_path, '_'.join(['distr_diffs', str(idx)]) + '.png'))

		if plot:
			plt.show()

		to_remove = sorted(hotels)[np.argmax(abs_diffs[np.argmin(evals), :])]
		
		print('\nRemoving hotel %s' % to_remove)
		
		hotels.remove(sorted(hotels)[np.argmax(abs_diffs[np.argmin(evals), :])])

	return distances[np.argmax(evals)]


def objective(capacity_distros, taxi_distribution, metric):
	'''
	The objective function captures how "close" we are to a perfect match between the
	distribution of hotel capacities (occupancies?) and the distribution of taxicab
	coordinates (nearby pickups / dropoffs / both).
	'''
	if metric == 'relative_entropy':
		return entropy(np.array(list(capacity_distros.values())), np.array(list(taxi_distribution.values())))
	elif metric == 'abs_diffs':
		return np.sum(np.abs(np.array(list(capacity_distros.values())) - \
								np.array(list(taxi_distribution.values()))))
	elif metric == 'inverse_weighted_abs_diffs':
		return np.sum(np.abs(np.array(list(capacity_distros.values())) - \
								np.array(list(taxi_distribution.values()))) / \
								np.array(list(capacity_distros.values())))
	elif metric == 'weighted_abs_diffs':
		return np.sum(np.abs(np.array(list(capacity_distros.values())) - \
								np.array(list(taxi_distribution.values()))) * \
								np.array(list(capacity_distros.values())))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--maximum', default=100, type=int)
	parser.add_argument('--minimum', default=25, type=int)
	parser.add_argument('--step', default=25, type=int)
	parser.add_argument('--coord_type', default='pickups', type=str)
	parser.add_argument('--start_date', type=int, nargs=3, default=[2013, 1, 1])
	parser.add_argument('--end_date', type=int, nargs=3, default=[2013, 1, 2])
	parser.add_argument('--metric', type=str, default='weighted_abs_diffs')
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
		fname = os.path.join('..', 'data', '_'.join(['all_preprocessed', str(maximum)]), 'destinations.csv')
	elif coord_type == 'dropoffs':
		fname = os.path.join('..', 'data', '_'.join(['all_preprocessed', str(maximum)]), 'all_preprocessed.csv')
	else:
		raise Exception('Expecting one of "pickups" or "dropoffs" for command-line argument coord_type.')

	taxi_rides = pd.read_csv(fname, header=0, usecols=usecols)

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

	best_distance = optimize_distance(hotel_capacities, rides_by_hotel, minimum, maximum, step, metric)
