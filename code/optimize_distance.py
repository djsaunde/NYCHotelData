from __future__ import print_function

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


def optimize_distance(capacity_data, taxi_data, min_distance, max_distance, minimizee):
	'''
	Sweep a range of distance values and calculate the differences between the proportion
	of nearby pickups / dropoffs / both (depending on command-line argument). We want the
	distance which minimizes these differences. At the same time, we wish to remove hotels
	from the optimization of the distance which are severe outliers.
	'''
	distances = range(min_distance, max_distance, 25)
	objective_evaluations = np.zeros(np.shape(distances))

	capacity_hotel_names = set([ hotel_name for hotel_name in capacity_data.keys() ])
	taxi_hotel_names = set([ hotel_name for hotel_name in taxi_data.keys() ])
	hotel_names = capacity_hotel_names.intersection(taxi_hotel_names)

	for backwards_stepwise_idx in xrange(len(hotel_names)):
		capacity_sum = sum([ value for (hotel_name, value) in capacity_data.items() if hotel_name in hotel_names ])
		capacity_distribution = { hotel_name : capacity / float(capacity_sum) \
			for (hotel_name, capacity) in sorted(capacity_data.items()) if hotel_name in hotel_names }

		taxi_distributions = []
		for distance in distances:
			taxi_subset = { hotel_name : len(data[data <= distance]) for \
				(hotel_name, data) in sorted(taxi_data.items()) if hotel_name in hotel_names }
			taxi_sum = sum([ value for value in taxi_subset.values()])

			if taxi_sum != 0:
				taxi_distributions.append({ hotel_name : n_trips / float(taxi_sum) \
					for (hotel_name, n_trips) in sorted(taxi_subset.items()) if hotel_name in hotel_names })
			else:
				taxi_distributions.append({ hotel_name : 0.0 for hotel_name in \
						sorted(taxi_subset.keys()) if hotel_name in hotel_names })

		for idx, taxi_distribution in enumerate(taxi_distributions):
			objective_evaluations[idx] = objective(capacity_distribution, taxi_distribution, minimizee)

		fig1 = plt.figure(figsize=(18, 9.5))
		ax1 = fig1.add_subplot(111)

		ax1.plot(distances, objective_evaluations, label='Entropy as a function of distance')
		ax1.set_xlim([0, max(distances)]); ax1.set_ylim([0, max(objective_evaluations)])
		ax1.axhline(np.min(objective_evaluations), color='r', linestyle='--')

		if minimizee == 'relative_entropy':
			fig1.suptitle('KL Divergence (relative entropy) between occupancy\n' + \
					'proportions and empirical taxicab distribution', fontsize=20)
		elif minimizee == 'abs_diffs':
			fig1.suptitle('Sum of absolute differences between occupancy\n' + \
					'proportions and empirical taxicab distribution', fontsize=20)
		elif minimizee == 'weighted_abs_diffs':
			fig1.suptitle('Weighted (by magnitude) sum of absolute differences between\n' + \
						'occupancy proportions and empirical taxicab distribution', fontsize=20)
		elif minimizee == 'inverse_weighted_abs_diffs':
			fig1.suptitle('Inversely weighted (by magnitude) sum of absolute differences between\n' + \
						'occupancy proportions and empirical taxicab distribution', fontsize=20)

		fig1.savefig(os.path.join(plots_path, '_'.join(['relative_entropy', str(backwards_stepwise_idx)]) + '.png'))

		absolute_differences = np.zeros([len(distances), len(hotel_names)])
		for distance_idx, distance in enumerate(distances):
			for hotel_idx, hotel_name in enumerate(sorted(hotel_names)):
				absolute_differences[distance_idx, hotel_idx] = np.abs(capacity_distribution[hotel_name] - \
																taxi_distributions[distance_idx][hotel_name])

		cm = plt.get_cmap('gist_rainbow')

		fig2 = plt.figure(figsize=(18, 9.5))
		ax2 = fig2.add_subplot(111, projection='3d')

		fig2.suptitle('Absolute value of occupancy, taxicab distribution\n' + \
				'differences per hotel, per distance criterion', fontsize=20)

		for idx, distance in enumerate(distances):
			ax2.bar(np.arange(np.shape(absolute_differences)[1]), absolute_differences[distance_idx, :], 
								zs=idx, zdir='y', alpha=0.8, color=cm(1.0 * idx / len(hotel_names)))

		ax2.set_yticks(xrange(0, len(distances), 10))
		ax2.set_yticklabels([ distances[idx] for idx in xrange(0, len(distances), 10) ])
		ax2.set_xticks(xrange(len(hotel_names)))
		ax2.set_xticklabels(sorted(hotel_names))
		ax2.set_zlim([0, 1])
		ax2.set_zticks(np.linspace(0, 1, 11))

		fig2.savefig(os.path.join(plots_path, '_'.join(['abs_diffs', str(backwards_stepwise_idx)]) + '.png'))

		fig3, [ax3, ax4] = plt.subplots(1, 2, sharey=True, figsize=(18, 9.5))

		ax3.bar(xrange(len(hotel_names)), absolute_differences\
			[np.argmin(objective_evaluations), :], label='Absolute differences')
		ax3.set_title('Absolute differences between distributions')
		ax3.set_xticks(xrange(len(hotel_names)))
		ax3.set_yticks(np.linspace(0, 1, 11))
		ax3.set_ylim([0, 1])

		width = 0.25
		
		rects1 = ax4.bar(np.arange(len(hotel_names)), [ capacity_distribution[key] \
											for key in sorted(capacity_distribution) ], 
											width, color='r', label='Capacity distribution')
		rects2 = ax4.bar(np.arange(len(hotel_names)) + width, \
			[ taxi_distributions[np.argmin(objective_evaluations)][key] for key in \
				sorted(taxi_distributions[np.argmin(objective_evaluations)]) ], \
				width, color='y', label='Best taxi rides distribution ' + \
			'(distance =' + str(distances[np.argmin(objective_evaluations)]) + ')')

		ax4.set_title('Capacity distribution, best taxi rides distribution')
		ax4.set_xticks(xrange(len(hotel_names)))

		fig3.savefig(os.path.join(plots_path, '_'.join(['distro_diffs', str(backwards_stepwise_idx)]) + '.png'))

		plt.ylim([0, 1])
		plt.legend()

		if plot:
			plt.show()

		to_remove = sorted(hotel_names)[np.argmax(absolute_differences[np.argmin(objective_evaluations), :])]
		
		print('\nRemoving hotel %s' % to_remove)
		
		hotel_names.remove(sorted(hotel_names)[np.argmax(absolute_differences[np.argmin(objective_evaluations), :])])

	return distances[np.argmax(objective_evaluations)]


def objective(capacity_distribution, taxi_distribution, minimizee):
	'''
	The objective function captures how "close" we are to a perfect match between the
	distribution of hotel capacities (occupancies?) and the distribution of taxicab
	coordinates (nearby pickups / dropoffs / both).
	'''
	if minimizee == 'relative_entropy':
		return entropy(np.array(capacity_distribution.values()), np.array(taxi_distribution.values()))
	elif minimizee == 'abs_diffs':
		return np.sum(np.abs(np.array(capacity_distribution.values()) - \
								np.array(taxi_distribution.values())))
	elif minimizee == 'inverse_weighted_abs_diffs':
		return np.sum(np.abs(np.array(capacity_distribution.values()) - \
								np.array(taxi_distribution.values())) / \
								np.array(capacity_distribution.values()))
	elif minimizee == 'weighted_abs_diffs':
		return np.sum(np.abs(np.array(capacity_distribution.values()) - \
								np.array(taxi_distribution.values())) * \
								np.array(capacity_distribution.values()))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--max_distance', default=100, type=int)
	parser.add_argument('--min_distance', default=25, type=int)
	parser.add_argument('--coord_type', default='pickups', type=str)
	parser.add_argument('--start_date', type=int, nargs=3, default=[2013, 1, 1], \
				help='The day on which to start looking for satisfying coordinates.')
	parser.add_argument('--end_date', type=int, nargs=3, default=[2013, 2, 1], \
				help='The day on which to stop looking for satisfying coordinates.')
	parser.add_argument('--minimizee', type=str, default='weighted_abs_diffs', help='One of \
		"abs_diffs", "weighted_abs_diffs", "inverse_weighted_abs_diffs", or "relative_entropy".')
	parser.add_argument('--plot', dest='plot', action='store_true')
	parser.add_argument('--no_plot', dest='plot', action='store_false')
	parser.set_defaults(plot=False)

	args = parser.parse_args()
	args = vars(args)

	locals().update(args)

	plots_path = os.path.join('..', 'plots', 'distance_optimization', '_'.join(map(str, [min_distance, \
												max_distance, coord_type, start_date[0], start_date[1], \
												start_date[2], end_date[0], end_date[1], end_date[2], minimizee])))

	if not os.path.exists(plots_path):
		os.makedirs(plots_path)

	start_date, end_date = date(*start_date), date(*end_date)

	# Load daily capacity data.
	print('\nLoading daily per-hotel capacity data.')
	start = timeit.default_timer()

	daily_capacity_data = pd.read_csv(os.path.join('..', 'data', 'Unmasked Daily Capacity.csv'), index_col=False)
	daily_capacity_data['Date'] = pd.to_datetime(daily_capacity_data['Date'], format='%Y-%m-%d')
	daily_capacity_data = daily_capacity_data.loc[(daily_capacity_data['Date'] >= start_date) & \
													(daily_capacity_data['Date'] <= end_date)]

	print('Time: %.4f' % (timeit.default_timer() - start))

	# Create a dictionary which contains per-hotel daily capacity data.
	print('\nOrganizing data into per-hotel, per-day dictionary structure.')
	start = timeit.default_timer()
	
	per_hotel_capacity_data = {}
	for hotel_name in daily_capacity_data['Share ID'].unique():
		per_hotel_capacity_data[hotel_name] = sum([ row['Room Demand'] for (_, row) in \
			daily_capacity_data.loc[daily_capacity_data['Share ID'] == hotel_name].iterrows() ])

	print('Time: %.4f' % (timeit.default_timer() - start))

	# Load preprocessed data according to command-line "distance" parameter.
	print('\nReading in the pre-processed taxicab data.')
	start = timeit.default_timer()

	usecols = ['Share ID', 'Hotel Name', 'Distance From Hotel', 'Latitude', 'Longitude', 'Pick-up Time',
								'Drop-off Time', 'Passenger Count', 'Trip Distance', 'Fare Amount']
	if coord_type == 'pickups':
		taxi_data = pd.read_csv(os.path.join('..', 'data', '_'.join(['all_preprocessed', str(max_distance)]),
												'destinations.csv'), header=0, index_col=False, usecols=usecols)
	elif coord_type == 'dropoffs':
		taxi_data = pd.read_csv(os.path.join('..', 'data', '_'.join(['all_preprocessed', str(max_distance)]),
											'starting_points.csv'), header=0, index_col=False, usecols=usecols)
	elif coord_type == 'both':
		taxi_data = pd.read_csv(os.path.join('..', 'data', '_'.join(['all_preprocessed', str(max_distance)]),
												'destinations.csv'), header=0, index_col=False, usecols=usecols)
		taxi_data = pd.concat([taxi_data, pd.read_csv(os.path.join('..', 'data',
				'_'.join(['all_preprocessed', str(max_distance)]), 'starting_points.csv'),
						header=0, index_col=False, usecols=usecols)], ignore_index=True)
	else:
		raise Exception('Expecting one of "pickups", "dropoffs", or "both" for command-line argument coord_type.')

	taxi_data['Hotel Name'] = taxi_data['Hotel Name'].apply(str.strip)
	taxi_data['Pick-up Time'] = pd.to_datetime(taxi_data['Pick-up Time'], format='%Y-%m-%d')
	taxi_data['Drop-off Time'] = pd.to_datetime(taxi_data['Drop-off Time'], format='%Y-%m-%d')
	taxi_data = taxi_data.loc[(taxi_data['Pick-up Time'] >= start_date) & \
									(taxi_data['Drop-off Time'] <= end_date)]

	per_hotel_taxi_data = {}
	for hotel_name in taxi_data['Hotel Name'].unique():
		per_hotel_taxi_data[hotel_name] = np.array([ row['Distance From Hotel'] for (_, row) in \
			taxi_data.loc[taxi_data['Hotel Name'] == hotel_name].iterrows() ])


	print('Time: %.4f' % (timeit.default_timer() - start))

	best_distance = optimize_distance(per_hotel_capacity_data, per_hotel_taxi_data, \
												min_distance, max_distance, minimizee)