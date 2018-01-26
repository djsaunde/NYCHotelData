from __future__ import print_function

import os
import sys
import timeit
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_path = os.path.join('..', 'data')
plots_path = os.path.join('..', 'plots', 'distance_histograms')

if not os.path.isdir(plots_path):
	os.makedirs(plots_path)

parser = argparse.ArgumentParser()
parser.add_argument('--distance', type=int, default=300)
parser.add_argument('--coord_type', type=str, default='pickups')
parser.add_argument('--plot', dest='plot', action='store_true')
parser.add_argument('--no_plot', dest='plot', action='store_false')
parser.set_defaults(plot=False)

args = parser.parse_args()
args = vars(args)
locals().update(args)

if coord_type == 'pickups':
	fname = os.path.join(data_path, 'all_preprocessed_%d' % distance, 'destinations.csv')
elif coord_type == 'dropoffs':
	fname = os.path.join(data_path, 'all_preprocessed_%d' % distance, 'starting_points.csv')

print('\nReading pre-processed taxi data (%d feet)' % distance); start = timeit.default_timer()
df = pd.read_csv(fname); print('Time: %.4f' % (timeit.default_timer() - start))

start = timeit.default_timer()
for hotel in map(str.strip, df['Hotel Name'].unique()):
	print('... Plotting distance histogram for %s (Time: %.4f)' % (hotel, (timeit.default_timer() - start)))

	df[df['Hotel Name'] == hotel]['Distance From Hotel'].hist(bins=np.arange(0, distance + 1, 1))
	plt.title('Binned distances by feet (%s)' % hotel)
	plt.xlabel('Distance in feet'); plt.ylabel('No. observations')
	plt.legend(); plt.tight_layout()
	plt.savefig(os.path.join(plots_path, hotel))

	if plot:
		plt.show()

	plt.clf()

print('\n... Plotting distance histogram for all hotels'); start = timeit.default_timer()

df['Distance From Hotel'].hist(bins=np.arange(0, distance + 1, 1));
plt.title('Binned distances by feet (all hotels)')
plt.xlabel('Distance in feet'); plt.ylabel('No. observations')
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(plots_path, hotel))

if plot:
	plt.show()

plt.clf(); print('Time: %.4f\n' % (timeit.default_timer() - start))