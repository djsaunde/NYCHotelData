from __future__ import print_function

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.pyplot import cm

parser = argparse.ArgumentParser()
parser.add_argument('--start_date', type=int, nargs=3, default=[2014, 1, 1])
parser.add_argument('--end_date', type=int, nargs=3, default=[2016, 6, 30])
locals().update(vars(parser.parse_args()))

removals_path = os.path.join('..', 'data', 'grid_search_taxi_mlp_removals')

dirs = os.listdir(removals_path)
distances = sorted(list(map(int, [d.split('_')[0] for d in dirs])))
colors = cm.rainbow(np.linspace(0, 1, len(distances)))

fig, axes = plt.subplots(1, 2, figsize=(9, 5))

# Plot removals from all MLP regression models trained with taxi data.
for d, c in zip(distances, colors):
	dirname = '_'.join(map(str, [d, start_date[0], start_date[1],
			start_date[2], end_date[0], end_date[1], end_date[2]]))
	path = os.path.join(removals_path, dirname, 'removals.csv')
	
	# Get removal results .csv file.
	results = pd.read_csv(path)
	
	# Plot the test mean-squared errors.
	axes[0].semilogy(results['Test MSE'][:11], c=c, label='d = %d' % d)
	axes[1].plot(results['Test R^2'][:11], c=c, label='d = %d' % d)

# Plot removals from naive MLP regression model.
removals_path = os.path.join('..', 'data', 'grid_search_naive_mlp_removals')
dirname = '_'.join(map(str, [start_date[0], start_date[1],
	start_date[2], end_date[0], end_date[1], end_date[2]]))
path = os.path.join(removals_path, dirname, 'removals.csv')

# Get removal results .csv file.
results = pd.read_csv(path)

# Plot the test mean-squared errors.
axes[0].semilogy(results['Test MSE'][:11], c='k', linestyle='--', label='Naive')
axes[1].plot(results['Test R^2'][:11], c='k', linestyle='--', label='Naive')

axes[0].set_title('Test MSE vs. no. hotels removed')
axes[1].set_title('Test R^2 vs. no. hotels removed')
axes[0].set_xlabel('No. of hotels removed'); axes[0].set_ylabel('Test MSE')
axes[1].set_xlabel('No. of hotels removed'); axes[1].set_ylabel('Test R^2')
axes[0].set_xticks(range(11)); axes[1].set_xticks(range(11))
axes[0].legend(fontsize='x-small'); axes[1].legend(fontsize='x-small');
plt.tight_layout()

plt.savefig(os.path.join('..', 'plots', 'taxi_mlp_removals_results.png'))
plt.show()