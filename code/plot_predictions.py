import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

from datetime import date

parser = argparse.ArgumentParser()
parser.add_argument('--distance', default=25, type=int)
parser.add_argument('--trip_type', default='pickups', type=str)
parser.add_argument('--start_date', type=int, nargs=3, default=[2013, 1, 1])
parser.add_argument('--end_date', type=int, nargs=3, default=[2016, 6, 30])
parser.add_argument('--metric', type=str, default='rel_diffs')
parser.add_argument('--nrows', type=int, default=None)
parser.add_argument('--n', type=int, default=1000)
parser.add_argument('--model', type=str, default='lr')
parser.add_argument('--plot_type', type=str, default='plot')

locals().update(vars(parser.parse_args()))

fname = '_'.join(map(str, [distance, start_date[0], start_date[1], start_date[2], end_date[0], end_date[1], end_date[2], metric]))

start_date, end_date = date(*start_date), date(*end_date)

predictions_path = os.path.join('..', 'data', '%s_predictions' % model, fname)

predictions = np.load(os.path.join(predictions_path, 'predictions.npy'))
targets = np.load(os.path.join(predictions_path, 'targets.npy'))

if plot_type == 'plot':
	if n == -1:
		plt.plot(range(len(predictions)), predictions, label='predictions')
		plt.plot(range(len(targets)), targets, label='targets')
	else:
		plt.plot(range(n), predictions[:n], label='predictions')
		plt.plot(range(n), targets[:n], label='targets')

elif plot_type == 'scatter':
	if n == -1:
		plt.scatter(range(len(predictions)), predictions, label='predictions')
		plt.scatter(range(len(targets)), targets, label='targets')
	else:
		plt.scatter(range(n), predictions[:n], label='predictions')
		plt.scatter(range(n), targets[:n], label='targets')

plt.legend(); plt.show()