from __future__ import print_function

import os
import argparse
import pandas as pd

from util             import *
from timeit           import default_timer


parser = argparse.ArgumentParser()
parser.add_argument('--distance', default=100, type=int)
parser.add_argument('--trip_type', default='pickups', type=str)
parser.add_argument('--start_date', type=int, nargs=3, default=[2013, 1, 1])
parser.add_argument('--end_date', type=int, nargs=3, default=[2015, 1, 1])
parser.add_argument('--metric', type=str, default='rel_diffs')

locals().update(vars(parser.parse_args()))

fname = '_'.join(map(str, [distance, start_date[0], start_date[1], start_date[2], end_date[0], end_date[1], end_date[2], metric]))

start_date, end_date = date(*start_date), date(*end_date)

taxi_occupancy_path = os.path.join('..', 'data', 'taxi_occupancy', fname)
predictions_path = os.path.join('..', 'data', 'taxi_lr_predictions', fname)

processed_path = os.path.join('..', 'data', 'all_preprocessed_%d' % distance)
if not os.path.isdir(processed_path):
	os.makedirs(processed_path)

print(); print('Loading pre-processed taxi data with distance criterion d = %d.' % old)
df = df.read_csv(os.path.join(large_distance_path, trip_type + '.csv'))

print('Reducing by distance criterion d = %d.' % distance)
start = default_timer()

df = df.where(df['Distance From Hotel'] <= distance).dropna().drop(['Unnamed: 0'], axis=1)
df = df.compute()

print('Time: %.4f' % (default_timer() - start))

print('Writing to disk.'); print()
df.to_csv(os.path.join(processed_path, trip_type + '.csv'), index=False)