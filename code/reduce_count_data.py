from __future__ import print_function

import os
import argparse
import pandas as pd

from datetime import date
from timeit   import default_timer


parser = argparse.ArgumentParser()
parser.add_argument('--old', default=100, type=int)
parser.add_argument('--distance', default=25, type=int)
parser.add_argument('--trip_type', default='pickups', type=str)
parser.add_argument('--start_date', type=int, nargs=3, default=[2013, 1, 1])
parser.add_argument('--end_date', type=int, nargs=3, default=[2015, 1, 1])
parser.add_argument('--metric', type=str, default='rel_diffs')

locals().update(vars(parser.parse_args()))

fname = '_'.join(map(str, [distance, start_date[0], start_date[1], start_date[2], end_date[0], end_date[1], end_date[2], metric]))
other_fname = '_'.join(map(str, [old, start_date[0], start_date[1], start_date[2], end_date[0], end_date[1], end_date[2], metric]))

start_date, end_date = date(*start_date), date(*end_date)

taxi_occupancy_path = os.path.join('..', 'data', 'taxi_occupancy', fname)
other_taxi_occupancy_path = os.path.join('..', 'data', 'taxi_occupancy', other_fname)

if not os.path.isdir(taxi_occupancy_path):
	os.makedirs(taxi_occupancy_path)

print(); print('Loading merged taxi and occupancy data with distance criterion d = %d.' % old)
df = pd.read_csv(os.path.join(other_taxi_occupancy_path, 'Taxi and occupancy data.csv'))

print('Reducing by distance criterion d = %d.' % distance, end=' ')
start = default_timer()

df = df[df['Distance From Hotel'] <= distance]

print('(Time: %.4f)' % (default_timer() - start))

print('Writing to disk.'); print()
df.to_csv(os.path.join(taxi_occupancy_path, 'Taxi and occupancy data.csv'), index=False)