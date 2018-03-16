from __future__ import print_function

import os
import argparse
import pandas as pd
import dask.dataframe as dd

from util             import *
from timeit           import default_timer


parser = argparse.ArgumentParser()
parser.add_argument('--old', type=int, default=100)
parser.add_argument('--distance', type=int, default=25)
parser.add_argument('--trip_type', type=str, default='destinations')

args = parser.parse_args()
old = args.old
distance = args.distance
trip_type = args.trip_type

large_distance_path = os.path.join('..', 'data', 'all_preprocessed_%d' % old)

processed_path = os.path.join('..', 'data', 'all_preprocessed_%d' % distance)
if not os.path.isdir(processed_path):
	os.makedirs(processed_path)

print(); print('Loading pre-processed taxi data with distance criterion d = %d.' % old)
df = dd.read_csv(os.path.join(large_distance_path, trip_type + '.csv'), dtype={'Fare Amount' : 'object'})

print('Reducing by distance criterion d = %d.' % distance)
start = default_timer()

df = df.where(df['Distance From Hotel'] <= distance).dropna().drop(['Unnamed: 0'], axis=1)
df = df.compute()

print('Time: %.4f' % (default_timer() - start))

print('Writing to disk.'); print()
df.to_csv(os.path.join(processed_path, trip_type + '.csv'), index=False)