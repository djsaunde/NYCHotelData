from __future__ import print_function

import os
import argparse
import pandas as pd

from util import *


parser = argparse.ArgumentParser()
parser.add_argument('--old_distance', type=int, default=100)
parser.add_argument('--distance', type=int, default=25)
parser.add_argument('--trip_type', type=str, default='destinations')

args = parser.parse_args()
old_distance = args.old_distance
distance = args.distance
trip_type = args.trip_type

large_distance_path = os.path.join('..', 'data', 'all_preprocessed_%d' % old_distance)

processed_path = os.path.join('..', 'data', 'all_preprocessed_%d' % distance)
if not os.path.isdir(processed_path):
	os.makedirs(processed_path)

print(); print('Loading pre-processed taxi data with distance criterion d = %d.' % old_distance)
df = pd.read_csv(os.path.join(large_distance_path, trip_type + '.csv'))

print('Reducing by distance criterion d = %d.' % distance)
df = df[df['Distance From Hotel'] <= distance]

print('Writing to disk.'); print()
df.to_csv(os.path.join(processed_path, trip_type + '.csv'))