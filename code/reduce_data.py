from __future__ import print_function

import os
import argparse
import pandas as pd

from util import *


parser = argparse.ArgumentParser()
parser.add_argument('--distance', type=int, default=25, help='Distance criterion (in feet) from hotels considered.')
parser.add_argument('--trip_type', type=str, default='destinations')
args = parser.parse_args()
distance = args.distance
trip_type = args.trip_type

processed_path = os.path.join('..', 'data', 'all_preprocessed_' + str(distance))
if not os.path.isdir(processed_path):
	os.makedirs(processed_path)

large_distance_path = os.path.join('..', 'data', 'all_preprocessed_300')
	
df = pd.read_csv(os.path.join(large_distance_path, trip_type + '.csv'))
df = df[df['Distance From Hotel'] <= distance]

df.to_csv(os.path.join(processed_path, trip_type + '.csv'))