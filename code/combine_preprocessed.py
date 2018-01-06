from __future__ import print_function

import os
import xlrd
import timeit
import argparse
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('--distance', type=int, default=300)
args = parser.parse_args()
distance = args.distance

data_path = os.path.join('..', 'data', '_'.join(['all_preprocessed', str(distance)]))

# Get names of preprocessed data files.
files = [ fname for fname in os.listdir(data_path) if '.csv' in fname and \
			('starting_points' in fname or 'destinations' in fname) and not \
			fname == 'starting_points.csv' and not fname == 'destinations.csv' ]

first_destinations, first_starting_points = True, True

# Read in each Excel file to a pandas.DataFrame and
# gradually append them to a single .csv file.
for idx, fname in enumerate(files):
	print('Filename: %s; Progress: %d / %d' % (fname, idx + 1, len(files)))
 
	start = timeit.default_timer()
	
	if 'destinations' in fname:
		destinations = pd.read_csv(os.path.join(data_path, fname), compression='gzip')
	
		if first_destinations:
			destinations.to_csv(os.path.join(data_path, 'destinations.csv'), mode='w', header=True, index=False)
			first_destinations = False
		else:
			destinations.to_csv(os.path.join(data_path, 'destinations.csv'), mode='a', header=False, index=False)
	elif 'starting_points' in fname:
		starting_points = pd.read_csv(os.path.join(data_path, fname), compression='gzip')
		
		if first_starting_points:
			starting_points.to_csv(os.path.join(data_path, 'starting_points.csv'), mode='w', header=True, index=False)
			first_starting_points = False
		else:
			starting_points.to_csv(os.path.join(data_path, 'starting_points.csv'), mode='a', header=False, index=False)

	print('Time: %.4f\n' % timeit.default_timer() - start)

print('Finished merging all preprocessed data to "destinations.csv" and "starting_points.csv".')
