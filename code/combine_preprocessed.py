import os
import xlrd
import timeit
import argparse
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('--distance', type=int, default=300)
args = parser.parse_args()

distance = args.distance

# Path to preprocessed data by month
data_path = os.path.join('..', 'data', '_'.join(['all_preprocessed', str(distance)]))

# Get names of preprocessed data files
files = [ fname for fname in os.listdir(data_path) if '.csv' in fname and ('starting_points' in fname or 'destinations' in fname) \
							and not fname == 'starting_points.csv' and not fname == 'destinations.csv' ]

first_destinations = True
first_starting_points = True
# Read in each Excel file as a pandas dataframe and append to a list
for idx, fname in enumerate(files):
	print ' '.join(['-', fname, '(', str(idx + 1), '/', str(len(files)), ')'])
 
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

	print ' '.join(['... It took', str(timeit.default_timer() - start), 'seconds to merge the last dataframe in.'])

print '\nDone!\n'
