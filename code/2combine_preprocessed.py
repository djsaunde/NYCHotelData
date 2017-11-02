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
data_path = os.path.join('..', 'data', '_'.join(['all_	preprocessed', str(distance)]))

# Get names of preprocessed data files
files = [ fname for fname in os.listdir(data_path) if '.csv' in fname and \
					('starting_points' in fname or 'destinations' in fname) ]

# Read in each Excel file as a pandas dataframe and append to a list
for idx, fname in enumerate(files):
	print ' '.join(['-', fname, '(', str(idx + 1), '/', str(len(files)), ')'])
 
	start = timeit.default_timer()

	if 'destinations' in fname and not 'destinations' in locals():
		destinations = pd.read_csv(os.path.join(data_path, fname))
	elif 'destinations' in fname and 'destinations' in locals():
		try:
			destinations = pd.concat([destinations, pd.read_csv(os.path.join(data_path, fname))])
		except pd.tools.merge.MergeError, e:
			pass
	elif 'destinations' in fname and not 'destinations' in locals():
		starting_points = pd.read_csv(os.path.join(data_path, fname))
	elif 'destinations' in fname and 'destinations' in locals():
		try:
			starting_points = pd.concat([starting_points, pd.read_csv(os.path.join(data_path, fname))])
		except pd.tools.merge.MergeError, e:
			pass

	if 'destinations' in locals():
		print ' '.join(['Shape of destinations worksheet:', str(destinations.shape)])
	if 'starting_points' in locals():
		print ' '.join(['Shape of starting points worksheet:', str(starting_points.shape)])

	print ' '.join(['... It took', str(timeit.default_timer() - start), 'seconds to merge the last dataframe in.'])

# Write the two dataframes out to CSV files
destinations.to_csv(os.path.join(data_path, 'destinations.csv'), index=False)
starting_points.to_csv(os.path.join(data_path, 'starting_points.csv'), index=False)

print '\nDone!\n'