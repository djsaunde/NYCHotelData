import os
import xlrd
import timeit
import pandas as pd

# Path to preprocessed data by month
data_path = os.path.join('..', 'data', 'preprocessed')

# Get names of preprocessed data files
files = [ fname for fname in os.listdir(data_path) if '.xlsx' in fname and 'Nearby Pickups and Dropoffs.xlsx' != fname ]

# Read in each Excel file as a pandas dataframe and append to a list
for idx, fname in enumerate(files):
	print ' '.join(['-', fname, '(', str(idx + 1), '/', str(len(files)), ')'])
 
	start = timeit.default_timer()

	if 'destinations' not in locals() or 'starting_points' not in locals():
		if 'destinations' not in locals():
			try:
				destinations = pd.read_excel(os.path.join(data_path, fname), sheetname='Destinations')
			except xlrd.biffh.XLRDError, e:
				pass
		if 'starting_points' not in locals():
			try:
				starting_points = pd.read_excel(os.path.join(data_path, fname), sheetname='Starting Points')
			except xlrd.biffh.XLRDError, e:
				pass
	else:
		try:
			destinations = pd.concat([destinations, pd.read_excel(os.path.join(data_path, fname), sheetname='Destinations')], ignore_index=True)
		except (xlrd.biffh.XLRDError, pd.tools.merge.MergeError), e:
			pass
		try:
			starting_points = pd.concat([starting_points, pd.read_excel(os.path.join(data_path, fname), sheetname='Starting Points')], ignore_index=True)
		except (xlrd.biffh.XLRDError, pd.tools.merge.MergeError), e:
			pass

	if 'destinations' in locals():
		print ' '.join(['Shape of destinations worksheet:', str(destinations.shape)])
	if 'starting_points' in locals():
		print ' '.join(['Shape of starting points worksheet:', str(starting_points.shape)])

	print ' '.join(['... It took', str(timeit.default_timer() - start), 'seconds to merge the last dataframe in.'])

# Write the composite dataframe out to a new Excel file
writer = pd.ExcelWriter(os.path.join('..', 'data', 'preprocessed', 'Nearby Pickups and Dropoffs.xlsx'))
destinations.to_excel(writer, 'Destinations', index=False)
starting_points.to_excel(writer, 'Starting Points', index=False)
writer.save()

