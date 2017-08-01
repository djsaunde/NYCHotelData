import os
import xlrd
import timeit
import pandas as pd

# Path to preprocessed data by month
data_path = os.path.join('..', 'data', 'preprocessed')

# Get names of preprocessed data files
files = [ fname for fname in os.listdir(data_path) if '.xlsx' in fname and 'Nearby Pickups and Dropoffs.xlsx' != fname ]

print '\n'

# Read in each Excel file as a pandas dataframe and append to a list
for idx, fname in enumerate(files):
	
	print '-', fname, '(', idx + 1, '/', len(files), ')'
 
	start = timeit.default_timer()

	if 'destinations' not in locals() or 'starting_points' not in locals():
		if 'destinations' not in locals():
			try:
				destinations = pd.read_excel(os.path.join(data_path, fname), sheetname='Destinations')
			except xlrd.biffh.XLRDError:
				pass
		else:
			try:
				starting_points = pd.read_excel(os.path.join(data_path, fname), sheetname='Starting Points')
			except xlrd.biffh.XLRDError:
				pass
	else:
		try:
			destinations.merge(pd.read_excel(os.path.join(data_path, fname), sheetname='Destinations'))
		except (xlrd.biffh.XLRDError, pd.tools.merge.MergeError):
			pass
		try:
			starting_points.merge(pd.read_excel(os.path.join(data_path, fname), sheetname='Starting Points'))
		except (xlrd.biffh.XLRDError, pd.tools.merge.MergeError):
			pass

	print '... It took', timeit.default_timer() - start, 'seconds to merge the last dataframe in.'

# Write the composite dataframe out to a new Excel file
writer = pd.ExcelWriter(os.path.join('..', 'data', 'preprocessed', 'Nearby Pickups and Dropoffs.xlsx'))
destinations.to_excel(writer, 'Destinations', index=False)
starting_points.to_excel(writer, 'Starting Points', index=False)
writer.save()

print '\n'
