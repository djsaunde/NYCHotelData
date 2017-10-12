import os
import xlrd
import timeit
import argparse
import xlsxwriter
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('--distance', type=int, default=300)
args = parser.parse_args()

distance = args.distance

# Path to preprocessed data by month
data_path = os.path.join('..', 'data', '_'.join(['preprocessed', str(distance)]))

# Get names of preprocessed data files
files = [ fname for fname in os.listdir(data_path) if '.xlsx' in fname and 'Nearby Pickups and Dropoffs.xlsx' != fname ]

dests_wb = xlsxwriter.Workbook(os.path.join(data_path, 'destinations.xlsx'), {'constant_memory': True})
dests_ws = dests_wb.add_worksheet()
dests_idx = 0

stpts_wb = xlsxwriter.Workbook(os.path.join(data_path, 'starting_points.xlsx'), {'constant_memory': True})
stpts_ws = stpts_wb.add_worksheet()
stpts_idx = 0

# Read in each Excel file as a pandas dataframe and append to a list
for idx, fname in enumerate(files):
	print ' '.join(['-', fname, '(', str(idx + 1), '/', str(len(files)), ')'])
 
	start = timeit.default_timer()

	try:
		dests = pd.read_excel(os.path.join(data_path, fname), sheetname='Destinations')
		
		for data_idx, row in dests.iterrows():
			for column_idx, column in enumerate(dests.columns.values):
				dests_ws.write(dests_idx, column_idx, row[column])
			dests_idx += 1

	except (xlrd.biffh.XLRDError, pd.tools.merge.MergeError), e:
		pass
	try:
		stpts = pd.read_excel(os.path.join(data_path, fname), sheetname='Starting Points')
		
		for data_idx, row in stpts.iterrows():
			for column_idx, column in enumerate(stpts.columns.values):
				stpts_ws.write(stpts_idx, column_idx, row[column])
			stpts_idx += 1

	except (xlrd.biffh.XLRDError, pd.tools.merge.MergeError), e:
		pass

	print ' '.join(['... It took', str(timeit.default_timer() - start), 'seconds to merge the last dataframe in.'])

print '\nDone!\n'
