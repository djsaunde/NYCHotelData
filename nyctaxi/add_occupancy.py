import os
import sys
import pandas as pd

from datetime import date

path = os.path.join('..', 'data', 'taxi_occupancy')

start_date = date(2014, 1, 1)
end_date = date(2016, 6, 30)

for d in range(25, 325, 25):
	directory = '%d_2014_1_1_2016_6_30' % d
	
	counts = pd.read_csv(os.path.join(path, directory, 'Taxi and occupancy counts.csv'))
	occ = pd.read_csv(os.path.join('..', 'data', 'Unmasked Capacity and Price Data.csv'),
					  usecols=['Share ID', 'Date', 'Occ', 'ADR'])
	
	occ = occ.rename(index=str, columns={'Share ID' : 'Hotel Name'})
	occ['Date'] = pd.to_datetime(occ['Date'], format='%Y-%m-%d')
	occ = occ.loc[(occ['Date'] >= start_date) & (occ['Date'] <= end_date)]
	occ['Date'] = occ['Date'].dt.date
	
	# print(counts['Date'])
	# print(occ['Date'])
	# print(counts['Hotel Name'].unique())
	# print(occ['Hotel Name'].unique())
	# print(set(counts['Hotel Name'].unique()) ^ set(occ['Hotel Name'].unique()))
	# sys.exit()
	
	counts[counts[['Hotel Name', 'Date']] == occ[['Hotel Name', 'Date']]][['Occ', 'ADR']] = occ[['Occ', 'ADR']]
	
	# df = pd.merge(counts, occ, how='outer')
	
	print(counts)