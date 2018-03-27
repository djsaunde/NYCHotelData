import os
import argparse
import numpy  as np
import pandas as pd

from tqdm     import tqdm
from datetime import date

parser = argparse.ArgumentParser()
parser.add_argument('--minimum', type=int, nargs=3, default=[2014, 1, 1])
parser.add_argument('--maximum', type=int, nargs=3, default=[2016, 6, 30])
locals().update(vars(parser.parse_args()))

minimum, maximum = date(*minimum), date(*maximum)

path = os.path.join('..', 'data', 'Unmasked Daily Capacity.csv')
df = pd.read_csv(path)
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

keep = []
hotels = df['Share ID'].unique()
for hotel in tqdm(hotels):
	min_date = min(df[df['Share ID'] == hotel]['Date'])
	max_date = max(df[df['Share ID'] == hotel]['Date'])
	
	if min_date.date() <= minimum and max_date.date() >= maximum:
		keep.append(hotel)

print('Hotels kept: %d' % len(keep))

df = df[df['Share ID'].isin(keep)]

path = os.path.join('..', 'data', 'Hotel Occupancy.csv')
df.to_csv(path, index=False)