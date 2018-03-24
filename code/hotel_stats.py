from __future__ import print_function

import os
import numpy as np
import pandas as pd

path = os.path.join('..', 'data', 'Unmasked Daily Capacity.csv')
df = pd.read_csv(path)

print()

n_hotels = len(df['Share ID'].unique())
print('No. hotels:', n_hotels)

max_mean, max_std = 0.0, 0.0
for hotel in df['Share ID'].unique():
	mean = df[df['Share ID'] == hotel]['Room Demand'].mean()
	std = df[df['Share ID'] == hotel]['Room Demand'].std()
	
	if mean > max_mean:
		max_mean, max_std = mean, std
	
	print()
	print('Hotel:', hotel)
	print('Mean room demand:', mean)
	print('Std. room demand:', std)

print()
print('Max mean room demand:', max_mean)
print('Corresponding std. room demand:', max_std)

print()