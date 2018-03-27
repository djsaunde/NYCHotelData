from __future__ import print_function

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = os.path.join('..', 'data', 'Hotel Occupancy.csv')
df = pd.read_csv(path)

print()

n_hotels = len(df['Share ID'].unique())
print('No. hotels:', n_hotels)

means = []
stds = []

# max_mean, max_std = 0.0, 0.0
for hotel in df['Share ID'].unique():
	mean = df[df['Share ID'] == hotel]['Room Demand'].mean()
	std = df[df['Share ID'] == hotel]['Room Demand'].std()
	
# 	if mean > max_mean:
# 		max_mean, max_std = mean, std
	
# 	print()
# 	print('Hotel:', hotel)
# 	print('Mean room demand:', mean)
# 	print('Std. room demand:', std)
	
	means.append(mean)
	stds.append(std)

# print()
# print('Max mean room demand:', max_mean)
# print('Corresponding std. room demand:', max_std)

fig, axes = plt.subplots(2, 1)

axes[0].hist(means, bins=100, label='means')
axes[1].hist(stds, bins=100, label='stds')
axes[0].set_title('Means histogram')
axes[1].set_title('Stds. histogram')

plt.tight_layout()

plt.show()

print()