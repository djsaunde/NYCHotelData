from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import cPickle as p
import pandas as pd
import random, timeit, os

plt.rcParams["figure.figsize"] = (18.5, 9.75)

print '\n'
print '...reading in pickups data'

start_time = timeit.default_timer()
pickups = p.load(open('../NYCHotelData/data/nearby_pickups.p', 'rb'))

print 'it took', timeit.default_timer() - start_time, 'seconds to load the pickups data'
print '\n'

if 'nyc_map.p' in os.listdir('../NYCHotelData/data/'):
	print '...loading basemap of NYC from pickled Python object'
	m = p.load(open('../NYCHotelData/data/nyc_map.p', 'rb'))
else:
	print '...getting basemap of NYC from ARCGIS'
	m = Basemap(projection='hammer', llcrnrlon=-74.025, llcrnrlat=40.63, urcrnrlon=-73.76, urcrnrlat=40.85, epsg=4269)
	m.arcgisimage(service='World_Street_Map', xpixels=1000, dpi=200)
	p.dump(m, open('../NYCHotelData/data/nyc_map.p', 'wb'))

m.scatter(pickups['Longitude'][:100000], pickups['Latitude'][:100000], marker='+', color='Blue')

plt.show()