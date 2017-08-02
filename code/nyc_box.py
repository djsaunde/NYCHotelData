from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import cPickle as p
import pandas as pd
import random, timeit, os

plt.rcParams["figure.figsize"] = (18.5, 9.75)

m = Basemap(projection='hammer', llcrnrlon=-74.025, llcrnrlat=40.63, urcrnrlon=-73.76, urcrnrlat=40.85, epsg=4269)
m.arcgisimage(service='World_Street_Map', xpixels=1000, dpi=200)

plt.show()
