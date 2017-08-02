# NYC Hotel Data Project

This repository contains code for projects associated with yellow and green taxicab data from New York City from 2008-2016. The data is necessarily sparse due to inconsistent data collection and removable of (latitude, longitude) pick-up and drop-off coordinates for certain months / years. One goal of this project is to use the dataset of taxicab rides, along with a dataset of NYC hotels and associated data, to quantify the competition between the top hotels in NYC and to determine which parts of the city are underserved by the hotel industry.

## Methods

For each hotel in NYC, we look at the taxicab trips which either originate from or end up at it. We say that those trips which begin or end within, say, 50 feet of the hotel are _close enough_ to provide a good estimation of this population. Luckily, the [New York City Taxi and Limousine Commission](http://www.nyc.gov/html/tlc/html/home/home.shtml) provides us with the aforementioned geospatial coordinates of these trips (along with other potentially useful information per trip), and, given the addresses of the hotels in NYC which we aim to investigate, we can use a geolocation service to get their coordinates as well. This project uses the [Google Maps Geolocation API](https://developers.google.com/maps/documentation/geolocation/intro), accessed from the convenient Python interface [geopy](https://github.com/geopy/geopy).

After discovering those trips which begin or end _close_ to each hotel of interest, we would then like to estimate the distribution of pick-up or drop-off locations by hotel; i.e., in the case of pick-ups, given a hotel, from where did the patron likely come from? Since we are only interested in the city of New York, we can first represent the city by choosing a geospatial bounding box around it (given by four sets of (latitude, longitude) coordinates),

![NYC Bounding Box](https://github.com/djsaunde/NYCHotelData/blob/master/nyc_box.png)

and then divide the box into regularly-sized "bins" of, say, 500 square feet. To estimate the distribution of pick-up locations for a particular hotel, we assign a count to each bin, incrementing it for each pick-up coordinate which lies within. To obtain a proper probability distribution, we divide each bin's count by the total number of trips in our dataset, ensuring that the bins' values sum to 1. 

We can then graft this empiricial probability distribution onto the map of NYC for viewing purposes.

Next, given some population of interest (pick-ups, drop-offs, or both), we want to investigate some measure of similarity between these populations per hotel. Once we have computed the distributions as described above, we can compute their pairwise [Kullbeck-Liebler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence), although we must be careful to assign small non-negative probabilities to the bins on the NYC map for which there are no data and renormalize before we do.

It is important to note that the estimation of these distributions and their comparison can be done for certain time intervals of interest: we may choose the year, month, day, and time of day, which may change the form of the per-hotel, per-population distributions. For example, we may expect more taxicab traffic in NYC's commerical district during the week than during the weekends, and more traffic in the downtown area at night and on the weekends.

## Setting Things Up

This guide will assume you are using a \*nix system. Clone the repository and change directory to the top level of the project. Issue

```
pip install -r requirements.txt
```

to install (some of) the project's dependencies.

Plotting heatmaps over a map of NYC requires a bit more setup. Namely, you will need to download and install ```mpl_toolkits.basemap```, one of the ```matplotlib``` toolkits, which doesn't seem to come standard as part of the ```matplotlib``` package anymore.

Thanks to [Github user robinkraft's gist](https://gist.github.com/robinkraft/2a8ee4dd7e9ee9126030) and [this matplotlib.org article](https://matplotlib.org/basemap/users/installing.html), I was able to install ```basemap```.

To compile and install GEOS, issue the following:

```
cd /tmp
wget http://download.osgeo.org/geos/geos-3.4.2.tar.bz2
bunzip2 geos-3.4.2.tar.bz2
tar xvf geos-3.4.2.tar

cd geos-3.4.2

./configure && make && sudo make install
sudo ldconfig  # links the GEOS libraries so basemap can find them later
```

You may wish to change the GEOS version (from 3.4.2 to a newer version), but these commands worked fine for me.

Now, navigate to the [basemap 1.0.7 downloads page](https://sourceforge.net/projects/matplotlib/files/matplotlib-toolkits/basemap-1.0.7/), and select __basemap-1.0.7.tar.gz__. Download it to a directory of your choice, denoted by ```<BASEMAPDIR>```, and issue the following:

```
cd <BASEMAPDIR>
tar xzvf basemap-1.0.7.tar.gz
cd basemap-1.0.7
python setup.py install
```

Now, ```basemap``` should be installed. To verify this, enter a Python interactive session and issue

```
from mpl_toolkits.basemap import Basemap
```
