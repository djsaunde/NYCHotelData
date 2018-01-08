# NYC Hotel Data Project

This repository contains code for projects making use of yellow and green taxicab data from New York City during the years 2009-2017. All years and months (except for July 2016 onward) have (latitude, longitude) coordinates associated with taxicab trips pickup and dropoff location. One goal of this project is to use the dataset of taxicab rides, along with a dataset of NYC hotel data, to quantify the competition between the top hotels in NYC and to determine which parts of the city are underserved by the hotel industry.

## Methods

For each hotel in NYC, we look at the taxicab trips which either originate from or end up at it. We say that those trips which begin or end within, say, 100 feet of the hotel are _close enough_ to provide a good estimation of these groups of trips. Luckily, the [New York City Taxi and Limousine Commission](http://www.nyc.gov/html/tlc/html/home/home.shtml) provides us with the aforementioned geospatial coordinates of these trips (along with other potentially useful details). Given the addresses of the hotels in NYC which we aim to investigate, we can use a geolocation service to get their coordinates as well. This project uses the [Google Maps Geolocation API](https://developers.google.com/maps/documentation/geolocation/intro), accessed from the convenient Python interface [geopy](https://github.com/geopy/geopy).

After discovering those trips which begin or end _close_ to each hotel of interest, we would then like to estimate the distribution of pick-up or drop-off locations by hotel; i.e., in the case of pick-ups, given a hotel, where did the patron likely come from? Since we are only interested in the city of New York, we can first represent the city by choosing a geospatial bounding box around it (given by four sets of (latitude, longitude) coordinates):

![NYC Bounding Box](https://github.com/djsaunde/NYCHotelData/blob/master/nyc_box.png)

We can then divide the box into regularly-sized "bins" of, say, 500 square feet. To estimate the distribution of pick-up locations for a particular hotel, we assign a count to each bin, incrementing it for each pick-up coordinate which lies inside. To obtain a proper probability distribution, we divide each bin's count by the total number of trips in our dataset, ensuring that the bins' values sum to 1. 

We can draw this __empiricial probability distribution__ onto the map of NYC for visualization purposes.

Next, given some population of interest (pick-ups, drop-offs, or both), we want to investigate some measure of similarity between these populations per hotel. Once we have computed the distributions as described above, we can compute their pairwise [Kullbeck-Liebler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence), although we must assign small non-negative probabilities to the bins on the NYC map for which there are no data and renormalize before we do.

The estimation of these distributions and their comparison can be done for any time interval of interest: we may choose the year, month, day, and time of day, or given a start and end date, which should change the per-hotel, per-population distributions. For example, we may expect more taxicab traffic in NYC's commerical district during the week than during the weekends, and more traffic in the downtown area at night and on the weekends.

## Setting Things Up

This guide will assume you are using a \*nix system. Clone the repository and change directory to the top level of the project. Issue

```
pip install -r requirements.txt
```

to install the project's dependencies.

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

## Running the code

These instructions are intended for all those involved with this project in the UMass Amherst Department of Resource Economics. In order to run the code, you must have certain sensitive NYC hotel data on your machine, which requires that you are a part of this project and have been granted access.

However, a large part of this codebase is general-purpose and does not rely on such specific credentials. Feel free to reuse and repurpose all code pertaining to the taxi data.

### Geolocating hotels

Ensure that you have a [Google Geocoding-enabled API](https://developers.google.com/maps/documentation/geocoding/start) key in a file titled "_key.txt_", located at the top level of the project directory. You must also have the file titled "_Final hotel Identification.xlsx_" in the `data` directory. This contains data on the "Share ID"s, names, and addresses of each NYC hotel being studied.

Navigate to the `code` directory and run `python geolocate_hotel.py` This will create a new file, titled "_Final hotel Identification (with coordinates).csv_", which adds latitude and longitude columns to the data from the original file, which have been calculated using the `GoogleV3` geolocation interface provided by the [`geopy`](https://pypi.python.org/pypi/geopy) library.

### Getting taxicab data

As a first step, we can download all the taxicab data we care to inspect. In the `code/bash` directory, one can find the `download_raw_data.sh` and `get_data_file.sh` bash scripts, which can be run by issuing `./download_raw_data.sh` or `./get_data_file.sh [color] [year] [month]` (replace "color" with "yellow" or "green", "year" with "2009", ..., "2017", and "month" with "01", ..., "12"). Note that only years 2009 - 2016 (up through June 2016) contain (latitude, longitude) coordinates; other years / months will cause errors in later processing steps.

These scripts download the indicated taxi data files to the `taxi/taxi_data` directory. 

### Pre-processing taxicab data

In order to reduce the large volume of the NYC taxi data, we can safely discard trips which don't begin at or end up *near* any of the hotels in our list. This nearness is specified by some *distance* criterion in feet. Also, we may only be interested in *pick-ups* or *drop-offs* near hotels, or the combination thereof.

Once some (or all) taxicab data is downloaded (A few hundred gigabytes! Consider using high-performance computing (HPC) resources), one can use the script `preprocess_data.py` in the `code` directory to throw away unneeded trips. This script accepts arguments `distance` (distance from hotel criterion), `file_name` (name of file to pre-process), `n_hotels` (number of hotels, in order, to pre-process with respect to; used for debugging purposes), `n_jobs` (number of CPU threads to use for parallel computing), and `file_idx` (index of data file in alphabetically ordered list of data file names; used in bash scripts for parallelization). An example run of this script is as follows:

```
python preprocess_data.py --distance 300 --file_name yellow_tripdata_2013-01.csv --n_jobs 8
``` 

The default values for all but the `distance` and `file_name` arguments will typically suffice.

To pre-process all taxi data using an HPC system with the [Slurm workload manager](https://slurm.schedmd.com/), one can use the bash script `all_preprocess.sh` which accepts command-line arguments for `distance` and `n_jobs`. For example,

```
./all_preprocess.sh 300 16
```

will submit a Slurm job (using the `sbatch` command) running `preprocess_data.py` for each taxi data file in the `data/taxi_data`, in which in individual process will use a 300 feet distance criterion, and 16 threads for parallel processing.

The `all_preprocess.sh` script submits jobs via the `one_preprocess.sh` script, which contains Slurm job arguments at the top of the file. Modify these according to the limitations of your HPC system, or for your desired configuration.

The `preprocess_data.py` writes out those trips satisfying the user-specified criteria at the command-line: Whether to look for nearby taxicab pick-up / drop-off trips or both, and how far trips need to *start at* or *end up* near a hotel to be considered *nearby*. Trips are written to a directory `data/all_preprocessed_[distance]` as `.csv` files with titles like `NPD_[coordinate_type]_[taxi data filename].csv`, where `coordinate_type` is replaced with one of `destinations` or `starting_points`, which correspond to the endpoints or nearby pick-ups and starting points of nearby drop-offs, respectively.

Finally, we can combine all pre-processed data thus far with the script `combine_preprocessed.py`, which can be run on an HPC system with `run_combine_preprocessed.sh`. Both the Python and bash scripts accept a `distance` argument, which then looks for pre-processed taxi data in the corresponding `data/all_preprocessed_[distance]` directory. The end result of these programs are the files `destinations.csv` and `starting_points.csv`, stored again in `data/all_preprocessed_[distance]`, which simply combined all pre-processed data with the distance criterion.

### Getting daily distributions of trip coordinates

The script `get_daily_coordinates.py` accepts arguments `start_date` (list of year, month, and day), `end_date` (list of year, month, and day), `coord_type` (string; one of "pickups", "dropoffs", or "both"), and `distance` (integer distance criterion in feet). This script makes use of the [`dask` parallel computing library](http://dask.pydata.org/en/latest/index.html) to generate CSV files for each day from `start_date` to `end_date` containing a single row of (pick-up, drop-off, or both) coordinates of rides (beginning, ending, or both) near all hotels being studied. This script can be run, for example, with:

```
python get_daily_coordinates.py --start_date 2014 6 14 --end_date 2014 6 21 --coord_type pickups --distance 100
```

This will generate CSV files of the form `pickups_100_[date].csv`, where `date` ranges from `start_date` to `end_date`.

Run the script `combine_daily_coordinates.py` (accepting the same command-line arguments as `get_daily_coordinates.py`, to retrieve the appropriate files) to combine all those single-row CSV files previously generated into a comprehensive CSV file containing one row for each day from `start_date` to `end_date`, where dates are indexed by the first column. The script stored the CSV file as `[coord_type]_[distance]_[start_date]_[end_date].csv`.

## Contact

The following is a list of the personnel working involved in this project, along with contact information and other details:

- Daniel Saunders ([email](mailto:djsaunde@cs.umass.edu) (djsaunde@cs.umass.edu) | [webpage](https://djsaunde.github.io) | [blog](https://djsaunde.wordpress.com))

- Professor Christian Rojas ([email](mailto:rojas@resecon.umass.edu) (rojas@resecon.umass.edu) | [webpage](https://www.umass.edu/resec/people/rojas))

- Professor Debi Mohapatra ([email](mailto:dmohapatra@umass.edu) (dmohapatra@umass.edu) | [webpage](https://sites.google.com/a/cornell.edu/debi-prasad/))
