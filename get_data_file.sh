if [ ! -d 'data/taxi_data' ]; then
	mkdir data/taxi_data
fi

color=${1:-yellow}
year=${2:-2013}
month=${3:-01}

wget -c -P data/taxi_data/ "https://s3.amazonaws.com/nyc-tlc/trip+data/${color}_tripdata_${year}-${month}.csv"
