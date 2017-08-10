distance=${1:-300}

mkdir ../data/preprocessed

declare -a fnames=("green_tripdata_2014-11.csv" "green_tripdata_2015-08.csv" "green_tripdata_2014-09.csv" "green_tripdata_2014-08.csv" "green_tripdata_2015-01.csv" \
			" green_tripdata_2014-12.csv" "green_tripdata_2014-02.csv" "green_tripdata_2014-07.csv" "green_tripdata_2015-12.csv")

for fname in "${fnames[@]}"
do
	sbatch filename_one_preprocess.sh $distance $fname
done
