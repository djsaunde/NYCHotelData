distance=${1:-100}
coord_type=${2:-'pickups'}

for year in $(seq 2013 2016)
do
	for month in $(seq 1 12)
	do
		for day in $(seq 1 31)
		do
			sbatch run_get_daily_coordinates.sh $distance $year $month $day $coord_type
		done
	done
done

