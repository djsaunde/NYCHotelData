distance=${1:-100}
coord_type=${2:-'pickups'}

for year in $(seq 2013 2016)
do
	for month in $(seq 1 12)
	do
		for day in $(seq 1 31)
		do
			date=$year' '$month' '$day
			sbatch run_get_daily_coordinates.sh $distance $date $coord_type
		done
	done
done

sbatch run_combine_daily_coordinates.sh $distance '2013 1 1' '2016 6 30' $coord_type
