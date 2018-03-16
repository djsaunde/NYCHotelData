distance=${1:-'100'}
trip_type=${2:-'pickups'}

dir='../../data/naive_mlp_results'
file='../../data/naive_mlp_results/results.csv'

if [ -f $file ]
then
	rm $file
else
	if [ ! -d $dir ]
	then
		mkdir $dir
		touch $file
	fi
fi

for hidden_layer_sizes in '128' '256' '512' '256 128' '512 256' '512 256 128'
do
	for alpha in '1e-3' '5e-4' '1e-4' '5e-5' '1e-5'
	do
		sbatch naive_mlp.sh $distance $trip_type $hidden_layer_sizes $alpha
	done
done
