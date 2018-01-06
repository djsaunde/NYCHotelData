distance=${1:-300}
n_jobs=${2:-8}

DIRECTORY="../../data/all_preprocessed_$distance"

if [ -d $DIRECTORY ]
then
	rm -r $DIRECTORY 	
fi

mkdir $DIRECTORY

n_files=$(ls ../../data/taxi_data/ | wc -l)

for (( file=0; file<$n_files; file++ ))
do
	for hotel in $(seq 0 177)
	do
		sbatch 2one_preprocess.sh $distance $file $hotel $n_jobs
	done
done
