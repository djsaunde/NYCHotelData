distance=${1:-100}

DIRECTORY="../../data/all_preprocessed_$distance"

if [ -d $DIRECTORY ]
then
	rm -r $DIRECTORY 	
fi

mkdir $DIRECTORY

n_files=$(ls ../../data/taxi_data/ | wc -l)

for (( file=0; file<$n_files; file++ ))
do
	sbatch 3one_preprocess.sh $distance $file
done
