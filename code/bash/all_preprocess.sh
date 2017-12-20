distance=${1:-300}
n_jobs=${2:-8}

mkdir "../../data/all_preprocessed_$distance"

n_files=$(ls ../../data/taxi_data/ | wc -l)

cd ..

for (( i=0; i<$n_files; i++ ))
do
	sbatch one_preprocess.sh $distance $i $n_jobs
done
