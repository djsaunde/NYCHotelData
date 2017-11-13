distance=${1:-300}
n_jobs=${2:-16}

mkdir "../data/all_preprocessed_$distance"

for i in {0..126}
do
	sbatch one_preprocess.sh $distance $i $n_jobs
done
