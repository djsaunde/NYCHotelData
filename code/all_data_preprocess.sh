distance=${1:-300}

mkdir "../data/all_preprocessed_$distance"

for i in {0..126}
do
	sbatch one_preprocess_all_data.sh $distance $i
done
