distance=${1:-300}

mkdir "../data/preprocessed_$distance"

for i in {0..126}
do
	sbatch one_preprocess.sh $distance $i
done
