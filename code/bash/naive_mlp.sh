#!/bin/bash
#
#SBATCH --partition=defq
#SBATCH --time=2:00:00
#SBATCH --account=rkozma
#SBATCH --mem=8000
#SBATCH --output=/mnt/nfs/work1/rkozma/djsaunde/nyctaxi/job_reports/naive_mlp_%j.out

cd ..

distance=${1:-'100'}
trip_type=${2:-'pickups'}
hidden_layer_sizes=${3:-'100'}
alpha=${4:-'1e-4'}

python naive_predict_occupancy_mlp.py --distance $distance --trip_type $trip_type \
				--hidden_layer_sizes $hidden_layer_sizes --alpha $alpha

