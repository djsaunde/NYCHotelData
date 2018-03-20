#!/bin/bash
#
#SBATCH --partition=longq
#SBATCH --time=24:00:00
#SBATCH --mem=64000
#SBATCH --account=rkozma
#SBATCH --output=/mnt/nfs/work1/rkozma/djsaunde/nyctaxi/job_reports/taxi_mlp_grid_search_%j.out
#SBATCH --ntasks-per-node=56

distance=${1:-100}
trip_type=${2:-pickups}

cd ..

python grid_search_taxi_mlp.py --distance $distance --trip_type $trip_type
