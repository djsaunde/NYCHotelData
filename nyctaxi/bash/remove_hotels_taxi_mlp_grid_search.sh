#!/bin/bash
#
#SBATCH --partition=longq
#SBATCH --time=48:00:00
#SBATCH --mem=64000
#SBATCH --account=rkozma
#SBATCH --output=/mnt/nfs/work1/rkozma/djsaunde/nyctaxi/job_reports/remove_hotels_taxi_mlp_grid_search_%j.out
#SBATCH --ntasks-per-node=56

distance=${1:-100}
trip_type=${2:-pickups}
removals=${3:-15}

cd ..

python grid_search_taxi_remove_hotels_mlp.py --distance $distance --trip_type $trip_type --removals $removals
