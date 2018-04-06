#!/bin/bash
#
#SBATCH --partition=longq
#SBATCH --time=48:00:00
#SBATCH --mem=64000
#SBATCH --account=rkozma
#SBATCH --output=/mnt/nfs/work1/rkozma/djsaunde/nyctaxi/job_reports/opt_remove_naive_mlp_grid_search_%j.out
#SBATCH --ntasks-per-node=56

trip_type=${1:-pickups}
removals=${2:-15}
metric=${3:-rel_diffs}

cd ..

python grid_search_naive_opt_remove_mlp.py --trip_type $trip_type --removals $removals --metric $metric
