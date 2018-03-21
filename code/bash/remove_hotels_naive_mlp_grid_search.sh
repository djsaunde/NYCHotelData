#!/bin/bash
#
#SBATCH --partition=longq
#SBATCH --time=48:00:00
#SBATCH --mem=64000
#SBATCH --account=rkozma
#SBATCH --output=/mnt/nfs/work1/rkozma/djsaunde/nyctaxi/job_reports/remove_hotels_naive_mlp_grid_search_%j.out
#SBATCH --ntasks-per-node=56

removals=${1:-15}

cd ..

python grid_search_naive_remove_hotels_mlp.py --removals $removals
