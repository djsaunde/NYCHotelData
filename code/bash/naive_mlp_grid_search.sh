#!/bin/bash
#
#SBATCH --partition=longq
#SBATCH --time=24:00:00
#SBATCH --mem=12000
#SBATCH --account=rkozma
#SBATCH --output=/mnt/nfs/work1/rkozma/djsaunde/nyctaxi/job_reports/naive_mlp_grid_search_%j.out
#SBATCH --ntasks-per-node=8

cd ..

python grid_search_naive_mlp.sh
