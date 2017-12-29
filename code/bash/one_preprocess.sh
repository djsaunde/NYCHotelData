#!/bin/bash
#
#SBATCH --partition=longq
#SBATCH --time=24:00:00
#SBATCH --mem=100000
#SBATCH --account=rkozma
#SBATCH --output=/mnt/nfs/work1/rkozma/djsaunde/NYCHotelData/job_reports/%j.out
#SBATCH --ntasks-per-node=32

cd ..

distance=${1:-300}
file_idx=${2:-0}
n_jobs=${3:-32}

python preprocess_data.py --distance $distance --file_idx $file_idx --n_jobs $n_jobs

exit
