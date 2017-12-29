#!/bin/bash
#
#SBATCH --partition=longq
#SBATCH --time=24:00:00
#SBATCH --mem=100000
#SBATCH --account=rkozma
#SBATCH --output=/mnt/nfs/work1/rkozma/djsaunde/NYCHotelData/code/job_reports/%j.out
#SBATCH --ntasks-per-node=32

cd ..

distance=${1:-100}
file_idx=${2:-0}
n_jobs=${3:-32}

python 3preprocess_data.py --distance $distance --file_idx $file_idx --n_jobs $n_jobs

exit
