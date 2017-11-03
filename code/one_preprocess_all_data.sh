#!/bin/bash
#
#SBATCH --partition=longq
#SBATCH --time=24:00:00
#SBATCH --mem=50000
#SBATCH --account=rkozma
#SBATCH --output=/mnt/nfs/work1/rkozma/djsaunde/NYCHotelData/code/job_reports/%j.out

distance=${1:-300}
file_idx=${2:-0}
n_jobs=${3:-16}

python preprocess_all_data.py --distance $distance --file_idx $file_idx --n_jobs $n_jobs

exit
