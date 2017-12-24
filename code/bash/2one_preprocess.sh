#!/bin/bash
#
#SBATCH --partition=longq
#SBATCH --time=24:00:00
#SBATCH --mem=100000
#SBATCH --account=rkozma
#SBATCH --output=/mnt/nfs/work1/rkozma/djsaunde/NYCHotelData/code/job_reports/%j.out
#SBATCH --ntasks-per-node=32

cd ..

distance=${1:-300}
file_idx=${2:-0}
hotel_idx=${3:-0}
n_jobs=${4:-32}

python 2preprocess_data.py --distance $distance --file_idx $file_idx \
				--hotel_idx $hotel_idx --n_jobs $n_jobs

exit
