#!/bin/bash
#
#SBATCH --partition=longq
#SBATCH --time=10:00:00
#SBATCH --mem=100000
#SBATCH --account=rkozma
#SBATCH --ntasks-per-node=8
#SBATCH --output=/mnt/nfs/work1/rkozma/djsaunde/NYCHotelData/code/job_reports/%j.out

distance=${1:-300}
file_idx=${2:-0}

python preprocess_data.py --distance $distance --file_idx $file_idx

exit
