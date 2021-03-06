#!/bin/bash
#
#SBATCH --partition=longq
#SBATCH --time=48:00:00
#SBATCH --mem=60000
#SBATCH --account=rkozma
#SBATCH --ntasks-per-node=1
#SBATCH --output=/mnt/nfs/work1/rkozma/djsaunde/NYCHotelData/job_reports/%j.out

cd ..

distance=${1:-300}

echo $distance

python combine_preprocessed.py --distance $distance

exit
