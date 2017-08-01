#!/bin/bash
#
#SBATCH --partition=longq
#SBATCH --time=10:00:00
#SBATCH --mem=100000
#SBATCH --account=rkozma
#SBATCH --ntasks-per-node=1
#SBATCH --output=/mnt/nfs/work1/rkozma/djsaunde/NYCHotelData/code/job_reports/%j.out

python combine_nearby.py

exit
