#!/bin/bash
#
#SBATCH --partition=longq
#SBATCH --time=10:00:00
#SBATCH --mem=100000
#SBATCH --account=rkozma
#SBATCH --ntasks-per-node=8
#SBATCH --output=/mnt/nfs/work1/rkozma/djsaunde/NYCHotelData/code/job_reports/%j.out

distance=${1:-100}
coord_type=${2:-pickups}

python get_daily_coordinates.py --start_date 2013 1 1 --end_date 2013 1 31 --distance $distance --coord_type $coord_type --n_jobs 8

exit
