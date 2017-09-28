#!/bin/bash
#
#SBATCH --partition=longq
#SBATCH --time=24:00:00
#SBATCH --mem=50000
#SBATCH --account=rkozma
#SBATCH --ntasks-per-node=8
#SBATCH --output=/mnt/nfs/work1/rkozma/djsaunde/NYCHotelData/code/job_reports/%j.out

distance=${1:-100}
start_date=${2:-'2013 1 1'}
end_date=${3:-'2016 12 31'}
coord_type=${4:-'pickups'}
n_jobs=${5:-8}

python get_daily_coordinates.py --distance $distance --start_date $start_date --end_date $end_date --coord_type $coord_type --n_jobs $n_jobs

exit
