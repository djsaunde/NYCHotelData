#!/bin/bash
#
#SBATCH --partition=longq
#SBATCH --time=1-00:00:00
#SBATCH --mem=60000
#SBATCH --account=rkozma
#SBATCH --ntasks-per-node=8
#SBATCH --output=/mnt/nfs/work1/rkozma/djsaunde/NYCHotelData/code/job_reports/%j.out

distance=${1:-100}
day=${2:-'2013 1 1'}
coord_type=${3:-'pickups'}

python get_daily_coordinates.py --distance $distance --day $day --coord_type $coord_type

exit
