#!/bin/bash
#
#SBATCH --partition=longq
#SBATCH --time=1-00:00:00
#SBATCH --mem=60000
#SBATCH --account=rkozma
#SBATCH --ntasks-per-node=8
#SBATCH --output=/mnt/nfs/work1/rkozma/djsaunde/nyctaxi/job_reports/daily_%j.out

cd ..

distance=${1:-300}
year=${2:-2013}
month=${3:-1}
day=${4:-1}
coord_type=${5:-'pickups'}

python get_daily_coordinates.py --distance $distance --day $year $month $day --coord_type $coord_type

exit
