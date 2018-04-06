#!/bin/bash
#
#SBATCH --partition=longq
#SBATCH --time=12:00:00
#SBATCH --mem=60000
#SBATCH --account=rkozma
#SBATCH --ntasks-per-node=8
#SBATCH --output=/mnt/nfs/work1/rkozma/djsaunde/NYCHotelData/code/%j.out

cd ..

distance=${1:-100}
file_name=${2:-''}

python preprocess_data.py --distance $distance --file_name $file_name

exit
