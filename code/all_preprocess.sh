#!/bin/bash
#
#SBATCH --partition=longq
#SBATCH --time=02-00:00:00
#SBATCH --mem-per-cpu=4000
#SBATCH --account=rkozma
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=8


mpiexec -np 50 python 2preprocess_data.py
exit
