#!/bin/bash 
#SBATCH --account=def-shi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=45G
#SBATCH --time=0-00:15
#SBATCH --output=print.out
mpirun ./abc_star_short












