#!/bin/bash
#SBATCH --job-name=exp_ID
#SBATCH --nodes=NUM
#SBATCH --output=%x.log

hostname
mpirun ./build/arnoldi -l 25 -matrix_path laplacian_SIZE.mat

