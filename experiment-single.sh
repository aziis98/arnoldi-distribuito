#!/bin/bash
#SBATCH --job-name=experiment-NUM
#SBATCH --nodes=NUM
#SBATCH --output=%x_%j.log

mpirun ./build/arnoldi -l 25
