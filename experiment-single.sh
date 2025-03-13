#!/bin/bash
#SBATCH --job-name=experiment-single
#SBATCH --nodes=20
#SBATCH --output=%x_%j.log

srun ./build/arnoldi -l 25
