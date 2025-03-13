#!/bin/bash
#SBATCH --job-name=experiment-1
#SBATCH --nodes=30
#SBATCH --output=%x_%j.log

for i in {1..30}
do
    echo "Iteration $i"
    mpirun -np $i ./build/arnoldi -l 100
    sleep 1
done

