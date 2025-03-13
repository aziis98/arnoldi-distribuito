#!/bin/bash
#SBATCH --job-name=experiment-1
#SBATCH --nodes=20
#SBATCH --output=%x_%j.log

for i in {1..20}
do
    echo "Iteration $i"
    mpirun -np $i ./build/arnoldi -l 400
    sleep 1
done

