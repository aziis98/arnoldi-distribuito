#!/bin/bash
#SBATCH --job-name=experiment-1
#SBATCH --nodes=20
#SBATCH --output=%x.log

for i in {1..20}
do
    echo "Node Count: $i"
    mpirun -d3 -np $i ./build/arnoldi -l 25
    sleep 1
done

