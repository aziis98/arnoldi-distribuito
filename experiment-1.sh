#!/bin/bash
#SBATCH --job-name=experiment-1
#SBATCH --nodes=20
#SBATCH --output=%x_%j.log

for i in {1..20}
do
    echo "Node Count: $i"
    srun --nodes=$i ./build/arnoldi -l 25
    sleep 1
done

