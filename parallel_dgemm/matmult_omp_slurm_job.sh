#!/bin/bash 
# 
#SBATCH --job-name=bash 
#SBATCH --output=matmult_output.log 
# 
#SBATCH --partition=cmp
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=28
#SBATCH --mem-per-cpu=4G
#SBATCH --time=02:00:00 
#
# Load the software to run the program 

module load GCC

for t in 1 2 4 8 12 14 16 20 24 28; do
    export OMP_NUM_THREADS=${t}
    ./matmult_omp 1000 1000 1000
done

for t in 1 2 4 8 12 14 16 20 24 28; do
    export OMP_NUM_THREADS=${t}
    ./matmult_omp 1000 2000 5000
done

for t in 1 2 4 8 12 14 16 20 24 28; do
    export OMP_NUM_THREADS=${t}
    ./matmult_omp 9000 2500 3750
done
