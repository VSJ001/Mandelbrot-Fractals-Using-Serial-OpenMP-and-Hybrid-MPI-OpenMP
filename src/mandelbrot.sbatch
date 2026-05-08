#!/bin/bash
#SBATCH --job-name=mandelbrot_full
#SBATCH --output=mandelbrot_full.out
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --mem=4G

# Load OpenMPI
module load OpenMPI/4.1.6

# Compile with mpicc
echo "Compiling mandelbrot.c..."
mpicc -fopenmp -o mandelbrot mandelbrot.c
echo "Compilation done."

# Create/clear the log file
echo "Mandelbrot Benchmark Started on $(date)" > results_log.txt

# Define sizes and iteration depths
SIZES=("1024 768" "1920 1080" "3840 2160")
ITERS=("500" "1000" "5000")

# Loop through all combinations
for size in "${SIZES[@]}"; do
  for iter in "${ITERS[@]}"; do
    WIDTH=$(echo $size | cut -d' ' -f1)
    HEIGHT=$(echo $size | cut -d' ' -f2)

    echo "=== Running serial $WIDTH x $HEIGHT, $iter ==="
    ./mandelbrot serial $WIDTH $HEIGHT $iter

    echo "=== Running OpenMP $WIDTH x $HEIGHT, $iter ==="
    export OMP_NUM_THREADS=4
    ./mandelbrot openmp $WIDTH $HEIGHT $iter

    echo "=== Running Hybrid MPI+OpenMP $WIDTH x $HEIGHT, $iter ==="
    export OMP_NUM_THREADS=2
    mpirun -np 4 ./mandelbrot hybrid $WIDTH $HEIGHT $iter
  done
done

echo "All 27 combinations completed. Log saved to results_log.txt"
