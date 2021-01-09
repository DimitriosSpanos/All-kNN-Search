#!/bin/bash
#SBATCH --job-name=test
#SBATCH --time=00:50:00               # Run time (days-hh:mm:ss) - (max 7days) 
#SBATCH --partition=testing             # Submit to queue/partition named batch
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=1
#SBATCH --output=my_job.stdout    


module load gcc openmpi openblas

export OMPI_MCA_btl_vader_single_copy_mechanism=none
make clean
make all
#./V0
srun -n 4 ./V1
srun -n 4 ./V2
make clean
