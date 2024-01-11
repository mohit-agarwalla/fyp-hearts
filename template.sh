#!!/bin/bash
# PBS -l walltime 02:00:00
#PBS -l select=1:ncpus=1:mem=4gb

module load anaconda3/personal
source activate fyp

cd $PBS_O_WORKDIR\fyp-hearts

python3 run_experiment.py