#!/bin/bash
#PBS -l select=1:ncpus=12:mem=600gb
#PBS -lwalltime=03:00:00


module load anaconda3/personal
source activate fyp

cd $PBS_O_WORKDIR

python3 run_demo_experiment.py
