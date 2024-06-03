#!/bin/bash
####
#a) Define slurm job parameters
####
#SBATCH --job-name=lcdata
#SBATCH --cpus-per-task=24
#SBATCH --partition=week
#SBATCH --gres=gpu:0
#SBATCH --mem-per-cpu=150
#SBATCH --time=7000:00
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=johannes.bertram@student.uni-tuebingen.de
####
#b) copy all needed data to the jobs scratch folder
####
echo START
cp -R /home/stud502/Cramming_NanoGPT/data/openwebtext /scratch/$SLURM_JOB_ID/
echo COPIED
####
#c) Load the conda environment and execute your code
#d) Write your checkpoints to your home directory
####
# Load the conda environment
source activate data
# Run your script
python prepare_lc.py
echo DONE
