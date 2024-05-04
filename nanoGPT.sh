#!/bin/bash
####
#a) Define slurm job parameters
####
#SBATCH --job-name=get_OWT
#SBATCH --cpus-per-task=1
#SBATCH --partition=test
#SBATCH --mem-per-cpu=3G
#SBATCH --gres=gpu:0
#SBATCH --time=15:00
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=johannes.bertram@student.uni-tuebingen.de
####
#b) copy all needed data to the jobs scratch folder
####
echo START
cp -R /home/stud502/Cramming_NanoGPT/ /scratch/$SLURM_JOB_ID/
echo COPIED
####
#c) Load the conda environment and execute your code
#d) Write your checkpoints to your home directory
####
# Load the conda environment
source activate myenv
echo ENV
ls
# Run your script
python data/openwebtext/prepare.py
echo DONE
