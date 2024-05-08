#!/bin/bash
####
#a) Define slurm job parameters
####
#SBATCH --job-name=test_nanoGPT
#SBATCH --cpus-per-task=1
#SBATCH --partition=day
#SBATCH --gres=gpu:1
#SBATCH --time=300:00
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
# Run your script
torchrun --standalone --nproc_per_node=1 train.py config/train_gpt2.py
echo DONE
