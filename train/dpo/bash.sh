#!/bin/bash

#SBATCH --job-name=ibh-hcc-
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -t 200
#SBATCH --output=/home/m.ismail/MMed-RAG/outputs/%j.out
#SBATCH --error=/home/m.ismail/MMed-RAG/outputs/%j.err
cd /home/m.ismail/MMed-RAG
conda activate mmedrag
bash scripts/train_dpo_2stages.sh