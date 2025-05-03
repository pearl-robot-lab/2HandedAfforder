#!/bin/bash
#SBATCH --job-name=pipeline_P01_03
#SBATCH --mem-per-cpu=5GB
#SBATCH --time=6-23:59:00
#SBATCH --partition normal
#SBATCH --output=output/%A/output_%a.txt
#SBATCH --error=output/%A/error_%a.txt
#SBATCH -n 1
#SBATCH --cpus-per-task 5
#SBATCH --gres gpu:a100

#SBATCH -A project02426                                                                                                                                                                                                 
#SBATCH -p project02426                                                                                                                                                                                      

python train_custom_sam.py --data-folder $1 --model-folder $2 --batchsize 2 --data-folder-validation $3 --out-images $4 --num-epochs 10 --single-image $5
