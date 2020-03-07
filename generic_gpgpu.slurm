#!/bin/bash
#SBATCH --partition gpgpu
#SBATCH --nodes 1
#SBATCH --cpus-per-task 6
#SBATCH --gres gpu:4
#SBATCH --time 48:00:00
#SBATCH --mem-per-cpu 20G
#SBATCH --job-name ffjord_hep
#SBATCH --output ffjord_hep_%J.log
#SBATCH -A punim0514

# Load modules
# module load Tensorflow/1.12.0-intel-2017.u2-GCC-6.2.0-CUDA9-Python-3.6.4-GPU
module load Tensorflow/1.13.1-intel-2017.u2-GCC-6.2.0-CUDA10-Python-3.6.4-GPU

python3 cnf_flow.py
