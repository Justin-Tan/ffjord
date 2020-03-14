#!/bin/bash
#SBATCH --partition gpgpu
#SBATCH --nodes 1
#SBATCH --cpus-per-task 6
#SBATCH --gres gpu:4
#SBATCH --time 168:00:00
#SBATCH --mem-per-cpu 20G
#SBATCH --job-name jupyter
#SBATCH --output jupyter-%J.log
#SBATCH -A punim0514

# Load modules
# module load Tensorflow/1.12.0-intel-2017.u2-GCC-6.2.0-CUDA9-Python-3.6.4-GPU
module load Tensorflow/1.13.1-intel-2017.u2-GCC-6.2.0-CUDA10-Python-3.6.4-GPU

## get tunneling info
XDG_RUNTIME_DIR=""
#ipnport=$(shuf -i8000-9999 -n1)
ipnport=4200
ipnip=$(hostname -i)

## print tunneling instructions to jupyter-{jobid}.log
echo -e "
    Copy/Paste this in your local terminal to ssh tunnel with remote
    -----------------------------------------------------------------
    ssh -N -f -L $ipnport:$ipnip:$ipnport jtan@spartan.hpc.unimelb.edu.au
    -----------------------------------------------------------------

    Then open a browser on your local machine to the following address
    ------------------------------------------------------------------
    https://localhost:$ipnport  (prefix w/ https:// if using password)
    ------------------------------------------------------------------
    "

## start an ipcluster instance and launch jupyter server
# jupyter-notebook --no-browser --port=$ipnport --ip=$ipnip
jupyter lab --no-browser --port=$ipnport --ip=$ipnip
