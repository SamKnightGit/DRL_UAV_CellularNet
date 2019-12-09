#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --partition=General_Usage
#SBATCH --gres=gpu:1
#SBATCH --mem=2000  # memory in Mb
#SBATCH --time=0-05:00:00


export CUDA_HOME=/opt/cuda-10.0.130/

export CUDNN_HOME=/opt/cuDNN-7.6.0.64_10.0/

export STUDENT_ID=s1503602

export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH

export CPATH=${CUDNN_HOME}/include:$CPATH

export PATH=${CUDA_HOME}/bin:${PATH}

export PYTHON_PATH=$PATH

source /home/${STUDENT_ID}/miniconda3/bin/activate disspy2

python ./a2c_single_thread.py
