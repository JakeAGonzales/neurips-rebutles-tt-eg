#!/bin/bash
#SBATCH --account=mlopt
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=80:00:00
#SBATCH --job-name=oipo1_cvar_a025
#SBATCH --output=oipo1_cvar_a025_%j.out
#SBATCH --chdir=/gscratch/mlopt/jakegonz/llm_tune

source /gscratch/mlopt/jakegonz/miniconda3/etc/profile.d/conda.sh
conda activate myenv
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

export MODEL_ROOT=/gscratch/mlopt/jakegonz/hf_cache/models
export DATA_ROOT=/gscratch/mlopt/jakegonz/hf_cache/data
export EXP_ROOT=/gscratch/mlopt/jakegonz/llm_tune/experiments

module load cuda
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))

MASTER_PORT=$((20000 + (SLURM_JOB_ID % 20000)))

# oipo1 with CVaR risk over K=8 y'' samples, alpha=0.25 (worst 25%).
deepspeed --num_gpus=1 --master_port=$MASTER_PORT risk_egpo/train.py \
    --alg oipo1 --lora \
    --ypp_samples 8 \
    --risk cvar \
    --risk_alpha 0.25
