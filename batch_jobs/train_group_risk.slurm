#!/bin/bash
#SBATCH --account=ctrl
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=80:00:00
#SBATCH --job-name=group_risk_egpo
#SBATCH --output=group_risk_egpo_%j.out
#SBATCH --chdir=/gscratch/mlopt/jakegonz/llm_tune

source /gscratch/mlopt/jakegonz/miniconda3/etc/profile.d/conda.sh
conda activate myenv
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

export MODEL_ROOT=/gscratch/mlopt/jakegonz/hf_cache/models
export DATA_ROOT=/gscratch/mlopt/jakegonz/hf_cache/data
export EXP_ROOT=/gscratch/mlopt/jakegonz/llm_tune/experiments

module load cuda
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))

# Unique master port per job to avoid EADDRINUSE when multiple jobs share a node.
MASTER_PORT=$((20000 + (SLURM_JOB_ID % 20000)))
deepspeed --num_gpus=1 --master_port=$MASTER_PORT risk_egpo/train.py \
    --alg oipo1 --lora \
    --ypp_samples 1 --risk neutral \
    --use_group_dro \
    --risk_beta 1.0 \
    --ema_alpha 0.9 \
    --beta_anneal_frac 0.2 \
    --min_per_group 2
