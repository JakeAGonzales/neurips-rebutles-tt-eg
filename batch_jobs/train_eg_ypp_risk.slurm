#!/bin/bash
#SBATCH --account=ctrl
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=80:00:00
#SBATCH --job-name=eg_ypp_risk
#SBATCH --output=eg_ypp_risk_%j.out
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

# Extragradient (EGPO) with RISK OVER Y'' SAMPLES.
# --alg eg enables the extragradient inner update (effective_factor=2,
# estimate_extra_grad=True). K=8 y'' samples per prompt, entropic risk c=1.0.
# No group-DRO axis (orthogonal).
# NOTE: eg is ~2x slower per epoch than oipo1; with K=8 this is ~16x the cost
# of the K=1 vanilla oipo1 baseline. Expect multi-day wall-clock.
deepspeed --num_gpus=1 --master_port=$MASTER_PORT risk_egpo/train.py \
    --alg eg --lora \
    --ypp_samples 8 \
    --risk entropic \
    --risk_c 5.0
