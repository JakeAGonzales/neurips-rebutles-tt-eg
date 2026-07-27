#!/bin/bash
#SBATCH --account=mlopt
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --job-name=tteg_omd_c10
#SBATCH --output=output_files/tteg_omd_c10_%j.out
#SBATCH --chdir=/gscratch/mlopt/jakegonz/llm_tune

# OMD (oipo1) + TT, headline K=8 config: entropic c=10, gamma=0.1, base bias point.
# Single-step alg => only the base bias point applies (no extragradient look-ahead).
# Apples-to-apples counterpart to the EG-based TT runs (tteg_k8_ab).

source /gscratch/mlopt/jakegonz/miniconda3/etc/profile.d/conda.sh
conda activate nlhf   # only env with deepspeed + full trl/torch stack
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

export MODEL_ROOT=/gscratch/mlopt/jakegonz/hf_cache/models
export DATA_ROOT=/gscratch/mlopt/jakegonz/hf_cache/data
export EXP_ROOT=/gscratch/mlopt/jakegonz/llm_tune/experiments

module load cuda
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))

set -e

MASTER_PORT=$((20000 + (SLURM_JOB_ID % 20000)))

echo "[tteg_omd_c10] alg=oipo1 gamma=0.1 bias_point=base c=10 K=8"
deepspeed --num_gpus=1 --master_port=$MASTER_PORT risk_egpo/tt_eg.py \
    --alg oipo1 --lora \
    --ypp_samples 8 \
    --risk entropic --risk_c 10.0 \
    --use_tteg --tteg_gamma 0.1 --tteg_bias_point base
