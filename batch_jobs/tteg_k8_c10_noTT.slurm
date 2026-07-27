#!/bin/bash
#SBATCH --account=mlopt
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --job-name=tteg_k8_c10_noTT
#SBATCH --output=output_files/tteg_k8_c10_noTT_%j.out
#SBATCH --chdir=/gscratch/mlopt/jakegonz/llm_tune

# Apples-to-apples EG (no TT) control at tau/c=10, matching the TT-EG runs.
# Same codebase (tt_eg.py), same generation/prefs/loss/EG snapshot-restore --
# only difference: --use_tteg is OMITTED, so the tracker is never allocated,
# no bias is computed, and the -<xi,theta> debias term is never added
# (equivalent to forcing xi=0 every step). This is the missing matched control:
# entropic c=10 Extragradient WITHOUT the correction.

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

echo "[tteg_k8_c10_noTT] EG (no TT) c=10 K=8  (xi=0 every step)"
deepspeed --num_gpus=1 --master_port=$MASTER_PORT risk_egpo/tt_eg.py \
    --alg eg --lora \
    --ypp_samples 8 \
    --risk entropic --risk_c 10.0
