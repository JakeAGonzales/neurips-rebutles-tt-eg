#!/bin/bash
#SBATCH --account=mlopt
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=tteg_smoke_xt
#SBATCH --output=output_files/tteg_smoke_xt_%j.out
#SBATCH --chdir=/gscratch/mlopt/jakegonz/llm_tune

source /gscratch/mlopt/jakegonz/miniconda3/etc/profile.d/conda.sh
conda activate nlhf   # only env with deepspeed + full trl/torch stack
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

export MODEL_ROOT=/gscratch/mlopt/jakegonz/hf_cache/models
export DATA_ROOT=/gscratch/mlopt/jakegonz/hf_cache/data
# Keep smoke checkpoints out of the real experiments tree.
export EXP_ROOT=/gscratch/mlopt/jakegonz/llm_tune/experiments_smoke

module load cuda
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
MASTER_PORT=$((20000 + (SLURM_JOB_ID % 20000)))

# Fail-fast for the train steps only (after env+lmod setup, which trips -u).
set -e

echo "============ [1/2] Gate A end-to-end (neutral, extrapolated bias) ========="
# Neutral => b_hat is exactly zero regardless of bias point; xi must stay exactly
# zero. This confirms the pre-restore extrapolated pass did not perturb Gate A.
deepspeed --num_gpus=1 --master_port=$MASTER_PORT risk_egpo/tt_eg.py \
    --alg eg --lora --ypp_samples 8 \
    --risk neutral \
    --use_tteg --tteg_gamma 0.1 --tteg_bias_point extrapolated \
    --tteg_log_every 1 --max_steps 6 --tteg_verify

echo "============ [2/2] xi-survival (entropic c=10, extrapolated bias) ========="
# A few EG pairs of the headline setting on the new path: confirms the tracker
# advances, survives the snapshot/restore, and b_hat is nonzero at c=10 when it
# is evaluated at theta_tilde (fresh forward before the rollback).
deepspeed --num_gpus=1 --master_port=$MASTER_PORT risk_egpo/tt_eg.py \
    --alg eg --lora --ypp_samples 8 \
    --risk entropic --risk_c 10.0 \
    --use_tteg --tteg_gamma 0.1 --tteg_bias_point extrapolated \
    --tteg_log_every 1 --max_steps 6 --tteg_verify

echo "============ SMOKE (extrapolated) PASSED =================================="
