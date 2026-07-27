#!/bin/bash
#SBATCH --account=mlopt
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=tteg_omd_smoke
#SBATCH --output=output_files/tteg_omd_smoke_%j.out
#SBATCH --chdir=/gscratch/mlopt/jakegonz/llm_tune

# Smoke test for the OMD (single-step) TT port. Validates that:
#  - the every-step cadence works when estimate_extra_grad=False,
#  - Gate A holds (neutral => xi stays exactly 0, no extragradient rollback), and
#  - the tracker advances (b_hat nonzero) at entropic c=10 on the oipo1 update.

source /gscratch/mlopt/jakegonz/miniconda3/etc/profile.d/conda.sh
conda activate nlhf
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

export MODEL_ROOT=/gscratch/mlopt/jakegonz/hf_cache/models
export DATA_ROOT=/gscratch/mlopt/jakegonz/hf_cache/data
export EXP_ROOT=/gscratch/mlopt/jakegonz/llm_tune/experiments_smoke

module load cuda
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
MASTER_PORT=$((20000 + (SLURM_JOB_ID % 20000)))

set -e

echo "==================== [1/2] Gate A end-to-end (OMD, neutral) =============="
# Neutral => b_hat is exactly zero; xi must stay exactly zero on the single-step
# path too. --tteg_verify asserts the invariant at train end.
deepspeed --num_gpus=1 --master_port=$MASTER_PORT risk_egpo/tt_eg.py \
    --alg oipo1 --lora --ypp_samples 8 \
    --risk neutral \
    --use_tteg --tteg_gamma 0.5 --tteg_log_every 1 \
    --max_steps 6 --tteg_verify

echo "==================== [2/2] xi advances (OMD, entropic c=10) =============="
# Confirms every-step tracker updates fire and b_hat is nonzero at c=10.
deepspeed --num_gpus=1 --master_port=$MASTER_PORT risk_egpo/tt_eg.py \
    --alg oipo1 --lora --ypp_samples 8 \
    --risk entropic --risk_c 10.0 \
    --use_tteg --tteg_gamma 0.5 --tteg_log_every 1 \
    --max_steps 6 --tteg_verify

echo "==================== OMD SMOKE PASSED ===================================="
