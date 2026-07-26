#!/bin/bash
#SBATCH --account=mlopt
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=tteg_smoke
#SBATCH --output=output_files/tteg_smoke_%j.out
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

# Fail-fast for the test/train steps only (after env+lmod setup, which trips -u).
set -e

echo "==================== [1/4] Offline gates (A, C, sign) ===================="
python tests/test_tteg.py

echo "==================== [2/4] Debias mechanism (known-xi) ==================="
deepspeed --num_gpus=1 --master_port=$MASTER_PORT tests/test_tteg_mechanism.py

echo "==================== [3/4] Gate A end-to-end (neutral, real stack) ======="
# Neutral => b_hat is exactly zero; xi must stay exactly zero (bit-identical to
# non-TT-EG). The in-run assertion in _tteg_log + --tteg_verify enforce this.
deepspeed --num_gpus=1 --master_port=$MASTER_PORT risk_egpo/tt_eg.py \
    --alg eg --lora --ypp_samples 8 \
    --risk neutral \
    --use_tteg --tteg_gamma 0.5 --tteg_log_every 1 \
    --max_steps 6 --tteg_verify

echo "==================== [4/4] xi-survival (entropic c=10, real stack) ======="
# A few optimizer steps of the headline setting: confirms the tracker advances,
# survives the extragradient snapshot/restore, and b_hat is nonzero at c=10.
deepspeed --num_gpus=1 --master_port=$MASTER_PORT risk_egpo/tt_eg.py \
    --alg eg --lora --ypp_samples 8 \
    --risk entropic --risk_c 10.0 \
    --use_tteg --tteg_gamma 0.5 --tteg_log_every 1 \
    --max_steps 6 --tteg_verify

echo "==================== SMOKE PASSED ========================================="
