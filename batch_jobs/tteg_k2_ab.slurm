#!/bin/bash
#SBATCH --account=mlopt
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --job-name=tteg_k2_ab
#SBATCH --output=output_files/tteg_k2_ab_%A_%a.out
#SBATCH --chdir=/gscratch/mlopt/jakegonz/llm_tune
#SBATCH --array=0-1

# Fast A/B pilot: entropic c=10, gamma=0.1, K=2 (smallest TT-EG-valid K, so ~2x
# faster than the K=8 headline). Task 0 = base bias point, task 1 = extrapolated.
# TT-EG requires ypp_samples >= 2; K=1 cannot exercise the tracker (b_hat == 0).

source /gscratch/mlopt/jakegonz/miniconda3/etc/profile.d/conda.sh
conda activate nlhf   # only env with deepspeed + full trl/torch stack
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

export MODEL_ROOT=/gscratch/mlopt/jakegonz/hf_cache/models
export DATA_ROOT=/gscratch/mlopt/jakegonz/hf_cache/data
export EXP_ROOT=/gscratch/mlopt/jakegonz/llm_tune/experiments

module load cuda
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))

set -e

BIAS_POINTS=(base extrapolated)
BP=${BIAS_POINTS[$SLURM_ARRAY_TASK_ID]}
# SLURM_JOB_ID is unique per array task, avoiding port clashes on shared nodes.
MASTER_PORT=$((20000 + (SLURM_JOB_ID % 20000)))

echo "[tteg_k2_ab] gamma=0.1 bias_point=$BP c=10 K=2"
deepspeed --num_gpus=1 --master_port=$MASTER_PORT risk_egpo/tt_eg.py \
    --alg eg --lora \
    --ypp_samples 2 \
    --risk entropic --risk_c 10.0 \
    --use_tteg --tteg_gamma 0.1 --tteg_bias_point $BP
