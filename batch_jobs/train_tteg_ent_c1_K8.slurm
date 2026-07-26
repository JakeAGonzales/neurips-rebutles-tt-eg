#!/bin/bash
#SBATCH --account=mlopt
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=80:00:00
#SBATCH --job-name=tteg_ent_c1_K8
#SBATCH --output=output_files/tteg_ent_c1_K8_%j.out
#SBATCH --chdir=/gscratch/mlopt/jakegonz/llm_tune

source /gscratch/mlopt/jakegonz/miniconda3/etc/profile.d/conda.sh
conda activate nlhf   # only env with deepspeed + full trl/torch stack
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

export MODEL_ROOT=/gscratch/mlopt/jakegonz/hf_cache/models
export DATA_ROOT=/gscratch/mlopt/jakegonz/hf_cache/data
export EXP_ROOT=/gscratch/mlopt/jakegonz/llm_tune/experiments

module load cuda
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))

MASTER_PORT=$((20000 + (SLURM_JOB_ID % 20000)))

# TT-EG CONTINUITY CONTROL (not the headline): entropic risk over y'' at c=1.0,
# K=8, tracker gamma=0.5. At c=1 the entropic bias is smallest, so this is the
# regime where the correction is tiny and most likely lost to bf16 -- useful as a
# low-c control against the headline c=10 run (train_tteg_ent_c10_K8.slurm).
# Axis-1 only (no group-DRO). eg is ~2x slower per epoch; K=8 => multi-day.
# Watch the .out for [TT-EG] lines: |xi| should rise then plateau, resid decay.
deepspeed --num_gpus=1 --master_port=$MASTER_PORT risk_egpo/tt_eg.py \
    --alg eg --lora \
    --ypp_samples 8 \
    --risk entropic --risk_c 1.0 \
    --use_tteg --tteg_gamma 0.5
