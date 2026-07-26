#!/bin/bash
#SBATCH --account=mlopt
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=80:00:00
#SBATCH --job-name=tteg_ent_c10_K8
#SBATCH --output=output_files/tteg_ent_c10_K8_%j.out
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

# HEADLINE TT-EG run: extragradient (EGPO) + entropic risk over y'' (K=8, c=10.0)
# with the two-timescale bias tracker (gamma=0.5). c=10 is the bias-dominated
# regime where the entropic bias constant is largest -> TT-EG should show the
# clearest gain (and where bf16 rounding is the main risk; run smoke_tteg first).
# Axis-1 only (no group-DRO). eg is ~2x slower per epoch; K=8 => multi-day.
# DO NOT launch until smoke_tteg.slurm passes (Gates A/C + mechanism check).
deepspeed --num_gpus=1 --master_port=$MASTER_PORT risk_egpo/tt_eg.py \
    --alg eg --lora \
    --ypp_samples 8 \
    --risk entropic --risk_c 10.0 \
    --use_tteg --tteg_gamma 0.5
