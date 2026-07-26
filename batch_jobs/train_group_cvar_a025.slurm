#!/bin/bash
#SBATCH --account=mlopt
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=80:00:00
#SBATCH --job-name=group_cvar_a025
#SBATCH --output=group_cvar_a025_%j.out
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

# Group-DRO with CVaR-over-groups weighting at alpha=0.25.
# With G=4 PKU severity groups this puts ~all weight on the worst-loss group
# every step (canonical Sagawa-style worst-group DRO). Compare against:
#   train_group_risk.slurm        (entropic, beta=1, soft worst-group)
#   train_group_risk_beta0.slurm  (entropic, beta=0, ERM control)
deepspeed --num_gpus=1 --master_port=$MASTER_PORT risk_egpo/train.py \
    --alg oipo1 --lora \
    --ypp_samples 1 --risk neutral \
    --use_group_dro \
    --group_risk_fn cvar \
    --group_risk_alpha 0.25 \
    --ema_alpha 0.9 \
    --min_per_group 2
