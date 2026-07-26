#!/bin/bash
#SBATCH --account=mlopt
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=80:00:00
#SBATCH --array=0-3
#SBATCH --job-name=risk_egpo
#SBATCH --output=risk_egpo_%A_%a.out
#SBATCH --chdir=/mmfs1/home/maxh24/llm_tune

source /mmfs1/gscratch/mlopt/maxh24/miniconda3/etc/profile.d/conda.sh
conda activate llm_tune

export MODEL_ROOT=/mmfs1/gscratch/mlopt/maxh24/llm_tune/I_models
export DATA_ROOT=/mmfs1/gscratch/mlopt/maxh24/llm_tune/I_data
export EXP_ROOT=/mmfs1/gscratch/mlopt/maxh24/llm_tune/I_experiments

module load cuda
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))

PORT=$(( 29500 + SLURM_ARRAY_TASK_ID ))
COMMON="--alg oipo1 --lora --ypp_samples 8"

case $SLURM_ARRAY_TASK_ID in
  0)
    deepspeed --num_gpus=1 --master_port=$PORT risk_egpo/train.py \
        $COMMON --risk cvar --risk_alpha 0.5
    ;;
  1)
    deepspeed --num_gpus=1 --master_port=$PORT risk_egpo/train.py \
        $COMMON --risk cvar --risk_alpha 0.25
    ;;
  2)
    deepspeed --num_gpus=1 --master_port=$PORT risk_egpo/train.py \
        $COMMON --risk cvar --risk_alpha 0.125
    ;;
  3)
    deepspeed --num_gpus=1 --master_port=$PORT risk_egpo/train.py \
        $COMMON --risk entropic --risk_c 0.1
    ;;
esac
