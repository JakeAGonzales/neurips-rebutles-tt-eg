#!/bin/bash
#SBATCH --account=mlopt-ckpt
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --job-name=plot_oipo1
#SBATCH --output=plot_oipo1_%j.out
#SBATCH --chdir=/gscratch/mlopt/jakegonz/llm_tune

source /gscratch/mlopt/jakegonz/miniconda3/etc/profile.d/conda.sh
conda activate myenv
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

OUTDIR=${1:-/gscratch/mlopt/jakegonz/llm_tune/plots_oipo1}
ROOT=/gscratch/mlopt/jakegonz/llm_tune

python plot_oipo1_compare.py \
    --run OIPO1:"$ROOT/oipo1_vanilla_34839616.out" \
    --run OIPO1_K8:"$ROOT/oipo1_neutral_K8_34871534.out" \
    --run OIPO1_ent_c1:"$ROOT/oipo1_ypp_risk_34840035.out" \
    --run OIPO1_ent_c2:"$ROOT/oipo1_ent_c2_34843630.out" \
    --run OIPO1_ent_c5:"$ROOT/oipo1_ent_c5_34843631.out" \
    --run OIPO1_ent_c10:"$ROOT/oipo1_ent_c10_34843634.out" \
    --run OIPO1_cvar_a050:"$ROOT/oipo1_cvar_a050_34843635.out" \
    --run OIPO1_cvar_a025:"$ROOT/oipo1_cvar_a025_34843637.out" \
    --run OIPO1_cvar_a0125:"$ROOT/oipo1_cvar_a0125_34843638.out" \
    --run EG_K1:"$ROOT/eg_neutral_K1_34910374.out" \
    --run EG_K8:"$ROOT/eg_neutral_K8_34910376.out" \
    --run EG_ent_c5:"$ROOT/eg_ypp_risk_34847719.out" \
    --run EG_cvar_a025:"$ROOT/eg_cvar_a025_34910480.out" \
    --run EG_cvar_a0125:"$ROOT/eg_cvar_a0125_34910398.out" \
    --run NashMD_K1:"$ROOT/nmd_vanilla_34905151.out" \
    --run NashMD_K8:"$ROOT/nmd_neutral_K8_34910378.out" \
    --run gDRO_b1:"$ROOT/group_risk_egpo_34815099.out" \
    --run gDRO_b0:"$ROOT/group_risk_egpo_beta0_34831175.out" \
    --run gDRO_cvar_a025:"$ROOT/group_cvar_a025_34911692.out" \
    --outdir "$OUTDIR" \
    --prefix oipo1
