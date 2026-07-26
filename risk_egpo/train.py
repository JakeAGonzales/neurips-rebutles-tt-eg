r"""
Risk-aware EGPO training script.

Two orthogonal risk axes are supported:

1. Risk over y'' samples (per-prompt preference aggregation)
     --risk {neutral|cvar|entropic}, --ypp_samples K, --risk_alpha, --risk_c
   Applied inside `_compute_prefs` when ypp_samples > 1.

2. Risk over safety category e (group-DRO at batch aggregation — Option A)
     --use_group_dro --c C [--ema_alpha A]
     [--beta_anneal_frac F] [--min_per_group K] [--num_groups G]
   Implemented as streaming group-DRO (Sagawa et al. 2020) with entropic-
   risk / soft-max weights
       w_e \propto p_hat(e) * exp(+c * Z_ema[e])
   over per-group mean IPO loss. Groups default to PKU-SafeRLHF severity
   levels {safe=0, unsafe-low=1, unsafe-med=2, unsafe-high=3} derived as
   e = max(response_0_severity_level, response_1_severity_level).

Regression sanity
-----------------
When --use_group_dro is *not* set, the loss path is byte-identical to the
base EGPO IPO loss. Use this to confirm no unintended regression.

Usage (baseline EGPO, identical loss to before):
    deepspeed --num_gpus=1 risk_egpo/train.py --alg oipo1 --lora \\
        --ypp_samples 4 --risk entropic --risk_c 1.0

Usage (group DRO over e, entropic-risk c=1.0, stratified k=4):
    deepspeed --num_gpus=1 risk_egpo/train.py --alg oipo1 --lora \\
        --ypp_samples 4 --risk neutral \\
        --use_group_dro --c 1.0 --ema_alpha 0.9 \\
        --beta_anneal_frac 0.2 --min_per_group 4

Risk-over-y'' types
-------------------
  neutral   : risk-neutral mean (baseline, identical to EGPO)
  cvar      : CVaR with --risk_alpha in (0, 1]
  entropic  : entropic risk with --risk_c > 0
"""

import argparse
import deepspeed
import json
import os
import sys
import torch

# PyTorch 2.6+ made zip(..., strict=True) the default in LRScheduler._update_lr,
# which trips when LoRA + DeepSpeed leaves scheduler.base_lrs (set at init from
# 2 param groups) longer than optimizer.param_groups (collapsed to 1 by
# DeepSpeed after a group goes empty). Patch _update_lr to truncate per-step
# while preserving torch's _enable_get_lr_call context so get_lr() reads
# last_epoch correctly.
from torch.optim import lr_scheduler as _lrs
from contextlib import contextmanager

@contextmanager
def _get_lr_ctx(scheduler):
    scheduler._get_lr_called_within_step = True
    try:
        yield
    finally:
        scheduler._get_lr_called_within_step = False

_LR_DEBUG = {"n": 0}
def _safe_update_lr(self, epoch=None):
    # Observed with TRL OnlineDPO + DeepSpeed + LoRA: `step()` gets invoked via
    # the `step(epoch=...)` branch with epoch=-1 somewhere in the wrapper stack,
    # so `last_epoch` stays pinned at -1 while `_step_count` advances normally.
    # `lr_lambda(-1) = -1/warmup_steps` then yields a tiny NEGATIVE LR forever.
    # Fix: derive the current step from the monotonic `_step_count` and write
    # it back into `last_epoch` so `get_lr()` computes the right fraction.
    step_idx = max(0, int(getattr(self, "_step_count", 1)) - 1)
    self.last_epoch = step_idx
    with _get_lr_ctx(self):
        values = self.get_lr()
    n = len(self.optimizer.param_groups)
    vals = list(values)[:n]
    if _LR_DEBUG["n"] < 3:
        _LR_DEBUG["n"] += 1
        try:
            print(f"[LR-PATCH] call#{_LR_DEBUG['n']} "
                  f"cls={type(self).__name__} "
                  f"last_epoch={self.last_epoch} "
                  f"_step_count={getattr(self, '_step_count', '?')} "
                  f"base_lrs={getattr(self, 'base_lrs', '?')} "
                  f"n_pg={n} values={list(values)} applied={vals}")
        except Exception as _e:
            print(f"[LR-PATCH] debug print failed: {_e}")
    for pg, lr in zip(self.optimizer.param_groups, vals):
        pg["lr"] = lr
    self._last_lr = [pg["lr"] for pg in self.optimizer.param_groups]
_lrs.LRScheduler._update_lr = _safe_update_lr

# Resolve paths — must happen before any local imports
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)          # llm_tune/
_LMS  = os.path.join(_ROOT, "EGPO", "lms")
sys.path.insert(0, _LMS)
sys.path.insert(0, _ROOT)               # makes `risk_egpo` importable as a package

from datasets import load_dataset
from peft import get_peft_model

from configs import lora_config, get_ds_config
from judges.pair_judge import PairJudge
from utils import (
    EXP_ROOT,
    DATA_ROOT,
    AverageLossCallback,
    load_model_tokenizer,
    format_names,
    transform_into_chat,
    get_latest_checkpoint,
)

from risk_egpo.config import RiskExtragradientConfig
from risk_egpo.loss import make_neutral, make_cvar, make_entropic
from risk_egpo.trainer import RiskExtragradientTrainer
from risk_egpo.group_dro import (
    add_group_labels,
    group_histogram,
    format_histogram,
    DEFAULT_GROUP_NAMES,
)

ALG_MAPPING = {
    "oipo1": "OnlineIPO1",
    "oipo2": "OnlineIPO2",
    "nmd":   "NashMD",
    "nmdpg": "NashMDPG",
    "eg":    "Extragradient",
}

# ── argument parsing ───────────────────────────────────────────────────────────

argparser = argparse.ArgumentParser()
argparser.add_argument("--alg", choices=list(ALG_MAPPING.keys()), required=True)
argparser.add_argument("--lora", action="store_true")
argparser.add_argument("--epochs", type=int, default=10)
argparser.add_argument("--lr", type=float, default=5e-7)
argparser.add_argument("--y_yp_mixture_coef", type=float, default=0)
argparser.add_argument("--y_yp_temperature", type=float, default=2.0)
argparser.add_argument("--y_yp_top_k", type=int, default=10)
argparser.add_argument("--y_yp_min_p", type=float, default=0.0)
argparser.add_argument("--seed", type=int, default=42)
argparser.add_argument("--local_rank", type=int, default=0)
argparser.add_argument("--load_dir", type=str, default=None)

# Risk-specific arguments
argparser.add_argument("--ypp_samples", type=int, default=1,
                       help="Number of y'' samples per prompt (K)")
argparser.add_argument("--risk", choices=["neutral", "cvar", "entropic"],
                       default="neutral",
                       help="Risk functional to apply over y'' preferences")
argparser.add_argument("--risk_alpha", type=float, default=0.25,
                       help="CVaR level alpha in (0,1] (only used when --risk cvar)")
argparser.add_argument("--risk_c", type=float, default=1.0,
                       help="Entropic risk concentration c>0 (only used when --risk entropic)")

# Group-DRO arguments (risk over safety category e)
argparser.add_argument("--use_group_dro", action="store_true",
                       help="Enable risk-over-e group-DRO at batch aggregation. "
                            "When off, loss reduces exactly to EGPO IPO loss.")
argparser.add_argument("--c", "--risk_beta", dest="c", type=float, default=0.0,
                       help="Entropic-risk temperature over safety categories "
                            "(named `c` for parity with response-level entropic risk; "
                            "distinct from IPO `beta`). 0 -> nominal p(e)-weighted, "
                            "inf -> worst-category. Sign: upweights HIGH-loss groups. "
                            "Old flag name `--risk_beta` is still accepted as an alias.")
argparser.add_argument("--ema_alpha", type=float, default=0.9,
                       help="EMA coefficient for streaming per-group loss estimates.")
argparser.add_argument("--beta_anneal_frac", type=float, default=0.0,
                       help="If >0, linearly ramp c from 0 to target over "
                            "the first `beta_anneal_frac` fraction of training.")
argparser.add_argument("--min_per_group", type=int, default=0,
                       help="If >0, use group-stratified sampler with >=k samples "
                            "per group per batch. 0 disables stratification.")
argparser.add_argument("--num_groups", type=int, default=4,
                       help="Number of safety categories. Default 4 matches "
                            "PKU-SafeRLHF severity levels {0,1,2,3}.")
argparser.add_argument("--group_risk_fn", choices=["entropic", "cvar"], default="entropic",
                       help="Group-DRO weight rule. 'entropic' uses softmax over "
                            "Z_ema with --c. 'cvar' uses p_nominal-weighted "
                            "CVaR over groups at level --group_risk_alpha.")
argparser.add_argument("--group_risk_alpha", type=float, default=1.0,
                       help="CVaR level over groups (only when --group_risk_fn cvar). "
                            "alpha=1 -> ERM; alpha->0 -> worst-group. With G=4 groups, "
                            "alpha=0.25 puts ~all weight on the worst severity group.")
argparser.add_argument("--p_nominal_override", type=str, default=None,
                       help="Comma-separated floats of length --num_groups overriding "
                            "the empirical p_hat used by the group-DRO weight rule and "
                            "the stratified sampler. Will be normalized. Examples: "
                            "'1,1,1,1' (uniform), '1,2,4,8' (severity-weighted, 2^e).")

args = argparser.parse_args()

# ── distributed setup ─────────────────────────────────────────────────────────

deepspeed.init_distributed()
local_rank = args.local_rank
torch.cuda.set_device(local_rank)
world = deepspeed.comm.get_world_size()

EXP_ROOT = os.path.join(EXP_ROOT, "risk_egpo")
os.environ["WANDB_PROJECT"] = "risk_egpo"

alg = args.alg.lower()
num_gpus = torch.cuda.device_count()

batch_size = 64
micro_batch_size = 8 if args.lora else 1
gen_micro_batch_size = min(2 * micro_batch_size, batch_size // num_gpus)
samples_per_prompt = 1
prefix_chunk_num = 1
gradient_accumulation_steps = batch_size // (micro_batch_size * num_gpus)

estimate_extra_grad = (alg == "eg")
effective_factor = 2 if alg == "eg" else 1

# ── per-algorithm config ───────────────────────────────────────────────────────

if alg == "oipo1":
    additional_config = {
        "y_yp_mixture_coef": args.y_yp_mixture_coef,
        "y_yp_temperature": args.y_yp_temperature,
        "y_yp_top_k": args.y_yp_top_k,
        "y_yp_min_p": args.y_yp_min_p,
    }
elif alg == "oipo2":
    additional_config = {
        "y_yp_mixture_coef": 0,
    }
elif alg == "nmd":
    additional_config = {
        "y_yp_mixture_coef": args.y_yp_mixture_coef,
        "y_yp_temperature": args.y_yp_temperature,
        "y_yp_top_k": args.y_yp_top_k,
        "y_yp_min_p": args.y_yp_min_p,
        "ypp_mixture_coef": 0.125,
    }
elif alg == "nmdpg":
    additional_config = {
        "y_yp_mixture_coef": args.y_yp_mixture_coef,
        "y_yp_temperature": args.y_yp_temperature,
        "y_yp_top_k": args.y_yp_top_k,
        "y_yp_min_p": args.y_yp_min_p,
        "mixture_coef": 0.125,
    }
elif alg == "eg":
    additional_config = {
        "y_yp_mixture_coef": args.y_yp_mixture_coef,
        "y_yp_temperature": args.y_yp_temperature,
        "y_yp_top_k": args.y_yp_top_k,
        "y_yp_min_p": args.y_yp_min_p,
    }

# ── build risk functional ──────────────────────────────────────────────────────

if args.risk == "neutral":
    ypp_risk_fn = make_neutral()
elif args.risk == "cvar":
    ypp_risk_fn = make_cvar(args.risk_alpha)
elif args.risk == "entropic":
    ypp_risk_fn = make_entropic(args.risk_c)

# ── model / dataset ───────────────────────────────────────────────────────────

model_name = "vectorzhou/gemma-2-2b-it-alpaca-cleaned-SFT"
dataset_name = "PKU-Alignment/PKU-SafeRLHF"

# Default k for stratified sampler: floor(B / (2 G)) per writeup recipe.
if args.min_per_group > 0 and args.min_per_group * args.num_groups > batch_size:
    raise ValueError(
        f"min_per_group ({args.min_per_group}) * num_groups ({args.num_groups}) "
        f"exceeds batch_size ({batch_size})."
    )

config = {
    "num_train_epochs": args.epochs * effective_factor,
    "per_device_train_batch_size": micro_batch_size,
    "gradient_accumulation_steps": gradient_accumulation_steps,
    "per_device_generate_batch_size": gen_micro_batch_size,
    "samples_per_prompt": samples_per_prompt,
    "learning_rate": args.lr,
    "weight_decay": 0.01,
    "bf16": True,
    "warmup_steps": 1000,
    "beta": 0.1,
    "estimate_extra_grad": estimate_extra_grad,
    "prefix_chunk_num": prefix_chunk_num,
    "seed": args.seed,
    "ypp_samples": args.ypp_samples,
    # Group-DRO (risk over e). When use_group_dro=False the trainer falls back
    # to the exact base IPO loss, so these are inert.
    "use_group_dro": args.use_group_dro,
    "group_risk_fn": args.group_risk_fn,
    "group_risk_alpha": args.group_risk_alpha,
    "c": args.c,
    "ema_alpha": args.ema_alpha,
    "beta_anneal_frac": args.beta_anneal_frac,
    "min_per_group": args.min_per_group,
    "num_groups": args.num_groups,
    # p_nominal populated below after dataset load
    **additional_config,
}

ds_config = get_ds_config(
    batch_size=batch_size,
    micro_batch_size=micro_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    learning_rate=args.lr,
)
ds_config.pop("optimizer", None)
ds_config.pop("scheduler", None)

if local_rank == 0:
    # Embed risk info in the run name so checkpoints are identifiable
    risk_tag = args.risk
    if args.risk == "cvar":
        risk_tag = f"cvar{args.risk_alpha}"
    elif args.risk == "entropic":
        risk_tag = f"entropic{args.risk_c}"
    if args.use_group_dro:
        if args.group_risk_fn == "cvar":
            risk_tag += f"-eDROcvar{args.group_risk_alpha}"
        else:
            risk_tag += f"-eDRO{args.c}"
        if args.min_per_group > 0:
            risk_tag += f"-strat{args.min_per_group}"
        if args.p_nominal_override is not None:
            _ovr = [float(x) for x in args.p_nominal_override.split(",")]
            if all(float(v).is_integer() for v in _ovr):
                _ptag = "".join(str(int(v)) for v in _ovr)
            else:
                _ptag = "-".join(f"{v:g}" for v in _ovr)
            risk_tag += f"-prior{_ptag}"

    exp_base_name, run_name = format_names(
        ALG=f"Risk-{ALG_MAPPING[alg]}-K{args.ypp_samples}-{risk_tag}",
        model_name=model_name,
        dataset_name=dataset_name,
        lora=args.lora,
        config=config,
    )
    msg_holder = [exp_base_name, run_name]
else:
    msg_holder = [None, None]

deepspeed.comm.barrier()
torch.distributed.broadcast_object_list(msg_holder, src=0)
exp_base_name, run_name = msg_holder

pref_model_name = "vectorzhou/gemma-2-2b-it-preference_dataset_mixture2_and_safe_pku-Preference"
judge = PairJudge(pref_model_name)

model, tokenizer = load_model_tokenizer(model_name)
if args.lora:
    model = get_peft_model(model, lora_config)

dataset = load_dataset(dataset_name, cache_dir=DATA_ROOT)
# Attach per-sample safety-category label `e` BEFORE transform_into_chat. The
# HF .map in transform_into_chat preserves extra columns, so `e` survives.
train_raw = add_group_labels(dataset["train"], key="e")
train_dataset = transform_into_chat(train_raw)

# Group histogram + nominal p_hat (used for DRO weights and optional sampler).
labels_list = train_dataset["e"]
counts, p_nominal = group_histogram(labels_list, num_groups=args.num_groups)
if local_rank == 0:
    print(format_histogram(counts, p_nominal, names=DEFAULT_GROUP_NAMES))
    print(f"use_group_dro={args.use_group_dro}  group_risk_fn={args.group_risk_fn}  "
          f"c={args.c}  group_risk_alpha={args.group_risk_alpha}  "
          f"ema_alpha={args.ema_alpha}  min_per_group={args.min_per_group}")
if args.p_nominal_override is not None:
    override_vals = [float(x) for x in args.p_nominal_override.split(",")]
    if len(override_vals) != args.num_groups:
        raise ValueError(
            f"--p_nominal_override has {len(override_vals)} values but "
            f"--num_groups is {args.num_groups}."
        )
    if any(v < 0 for v in override_vals) or sum(override_vals) <= 0:
        raise ValueError(
            f"--p_nominal_override must be non-negative with positive sum, "
            f"got {override_vals}."
        )
    s = sum(override_vals)
    p_override = [v / s for v in override_vals]
    if local_rank == 0:
        print(f"[p_nominal] override active: raw={override_vals}  "
              f"normalized={['%.4f' % v for v in p_override]}  "
              f"(empirical p_hat={['%.4f' % v for v in p_nominal]})")
    config["p_nominal"] = p_override
else:
    config["p_nominal"] = p_nominal

# Set the output directory
if args.load_dir:
    if args.load_dir[-1] == "/":
        args.load_dir = args.load_dir[:-1]
    run_name = args.load_dir.split("/")[-1]
    exp_base_name = run_name[:run_name.rfind("-")]
output_dir = os.path.join(EXP_ROOT, run_name)

training_args = RiskExtragradientConfig(
    run_name=run_name,
    output_dir=output_dir,
    **config,
    save_steps=0.1,
    save_total_limit=None,
    logging_steps=10,
    logging_dir="./logs",
    report_to="none",
    deepspeed=json.dumps(ds_config),
)

resume_checkpoint = get_latest_checkpoint(output_dir)
if args.local_rank == 0:
    print(f"Resuming from {resume_checkpoint}")
    print(f"Risk functional: {args.risk}  ypp_samples={args.ypp_samples}")

trainer = RiskExtragradientTrainer(
    model=model,
    judge=judge,
    args=training_args,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    ypp_risk_fn=ypp_risk_fn,
    callbacks=[AverageLossCallback(gradient_accumulation_steps)],
)

trainer.train(resume_from_checkpoint=resume_checkpoint)
