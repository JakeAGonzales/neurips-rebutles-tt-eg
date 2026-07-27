# Handoff: why does the c=10 TT-EG run under-perform the old EG-Ent (τ=5) baseline?

This is an investigation guide. The **symptom** is that the TT-EG runs show a much
smaller KL (distance from the reference policy) at a given step than the paper's
extragradient baselines, which looks like "worse" training. The **puzzle** is that
the TT-EG bias correction it adds is tiny (noise scale), so the correction *itself*
should not be able to cause a large performance change. Therefore something else
must differ between the old and new runs. Find it.

## The runs (exact launch commands)

| Run | Command | Env | Steps reached | KL @ step 4680 |
|-----|---------|-----|---------------|----------------|
| OLD "EG-Ent (τ=5)" (paper) | `risk_egpo/train.py --alg eg --lora --ypp_samples 8 --risk entropic --risk_c 5.0` | `myenv` | ~7300 | 0.89 |
| NEW TT-EG (base) | `risk_egpo/tt_eg.py --alg eg --lora --ypp_samples 8 --risk entropic --risk_c 10.0 --use_tteg --tteg_gamma 0.1 --tteg_bias_point base` | `nlhf` | ~2470 | n/a (ends ~0.23) |
| NEW TT-EG (extrapolated) | `... --tteg_bias_point extrapolated` | `nlhf` | ~1520 | n/a (ends ~0.12) |
| NEW no-TT c=10 control | `risk_egpo/tt_eg.py --alg eg --lora --ypp_samples 8 --risk entropic --risk_c 10.0` (no `--use_tteg`) | `nlhf` | (running) | TBD |

## Four things change simultaneously between OLD and NEW — isolate them

1. **Risk coefficient `risk_c`: 5.0 → 10.0.** Highest-prior cause. At c=10 the entropic
   weights `w_i ∝ exp(-c·P_i)` concentrate hard: effective sample size
   `ESS = 1/Σ w_i² ≈ 2.6 out of 8` (recovered from the logged delta bias
   `b = -Var̂(g)/(2cK q̂²)`, so `ESS = K/(1 − 2c(K−1)·b)`). Fewer effective samples →
   higher-variance, larger raw gradients and slower KL growth per step. This alone can
   explain most of the gap.
2. **Entry point `train.py` → `tt_eg.py` (`TTEGTrainer`).** Confirm the `use_tteg=False`
   path is behaviourally identical to the old EG path.
3. **Conda env `myenv` → `nlhf`.** Version skew: `nlhf` = transformers 4.48.0, trl 0.13.0,
   deepspeed 0.16.2, torch 2.2.1+cu121, peft 0.14.0. The May-2025 baselines were trained
   in `myenv` (which then still had deepspeed). TRL DPO/KL internals changed across
   versions and could rescale KL / reward.
4. **The TT correction itself (`--use_tteg`).** Expected to be negligible (tiny residual).

The config hyperparameters are **verified identical** between `train.py` and `tt_eg.py`
(lr, `warmup_steps=1000`, batch 64, micro 8, `num_train_epochs = epochs*2` for EG,
`weight_decay=0.01`, `beta=0.1`). So the gap is NOT a hyperparameter mismatch.

## Where to look (prioritized), with file references

- **A. `risk_c` 5 vs 10 (start here).** `risk_egpo/tt_eg.py:722-727` (`make_entropic(args.risk_c)`)
  and the entropic risk in `risk_egpo/loss.py` / `risk_egpo/trainer.py`. Verify c=10 is
  intended and quantify KL-vs-step sensitivity to c.
- **B. `TTEGTrainer.training_step` vs base `ExtragradientTrainer.training_step`.**
  - `risk_egpo/tt_eg.py` `training_step` (~lines 488-620): the `is_correction =
    global_step % 2 == 1` gating and `do_bias` path.
  - `EGPO/lms/trainers/extragradient_trainer.py` `training_step` (~lines 601-760): the
    snapshot / restore two-step update.
  - **Check:** with `use_tteg=False`, does `TTEGTrainer` reduce *exactly* to the base EG
    step? Same number of `optimizer.step()` calls per logged step? Same
    snapshot/restore order? Same logging cadence and `global_step` accounting? Any
    off-by-one in the `%2` gating would halve effective updates per logged step and make
    KL-vs-step look artificially slow.
- **C. Env skew.** `pip freeze` in `myenv` vs `nlhf`; diff transformers/trl/deepspeed.
  Look at how TRL computes `kl` and `rewards/*` in each version.
- **D. Step budget / x-axis.** Old EG ran to ~7300 steps; the TT pilots are short. Compare
  ONLY at matched steps or (better) on **metric-vs-KL** axes, not vs-step. The paper KL
  figure is produced by `batch_jobs/plot_for_paper.slurm` → `plot_oipo1_compare.py` with
  `--xclip 4680`; the same `.out` logs continue past 4680 to ~1.1–1.2, which is expected.

## The decisive experiment

Reproduce the OLD EG config through the NEW code path + env:

```bash
# in nlhf, through tt_eg.py, NO correction, but c=5 (old value)
deepspeed --num_gpus=1 risk_egpo/tt_eg.py --alg eg --lora \
    --ypp_samples 8 --risk entropic --risk_c 5.0
```

- If this **matches** the old EG (KL@4680 ≈ 0.89) → code path + env are innocent, and the
  entire gap is `risk_c` 5 → 10 (expected, not a bug).
- If it **diverges** from the old EG → the `tt_eg.py` path and/or `nlhf` env is the culprit;
  proceed to diff B and C precisely.

Independently, the **no-TT c=10 control** vs **TT-on c=10** isolates suspect #4: if their
KL/loss/accuracy trajectories overlap, the correction is confirmed negligible.

## Key files

- `risk_egpo/tt_eg.py` — `TTEGTrainer`, tracker, bias surrogate, `compute_bias_payoff`
  (entropic delta bias, ~line 94), diagnostics, and `main()` config (~line 690+).
- `risk_egpo/train.py` — old entry point; config block ~line 280 (identical hyperparams).
- `EGPO/lms/trainers/extragradient_trainer.py` — base EG snapshot/restore step.
- `EGPO/lms/loss` / `risk_egpo/loss.py` — IPO/DPO loss and risk functionals.
- `plot_training_metrics.py` / `plot_oipo1_compare.py` — training-dynamics plotting
  (EMA α=0.1 smoothing; `--xclip` to match the paper's 4680-step cutoff).
