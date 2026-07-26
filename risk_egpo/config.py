from dataclasses import dataclass, field
from typing import Optional

from trainers.extragradient_config import ExtragradientConfig


@dataclass
class RiskExtragradientConfig(ExtragradientConfig):
    r"""
    Configuration for RiskExtragradientTrainer.

    Extends ExtragradientConfig with two orthogonal risk axes:

    1. Risk over y'' (per-sample preference aggregation over K y'' draws).
       Controlled by `ypp_samples` and the `ypp_risk_fn` passed to the trainer.

    2. Risk over safety categories e (group-DRO at the batch-aggregation step).
       Enabled by `use_group_dro=True`. Implements Option A from the writeup
       as streaming group-DRO. The group-weight rule is selected by
       `group_risk_fn`:
         - "entropic" (default): soft-max weighting parameterized by `c`.
         - "cvar": p_nominal-weighted CVaR over groups at level
           `group_risk_alpha`; with G=4 groups and alpha=0.25 this is the
           classical worst-group DRO (Sagawa et al. 2020).
       When `use_group_dro=False` the loss reduces exactly to the base EGPO
       IPO loss (regression sanity).

    Parameters:
        ypp_samples (`int`, *optional*, defaults to `1`):
            Number of y'' samples drawn per prompt for preference estimation
            (risk-over-y'' axis).

        use_group_dro (`bool`, *optional*, defaults to `False`):
            Master switch for risk-over-e (group DRO). When False, the trainer
            behaves identically to the base ExtragradientTrainer at the batch
            aggregation step.
        c (`float`, *optional*, defaults to `0.0`):
            Entropic-risk temperature over safety categories. Named `c`
            for parity with the response-level entropic risk; not the same
            as the IPO KL coefficient (which is called `beta`).
            Sign convention: IPO loss is lower-is-better, so weights
            w_e \propto p(e) * exp(+c * Z_hat_e) upweight high-loss
            groups. c -> 0 gives the nominal p(e)-weighted loss;
            c -> inf concentrates on the worst group.
        ema_alpha (`float`, *optional*, defaults to `0.9`):
            EMA coefficient for per-group loss estimates used to compute the
            detached softmax weights. Z_ema[e] = alpha * Z_ema[e] + (1-alpha) * Z_hat[e].
        num_groups (`int`, *optional*, defaults to `4`):
            Number of safety categories. For PKU-SafeRLHF with severity in
            {0,1,2,3}, use 4 groups: {safe, unsafe-low, unsafe-med, unsafe-high}.
        p_nominal (`list[float]`, *optional*, defaults to uniform):
            Nominal category frequencies p_hat(e), length `num_groups`. If None,
            uniform 1/num_groups is used. Normally set from dataset statistics
            at train-time.
        beta_anneal_frac (`float`, *optional*, defaults to `0.0`):
            If > 0, linearly ramp c from 0 to the target value over the
            first `beta_anneal_frac` fraction of training. 0.0 disables annealing.
            (Field name kept as `beta_anneal_frac` for backward compatibility;
            only the per-group risk parameter itself was renamed risk_beta -> c.)
        min_per_group (`int`, *optional*, defaults to `0`):
            If > 0, use a group-stratified sampler that guarantees at least
            `min_per_group` samples from each group per batch. 0 disables
            stratified sampling (standard iid sampling).
    """

    ypp_samples: int = 1

    use_group_dro: bool = False
    # Selects the group-weight rule used by the DRO aggregator:
    #   "entropic" (default) -> w_e \propto p(e) * exp(+c * Z_ema[e])
    #   "cvar"                -> p_nominal-weighted CVaR over groups at level
    #                            `group_risk_alpha` in (0, 1].
    group_risk_fn: str = "entropic"
    group_risk_alpha: float = 1.0  # only used when group_risk_fn == "cvar"
    c: float = 0.0                  # only used when group_risk_fn == "entropic"
    ema_alpha: float = 0.9
    num_groups: int = 4
    p_nominal: Optional[list] = None
    beta_anneal_frac: float = 0.0
    min_per_group: int = 0

    # Prevent HF Trainer from stripping the `e` safety-category column before
    # it reaches the training step. Without this, `remove_unused_columns=True`
    # (HF default) drops every column not in the model's forward signature —
    # including `e` — and every sample falls through to the `e=0` fallback
    # in training_step, which would make group-DRO inert (all samples in
    # group 0).
    remove_unused_columns: bool = False
