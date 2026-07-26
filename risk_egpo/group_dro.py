r"""
Group-DRO ("risk over safety categories e") machinery for risk-aware EGPO.

Implements Option A from the writeup:

    L_beta^e(pi, pi') = (1/beta) log sum_e p(e) exp(beta * Z_e(pi, pi'))

where Z_e is the *IPO squared loss* restricted to samples with category e,
and p(e) is the nominal category frequency. For a lower-is-better loss Z the
gradient weights are

    w_e \propto p(e) * exp(+beta * Z_e)                                 (*)

i.e. upweight high-loss groups. This is the sign-correct dual of the
payoff-form "soft-min" in the writeup (Eq. L-robust with -beta).

We use the streaming / EMA variant of Sagawa et al. 2020:

    Z_ema[e]  = alpha * Z_ema[e] + (1-alpha) * Z_hat[e].detach()
    w[e]      = softmax_with_prior(+c * Z_ema[e], p_nominal)            # detached
    loss      = sum_e w[e] * Z_hat[e]                                   # grad flows

so gradient weights are stable across noisy batches while the gradient itself
stays on the current minibatch (unbiased).

Module contents:
    - `compute_group_label`       : int label from PKU-SafeRLHF row (max severity).
    - `add_group_labels`          : HF dataset .map helper.
    - `group_histogram`           : diagnostic counts / p_hat.
    - `GroupDROState`             : per-trainer EMA/buffer + loss aggregator.
    - `GroupStratifiedSampler`    : optional min-per-group batch sampler.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import math
import numpy as np
import torch
from torch.utils.data import Sampler


# ─────────────────────────────────────────────────────────────────────────────
# Group labels from PKU-SafeRLHF
# ─────────────────────────────────────────────────────────────────────────────

def compute_group_label(example: dict) -> int:
    """
    Pair-level severity label in {0, 1, 2, 3}:

        e = max(response_0_severity_level, response_1_severity_level)

    Maps to the 4-group scheme {safe, unsafe-low, unsafe-med, unsafe-high}.
    Severity 0 is the "safe" group; is_response_*_safe is not needed since
    the severity scale already encodes safety (0 == safe).

    Raises KeyError if the severity fields are not present; this is intentional
    so we fail loudly when the dataset schema changes.
    """
    return int(max(example["response_0_severity_level"],
                   example["response_1_severity_level"]))


def add_group_labels(dataset, key: str = "e"):
    """
    Return a new HF dataset with an int column `key` = compute_group_label(row).
    """
    def _fn(ex):
        ex[key] = compute_group_label(ex)
        return ex
    return dataset.map(_fn)


def group_histogram(labels, num_groups: int):
    """
    Returns (counts, p_hat) as lists of length num_groups.
    `labels` is an iterable of int group ids in [0, num_groups).
    """
    counts = [0] * num_groups
    for e in labels:
        counts[int(e)] += 1
    total = sum(counts)
    p_hat = [c / total if total > 0 else 1.0 / num_groups for c in counts]
    return counts, p_hat


def format_histogram(counts, p_hat, names=None) -> str:
    G = len(counts)
    if names is None:
        names = [f"e={g}" for g in range(G)]
    total = sum(counts)
    lines = [f"  {names[g]:<16s} n={counts[g]:6d}  p_hat={p_hat[g]:.4f}"
             for g in range(G)]
    return (f"Group histogram (total={total}):\n" + "\n".join(lines))


# Default names for the 4-group PKU-SafeRLHF scheme
DEFAULT_GROUP_NAMES = ["safe", "unsafe-low", "unsafe-med", "unsafe-high"]


# ─────────────────────────────────────────────────────────────────────────────
# DRO aggregator
# ─────────────────────────────────────────────────────────────────────────────

class GroupDROState:
    """
    Holds per-group EMA loss estimates and produces (weighted_loss, diagnostics)
    from a per-sample loss vector and an integer group-label vector.

    Weights follow the lower-is-better loss convention:
        w_e proportional to p_nominal[e] * exp(+c * Z_ema[e])

    so groups with high EMA loss get upweighted. Weights are detached; the
    gradient flows through Z_hat computed on the current batch.

    "Warmup" behavior: until every group has been observed at least once
    in a batch (across the history), we fall back to uniform w to avoid
    chasing noise from an uninitialized EMA slot. The first time a group is
    seen, its EMA is initialized to that batch's Z_hat[e] (no mixing).
    """

    def __init__(
        self,
        num_groups: int,
        p_nominal: Optional[list] = None,
        ema_alpha: float = 0.9,
    ):
        self.num_groups = int(num_groups)
        self.ema_alpha = float(ema_alpha)
        # lazy-initialized on first call so we land on the right device
        self.Z_ema: Optional[torch.Tensor] = None
        self.Z_ema_seen: Optional[torch.Tensor] = None
        if p_nominal is None:
            p = [1.0 / num_groups] * num_groups
        else:
            assert len(p_nominal) == num_groups, \
                f"p_nominal length {len(p_nominal)} != num_groups {num_groups}"
            s = sum(p_nominal)
            p = [x / s for x in p_nominal]
        self._p_nominal_list = p
        self.p_nominal: Optional[torch.Tensor] = None  # lazy
        # diagnostics (populated on each call)
        self.last_w: Optional[torch.Tensor] = None
        self.last_Z_hat: Optional[torch.Tensor] = None
        self.last_counts: Optional[torch.Tensor] = None

    def _lazy_init(self, device, dtype):
        if self.Z_ema is None:
            self.Z_ema = torch.zeros(self.num_groups, device=device, dtype=dtype)
            self.Z_ema_seen = torch.zeros(self.num_groups, device=device,
                                          dtype=torch.bool)
            self.p_nominal = torch.tensor(
                self._p_nominal_list, device=device, dtype=dtype
            )

    def compute_loss(
        self,
        per_sample_loss: torch.Tensor,   # (B,)
        e: torch.Tensor,                 # (B,) long
        c: float,
        risk_fn: str = "entropic",
        risk_alpha: float = 1.0,
    ):
        """
        Returns (loss_scalar, diagnostics_dict).

        Aggregates per-sample loss into per-group means Z_hat[e], updates the
        EMA (detached), computes detached weights w[e], and returns
        sum_{e present in batch} w[e] * Z_hat[e].

        Groups not present in the current batch contribute 0 to the gradient
        (standard streaming group-DRO behavior).

        Weight rules (selected by `risk_fn`):
            - "entropic" (default): w_e \propto p_nominal[e] * exp(+c * Z_ema[e]).
              c=0 -> nominal-weighted ERM; c -> inf concentrates on the worst group.
            - "cvar": p_nominal-weighted CVaR over groups at level `risk_alpha`.
              Sort groups by Z_ema descending, pour cumulative p_nominal mass
              until reaching alpha, allocate w_e = p_nominal[e]/alpha to fully-
              covered groups and a partial weight to the boundary group so that
              sum w_e = 1. risk_alpha=1.0 reduces to nominal ERM.
        """
        device, dtype = per_sample_loss.device, per_sample_loss.dtype
        self._lazy_init(device, dtype)

        G = self.num_groups
        # per-group mean of current batch
        Z_hat = torch.zeros(G, device=device, dtype=dtype)
        counts = torch.zeros(G, device=device, dtype=torch.long)
        for g in range(G):
            mask = (e == g)
            n_g = int(mask.sum().item())
            counts[g] = n_g
            if n_g > 0:
                Z_hat[g] = per_sample_loss[mask].mean()

        present = counts > 0

        # EMA update (detached, in-place on buffer)
        with torch.no_grad():
            Zd = Z_hat.detach()
            first_time = present & (~self.Z_ema_seen)
            update = present & self.Z_ema_seen
            # init from first batch each group is seen
            self.Z_ema[first_time] = Zd[first_time]
            a = self.ema_alpha
            self.Z_ema[update] = a * self.Z_ema[update] + (1.0 - a) * Zd[update]
            self.Z_ema_seen = self.Z_ema_seen | present

            # weights from EMA
            if bool(self.Z_ema_seen.all().item()):
                if risk_fn == "entropic":
                    w = _entropic_group_weights(
                        self.Z_ema, self.p_nominal, c
                    )
                elif risk_fn == "cvar":
                    w = _cvar_group_weights(
                        self.Z_ema, self.p_nominal, risk_alpha
                    )
                else:
                    raise ValueError(f"Unknown group risk_fn: {risk_fn!r}")
            else:
                w = torch.ones(G, device=device, dtype=dtype) / G

        # weighted-sum loss: gradient flows through Z_hat (present groups only)
        loss = (w[present] * Z_hat[present]).sum()

        # stash diagnostics
        self.last_w = w.detach()
        self.last_Z_hat = Z_hat.detach()
        self.last_counts = counts.detach()

        diag = {
            "dro/w": w.detach().cpu().tolist(),
            "dro/Z_hat": Z_hat.detach().cpu().tolist(),
            "dro/Z_ema": self.Z_ema.detach().cpu().tolist(),
            "dro/counts": counts.detach().cpu().tolist(),
            "dro/c": float(c),
            "dro/risk_fn": str(risk_fn),
            "dro/risk_alpha": float(risk_alpha),
            "dro/all_groups_seen": bool(self.Z_ema_seen.all().item()),
        }
        return loss, diag


# ───────────────────────────────────────────────────────────────────
# Group-weight rules (entropic and CVaR)
# ───────────────────────────────────────────────────────────────────

def _entropic_group_weights(
    Z_ema: torch.Tensor,       # (G,) per-group EMA loss (higher = worse)
    p_nominal: torch.Tensor,   # (G,) base group probabilities
    c: float,
) -> torch.Tensor:
    """
    w_e \propto p_nominal[e] * exp(+c * Z_ema[e]), normalized to sum 1.
    Computed in log-space for numerical stability.
    """
    logits = c * Z_ema
    log_p = torch.log(p_nominal.clamp_min(1e-30))
    z = logits + log_p
    z = z - z.max()
    w = torch.exp(z)
    w = w / w.sum()
    return w


def _cvar_group_weights(
    Z_ema: torch.Tensor,       # (G,) per-group EMA loss (higher = worse)
    p_nominal: torch.Tensor,   # (G,) base group probabilities
    alpha: float,
) -> torch.Tensor:
    """
    p_nominal-weighted CVaR over groups at level alpha in (0, 1].

    Sort groups by Z_ema descending; allocate w_e = p_nominal[e]/alpha to
    groups whose cumulative p_nominal mass <= alpha, then a partial weight to
    the boundary group so that sum w_e == 1. alpha == 1 reduces to nominal
    p_nominal weights (i.e. ERM).
    """
    G = Z_ema.shape[0]
    device, dtype = Z_ema.device, Z_ema.dtype
    a = float(max(alpha, 1e-12))

    if a >= 1.0 - 1e-9:
        return p_nominal / p_nominal.sum().clamp_min(1e-30)

    # sort groups by Z_ema descending (worst-loss first)
    order = torch.argsort(Z_ema, descending=True)
    p_sorted = p_nominal[order]
    cum = torch.cumsum(p_sorted, dim=0)

    # allocate full p_e/alpha weight where cumulative mass <= alpha,
    # partial weight to the boundary group (where cumulative crosses alpha).
    full_mask = cum <= a
    w_sorted = torch.zeros(G, device=device, dtype=dtype)
    w_sorted[full_mask] = p_sorted[full_mask] / a

    # boundary index: first group where cumulative mass first exceeds alpha
    crossed = (~full_mask).nonzero(as_tuple=False)
    if crossed.numel() > 0:
        b = int(crossed[0].item())
        used = cum[b - 1].item() if b > 0 else 0.0
        remaining = max(a - used, 0.0)
        w_sorted[b] = remaining / a

    # invert sort to align with original group indexing
    w = torch.zeros(G, device=device, dtype=dtype)
    w[order] = w_sorted
    # numerical safety: enforce nonneg, normalize to sum 1
    w = w.clamp_min(0.0)
    w = w / w.sum().clamp_min(1e-30)
    return w


def anneal_c(
    target: float,
    global_step: int,
    max_steps: Optional[int],
    anneal_frac: float,
) -> float:
    """
    Linear ramp: c = target * min(1, step / (anneal_frac * max_steps)).
    If anneal_frac <= 0 or max_steps is falsy, returns target.
    """
    if anneal_frac <= 0 or not max_steps:
        return target
    warmup = max(1, int(anneal_frac * max_steps))
    frac = min(1.0, float(global_step) / float(warmup))
    return target * frac


# ─────────────────────────────────────────────────────────────────────────────
# Stratified sampler (optional, min-per-group per batch, rest proportional)
# ─────────────────────────────────────────────────────────────────────────────

class GroupStratifiedSampler(Sampler):
    """
    Single-process index sampler: each contiguous window of `batch_size` yields
    at least `min_per_group` samples from each of `num_groups` groups, with the
    remaining slots drawn proportional to `p_nominal` (or empirical frequencies
    if `p_nominal` is None).

    Notes
    -----
    * Sampling is with replacement inside each group (so rare groups do not
      block batch formation). Overall epoch length is approximately
      `ceil(len(dataset) / batch_size) * batch_size`.
    * This sampler is *not* distributed-aware. When the HF Trainer is launched
      on world_size > 1 the DataLoader will shard per-rank, which degrades
      the per-batch stratification guarantee but does not break training.
      Intended use is single-GPU runs (matches current SLURM config).
    """

    def __init__(
        self,
        group_labels,
        batch_size: int,
        num_groups: int,
        min_per_group: int,
        p_nominal: Optional[list] = None,
        seed: int = 0,
        drop_last: bool = False,
    ):
        self.num_groups = int(num_groups)
        self.batch_size = int(batch_size)
        self.min_per_group = int(min_per_group)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)

        labels = np.asarray(list(group_labels), dtype=np.int64)
        self.n = int(labels.shape[0])
        self.by_group = [np.where(labels == g)[0] for g in range(num_groups)]
        counts = np.array([len(ix) for ix in self.by_group], dtype=np.float64)
        if p_nominal is None:
            # empirical
            p = counts / max(counts.sum(), 1.0)
        else:
            assert len(p_nominal) == num_groups
            p = np.asarray(p_nominal, dtype=np.float64)
            p = p / max(p.sum(), 1e-30)
        self.p_nominal = p

        # guard: groups with zero samples break sampling-with-replacement
        self._nonempty = [len(ix) > 0 for ix in self.by_group]
        if not any(self._nonempty):
            raise ValueError("GroupStratifiedSampler: all groups are empty")

        assert self.min_per_group * self.num_groups <= self.batch_size, (
            f"min_per_group*num_groups ({self.min_per_group * self.num_groups}) "
            f"exceeds batch_size ({self.batch_size})"
        )

    def _draw_batch(self, rng: np.random.Generator) -> np.ndarray:
        B = self.batch_size
        G = self.num_groups
        slots = np.zeros(G, dtype=np.int64)
        # minimum per group (only for non-empty groups; empty groups skip)
        for g in range(G):
            if self._nonempty[g]:
                slots[g] = self.min_per_group
        remaining = B - int(slots.sum())
        if remaining > 0:
            # distribute remaining slots ~proportional to p_nominal, restricted
            # to non-empty groups
            p = self.p_nominal.copy()
            for g in range(G):
                if not self._nonempty[g]:
                    p[g] = 0.0
            s = p.sum()
            if s <= 0:
                # fall back to uniform over non-empty groups
                p = np.array([1.0 if ne else 0.0 for ne in self._nonempty],
                             dtype=np.float64)
                p /= p.sum()
            else:
                p /= s
            extras = rng.multinomial(remaining, p)
            slots += extras

        # sample `slots[g]` indices from each group with replacement
        out = []
        for g in range(G):
            k = int(slots[g])
            if k <= 0:
                continue
            ixs = self.by_group[g]
            if len(ixs) == 0:
                continue
            picks = ixs[rng.integers(0, len(ixs), size=k)]
            out.append(picks)
        batch = np.concatenate(out) if out else np.array([], dtype=np.int64)
        rng.shuffle(batch)
        return batch

    def __iter__(self):
        # Derive a per-epoch seed: HF Trainer bumps epoch; we just use seed.
        rng = np.random.default_rng(self.seed)
        num_batches = self.n // self.batch_size
        if not self.drop_last and self.n % self.batch_size != 0:
            num_batches += 1
        for _ in range(num_batches):
            batch = self._draw_batch(rng)
            for idx in batch.tolist():
                yield int(idx)

    def __len__(self):
        num_batches = self.n // self.batch_size
        if not self.drop_last and self.n % self.batch_size != 0:
            num_batches += 1
        return num_batches * self.batch_size
