"""
Risk functionals for preference aggregation.

Convention
----------
Every callable follows the PAYOFF → LOSS convention:

    rho(x: Tensor[..., K]) -> Tensor[...]

    - Input  x: K payoff values in [0, 1] (e.g. P(y > y'') for K y'' samples)
    - Output  : a scalar LOSS per batch element (higher = worse)

To obtain a risk-adjusted PAYOFF for the IPO formula, negate the output:

    risk_payoff = -rho(raw_preferences)

Built-in functionals
--------------------
make_neutral()          risk-neutral: rho(x) = -mean(x)
make_cvar(alpha)        CVaR at level alpha: rho(x) = -mean of bottom-alpha fraction
make_entropic(c)        Entropic risk: rho(x) = -(1/c) log sum exp(-c x)
"""

import math
import torch
from typing import Callable


def make_neutral() -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Risk-neutral aggregation: rho(x) = -mean(x, dim=-1).

    PAYOFF version: -rho = mean(x), identical to original EGPO expectation.
    """
    def neutral(x: torch.Tensor) -> torch.Tensor:
        return -x.mean(dim=-1)
    return neutral


def make_cvar(alpha: float) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Conditional Value-at-Risk (CVaR) at level alpha.

    rho(x) = -mean of the bottom-alpha fraction of payoffs.

    alpha in (0, 1]: fraction of worst-case payoffs to average over.
    alpha=1.0 degenerates to risk-neutral.
    alpha->0 approaches the minimum payoff (most pessimistic).

    PAYOFF version: -rho = mean of worst-alpha fraction = CVaR payoff.
    This emphasises prompts where y rarely beats y''.
    """
    if not (0 < alpha <= 1.0):
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")

    def cvar(x: torch.Tensor) -> torch.Tensor:
        K = x.shape[-1]
        k = max(1, math.ceil(alpha * K))
        # smallest k values are the worst payoffs (bottom-alpha)
        worst, _ = x.topk(k, dim=-1, largest=False, sorted=False)
        return -worst.mean(dim=-1)
    return cvar


def make_entropic(c: float) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Entropic risk measure with concentration parameter c > 0.

    rho(x) = (1/c) * logsumexp(-c * x, dim=-1)

    Good y (high consistent payoffs) → small logsumexp → low rho (LOSS convention ✓).
    Bad y (low payoffs) → large logsumexp → high rho.

    c -> 0  : approaches risk-neutral mean (L'Hôpital)
    c -> inf: approaches -min(x)  (most risk-averse)
    """
    if c == 0:
        raise ValueError("c must be non-zero; use make_neutral() for risk-neutral case")

    def entropic(x: torch.Tensor) -> torch.Tensor:
        # (1/c) * log(mean(exp(-c*x))) = logsumexp - log(K))/c
        return (1.0 / c) * (torch.logsumexp(-c * x, dim=-1) - math.log(x.shape[-1]))
    return entropic
