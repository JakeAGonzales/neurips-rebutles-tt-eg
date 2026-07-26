"""Offline TT-EG correctness tests (no model / judge / GPU required).

Run:
    python tests/test_tteg.py          # prints a report, exits nonzero on failure
    pytest tests/test_tteg.py -q

Covers:
  Gate A  -- neutral/cvar bias estimators are exactly zero (arithmetic invariant).
  Gate C  -- entropic bias estimator matches Monte-Carlo truth, error shrinks in K.
  Sign    -- the bias surrogate's gradient equals grad(r_biased^2) - grad(r_debiased^2),
             i.e. b_hat = -(2/beta) mean(db * grad s). This is THE sign gate.
"""

import math
import os
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "EGPO", "lms"))  # for `trainers`, `utils`
sys.path.insert(0, _ROOT)

from risk_egpo.tt_eg import compute_bias_payoff  # noqa: E402


# ---------------------------------------------------------------------------
# Gate A: neutral & cvar residual bias is exactly zero.
# ---------------------------------------------------------------------------
def test_gate_a_zero_bias():
    x = torch.rand(5, 3, 8)  # (B, chunks, K)
    for kind in ("neutral", "cvar"):
        b = compute_bias_payoff(x, kind, c=1.0, alpha=0.25)
        assert b.shape == x.shape[:-1]
        assert torch.all(b == 0.0), f"{kind} bias must be exactly zero, got {b.abs().max()}"
    print("[Gate A] PASS: neutral & cvar bias estimators are exactly zero")


# ---------------------------------------------------------------------------
# Gate C: entropic delta-method estimator vs analytic leading bias term.
# ---------------------------------------------------------------------------
def _analytic_leading_bias(dist, c, K, n_big=2_000_000, seed=1):
    """Ground-truth leading bias term for h(q_hat), q_hat=mean(exp(-c P)):
        E[h(q_hat)] - h(q) ~= (1/2) h''(q) Var(g)/K,  h''(q) = -1/(c q^2)
    i.e. -Var(g)/(2 c q^2 K). q and Var(g) are means, so a large sample pins them
    to high precision (unlike the bias itself, which is a tiny O(1/K) difference)."""
    g = torch.distributions.Beta(dist[0], dist[1]).sample(
        (n_big,), )  # population draw
    gg = torch.exp(-c * g.double())
    q = gg.mean()
    var_g = gg.var(unbiased=True)
    return (-var_g / (2.0 * c * q ** 2 * K)).item()


def _gate_c_for_c(c, ks=(2, 4, 8, 16, 32, 64), n_rep=40000, seed=0):
    torch.manual_seed(seed)
    a, b = torch.tensor(2.0), torch.tensor(3.0)  # Beta payoff population, real spread
    dist = torch.distributions.Beta(a, b)

    rows = []
    for K in ks:
        analytic = _analytic_leading_bias((a, b), c, K)          # low-noise truth
        samples = dist.sample((n_rep, K))                        # (n_rep, K)
        est = compute_bias_payoff(samples, "entropic", c=c).mean().item()
        rel = abs(est - analytic) / (abs(analytic) + 1e-30)
        rows.append((K, analytic, est, rel))
    return rows


def test_gate_c_entropic_estimator():
    ok = True
    for c in (1.0, 5.0, 10.0):
        rows = _gate_c_for_c(c)
        print(f"\n[Gate C] c={c}:  K   analytic_bias  E[est_bias]    rel_err")
        for K, ab, eb, rel in rows:
            print(f"          {K:>4d}  {ab: .3e}   {eb: .3e}   {rel: .3f}")
        # Sign must match at every K (both negative: rho is concave in q). This is
        # the hard, c-independent invariant -- a sign flip is the doubling bug.
        for K, ab, eb, _ in rows:
            if not (ab < 0 and eb < 0):
                ok = False
                print(f"          SIGN FAIL at c={c}, K={K}: analytic={ab}, est={eb}")
        # The estimator must converge to the leading term as K grows (higher-order
        # O(1/K^2) corrections, larger at high c, vanish with K).
        assert rows[-1][3] < rows[0][3], f"est should converge to leading term in K (c={c})"
        assert rows[-1][3] < 0.15, f"c={c}: rel_err at K=64 = {rows[-1][3]:.3f} too large"
        # Tame regime (c=1): tight already at the K we train with.
        if c == 1.0:
            for K, _, _, rel in rows:
                if K >= 8:
                    assert rel < 0.15, f"c={c} K={K}: rel_err {rel:.3f} too large"
    assert ok, "Gate C sign check failed"
    print("\n[Gate C] PASS: estimator matches analytic leading bias, correct sign, converges in K")


# ---------------------------------------------------------------------------
# Sign gate: surrogate gradient == grad(r_biased^2) - grad(r_debiased^2).
# Mirrors TTEGTrainer._bias_surrogate: S = -(2/beta) mean(db * s), db detached.
# ---------------------------------------------------------------------------
def test_surrogate_sign():
    torch.manual_seed(0)
    beta = 0.1
    N = 16
    theta = torch.randn(N, requires_grad=True)

    # s carries grad through theta (here s = theta for a clean, exact check).
    def s_of(t):
        return t
    p = torch.randn(N)               # pref_y - pref_yp (detached target)
    db = 0.01 * torch.randn(N)       # bias_y - bias_yp (detached)

    # Reference: b_hat = grad(mean r_biased^2) - grad(mean r_debiased^2).
    r_b = s_of(theta) - p / beta
    (gb,) = torch.autograd.grad((r_b ** 2).mean(), theta, retain_graph=True)
    r_d = s_of(theta) - (p - db) / beta
    (gd,) = torch.autograd.grad((r_d ** 2).mean(), theta, retain_graph=True)
    b_hat_ref = gb - gd

    # Surrogate as implemented in the trainer.
    S = -(2.0 / beta) * (db.detach() * s_of(theta)).mean()
    (b_hat_surrogate,) = torch.autograd.grad(S, theta)

    assert torch.allclose(b_hat_surrogate, b_hat_ref, atol=1e-6), (
        f"surrogate grad != grad(r_b^2)-grad(r_d^2)\n"
        f" surrogate={b_hat_surrogate[:4]}\n ref      ={b_hat_ref[:4]}")
    # And confirm the closed form b_hat = -(2/beta) mean(db * grad s); grad s = 1 here.
    expected = -(2.0 / beta) * db / N
    assert torch.allclose(b_hat_surrogate, expected, atol=1e-6)
    print("[Sign] PASS: surrogate grad == grad(r_biased^2) - grad(r_debiased^2) = b_hat")


if __name__ == "__main__":
    test_gate_a_zero_bias()
    test_surrogate_sign()
    test_gate_c_entropic_estimator()
    print("\nALL OFFLINE TT-EG TESTS PASSED")
