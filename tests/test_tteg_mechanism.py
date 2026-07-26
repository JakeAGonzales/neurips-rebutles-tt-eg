r"""TT-EG debias-mechanism verification under the real DeepSpeed stack.

This is the one-off "known-xi constant" check: it confirms that adding the
surrogate  -<xi, theta>  to *every* microbatch loss produces a parameter delta of
exactly  lr * xi  relative to the xi=0 case -- i.e. the correction survives
(a) DeepSpeed's 1/gas gradient-accumulation scaling and (b) bf16 rounding. It is
the offline substitute for the `debias_effective` metric we cannot expose under
ZeRO-0 + BF16 (no pre/post gradients).

Uses plain SGD (momentum 0, no clipping) so the applied update is exactly
-lr*(g_main - xi); with AdamW in production the *gradient*-level correction is the
same, only the per-parameter scaling differs.

Run:
    deepspeed --num_gpus=1 tests/test_tteg_mechanism.py
"""

import sys

import torch
import torch.nn as nn
import deepspeed

GAS = 4          # gradient accumulation steps (== #microbatches)
LR = 0.1
D = 8


def _ds_config(clip):
    return {
        "train_batch_size": GAS,
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": GAS,
        "optimizer": {"type": "SGD", "params": {"lr": LR, "momentum": 0.0}},
        "bf16": {"enabled": True},
        "zero_optimization": {"stage": 0},
        "gradient_clipping": clip,
        "steps_per_print": 10 ** 12,
        "wall_clock_breakdown": False,
    }


def _build_model():
    torch.manual_seed(1234)          # identical init on every call
    return nn.Linear(D, 1, bias=False)


def _fixed_data(device):
    g = torch.Generator().manual_seed(7)
    X = torch.randn(GAS, D, generator=g)
    y = torch.randn(GAS, 1, generator=g)
    return X.to(device), y.to(device)


def _debias_term(engine, xi):
    """-<xi, theta>; grad wrt theta is -xi (cast to param dtype, as in production)."""
    term = None
    for n, p in engine.named_parameters():
        if p.requires_grad and n in xi:
            c = (xi[n].to(p.device, dtype=p.dtype) * p).sum()
            term = c if term is None else term + c
    return -term


def _run(use_xi, xi_value, clip=0.0):
    model = _build_model()
    engine, _, _, _ = deepspeed.initialize(
        model=model, model_parameters=model.parameters(), config=_ds_config(clip))
    device = engine.device
    X, y = _fixed_data(device)

    xi = {n: torch.full_like(p, xi_value) for n, p in engine.named_parameters()
          if p.requires_grad}
    param_dtype = next(engine.parameters()).dtype  # bf16 under this config

    for i in range(GAS):
        pred = engine(X[i:i + 1].to(dtype=param_dtype))
        loss = 0.5 * ((pred.float() - y[i:i + 1].float()) ** 2).mean()
        if use_xi:
            loss = loss + _debias_term(engine, xi)
        engine.backward(loss)
        engine.step()

    theta = {n: p.detach().float().cpu().clone()
             for n, p in engine.named_parameters() if p.requires_grad}
    del engine
    torch.cuda.empty_cache()
    return theta, xi


def test_known_xi_param_delta():
    # Clear, above-bf16-noise magnitude: expect delta == lr * xi exactly.
    xi_val = 1.0
    theta_a, _ = _run(use_xi=False, xi_value=xi_val)
    theta_b, xi = _run(use_xi=True, xi_value=xi_val)

    ok = True
    for n in theta_a:
        delta = theta_b[n] - theta_a[n]
        expected = LR * xi[n].float().cpu()
        rel = (delta - expected).abs().max().item() / (expected.abs().max().item() + 1e-12)
        print(f"[mechanism] {n}: max|delta-lr*xi|/|lr*xi| = {rel:.4e} "
              f"(delta~{delta.abs().mean():.4e}, expected~{expected.abs().mean():.4e})")
        if rel > 0.10:  # loose enough for bf16 weight rounding, tight enough to
            ok = False  # catch 1/gas (~0.75 off) or rounding-away (~1.0 off)
    assert ok, ("[mechanism] FAIL: param delta != lr*xi -> the -<xi,theta> "
                "correction is not reaching the gradient at full strength "
                "(1/gas scaling or bf16 rounding). If delta ~= lr*xi/gas, the "
                "term is only being added to one microbatch.")
    print("[mechanism] PASS: correction nets exactly lr*xi (1/gas + bf16 verified)")


def sweep_bf16_threshold():
    """Informational: smallest xi whose correction survives bf16. Substitutes for
    the online debias_effective metric."""
    print("\n[mechanism] bf16 survival sweep (delta/(lr*xi) should be ~1):")
    for xi_val in (1.0, 1e-1, 1e-2, 1e-3, 1e-4):
        theta_a, _ = _run(use_xi=False, xi_value=xi_val)
        theta_b, xi = _run(use_xi=True, xi_value=xi_val)
        n = next(iter(theta_a))
        delta = (theta_b[n] - theta_a[n]).abs().mean().item()
        expected = (LR * xi[n].float().cpu()).abs().mean().item()
        ratio = delta / (expected + 1e-30)
        print(f"    xi={xi_val:.0e}  delta/(lr*xi)={ratio:.3f}"
              f"{'  <-- rounded away' if ratio < 0.1 else ''}")


if __name__ == "__main__":
    test_known_xi_param_delta()
    sweep_bf16_threshold()
    print("\nMECHANISM TEST PASSED")
