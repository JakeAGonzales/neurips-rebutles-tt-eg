r"""
TT-EG: Two-Timescale Extragradient for risk-aware EGPO.

The risk functional `rho` over K=ypp_samples opponent payoffs is nonlinear, so the
risk-adjusted payoff is a biased (O(1/K)) estimate of the population value; that bias
propagates into the IPO target and floors the extragradient update. TT-EG adds a
fast-timescale tracker `xi` (parameter-shaped over LoRA params) estimating the
gradient-space bias and subtracts it from the applied gradient.

  entropic : delta-method estimator (nonzero, grows with c)
  cvar     : exact zero residual (Rockafellar-Uryasev) -> TT-EG is a no-op (control)

Mapped to this repo's even/odd extragradient loop (paper Algorithm 2):
  even (global_step even, extrapolation): apply g_main - xi_prev
  odd  (global_step odd,  correction):    apply g_main - xi_prev, then
                                           xi <- (1-gamma) xi_prev + gamma b_hat
Both steps of a pair use the same xi (mutated only at the end of the odd step, from
the odd batch); xi_0 = 0. The correction is realized by adding -<xi,theta> to the
loss (grad = -xi), routing through DeepSpeed's normal backward (robust under the
BF16 optimizer where manual .grad edits are unsafe). b_hat comes from an
autograd.grad of the bias surrogate; tteg_bias_point selects the iterate:
  base         : reuse the correction-step activations at theta_t (free).
  extrapolated : a fresh forward at theta_tilde before the rollback (textbook EG).
Neither option triggers an extra generation/judge call. Tracking quality is logged
as tteg/tracking_residual{,_sq} = ||xi - b_hat||{,^2}.

Prereqs (asserted): --alg eg, --lora, ypp_samples >= 2. Orthogonal to group-DRO.

Usage:
    deepspeed --num_gpus=1 risk_egpo/tt_eg.py --alg eg --lora \
        --ypp_samples 8 --risk entropic --risk_c 1.0 --use_tteg --tteg_gamma 0.5
"""

import argparse
import json
import math
import os
import sys
from contextlib import contextmanager
from collections import deque
from dataclasses import dataclass

import deepspeed
import jinja2
import torch

# --- DeepSpeed + LoRA negative-LR patch (same as risk_egpo/train.py) ---------
from torch.optim import lr_scheduler as _lrs


@contextmanager
def _get_lr_ctx(scheduler):
    scheduler._get_lr_called_within_step = True
    try:
        yield
    finally:
        scheduler._get_lr_called_within_step = False


def _safe_update_lr(self, epoch=None):
    step_idx = max(0, int(getattr(self, "_step_count", 1)) - 1)
    self.last_epoch = step_idx
    with _get_lr_ctx(self):
        values = self.get_lr()
    n = len(self.optimizer.param_groups)
    vals = list(values)[:n]
    for pg, lr in zip(self.optimizer.param_groups, vals):
        pg["lr"] = lr
    self._last_lr = [pg["lr"] for pg in self.optimizer.param_groups]


_lrs.LRScheduler._update_lr = _safe_update_lr

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "EGPO", "lms"))
sys.path.insert(0, _ROOT)

from transformers.training_args import OptimizerNames  # noqa: E402
from trl.data_utils import is_conversational, maybe_apply_chat_template  # noqa: E402
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE, empty_cache  # noqa: E402

from utils import (  # noqa: E402
    move_state_dict_to_cpu_backup,
    move_state_dict_to_device,
)
from risk_egpo.config import RiskExtragradientConfig  # noqa: E402
from risk_egpo.trainer import RiskExtragradientTrainer  # noqa: E402


# ===========================================================================
# Bias estimator (payoff space, PAYOFF -> LOSS convention like rho)
# ===========================================================================
def compute_bias_payoff(payoffs, risk_kind, c=None, alpha=None):
    """Delta-method bias of the risk-adjusted payoff, per side.

    payoffs: (..., K) raw judge probabilities P(y > y''_k). Returns (...,) in rho's
    pre-negation LOSS space (`_compute_prefs` negates it with the payoff).

    entropic: rho(x)=h(E[g]), g(P)=exp(-c P), h(q)=(1/c)log q; h''=-1/(c q^2)
        => b = -Var_hat(g) / (2 c K q_hat^2),  Var_hat unbiased (ddof=1).
    neutral/cvar: plain sample mean -> zero residual bias.
    """
    if risk_kind in ("neutral", "cvar"):
        return torch.zeros_like(payoffs[..., 0])
    if risk_kind == "entropic":
        if not c:
            raise ValueError("entropic bias requires nonzero c")
        K = payoffs.shape[-1]
        if K < 2:
            raise ValueError("entropic bias needs K >= 2 for unbiased variance")
        p64 = payoffs.double()
        log_q = torch.logsumexp(-c * p64, dim=-1) - math.log(K)  # stable log q_hat
        var_g = torch.exp(-c * p64).var(dim=-1, unbiased=True)   # ddof=1, REQUIRED
        log_term = torch.log(var_g.clamp_min(1e-300)) - 2.0 * log_q
        b = -torch.exp(log_term) / (2.0 * c * K)
        return b.to(payoffs.dtype)
    raise ValueError(f"unknown risk_kind {risk_kind!r}")


# ===========================================================================
# Config
# ===========================================================================
@dataclass
class TTEGConfig(RiskExtragradientConfig):
    r"""RiskExtragradientConfig + TT-EG fast-timescale tracker options."""

    use_tteg: bool = False
    tteg_gamma: float = 0.1                 # fast-timescale EMA gain (tau_eff ~ 1/gamma)
    tteg_gamma_schedule: str = "const"      # {const, decay}: decay = gamma_0 t^-1/3
    tteg_bias_every: int = 1                # stride for b_hat computation
    tteg_bias_point: str = "base"           # {base, extrapolated}: iterate for grad(b_hat)
    tteg_log_every: int = 10


# ===========================================================================
# Trainer
# ===========================================================================
class TTEGTrainer(RiskExtragradientTrainer):
    """Risk-aware extragradient trainer with a two-timescale bias tracker."""

    def __init__(self, *args, risk_kind="neutral", risk_c=1.0, risk_alpha=0.25,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.risk_kind = risk_kind
        self.risk_c = risk_c
        self.risk_alpha = risk_alpha

        self.use_tteg = bool(getattr(self.args, "use_tteg", False))
        self.tteg_gamma = float(getattr(self.args, "tteg_gamma", 0.1))
        self.tteg_gamma_schedule = getattr(self.args, "tteg_gamma_schedule", "const")
        self.tteg_bias_every = int(getattr(self.args, "tteg_bias_every", 1))
        self.tteg_bias_point = getattr(self.args, "tteg_bias_point", "base")
        self.tteg_log_every = int(getattr(self.args, "tteg_log_every", 10))

        # tracker state (lazily allocated at first step, keyed by runtime names)
        self.xi = None
        self._tteg_names = None
        self._xi_version = 0
        self._xi_fp_cache = 0.0

        if self.use_tteg:
            assert 0.0 < self.tteg_gamma <= 1.0, "tteg_gamma must be in (0, 1]"
            assert self.args.ypp_samples >= 2, "TT-EG needs ypp_samples >= 2"
            assert self.risk_kind in ("neutral", "cvar", "entropic")
            assert self.tteg_bias_point in ("base", "extrapolated")
            # The extrapolated bias point evaluates grad(b_hat) at the EG
            # look-ahead iterate, so it only exists for extragradient. The base
            # bias point (grad at theta_t, reusing the main forward) is
            # algorithm-agnostic and works for single-step algs (oipo1/nmd/...).
            if self.tteg_bias_point == "extrapolated":
                assert self.args.estimate_extra_grad, (
                    "tteg_bias_point=extrapolated requires --alg eg "
                    "(extragradient look-ahead); use base for single-step algs")

    # ------------------------------------------------------------------
    # tracker bookkeeping
    # ------------------------------------------------------------------
    def _lazy_init_tracker(self, model):
        if self.xi is not None:
            return
        self.xi, self._tteg_names = {}, []
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.xi[n] = torch.zeros_like(p, device="cpu", dtype=torch.float32)
                self._tteg_names.append(n)
        self._xi_fp_cache = self._xi_fp()

    def _xi_fp(self):
        return float(sum(t.sum().item() for t in self.xi.values()))

    def _assert_xi_survived(self):
        """Fire immediately if anything (e.g. the EG snapshot/restore) mutated xi
        outside `_update_tracker`. Catches the repo-specific rollback trap."""
        if self.xi is None or self._xi_version == 0:
            return
        fp = self._xi_fp()
        if abs(fp - self._xi_fp_cache) > 1e-9 * max(1.0, abs(self._xi_fp_cache)):
            raise RuntimeError(
                f"[TT-EG] xi modified outside _update_tracker (expected "
                f"{self._xi_fp_cache:.9e}, got {fp:.9e}); the extragradient "
                f"snapshot/restore likely captured xi."
            )

    def _current_gamma(self):
        if self.tteg_gamma_schedule == "decay":
            return self.tteg_gamma * (max(1, self._xi_version + 1) ** (-1.0 / 3.0))
        return self.tteg_gamma

    def _update_tracker(self, b_hat):
        g = self._current_gamma()
        for n in self.xi:
            self.xi[n].mul_(1.0 - g).add_(b_hat[n].to("cpu", torch.float32), alpha=g)
        self._xi_version += 1
        self._xi_fp_cache = self._xi_fp()

    # ------------------------------------------------------------------
    # surrogates
    # ------------------------------------------------------------------
    def _debias_term(self, model):
        """Loss add-on whose grad w.r.t. theta is exactly -xi (added per minibatch;
        DeepSpeed's 1/gas scaling over `gas` minibatches nets -xi)."""
        term = None
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.xi:
                contrib = (self.xi[n].to(p.device, dtype=p.dtype) * p).sum()
                term = contrib if term is None else term + contrib
        return torch.zeros((), device=self.args.device) if term is None else -term

    def _sample_weights(self, e, B, device, dtype):
        """Per-sample weights matching the main objective's reduction. Non-DRO ->
        uniform 1/B (== mean). DRO -> detached w[g]/n_g, matching _compute_losses_dro."""
        if self.dro_state is None or e is None or self.dro_state.last_w is None:
            return torch.full((B,), 1.0 / B, device=device, dtype=dtype)
        w = self.dro_state.last_w.to(device, dtype)
        out = torch.zeros(B, device=device, dtype=dtype)
        for g in range(self.dro_state.num_groups):
            mask = e == g
            n_g = int(mask.sum())
            if n_g > 0:
                out[mask] = w[g] / n_g
        return out

    def _bias_surrogate(self, logps, ref_logps, bias, e):
        """Surrogate S with grad_theta S = b_hat.

        r_biased=s-(pref_y-pref_yp)/beta, r_debiased adds db/beta (db=bias_y-bias_yp).
        b_hat = grad(r_biased^2)-grad(r_debiased^2) = -(2/beta) mean(db*grad s)
             => S = -(2/beta) weighted_mean(db*s)  (bias detached, s carries grad).
        """
        s = logps[:, 0] - logps[:, 1] - ref_logps[:, 0] + ref_logps[:, 1]   # (B,C)
        db = (bias[:, :, 0] - bias[:, :, 1]).detach()                       # (B,C)
        per_sample = (db * s).mean(dim=-1)                                  # (B,)
        w = self._sample_weights(e, per_sample.shape[0],
                                 per_sample.device, per_sample.dtype)
        return -(2.0 / self.beta) * (w.detach() * per_sample).sum()

    def _accumulate_bias_grad_fresh(self, model, minibatch, b_hat, tteg_params):
        """Accumulate grad_theta of the bias surrogate into b_hat using a FRESH
        forward at the model's CURRENT params (theta_tilde for the extrapolated
        option). autograd.grad returns grads without touching .grad, so the
        pending EG restore and the main correction backward are unaffected."""
        y_yp, input_length = minibatch["y_yp"], minibatch["input_length"]
        logps = self._compute_logps(model, y_yp, input_length)
        with torch.no_grad():
            if self.ref_model is None:
                with model.disable_adapter():
                    ref_logps = self._compute_logps(model, y_yp, input_length)
            else:
                ref_logps = self._compute_logps(self.ref_model, y_yp, input_length)
        surrogate = self._bias_surrogate(logps, ref_logps, minibatch["bias"],
                                         minibatch["e"])
        grads = torch.autograd.grad(surrogate, tteg_params, allow_unused=True)
        for n, gg in zip(self._tteg_names, grads):
            if gg is not None:
                b_hat[n].add_(gg.detach())

    # ------------------------------------------------------------------
    # _compute_prefs: same K-sample judge call as the parent, plus per-sample
    # bias computed from the SAME payoffs (free). Returns (prefs, bias_pref).
    # ------------------------------------------------------------------
    def _compute_prefs(self, y_yp, ypp, input_length):
        num_chunks = y_yp["chunk_pos"].shape[-1]
        prompts = y_yp["raw"]
        device = y_yp["input_ids"].device
        completions, rendered_prompts = [], prompts
        is_conv = is_conversational({"prompt": prompts[0]})
        template = None
        if is_conv:
            template = jinja2.Environment().from_string(SIMPLE_CHAT_TEMPLATE)
            rendered_prompts = [template.render(messages=p) for p in prompts]

        def _tmpl(strings):
            if not is_conv:
                return strings
            return [template.render(messages=[{"role": "assistant", "content": c}])
                    for c in strings]

        for i in range(num_chunks):
            end = y_yp["chunk_pos"][i]
            y = _tmpl([c.strip() for c in self.processing_class.batch_decode(
                y_yp["input_ids"][:, 0, input_length:end], skip_special_tokens=True)])
            yp = _tmpl([c.strip() for c in self.processing_class.batch_decode(
                y_yp["input_ids"][:, 1, input_length:end], skip_special_tokens=True)])
            if ypp is not None:
                K = ypp["input_ids"].shape[1]
                end_ypp = ypp["chunk_pos"][i]
                for k in range(K):
                    ypp_k = _tmpl([c.strip() for c in self.processing_class.batch_decode(
                        ypp["input_ids"][:, k, input_length:end_ypp],
                        skip_special_tokens=True)])
                    completions += list(zip(y + yp, ypp_k + ypp_k))
            else:
                completions += list(zip(y, yp))

        n_prompts = len(prompts)
        judge_prompts = rendered_prompts * (len(completions) // n_prompts)
        prefs = torch.tensor(self.judge.judge(judge_prompts, completions),
                             dtype=torch.float32, device=device)

        if ypp is not None:
            raw = prefs.view(num_chunks, K, 2, n_prompts).permute(3, 0, 2, 1)  # (B,C,2,K)
            prefs_out = torch.stack([-self.ypp_risk_fn(raw[:, :, 0, :]),
                                     -self.ypp_risk_fn(raw[:, :, 1, :])], dim=-1)
            bias_out = torch.stack(
                [-compute_bias_payoff(raw[:, :, 0, :], self.risk_kind,
                                      c=self.risk_c, alpha=self.risk_alpha),
                 -compute_bias_payoff(raw[:, :, 1, :], self.risk_kind,
                                      c=self.risk_c, alpha=self.risk_alpha)], dim=-1)
        else:
            prefs_out = prefs.view(num_chunks, 1, n_prompts).permute(2, 0, 1)
            prefs_out = torch.cat((prefs_out, 0.5 * torch.ones_like(prefs_out)), dim=-1)
            bias_out = torch.zeros_like(prefs_out)
        self._last_bias_absmax = float(bias_out.abs().max().item())
        # Empirical MC sanity (free; no extra forwards/judge). rho_plugin is the
        # advisor's plug-in entropic risk (1/c)(logsumexp(-c x)-log K) == ypp_risk_fn,
        # i.e. rho = -prefs_out. bias_delta is the delta-method bias b of that same
        # rho == compute_bias_payoff == -bias_out. The "diff of the MC estimate" the
        # advisor asked for is exactly this correction b relative to rho.
        if ypp is not None:
            self._last_rho_plugin = float((-prefs_out).mean())
            self._last_bias_delta = float((-bias_out).mean())
        return prefs_out, bias_out

    # ------------------------------------------------------------------
    # window-averaged VECTOR diagnostics (advisor spec)
    # ------------------------------------------------------------------
    def _window_bias_stats(self, b_hat):
        r"""Trailing-window diagnostics that average the bias VECTORS first, then
        take norms/cosines/ratios (since ||mean(b_hat)|| != mean||b_hat||).

        For each W in self.tteg_windows compute, elementwise in parameter space,
            b_bar_W = (1/W) sum_{s=t-W+1..t} b_hat_s      (uses <W samples early)
        and log, against the current tracker xi_t (pre-update):
            resid_ratio_W = ||b_bar_W - xi|| / (||b_bar_W|| + eps)   [PRIMARY]
            cos_W         = <xi, b_bar_W> / (||xi|| ||b_bar_W|| + eps)
            scale_W       = ||xi|| / (||b_bar_W|| + eps)
            bbar_W_norm   = ||b_bar_W||          (uncorrected estimated bias)
            resid_bias_W  = ||b_bar_W - xi||     (residual after TT correction)
        Inner products / squared norms are aggregated across tensors (no flatten).
        """
        windows = getattr(self, "tteg_windows", (20, 50, 100))
        if not windows:
            return
        w_max = max(windows)
        if not hasattr(self, "_bhat_buf"):
            self._bhat_buf = deque(maxlen=w_max)
        # snapshot current b_hat_t as CPU fp32 (same space/keys as xi)
        self._bhat_buf.append(
            {n: b_hat[n].detach().to("cpu", torch.float32).clone() for n in self.xi})
        buf = self._bhat_buf
        eps = 1e-8

        # single newest->oldest pass; snapshot running mean at each window size
        cum = {n: torch.zeros_like(self.xi[n]) for n in self.xi}
        want = set(windows)
        means, count = {}, 0
        for snap in reversed(buf):
            count += 1
            for n in cum:
                cum[n].add_(snap[n])
            if count in want:
                means[count] = {n: (cum[n] / count) for n in cum}
        full_mean = {n: (cum[n] / count) for n in cum}  # count == len(buf)

        xi_norm = math.sqrt(sum(float((self.xi[n] * self.xi[n]).sum()) for n in self.xi))

        def _put(k, v):
            self.stats.setdefault(k, []).append(float(v))

        for W in windows:
            bbar = means.get(W, full_mean)  # fall back to all history if < W seen
            bbar_norm = math.sqrt(sum(float((bbar[n] * bbar[n]).sum()) for n in bbar))
            dot = sum(float((self.xi[n] * bbar[n]).sum()) for n in bbar)
            resid = math.sqrt(sum(float(((bbar[n] - self.xi[n]) ** 2).sum()) for n in bbar))
            _put(f"tteg/bbar{W}_norm", bbar_norm)
            _put(f"tteg/resid_bias{W}", resid)
            _put(f"tteg/resid_ratio{W}", resid / (bbar_norm + eps))
            _put(f"tteg/cos{W}", dot / (xi_norm * bbar_norm + eps))
            _put(f"tteg/scale{W}", xi_norm / (bbar_norm + eps))

    # ------------------------------------------------------------------
    # metrics / halts
    # ------------------------------------------------------------------
    def _tteg_log(self, b_hat, grad_norm=None):
        xi_norm = math.sqrt(sum(float((t * t).sum()) for t in self.xi.values()))
        b_norm = math.sqrt(sum(float((g * g).sum()) for g in b_hat.values()))
        resid = math.sqrt(sum(float(((self.xi[n] - b_hat[n].to("cpu", torch.float32)) ** 2).sum())
                              for n in self.xi))
        if any(not torch.isfinite(g).all() for g in b_hat.values()):
            raise RuntimeError("[TT-EG] HALT: b_hat is non-finite")
        # Gate A (arithmetic invariant): neutral rho is linear, so b_hat is
        # exactly zero and xi must never leave zero -> run is bit-identical to
        # non-TT-EG. Stronger than the CVaR (Gate B) control.
        if self.risk_kind == "neutral":
            assert b_norm == 0.0 and xi_norm == 0.0, (
                f"[TT-EG] Gate A violated: neutral must keep xi=0 "
                f"(|b_hat|={b_norm:.3e}, |xi|={xi_norm:.3e})")

        def _put(k, v):
            self.stats.setdefault(k, []).append(float(v))

        _put("tteg/xi_norm", xi_norm)
        _put("tteg/b_hat_norm", b_norm)
        _put("tteg/tracking_residual", resid)
        _put("tteg/tracking_residual_sq", resid * resid)
        _put("tteg/tracking_residual_rel", resid / (b_norm + 1e-12))
        _put("tteg/b_hat_scale_ratio", b_norm * self.args.ypp_samples)
        _put("tteg/b_hat_payoff_absmax", getattr(self, "_last_bias_absmax", 0.0))
        _put("tteg/xi_version", self._xi_version)
        _put("tteg/gamma", self._current_gamma())
        _put("tteg/eta_over_gamma", self._get_learning_rate() / max(self._current_gamma(), 1e-12))
        # empirical MC plug-in sanity (payoff space): plug-in risk rho, its
        # delta-method bias b, and the relative correction |b|/|rho|.
        rho = getattr(self, "_last_rho_plugin", 0.0)
        bdel = getattr(self, "_last_bias_delta", 0.0)
        _put("tteg/rho_plugin", rho)
        _put("tteg/bias_payoff_delta", bdel)
        _put("tteg/bias_rel", abs(bdel) / (abs(rho) + 1e-12))
        # window-averaged vector diagnostics (xi here is pre-update xi_t)
        self._window_bias_stats(b_hat)
        if grad_norm is not None:
            frac = xi_norm / (grad_norm + 1e-12)
            _put("tteg/grad_norm_main", grad_norm)
            _put("tteg/debias_fraction", frac)
            if frac > 1.0:
                raise RuntimeError(f"[TT-EG] HALT: debias_fraction {frac:.3f} > 1 "
                                   f"(sign/scale error)")
        if self.state.global_step % self.tteg_log_every <= 1:
            print(f"[TT-EG] step={self.state.global_step} v={self._xi_version} "
                  f"|xi|={xi_norm:.3e} |b_hat|={b_norm:.3e} resid={resid:.3e} "
                  f"gamma={self._current_gamma():.3f}", flush=True)

    # ------------------------------------------------------------------
    # training_step: parent's K-sample loop + TT-EG debias/tracker.
    # ------------------------------------------------------------------
    def training_step(self, model, inputs, num_items_in_batch=None):
        self._assert_xi_survived()

        self.accumulated_steps += 1
        if self.accumulated_steps == 1:
            self.batch_inputs, self.batch_datas = [], []
        is_last_minibatch = self.accumulated_steps == self.args.gradient_accumulation_steps
        if is_last_minibatch:
            self.accumulated_steps = 0

        batch_size = len(next(iter(inputs.values())))
        prompts = inputs["prompt"]
        inputs = [{k: v[i] for k, v in inputs.items()} for i in range(batch_size)]
        e_list = [int(x["e"]) if "e" in x else 0 for x in inputs]
        inputs = [maybe_apply_chat_template(x, self.processing_class) for x in inputs]
        inputs = [{
            "raw": p, "e": e_val,
            **self.tokenize_row(x, self.model.config.is_encoder_decoder, self.processing_class),
        } for p, x, e_val in zip(prompts, inputs, e_list)]
        self.batch_inputs.extend(inputs)

        if not is_last_minibatch:
            return torch.tensor(0.0, device=self.args.device)

        batch_size = self.args.per_device_train_batch_size
        gen_batch_size = max(self.args.per_device_generate_batch_size, batch_size)
        for i in range(0, len(self.batch_inputs), gen_batch_size):
            inputs = self.batch_inputs[i:i + gen_batch_size]
            cur_batch_size = len(inputs)
            e_batch = torch.tensor([inp.pop("e") for inp in inputs],
                                   dtype=torch.long, device=self.args.device)
            inputs = self.data_collator(inputs)
            inputs = self._prepare_inputs(inputs)
            input_length = inputs["prompt_input_ids"].shape[1]
            prompts = {
                "input_ids": inputs["prompt_input_ids"],
                "attention_mask": inputs["prompt_attention_mask"],
                "raw": inputs["raw"],
            }
            y_yp = self._generate_completions(
                model, prompts, num_samples=2 * self.args.samples_per_prompt,
                mixture_coef=self.args.y_yp_mixture_coef,
                temperature=self.args.y_yp_temperature, top_k=self.args.y_yp_top_k,
                top_p=self.args.y_yp_top_p, min_p=self.args.y_yp_min_p)
            ypp = self._generate_completions(
                model, prompts, num_samples=self.args.ypp_samples,
                mixture_coef=self.args.ypp_mixture_coef,
                temperature=1.0, top_k=0, top_p=1.0, min_p=0.0)

            y_yp = self._process_completions(y_yp, prompts)
            self._process_data(y_yp, self.args.samples_per_prompt)
            ypp = self._process_completions(ypp, prompts)
            self._process_data(ypp, 1)

            prefs, bias = self._compute_prefs(y_yp, ypp, input_length)
            for j in range(0, cur_batch_size, batch_size):
                self.batch_datas.append({
                    "y_yp": self.sub_data(y_yp, j, j + batch_size),
                    "ypp": self.sub_data(ypp, j, j + batch_size),
                    "input_length": input_length,
                    "prefs": prefs[j:j + batch_size],
                    "bias": bias[j:j + batch_size],
                    "e": e_batch[j:j + batch_size],
                })
        del self.batch_inputs

        model.train()
        if self.use_tteg:
            self._lazy_init_tracker(model)

        # TT-EG bias bookkeeping. Set up BEFORE the snapshot/restore so the
        # extrapolated option can evaluate grad(b_hat) while the model still
        # holds the extrapolated iterate theta_tilde.
        # Cadence. Extragradient runs extrapolate(even)+update(odd) pairs, so the
        # bias/tracker fire once per pair on the odd (correction) step, indexed by
        # global_step // 2. Single-step algs (oipo1/nmd/...) update every step, so
        # every step is a tracker step, indexed by global_step directly.
        if self.args.estimate_extra_grad:
            is_correction = self.state.global_step % 2 == 1
            update_idx = self.state.global_step // 2
        else:
            is_correction = True
            update_idx = self.state.global_step
        do_bias = (self.use_tteg and is_correction
                   and update_idx % self.tteg_bias_every == 0)
        b_hat = ({n: torch.zeros_like(self.xi[n], device=self.args.device)
                  for n in self._tteg_names} if do_bias else None)
        if do_bias:
            name2p = {n: p for n, p in model.named_parameters() if p.requires_grad}
            tteg_params = [name2p[n] for n in self._tteg_names]

        # (extrapolated) grad(b_hat) at theta_tilde, before the rollback. Matches
        # textbook EG (oracle at the extrapolated iterate); needs a fresh forward.
        if do_bias and self.tteg_bias_point == "extrapolated":
            for minibatch in self.batch_datas:
                self._accumulate_bias_grad_fresh(model, minibatch, b_hat, tteg_params)

        # Extragradient snapshot/restore (identical to parent; xi is NOT touched).
        if self.args.estimate_extra_grad:
            if self.state.global_step % 2 == 1:
                gpu_opt = move_state_dict_to_device(self.optimizer_backup,
                                                    next(model.parameters()).device)
                opt_sd = [None] * self.args.local_rank + [gpu_opt]
                # DeepSpeed's BF16_Optimizer.load_state_dict indexes its arg by
                # dp_rank internally (state_dict_list[dp_rank]); pass the LIST,
                # not opt_sd[local_rank], else it does dict[0] -> KeyError: 0.
                self.optimizer.load_state_dict(opt_sd)
                if self.model_backup is not None:
                    with torch.no_grad():
                        for name, param in model.named_parameters():
                            if param.requires_grad and name in self.model_backup:
                                param.data.copy_(self.model_backup[name].to(param.device))
                                param.grad = None
            else:
                self.model_backup = {name: param.data.clone().detach().cpu()
                                     for name, param in model.named_parameters()
                                     if param.requires_grad}
                self.optimizer_backup = move_state_dict_to_cpu_backup(
                    self.optimizer.state_dict())

        n_mb = len(self.batch_datas)
        loss_sum = 0
        for minibatch in self.batch_datas:
            y_yp, ypp = minibatch["y_yp"], minibatch["ypp"]
            input_length = minibatch["input_length"]
            prefs, e = minibatch["prefs"], minibatch["e"]

            logps = self._compute_logps(model, y_yp, input_length)
            with torch.no_grad():
                if self.ref_model is None:
                    with model.disable_adapter():
                        ref_logps = self._compute_logps(model, y_yp, input_length)
                else:
                    ref_logps = self._compute_logps(self.ref_model, y_yp, input_length)

            if self.dro_state is not None:
                loss, kl = self._compute_losses_dro(logps, ref_logps, prefs, e)
            else:
                loss, kl = self._compute_losses(logps, ref_logps, prefs)

            # (base) grad(b_hat) at theta_t, reusing the SAME activations as the
            # main loss (free). The extrapolated option accumulated b_hat above.
            if do_bias and self.tteg_bias_point == "base":
                surrogate = self._bias_surrogate(logps, ref_logps, minibatch["bias"], e)
                grads = torch.autograd.grad(surrogate, tteg_params,
                                            retain_graph=True, allow_unused=True)
                for n, gg in zip(self._tteg_names, grads):
                    if gg is not None:
                        b_hat[n].add_(gg.detach())

            self._log_statistics(
                y_yp=y_yp, ypp=ypp,
                logps=logps.detach().permute(0, 2, 1).reshape(-1, 2),
                ref_logps=ref_logps.permute(0, 2, 1).reshape(-1, 2),
                prefs=prefs.reshape(-1, 2), kl=kl, input_length=input_length)

            # Debias: -<xi, theta> per minibatch -> nets -xi after DeepSpeed 1/gas.
            if self.use_tteg:
                loss = loss + self._debias_term(model)

            kwargs = {}
            if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
                kwargs["learning_rate"] = self._get_learning_rate()
            if self.args.n_gpu > 1:
                loss = loss.mean()
            self.accelerator.backward(loss, **kwargs)
            loss_sum += loss.detach() / self.args.gradient_accumulation_steps

        if do_bias:
            for n in b_hat:
                b_hat[n] /= n_mb
            grad_norm = None
            try:  # best-effort; .grad may be unavailable under some DS configs
                sq = sum(float((p.grad ** 2).sum()) for _, p in model.named_parameters()
                         if p.requires_grad and p.grad is not None)
                grad_norm = math.sqrt(sq) if sq > 0 else None
            except Exception:
                grad_norm = None
            self._tteg_log(b_hat, grad_norm)
            self._update_tracker(b_hat)

        if (self.args.torch_empty_cache_steps is not None
                and self.state.global_step % self.args.torch_empty_cache_steps == 0):
            empty_cache()
        return loss_sum


# ===========================================================================
# Training entrypoint (mirrors risk_egpo/train.py; TT-EG defaults to --alg eg)
# ===========================================================================
def main():
    from datasets import load_dataset
    from peft import get_peft_model

    from configs import lora_config, get_ds_config
    from judges.pair_judge import PairJudge
    from utils import (EXP_ROOT, DATA_ROOT, AverageLossCallback,
                       load_model_tokenizer, format_names, transform_into_chat,
                       get_latest_checkpoint)
    from risk_egpo.loss import make_neutral, make_cvar, make_entropic
    from risk_egpo.group_dro import (add_group_labels, group_histogram,
                                     format_histogram, DEFAULT_GROUP_NAMES)

    ALG_MAPPING = {"oipo1": "OnlineIPO1", "oipo2": "OnlineIPO2",
                   "nmd": "NashMD", "nmdpg": "NashMDPG", "eg": "Extragradient"}

    ap = argparse.ArgumentParser()
    ap.add_argument("--alg", choices=list(ALG_MAPPING.keys()), default="eg")
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=5e-7)
    ap.add_argument("--y_yp_mixture_coef", type=float, default=0)
    ap.add_argument("--y_yp_temperature", type=float, default=2.0)
    ap.add_argument("--y_yp_top_k", type=int, default=10)
    ap.add_argument("--y_yp_min_p", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--local_rank", type=int, default=0)
    ap.add_argument("--load_dir", type=str, default=None)
    # risk over y''
    ap.add_argument("--ypp_samples", type=int, default=8)
    ap.add_argument("--risk", choices=["neutral", "cvar", "entropic"], default="entropic")
    ap.add_argument("--risk_alpha", type=float, default=0.25)
    ap.add_argument("--risk_c", type=float, default=1.0)
    # group-DRO (Axis 2, optional; validate Axis 1 alone first)
    ap.add_argument("--use_group_dro", action="store_true")
    ap.add_argument("--group_risk_fn", choices=["entropic", "cvar"], default="entropic")
    ap.add_argument("--group_risk_alpha", type=float, default=1.0)
    ap.add_argument("--c", "--risk_beta", dest="c", type=float, default=0.0)
    ap.add_argument("--ema_alpha", type=float, default=0.9)
    ap.add_argument("--beta_anneal_frac", type=float, default=0.0)
    ap.add_argument("--min_per_group", type=int, default=0)
    ap.add_argument("--num_groups", type=int, default=4)
    ap.add_argument("--p_nominal_override", type=str, default=None)
    # TT-EG
    ap.add_argument("--use_tteg", action="store_true")
    ap.add_argument("--tteg_gamma", type=float, default=0.1)
    ap.add_argument("--tteg_gamma_schedule", choices=["const", "decay"], default="const")
    ap.add_argument("--tteg_bias_every", type=int, default=1)
    ap.add_argument("--tteg_bias_point", choices=["base", "extrapolated"], default="base")
    ap.add_argument("--tteg_log_every", type=int, default=10)
    # smoke / verification
    ap.add_argument("--max_steps", type=int, default=-1,
                    help="If >0, cap optimizer steps (overrides epochs). For smoke runs.")
    ap.add_argument("--tteg_verify", action="store_true",
                    help="Print/assert tracker invariants at train end (Gate A, xi-survival).")
    args = ap.parse_args()

    if args.use_tteg:
        assert args.lora, "TT-EG requires --lora (tracker is over LoRA params)"
        assert args.ypp_samples >= 2, "TT-EG requires ypp_samples >= 2"
        # EG supports both bias points; single-step algs (oipo1/nmd/...) only the
        # base bias point (no extragradient look-ahead iterate to evaluate at).
        if args.alg != "eg":
            assert args.tteg_bias_point == "base", (
                f"TT with --alg {args.alg} requires --tteg_bias_point base "
                "(extrapolated needs extragradient)")

    deepspeed.init_distributed()
    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)

    exp_root = os.path.join(EXP_ROOT, "risk_egpo")
    os.environ["WANDB_PROJECT"] = "risk_egpo"
    alg = args.alg.lower()
    num_gpus = torch.cuda.device_count()

    batch_size = 64
    micro_batch_size = 8 if args.lora else 1
    gen_micro_batch_size = min(2 * micro_batch_size, batch_size // num_gpus)
    gradient_accumulation_steps = batch_size // (micro_batch_size * num_gpus)
    estimate_extra_grad = (alg == "eg")
    effective_factor = 2 if alg == "eg" else 1

    additional_config = {
        "y_yp_mixture_coef": args.y_yp_mixture_coef,
        "y_yp_temperature": args.y_yp_temperature,
        "y_yp_top_k": args.y_yp_top_k,
        "y_yp_min_p": args.y_yp_min_p,
    }
    if alg == "oipo2":
        additional_config = {"y_yp_mixture_coef": 0}
    elif alg == "nmd":
        additional_config["ypp_mixture_coef"] = 0.125
    elif alg == "nmdpg":
        additional_config["mixture_coef"] = 0.125

    if args.risk == "neutral":
        ypp_risk_fn = make_neutral()
    elif args.risk == "cvar":
        ypp_risk_fn = make_cvar(args.risk_alpha)
    else:
        ypp_risk_fn = make_entropic(args.risk_c)

    model_name = "vectorzhou/gemma-2-2b-it-alpaca-cleaned-SFT"
    dataset_name = "PKU-Alignment/PKU-SafeRLHF"

    config = {
        "num_train_epochs": args.epochs * effective_factor,
        "per_device_train_batch_size": micro_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "per_device_generate_batch_size": gen_micro_batch_size,
        "samples_per_prompt": 1,
        "learning_rate": args.lr,
        "weight_decay": 0.01,
        "bf16": True,
        "warmup_steps": 1000,
        "beta": 0.1,
        "estimate_extra_grad": estimate_extra_grad,
        "prefix_chunk_num": 1,
        "seed": args.seed,
        "ypp_samples": args.ypp_samples,
        "use_group_dro": args.use_group_dro,
        "group_risk_fn": args.group_risk_fn,
        "group_risk_alpha": args.group_risk_alpha,
        "c": args.c,
        "ema_alpha": args.ema_alpha,
        "beta_anneal_frac": args.beta_anneal_frac,
        "min_per_group": args.min_per_group,
        "num_groups": args.num_groups,
        "use_tteg": args.use_tteg,
        "tteg_gamma": args.tteg_gamma,
        "tteg_gamma_schedule": args.tteg_gamma_schedule,
        "tteg_bias_every": args.tteg_bias_every,
        "tteg_bias_point": args.tteg_bias_point,
        "tteg_log_every": args.tteg_log_every,
        "max_steps": args.max_steps,
        **additional_config,
    }

    ds_config = get_ds_config(batch_size, micro_batch_size,
                              gradient_accumulation_steps, args.lr)
    ds_config.pop("optimizer", None)
    ds_config.pop("scheduler", None)

    if local_rank == 0:
        risk_tag = args.risk
        if args.risk == "cvar":
            risk_tag = f"cvar{args.risk_alpha}"
        elif args.risk == "entropic":
            risk_tag = f"entropic{args.risk_c}"
        if args.use_tteg:
            risk_tag += f"-TTEG{args.tteg_gamma}"
            if args.tteg_bias_point == "extrapolated":
                risk_tag += "-xt"
        exp_base_name, run_name = format_names(
            ALG=f"Risk-{ALG_MAPPING[alg]}-K{args.ypp_samples}-{risk_tag}",
            model_name=model_name, dataset_name=dataset_name,
            lora=args.lora, config=config)
        msg_holder = [exp_base_name, run_name]
    else:
        msg_holder = [None, None]
    deepspeed.comm.barrier()
    torch.distributed.broadcast_object_list(msg_holder, src=0)
    exp_base_name, run_name = msg_holder

    judge = PairJudge("vectorzhou/gemma-2-2b-it-preference_dataset_mixture2_and_safe_pku-Preference")
    model, tokenizer = load_model_tokenizer(model_name)
    if args.lora:
        model = get_peft_model(model, lora_config)

    dataset = load_dataset(dataset_name, cache_dir=DATA_ROOT)
    train_raw = add_group_labels(dataset["train"], key="e")
    train_dataset = transform_into_chat(train_raw)

    counts, p_nominal = group_histogram(train_dataset["e"], num_groups=args.num_groups)
    if local_rank == 0:
        print(format_histogram(counts, p_nominal, names=DEFAULT_GROUP_NAMES))
    if args.p_nominal_override is not None:
        ovr = [float(x) for x in args.p_nominal_override.split(",")]
        assert len(ovr) == args.num_groups and sum(ovr) > 0 and all(v >= 0 for v in ovr)
        s = sum(ovr)
        config["p_nominal"] = [v / s for v in ovr]
    else:
        config["p_nominal"] = p_nominal

    if args.load_dir:
        args.load_dir = args.load_dir.rstrip("/")
        run_name = args.load_dir.split("/")[-1]
    output_dir = os.path.join(exp_root, run_name)

    training_args = TTEGConfig(
        run_name=run_name, output_dir=output_dir, **config,
        save_steps=0.1, save_total_limit=None, logging_steps=10,
        logging_dir="./logs", report_to="none", deepspeed=json.dumps(ds_config))

    resume_checkpoint = get_latest_checkpoint(output_dir)
    if local_rank == 0:
        print(f"Resuming from {resume_checkpoint}")
        print(f"TT-EG={args.use_tteg} gamma={args.tteg_gamma} risk={args.risk} "
              f"K={args.ypp_samples}")

    trainer = TTEGTrainer(
        model=model, judge=judge, args=training_args, processing_class=tokenizer,
        train_dataset=train_dataset, ypp_risk_fn=ypp_risk_fn,
        risk_kind=args.risk, risk_c=args.risk_c, risk_alpha=args.risk_alpha,
        callbacks=[AverageLossCallback(gradient_accumulation_steps)])
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    if args.tteg_verify and local_rank == 0:
        xi_fp = trainer._xi_fp() if trainer.xi is not None else 0.0
        expected_updates = trainer.state.global_step // 2  # odd steps, bias_every=1
        print(f"[TT-EG verify] global_step={trainer.state.global_step} "
              f"xi_version={trainer._xi_version} (expected ~{expected_updates}) "
              f"xi_fp={xi_fp:.6e}")
        if args.risk == "neutral":
            assert trainer.xi is None or all(
                float(t.abs().sum()) == 0.0 for t in trainer.xi.values()), \
                "[TT-EG verify] Gate A FAIL: neutral run left nonzero xi"
            print("[TT-EG verify] Gate A PASS: xi stayed exactly zero (neutral)")
        else:
            assert trainer._xi_version > 0 and xi_fp != 0.0, \
                "[TT-EG verify] tracker never moved; b_hat may be identically zero"
            print("[TT-EG verify] xi-survival PASS: tracker advanced and is nonzero")


if __name__ == "__main__":
    main()
