"""
RiskExtragradientTrainer

Extends ExtragradientTrainer with risk-aware preference aggregation over
multiple y'' samples.  The only structural differences from the base trainer:

1. ypp is always generated (no can_reduce_variance shortcut) and K=ypp_samples
   completions are produced instead of 1.

2. _compute_prefs applies a user-supplied risk functional rho over the K
   per-sample preferences P(y > y''_k) before computing the IPO target.
   The convention is: rho maps PAYOFFS -> LOSS; we negate to get back a
   risk-adjusted PAYOFF.

All other behaviour (extragradient optimizer backup/restore, logging, loss
computation, etc.) is inherited from ExtragradientTrainer unchanged.
"""

import jinja2
import torch
import torch.nn as nn

from typing import Any, Callable, Optional, Union

from transformers import (
    BaseImageProcessor,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
)
from transformers.trainer_utils import EvalPrediction
from transformers.training_args import OptimizerNames
from trl.data_utils import is_conversational, maybe_apply_chat_template
from trl.trainer.utils import (
    SIMPLE_CHAT_TEMPLATE,
    empty_cache,
)
from trl.trainer.judges import BasePairwiseJudge

from trainers.extragradient_trainer import ExtragradientTrainer
from trainers.extragradient_config import ExtragradientConfig
from utils import (
    move_state_dict_to_cpu_backup,
    move_state_dict_to_device,
)

from .config import RiskExtragradientConfig
from .loss import make_neutral
from .group_dro import GroupDROState, GroupStratifiedSampler, anneal_c


class RiskExtragradientTrainer(ExtragradientTrainer):
    """
    Risk-aware variant of ExtragradientTrainer.

    Parameters
    ----------
    ypp_risk_fn : Callable[[Tensor], Tensor], optional
        Risk functional following the PAYOFF -> LOSS convention.
        Signature: (x: Tensor[..., K]) -> Tensor[...]
        Defaults to make_neutral() (risk-neutral mean, identical to base).
    All other parameters are identical to ExtragradientTrainer.
    """

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        ref_model: Union[PreTrainedModel, nn.Module] = None,
        reward_model: Union[PreTrainedModel, nn.Module, None] = None,
        judge: Optional[BasePairwiseJudge] = None,
        args: Optional[RiskExtragradientConfig] = None,
        data_collator: Optional[Callable] = None,
        train_dataset=None,
        eval_dataset=None,
        processing_class: Optional[
            Union[PreTrainedTokenizerBase, BaseImageProcessor,
                  FeatureExtractionMixin, ProcessorMixin]
        ] = None,
        peft_config: Optional[dict] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], dict]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple = (None, None),
        preprocess_logits_for_metrics: Optional[Callable] = None,
        ypp_risk_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__(
            model=model,
            ref_model=ref_model,
            reward_model=reward_model,
            judge=judge,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            peft_config=peft_config,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        self.ypp_risk_fn = ypp_risk_fn if ypp_risk_fn is not None else make_neutral()

        # Risk-over-e (group DRO) state. Inert unless args.use_group_dro=True.
        if getattr(self.args, "use_group_dro", False):
            self.dro_state = GroupDROState(
                num_groups=self.args.num_groups,
                p_nominal=self.args.p_nominal,
                ema_alpha=self.args.ema_alpha,
            )
        else:
            self.dro_state = None

    # ------------------------------------------------------------------
    # Core override: preference computation with K y'' samples + risk fn
    # ------------------------------------------------------------------

    def _compute_prefs(self, y_yp, ypp, input_length):
        """
        Args:
            y_yp : dict, shape (B, 2, seq_len) — (y, y') completions
            ypp  : dict, shape (B, K, seq_len) — K y'' completions, or None
            input_length : int

        Returns:
            prefs : Tensor (B, num_chunks, 2)
                (risk_payoff(y vs y''), risk_payoff(y' vs y''))
                or (0.5, P(y > y')) when ypp is None (fallback pairwise)
        """
        num_chunks = y_yp["chunk_pos"].shape[-1]
        prompts = y_yp["raw"]
        device = y_yp["input_ids"].device

        completions = []
        is_conv = is_conversational({"prompt": prompts[0]})

        for i in range(num_chunks):
            chunk_end_yy = y_yp["chunk_pos"][i]

            y = self.processing_class.batch_decode(
                y_yp["input_ids"][:, 0, input_length:chunk_end_yy],
                skip_special_tokens=True,
            )
            y = [c.strip() for c in y]

            yp = self.processing_class.batch_decode(
                y_yp["input_ids"][:, 1, input_length:chunk_end_yy],
                skip_special_tokens=True,
            )
            yp = [c.strip() for c in yp]

            if ypp is not None:
                K = ypp["input_ids"].shape[1]
                chunk_end_ypp = ypp["chunk_pos"][i]

                # Optionally apply chat template (once per chunk)
                if is_conv:
                    environment = jinja2.Environment()
                    template = environment.from_string(SIMPLE_CHAT_TEMPLATE)
                    rendered_prompts = [template.render(messages=p) for p in prompts]
                    y = [template.render(messages=[{"role": "assistant", "content": c}])
                         for c in y]
                    yp = [template.render(messages=[{"role": "assistant", "content": c}])
                          for c in yp]
                else:
                    rendered_prompts = prompts

                for k in range(K):
                    ypp_k = self.processing_class.batch_decode(
                        ypp["input_ids"][:, k, input_length:chunk_end_ypp],
                        skip_special_tokens=True,
                    )
                    ypp_k = [c.strip() for c in ypp_k]

                    if is_conv:
                        ypp_k = [template.render(
                            messages=[{"role": "assistant", "content": c}]
                        ) for c in ypp_k]

                    # 2B pairs per ypp sample: (y vs ypp_k) + (yp vs ypp_k)
                    completions += list(zip(y + yp, ypp_k + ypp_k))

            else:
                # Fallback: pairwise (y vs y') — same as base class
                if is_conv:
                    environment = jinja2.Environment()
                    template = environment.from_string(SIMPLE_CHAT_TEMPLATE)
                    rendered_prompts = [template.render(messages=p) for p in prompts]
                    y = [template.render(messages=[{"role": "assistant", "content": c}])
                         for c in y]
                    yp = [template.render(messages=[{"role": "assistant", "content": c}])
                          for c in yp]
                else:
                    rendered_prompts = prompts

                completions += list(zip(y, yp))

        # Single batched judge call
        n_prompts = len(prompts)
        judge_prompts = rendered_prompts * (len(completions) // n_prompts)
        raw_prefs = self.judge.judge(judge_prompts, completions)
        prefs = torch.tensor(raw_prefs, dtype=torch.float32, device=device)

        if ypp is not None:
            # prefs layout: [chunk0_k0_y, chunk0_k0_yp, ..., chunk0_kK-1_yp, chunk1_...]
            # each block of 2B: first B = y vs ypp_k, last B = yp vs ypp_k
            # total length: num_chunks * K * 2 * B
            B = n_prompts
            # view: (num_chunks, K, 2, B) -> permute -> (B, num_chunks, 2, K)
            prefs = prefs.view(num_chunks, K, 2, B).permute(3, 0, 2, 1)
            # prefs shape: (B, num_chunks, 2, K)
            # dim -2 == 0: y vs y'', dim -2 == 1: yp vs y''

            # Apply risk functional: PAYOFF -> LOSS, then negate to get PAYOFF
            risk_y  = -self.ypp_risk_fn(prefs[:, :, 0, :])   # (B, num_chunks)
            risk_yp = -self.ypp_risk_fn(prefs[:, :, 1, :])   # (B, num_chunks)
            prefs = torch.stack([risk_y, risk_yp], dim=-1)    # (B, num_chunks, 2)
        else:
            # Fallback pairwise: same as base (0.5, P(y > y'))
            prefs = prefs.view(num_chunks, 1, B).permute(2, 0, 1)
            # (B, num_chunks, 1)
            prefs = torch.cat((prefs, 0.5 * torch.ones_like(prefs)), dim=-1)

        return prefs

    # ------------------------------------------------------------------
    # training_step override — identical to base except ypp generation:
    #   • always generates ypp (no can_reduce_variance shortcut)
    #   • uses self.args.ypp_samples instead of samples_per_prompt
    #   • _process_data(ypp, 1) to preserve K in dim 1
    # ------------------------------------------------------------------

    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        num_items_in_batch: Optional[int] = None,
    ) -> torch.Tensor:
        self.accumulated_steps += 1
        if self.accumulated_steps == 1:
            self.batch_inputs = []
            self.batch_datas = []
        if self.accumulated_steps == self.args.gradient_accumulation_steps:
            self.accumulated_steps = 0
            is_last_minibatch = True
        else:
            is_last_minibatch = False

        batch_size = len(next(iter(inputs.values())))
        prompts = inputs["prompt"]

        # One-time sanity print: confirm the `e` column survives HF's
        # remove_unused_columns filtering before we silently fall through to
        # e=0 for every sample.
        if not getattr(self, "_e_sanity_printed", False):
            self._e_sanity_printed = True
            has_e = "e" in inputs
            print(f"[RiskTrainer] batch keys: {list(inputs.keys())}  "
                  f"has_e={has_e}")
            if has_e:
                print(f"[RiskTrainer] first e values: {list(inputs['e'])[:8]}")

        inputs = [{k: v[i] for k, v in inputs.items()} for i in range(batch_size)]
        # Extract `e` BEFORE maybe_apply_chat_template: TRL's apply_chat_template
        # returns a new dict containing only the templated keys (prompt / chosen /
        # rejected / completion) and silently drops everything else, including
        # our group-label column. Snapshot here and reattach after templating.
        e_list = [int(x["e"]) if "e" in x else 0 for x in inputs]
        inputs = [maybe_apply_chat_template(x, self.processing_class) for x in inputs]
        inputs = [{
            "raw": p,
            "e": e_val,
            **self.tokenize_row(x,
                                self.model.config.is_encoder_decoder,
                                self.processing_class),
        } for p, x, e_val in zip(prompts, inputs, e_list)]
        self.batch_inputs.extend(inputs)

        if not is_last_minibatch:
            return torch.tensor(0.0, device=self.args.device)

        batch_size = self.args.per_device_train_batch_size
        gen_batch_size = max(self.args.per_device_generate_batch_size, batch_size)

        for i in range(0, len(self.batch_inputs), gen_batch_size):
            inputs = self.batch_inputs[i:i + gen_batch_size]
            cur_batch_size = len(inputs)
            # Pop per-sample group labels BEFORE the data collator (which does
            # not know how to collate int scalars).
            e_batch = torch.tensor(
                [inp.pop("e") for inp in inputs],
                dtype=torch.long,
                device=self.args.device,
            )
            inputs = self.data_collator(inputs)

            inputs = self._prepare_inputs(inputs)
            input_length = inputs["prompt_input_ids"].shape[1]
            prompts = {
                "input_ids": inputs["prompt_input_ids"],
                "attention_mask": inputs["prompt_attention_mask"],
                "raw": inputs["raw"],
            }

            # Generate (y, y') with configured mixture / sampling
            y_yp = self._generate_completions(
                model,
                prompts,
                num_samples=2 * self.args.samples_per_prompt,
                mixture_coef=self.args.y_yp_mixture_coef,
                temperature=self.args.y_yp_temperature,
                top_k=self.args.y_yp_top_k,
                top_p=self.args.y_yp_top_p,
                min_p=self.args.y_yp_min_p,
            )

            # Always generate K y'' samples (risk requires ypp)
            ypp = self._generate_completions(
                model,
                prompts,
                num_samples=self.args.ypp_samples,
                mixture_coef=self.args.ypp_mixture_coef,
                temperature=1.0,
                top_k=0,
                top_p=1.0,
                min_p=0.0,
            )

            y_yp = self._process_completions(y_yp, prompts)
            self._process_data(y_yp, self.args.samples_per_prompt)

            ypp = self._process_completions(ypp, prompts)
            # Pass num_samples=1 to preserve all K ypp samples in dim 1
            self._process_data(ypp, 1)

            prefs = self._compute_prefs(y_yp, ypp, input_length)

            for j in range(0, cur_batch_size, batch_size):
                self.batch_datas.append({
                    "y_yp": self.sub_data(y_yp, j, j + batch_size),
                    "ypp": self.sub_data(ypp, j, j + batch_size),
                    "input_length": input_length,
                    "prefs": prefs[j:j + batch_size],
                    "e": e_batch[j:j + batch_size],
                })

        del self.batch_inputs

        model.train()

        if self.args.estimate_extra_grad:
            if self.state.global_step % 2 == 1:
                gpu_optimizer_state_dict = move_state_dict_to_device(
                    self.optimizer_backup,
                    next(model.parameters()).device,
                )
                optimizer_state_dict = [None] * self.args.local_rank + [gpu_optimizer_state_dict]
                self.optimizer.load_state_dict(optimizer_state_dict[self.args.local_rank])

                if self.model_backup is not None:
                    with torch.no_grad():
                        for name, param in model.named_parameters():
                            if param.requires_grad and name in self.model_backup:
                                param.data.copy_(self.model_backup[name].to(param.device))
                                param.grad = None
            else:
                self.model_backup = {
                    name: param.data.clone().detach().cpu()
                    for name, param in model.named_parameters()
                    if param.requires_grad
                }
                optimizer_state_dict = self.optimizer.state_dict()
                cpu_optimizer_state_dict = move_state_dict_to_cpu_backup(optimizer_state_dict)
                self.optimizer_backup = cpu_optimizer_state_dict

        loss_sum = 0
        for minibatch in self.batch_datas:
            y_yp = minibatch["y_yp"]
            ypp = minibatch["ypp"]
            input_length = minibatch["input_length"]
            prefs = minibatch["prefs"]
            e = minibatch["e"]

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
                # Exact base-class behavior (regression sanity: identical loss
                # to vanilla EGPO when use_group_dro=False).
                loss, kl = self._compute_losses(logps, ref_logps, prefs)

            self._log_statistics(
                y_yp=y_yp,
                ypp=ypp,
                logps=logps.detach().permute(0, 2, 1).reshape(-1, 2),
                ref_logps=ref_logps.permute(0, 2, 1).reshape(-1, 2),
                prefs=prefs.reshape(-1, 2),
                kl=kl,
                input_length=input_length,
            )

            kwargs = {}
            if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
                kwargs["learning_rate"] = self._get_learning_rate()

            if self.args.n_gpu > 1:
                loss = loss.mean()

            if self.use_apex:
                from apex import amp
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss, **kwargs)

            loss_sum += loss.detach() / self.args.gradient_accumulation_steps

        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            empty_cache()

        return loss_sum

    # ------------------------------------------------------------------
    # Group-DRO (risk over safety category e) loss aggregation
    # ------------------------------------------------------------------
    def _compute_losses_dro(self, logps, ref_logps, prefs, e):
        """
        Same IPO residual as the base class (squared loss — lower is better),
        but aggregated across the batch via streaming group-DRO:

            per_sample_loss[i] = mean over chunks of residual_i^2
            Z_hat[g]           = mean_{i: e_i==g} per_sample_loss[i]
            Z_ema[g]           = alpha*Z_ema[g] + (1-alpha)*Z_hat[g].detach()
            w[g]          proportional to  p_nominal[g] * exp(+c * Z_ema[g])
            loss               = sum_{g present} w[g] * Z_hat[g]

        Gradient flows through Z_hat (which depends on the current-batch
        logps); w is detached. Groups not present in the batch contribute 0.

        Sign convention: +c because Z is a LOSS (lower-is-better).
        This upweights high-loss groups, i.e. the distributionally-robust
        direction of the writeup's Option A.
        """
        # Per-(sample, chunk) IPO residual. Identical to base `_compute_losses`
        # except we keep the per-sample axis instead of taking .mean().
        residual = (logps[:, 0] - logps[:, 1]
                    - ref_logps[:, 0] + ref_logps[:, 1]
                    - (prefs[:, :, 0] - prefs[:, :, 1]) / self.beta)   # (B, num_chunks)
        per_sample = (residual ** 2).mean(dim=-1)                       # (B,)

        kl = 0.5 * torch.abs(
            ref_logps[:, 0] + ref_logps[:, 1]
            - logps[:, 0].detach() - logps[:, 1].detach()
        ).mean()

        # Current c (with optional linear anneal). Only relevant for
        # entropic group weights; ignored when group_risk_fn == "cvar".
        c = anneal_c(
            target=self.args.c,
            global_step=self.state.global_step,
            max_steps=self.state.max_steps,
            anneal_frac=self.args.beta_anneal_frac,
        )

        group_risk_fn = getattr(self.args, "group_risk_fn", "entropic")
        group_risk_alpha = float(getattr(self.args, "group_risk_alpha", 1.0))

        loss, diag = self.dro_state.compute_loss(
            per_sample_loss=per_sample, e=e, c=c,
            risk_fn=group_risk_fn, risk_alpha=group_risk_alpha,
        )

        # Log per-group diagnostics (idempotent: setdefault keeps ordering).
        G = self.dro_state.num_groups
        for g in range(G):
            self.stats.setdefault(f"dro/w_{g}", []).append(diag["dro/w"][g])
            self.stats.setdefault(f"dro/Z_ema_{g}", []).append(diag["dro/Z_ema"][g])
            self.stats.setdefault(f"dro/count_{g}", []).append(float(diag["dro/counts"][g]))
        self.stats.setdefault("dro/c", []).append(diag["dro/c"])
        self.stats.setdefault("dro/risk_alpha", []).append(diag["dro/risk_alpha"])

        return loss, kl

    # ------------------------------------------------------------------
    # Optional: stratified sampler (min-per-group per batch)
    # ------------------------------------------------------------------
    def _get_train_sampler(self, *args, **kwargs):
        """
        If `args.min_per_group > 0` AND the dataset carries an `e` column, use
        the single-process GroupStratifiedSampler. Otherwise defer to HF's
        default sampler.

        Note: this sampler is not distributed-aware. Under DDP / DeepSpeed with
        world_size > 1, per-rank sharding will degrade the per-batch
        stratification guarantee (but will not break training).
        """
        min_per_group = getattr(self.args, "min_per_group", 0)
        if min_per_group <= 0 or self.train_dataset is None:
            return super()._get_train_sampler(*args, **kwargs)
        if "e" not in self.train_dataset.column_names:
            return super()._get_train_sampler(*args, **kwargs)

        return GroupStratifiedSampler(
            group_labels=self.train_dataset["e"],
            batch_size=self.args.per_device_train_batch_size
                       * max(1, self.args.world_size),
            num_groups=self.args.num_groups,
            min_per_group=min_per_group,
            p_nominal=self.args.p_nominal,
            seed=int(getattr(self.args, "seed", 0)),
            drop_last=False,
        )
