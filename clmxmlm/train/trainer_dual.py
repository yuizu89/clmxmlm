from __future__ import annotations

import math
import time
import inspect
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import Trainer

from ..masking import get_backbone_from_causallm, MaskController
from ..modeling import get_lm_head_module, special_ids_from_tokenizer
from ..data.mlm import build_mlm_batch


def _unwrap_model(model):
    return getattr(model, "module", model)


def local_grad_sumsq_and_numel(model) -> Tuple[float, int]:
    m = _unwrap_model(model)
    sumsq = 0.0
    numel = 0
    for p in m.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach().float()
        sumsq += float((g * g).sum().item())
        numel += int(g.numel())
    return sumsq, numel


def local_grad_norm(model) -> float:
    sumsq, _ = local_grad_sumsq_and_numel(model)
    return math.sqrt(max(sumsq, 1e-30))


def local_grad_rms(model) -> float:
    sumsq, n = local_grad_sumsq_and_numel(model)
    if n <= 0:
        return float("nan")
    return math.sqrt(max(sumsq / float(n), 1e-30))


def local_weight_norm(model) -> float:
    m = _unwrap_model(model)
    sq = 0.0
    for p in m.parameters():
        v = p.detach().float()
        sq += float((v * v).sum().item())
    return math.sqrt(max(sq, 1e-30))


class DualCLMMLMTrainer(Trainer):
    """
    CLM+MLM dual-mode Trainer (FlashAttention2-friendly):

    - Always call BACKBONE directly (e.g., Qwen3Model) and apply lm_head ourselves.
      This ensures `is_causal` really reaches the attention kernel, as in your successful test.

    - For CLM:
        controller.set(True) + backbone_forward(is_causal=True)
        CE on shifted tokens (next-token prediction)

    - For MLM (bidir-ish):
        controller.set(False) + backbone_forward(is_causal=False)
        CE only on masked positions (labels != -100)

    - attention_mask kept 2D (B,S) to keep FA2 varlen path happy.
    """

    def __init__(
        self,
        *args,
        mask_policy: str = "both",
        mlm_mask_ratio: float = 0.15,
        both_weights: str = "1.0,1.0",
        mask_token_id: Optional[int] = None,
        grad_sim: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        alias = {"causal": "clm", "bidir": "mlm"}
        self.mask_policy = alias.get(mask_policy, mask_policy)
        self.mlm_mask_ratio = float(mlm_mask_ratio)

        w = [float(x.strip()) for x in both_weights.split(",")]
        if len(w) != 2:
            raise ValueError("--both_weights must be 'w_clm,w_mlm'")
        self.w_clm, self.w_mlm = w

        need_mlm = self.mask_policy in ("mlm", "alternate", "both")
        self.mask_token_id = int(mask_token_id) if (need_mlm and mask_token_id is not None) else None
        if need_mlm and self.mask_token_id is None:
            raise RuntimeError("MLM requires mask_token_id")

        # backbone / head / controller
        self.backbone = get_backbone_from_causallm(self.model)
        self.lm_head = get_lm_head_module(self.model)
        self.controller = MaskController(self.backbone)

        # RNG for MLM masking on device
        dev = next(self.model.parameters()).device
        self._rng = torch.Generator(device=dev)
        base_seed = int(getattr(self.args, "seed", 42))
        local_rank = int(getattr(self.args, "local_rank", 0) or 0)
        self._rng.manual_seed(base_seed + local_rank)

        # special ids for masking
        tok = getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)
        self._special_ids = special_ids_from_tokenizer(tok) if tok is not None else []

        # perf stats
        self._n_params = int(sum(p.numel() for p in self.model.parameters()))
        self._reset_interval()
        self._step_start_time: Optional[float] = None
        self._micro_in_step: int = 0
        self._pending_grad_stats: Optional[Dict[str, float]] = None

        # ---- grad_sim ----
        self.grad_sim_enabled = bool(grad_sim)
        self._can_grad_sim = (self.mask_policy == "both")  # both のときだけ意味がある想定
        self._pending_grad_sim: Optional[Dict[str, float]] = None
        self._last_mlm_xy: Optional[Tuple[torch.Tensor, torch.Tensor]] = None  # (x_mlm, y_mlm)

    def _reset_interval(self):
        self._int_micro = 0
        self._int_loss = 0.0
        self._int_loss_clm = 0.0
        self._int_loss_mlm = 0.0
        self._int_loss_clm_n = 0
        self._int_loss_mlm_n = 0
        self._int_active_tokens = 0
        self._int_tokens_base = 0
        self._int_step_time = 0.0

    def _mode_now(self) -> str:
        if self.mask_policy in ("clm", "mlm", "both"):
            return self.mask_policy
        # alternate
        return "mlm" if (self.state.global_step % 2 == 1) else "clm"

    def _clm_loss(self, logits: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        logits_s = logits[:, :-1, :].contiguous()
        labels_s = input_ids[:, 1:].contiguous()
        am_s = attention_mask[:, 1:].contiguous().to(torch.bool)
        V = logits_s.size(-1)
        if am_s.any():
            return F.cross_entropy(logits_s[am_s].view(-1, V), labels_s[am_s].view(-1), reduction="mean")
        return logits.sum() * 0.0

    def _mlm_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        mask = labels != -100
        if not mask.any():
            return logits.sum() * 0.0
        V = logits.size(-1)
        return F.cross_entropy(logits[mask].view(-1, V), labels[mask].view(-1), reduction="mean")

    def _backbone_logits(self, input_ids, attention_mask, is_causal: bool) -> torch.Tensor:
        with self.controller.set(is_causal):
            out = self.controller.backbone_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                is_causal=is_causal,
                use_cache=False,
                return_dict=True,
                output_hidden_states=False,
            )
        h = out.last_hidden_state
        return self.lm_head(h)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        mode = self._mode_now()
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))

        B, S = input_ids.shape
        passes = 2 if mode == "both" else 1

        loss_clm = None
        loss_mlm = None
        active = 0

        if mode in ("clm", "both"):
            logits = self._backbone_logits(input_ids, attention_mask, is_causal=True)
            loss_clm = self._clm_loss(logits, input_ids, attention_mask)
            active += int(attention_mask[:, 1:].sum().item())

        if mode in ("mlm", "both"):
            x_mlm, y_mlm, nmask = build_mlm_batch(
                input_ids=input_ids,
                mask_token_id=int(self.mask_token_id),
                mask_ratio=float(self.mlm_mask_ratio),
                special_ids=self._special_ids,
                rng=self._rng,
            )

            # ---- grad_sim 用に「このステップで使った MLM マスク」を保存 ----
            if self.grad_sim_enabled and self._can_grad_sim:
                self._last_mlm_xy = (x_mlm.detach(), y_mlm.detach())

            logits = self._backbone_logits(x_mlm, attention_mask, is_causal=False)
            loss_mlm = self._mlm_loss(logits, y_mlm)
            active += int(nmask)

        if mode == "clm":
            loss_total = loss_clm
        elif mode == "mlm":
            loss_total = loss_mlm
        else:
            loss_total = self.w_clm * loss_clm + self.w_mlm * loss_mlm

        # interval stats
        self._int_micro += 1
        self._int_loss += float(loss_total.detach().float().cpu())

        if loss_clm is not None:
            self._int_loss_clm += float(loss_clm.detach().float().cpu())
            self._int_loss_clm_n += 1
        if loss_mlm is not None:
            self._int_loss_mlm += float(loss_mlm.detach().float().cpu())
            self._int_loss_mlm_n += 1

        self._int_active_tokens += int(active)
        self._int_tokens_base += int(B * S * passes)

        return (loss_total, None) if return_outputs else loss_total

    def _compute_grad_sim_all_params(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        grad_sim: cosine similarity between grad(loss_clm) and grad(loss_mlm).
        - 追加の forward + autograd.grad を行う（=重い）
        - .grad を汚さない（autograd.gradなので optimizer 勾配には影響しない）
        """
        device = next(self.model.parameters()).device
        t0 = time.time()
        out: Dict[str, float] = {"grad_sim": float("nan"), "grad_sim_oom": 0.0}

        params = [p for p in self.model.parameters() if p.requires_grad]

        try:
            with self.compute_loss_context_manager():
                input_ids = inputs["input_ids"]
                attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))

                # ---- CLM ----
                logits_c = self._backbone_logits(input_ids, attention_mask, is_causal=True)
                loss_clm = self._clm_loss(logits_c, input_ids, attention_mask)

                # ---- MLM (保存しておいたマスクがあればそれを使う) ----
                if self._last_mlm_xy is not None:
                    x_mlm, y_mlm = self._last_mlm_xy
                else:
                    x_mlm, y_mlm, _ = build_mlm_batch(
                        input_ids=input_ids,
                        mask_token_id=int(self.mask_token_id),
                        mask_ratio=float(self.mlm_mask_ratio),
                        special_ids=self._special_ids,
                        rng=self._rng,
                    )

                logits_m = self._backbone_logits(x_mlm, attention_mask, is_causal=False)
                loss_mlm = self._mlm_loss(logits_m, y_mlm)

            grads_c = torch.autograd.grad(loss_clm, params, allow_unused=True)
            grads_m = torch.autograd.grad(loss_mlm, params, allow_unused=True)

            dot = torch.zeros((), device=device, dtype=torch.float64)
            na  = torch.zeros((), device=device, dtype=torch.float64)
            nb  = torch.zeros((), device=device, dtype=torch.float64)
            den = torch.zeros((), device=device, dtype=torch.float64)

            for gc, gm in zip(grads_c, grads_m):
                if gc is None or gm is None:
                    continue
                gc = gc.detach().float()
                gm = gm.detach().float()
                dot = dot + (gc * gm).sum(dtype=torch.float64)
                na  = na  + (gc * gc).sum(dtype=torch.float64)
                nb  = nb  + (gm * gm).sum(dtype=torch.float64)
                den = den + float(gc.numel())

            dotv = float(dot.item())
            nav  = float(na.item())
            nbv  = float(nb.item())
            denv = float(den.item())

            cos = dotv / (math.sqrt(max(nav, 1e-30)) * math.sqrt(max(nbv, 1e-30)) + 1e-30)

            out["grad_sim"] = float(cos)
            out["grad_norm_clm"] = math.sqrt(max(nav, 1e-30))
            out["grad_norm_mlm"] = math.sqrt(max(nbv, 1e-30))

            if denv > 0.0:
                out["grad_norm_clm_rms"] = math.sqrt(max(nav / denv, 1e-30))
                out["grad_norm_mlm_rms"] = math.sqrt(max(nbv / denv, 1e-30))

        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg or "cuda oom" in msg:
                out["grad_sim_oom"] = 1.0
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            else:
                raise

        out["grad_sim_ms"] = (time.time() - t0) * 1000.0
        return out


    def _compute_lr(self) -> float:
        try:
            return float(self._get_learning_rate())
        except Exception:
            opt = getattr(self, "optimizer", None)
            if opt is None or len(opt.param_groups) == 0:
                return float("nan")
            return float(opt.param_groups[0].get("lr", float("nan")))

    def _is_end_of_step(self) -> bool:
        if getattr(self, "accelerator", None) is not None:
            return bool(self.accelerator.sync_gradients)
        gas = int(getattr(self.args, "gradient_accumulation_steps", 1))
        return (self._micro_in_step % max(1, gas) == 0)

    def training_step(self, model, inputs, num_items_in_batch=None):
        self._micro_in_step += 1
        if self._step_start_time is None:
            self._step_start_time = time.time()

        # call parent training_step (handles amp/deepspeed)
        try:
            loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
        except TypeError:
            loss = super().training_step(model, inputs)

        if self._is_end_of_step():
            next_step = int(self.state.global_step) + 1
            is_log_step = (self.args.logging_steps > 0) and (next_step % int(self.args.logging_steps) == 0)

            if is_log_step:
                gnorm = float(local_grad_norm(self.model))
                grms = float(local_grad_rms(self.model))
                wnorm = float(self._weight_norm())
                lr = float(self._compute_lr())
                upd2w = lr * float(gnorm) / max(wnorm, 1e-30)
                self._pending_grad_stats = {
                    "grad_norm": float(gnorm),
                    "grad_norm_rms": float(grms),
                    "update_to_weight": float(upd2w),
                }
                # ---- grad_sim（重いので log step のみ）----
                if self.grad_sim_enabled and self._can_grad_sim:
                    self._pending_grad_sim = self._compute_grad_sim_all_params(inputs)

            step_dt = max(1e-9, time.time() - self._step_start_time)
            self._int_step_time += float(step_dt)
            self._step_start_time = None
            self._micro_in_step = 0

        return loss

    def _weight_norm(self) -> float:
        m = _unwrap_model(self.model)
        sq = 0.0
        for p in m.parameters():
            v = p.detach().float()
            sq += float((v * v).sum().item())
        return math.sqrt(max(sq, 1e-30))

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None):
        # keep eval logs as-is
        if any(k.startswith("eval_") for k in logs.keys()):
            try:
                return super().log(logs, start_time=start_time)
            except TypeError:
                return super().log(logs)

        dt = max(1e-9, float(self._int_step_time))
        micro = max(1, int(self._int_micro))

        loss_avg = self._int_loss / micro
        loss_clm = (self._int_loss_clm / max(1, self._int_loss_clm_n)) if self._int_loss_clm_n > 0 else float("nan")
        loss_mlm = (self._int_loss_mlm / max(1, self._int_loss_mlm_n)) if self._int_loss_mlm_n > 0 else float("nan")

        tps_active = float(self._int_active_tokens) / dt
        flops_est = 6.0 * float(self._n_params) * float(self._int_tokens_base)
        tflops = (flops_est / dt) / 1e12

        step = int(self.state.global_step)
        epoch = float(self.state.epoch) if self.state.epoch is not None else float("nan")
        lr = float(self._compute_lr())
        lr_disp = float(f"{lr:.8f}") if lr == lr else lr

        out: Dict[str, float] = {
            "step": step,
            "epoch": round(epoch, 4) if epoch == epoch else epoch,
            "lr": lr_disp,
            "loss": round(float(loss_avg), 4),
            "loss_clm": round(float(loss_clm), 4) if loss_clm == loss_clm else loss_clm,
            "loss_mlm": round(float(loss_mlm), 4) if loss_mlm == loss_mlm else loss_mlm,
            "tps_active": round(float(tps_active), 1),
            "tflops": round(float(tflops), 2),
        }

        if self._pending_grad_stats is not None:
            out["grad_norm"] = round(float(self._pending_grad_stats["grad_norm"]), 4)
            out["grad_norm_rms"] = round(float(self._pending_grad_stats["grad_norm_rms"]), 6)
            out["update_to_weight"] = round(float(self._pending_grad_stats["update_to_weight"]), 6)
            self._pending_grad_stats = None

        if self._pending_grad_sim is not None:
            gs = self._pending_grad_sim
            if "grad_sim" in gs and gs["grad_sim"] == gs["grad_sim"]:
                out["grad_sim"] = round(float(gs["grad_sim"]), 6)
            else:
                out["grad_sim"] = gs.get("grad_sim", float("nan"))

            if "grad_norm_clm" in gs:
                out["grad_norm_clm"] = round(float(gs["grad_norm_clm"]), 4)
            if "grad_norm_mlm" in gs:
                out["grad_norm_mlm"] = round(float(gs["grad_norm_mlm"]), 4)
            if "grad_norm_clm_rms" in gs:
                out["grad_norm_clm_rms"] = round(float(gs["grad_norm_clm_rms"]), 6)
            if "grad_norm_mlm_rms" in gs:
                out["grad_norm_mlm_rms"] = round(float(gs["grad_norm_mlm_rms"]), 6)

            out["grad_sim_ms"] = round(float(gs.get("grad_sim_ms", 0.0)), 2)
            out["grad_sim_oom"] = float(gs.get("grad_sim_oom", 0.0))
            self._pending_grad_sim = None


        self._reset_interval()

        try:
            return super().log(out, start_time=start_time)
        except TypeError:
            return super().log(out)
