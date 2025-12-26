#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Qwen3 CLM + MLM (same batch) trainer (HF Trainer)
- Same data -> build CLM & MLM variants from the same batch
- MLM loss is computed manually (NO shift) over masked positions
- Logs are MACRO-step (optimizer update) oriented; no micro-batch spam
- TFLOPs is step-based (accounts passes=2 for CLM+MLM)
- Optional grad similarity (CLM grad vs MLM grad) computed only at logging steps

Run:
python run_qwen3_clm_mlm_trainer.py \
  --model_name_or_path Qwen/Qwen3-0.6B \
  --dataset_name HuggingFaceFW/fineweb-edu --dataset_split train --text_column text --streaming \
  --seqlen 2048 --per_device_train_batch_size 2 --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 --warmup_steps 100 --max_steps 1000 \
  --attn_impl flex_attention \
  --mask_policy both --both_weights 1.0,1.0 --mlm_mask_ratio 0.15 \
  --bf16 True --report_to none --logging_steps 20 \
  --grad_sim False

Notes:
- MLM/both requires --attn_impl flex_attention (we enforce)
- Adds <mask> token if missing (special token) and resizes model embeddings
"""

import os
import time
import math
import argparse
from typing import Optional, Iterator, List, Dict, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)

try:
    import torch.distributed as dist
except Exception:
    dist = None

# -------------------------- FlexAttention mask functions --------------------------
def causal_mask_fn(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
    return kv_idx <= q_idx

def full_visible_mask_fn(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
    return True

# -------------------------- DDP helpers --------------------------
def dist_is_init() -> bool:
    return dist is not None and dist.is_available() and dist.is_initialized()

def allreduce_sum_scalar(x: float, device: torch.device) -> float:
    if not dist_is_init():
        return float(x)
    t = torch.tensor(float(x), device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item())

# -------------------------- Tokenizer / mask token setup --------------------------
def ensure_mask_token(tokenizer, model, mask_token: str = "<mask>") -> int:
    """
    Ensure tokenizer has mask_token_id. If missing, add special token <mask> and resize model embeddings.
    Return mask_token_id.
    """
    mask_id = getattr(tokenizer, "mask_token_id", None)
    if mask_id is None:
        num_added = tokenizer.add_special_tokens({"mask_token": mask_token})
        mask_id = tokenizer.mask_token_id
        if num_added > 0:
            model.resize_token_embeddings(len(tokenizer))
            if hasattr(model, "tie_weights"):
                model.tie_weights()
    if mask_id is None:
        raise RuntimeError("mask_token_id を確保できませんでした。tokenizer / model の状態を確認してください。")
    return int(mask_id)

# -------------------------- Dataset: streaming fixed-length packing --------------------------
class PackedStreamingIterable(IterableDataset):
    """
    streaming -> tokenize(add_special_tokens=False) -> append EOS per document -> pack into fixed seqlen chunks.
    """
    def __init__(
        self,
        *,
        tokenizer,
        dataset_name: str,
        dataset_config: Optional[str],
        dataset_split: str,
        text_column: str,
        seqlen: int,
        streaming: bool,
        shuffle_buffer: int,
        seed: int,
        world_size: int,
        rank: int,
        add_eos_between_docs: bool = True,
        add_bos_at_chunk_start: bool = False,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        self.tok = tokenizer
        self.name = dataset_name
        self.config = dataset_config
        self.split = dataset_split
        self.text_key = text_column
        self.seqlen = int(seqlen)
        self.streaming = bool(streaming)
        self.shuffle_buffer = int(shuffle_buffer)
        self.seed = int(seed)
        self.world_size = max(1, int(world_size))
        self.rank = max(0, int(rank))
        self.add_eos_between_docs = bool(add_eos_between_docs)
        self.add_bos_at_chunk_start = bool(add_bos_at_chunk_start)
        self.cache_dir = cache_dir
        assert self.tok.eos_token_id is not None, "tokenizer に eos_token_id が必要です。"

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        ds = load_dataset(
            self.name,
            self.config if self.config else None,
            split=self.split,
            streaming=self.streaming,
            cache_dir=self.cache_dir,
        )
        if hasattr(ds, "shard"):
            ds = ds.shard(num_shards=self.world_size, index=self.rank)
        if hasattr(ds, "shuffle") and self.shuffle_buffer > 0:
            ds = ds.shuffle(buffer_size=self.shuffle_buffer, seed=self.seed)

        eos = int(self.tok.eos_token_id)
        buf: List[int] = []

        for ex in ds:
            txt = ex.get(self.text_key, "")
            if not isinstance(txt, str) or not txt:
                continue
            ids = self.tok(txt, add_special_tokens=False, return_attention_mask=False)["input_ids"]
            if self.add_eos_between_docs:
                ids = ids + [eos]
            buf.extend(ids)

            while len(buf) >= self.seqlen:
                chunk = buf[: self.seqlen]
                buf = buf[self.seqlen :]
                if self.add_bos_at_chunk_start and getattr(self.tok, "bos_token_id", None) is not None:
                    chunk[0] = int(self.tok.bos_token_id)
                yield {"input_ids": torch.tensor(chunk, dtype=torch.long)}

# -------------------------- Collator (simple) --------------------------
def clm_collate(examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    input_ids = torch.stack([ex["input_ids"] for ex in examples], dim=0)  # (B,S)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    labels = input_ids.clone()  # CLM: built-in loss shifts internally
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

# -------------------------- MLM helpers --------------------------
def special_ids_from_tokenizer(tok) -> List[int]:
    ids = []
    for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
        v = getattr(tok, name, None)
        if v is not None:
            ids.append(int(v))
    return sorted(set(ids))

def choose_mlm_positions_random(
    input_ids: torch.LongTensor,
    mask_ratio: float,
    special_ids: List[int],
    rng: torch.Generator,
) -> torch.BoolTensor:
    """
    Bernoulli(mask_ratio) per position; exclude special token ids; ensure at least one masked position per sample.
    """
    B, S = input_ids.shape
    device = input_ids.device
    if mask_ratio <= 0.0:
        return torch.zeros((B, S), dtype=torch.bool, device=device)

    cand = torch.ones((B, S), dtype=torch.bool, device=device)
    for sid in special_ids:
        cand &= (input_ids != sid)

    rnd = torch.rand((B, S), generator=rng, device=device)
    sel = (rnd < mask_ratio) & cand

    any_row = sel.any(dim=1)
    if (~any_row).any():
        bad = torch.where(~any_row)[0].tolist()
        for b in bad:
            idxs = torch.where(cand[b])[0]
            if idxs.numel() == 0:
                idxs = torch.arange(S, device=device)
            j = idxs[torch.randint(0, idxs.numel(), (1,), generator=rng, device=device)]
            sel[b, j] = True
    return sel

# -------------------------- Norm helpers --------------------------
def global_grad_norm(model) -> float:
    device = next(model.parameters()).device
    sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            sq += p.grad.detach().float().pow(2).sum().item()
    sq = allreduce_sum_scalar(sq, device)
    return math.sqrt(max(sq, 1e-30))

def global_weight_norm(model) -> float:
    device = next(model.parameters()).device
    sq = 0.0
    for p in model.parameters():
        sq += p.detach().float().pow(2).sum().item()
    sq = allreduce_sum_scalar(sq, device)
    return math.sqrt(max(sq, 1e-30))

# -------------------------- Trainer --------------------------
class CLMMLMTrainer(Trainer):
    """
    mask_policy:
      - clm       : CLM only
      - mlm       : MLM only
      - alternate : alternate clm/mlm by macro-step (global_step parity)
      - both      : compute both losses from the same batch and sum with weights
    """

    def __init__(
        self,
        *args,
        mask_policy: str = "both",
        mlm_mask_ratio: float = 0.15,
        both_weights: str = "1.0,1.0",  # "w_clm,w_mlm"
        attn_impl: str = "flex_attention",
        mask_token_id: Optional[int] = None,
        grad_sim: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        alias = {"causal": "clm", "bidir": "mlm"}
        self.mask_policy = alias.get(mask_policy, mask_policy)
        self.mlm_mask_ratio = float(mlm_mask_ratio)
        self.attn_impl = attn_impl
        self.is_flex = (attn_impl == "flex_attention")

        need_mlm = self.mask_policy in ("mlm", "alternate", "both")
        if need_mlm and not self.is_flex:
            raise RuntimeError("MLM を含む学習には --attn_impl flex_attention が必要です。")

        w = [float(x.strip()) for x in both_weights.split(",")]
        if len(w) != 2:
            raise ValueError("--both_weights は 'w_clm,w_mlm' の2要素で指定してください（例: 1.0,1.0）")
        self.w_clm, self.w_mlm = w

        self.mask_token_id = mask_token_id if need_mlm else None
        if need_mlm and self.mask_token_id is None:
            raise RuntimeError("MLMを使うには mask_token_id が必要です（<mask>追加/resizeが必要）。")

        self.grad_sim_enabled = bool(grad_sim)
        # grad_sim は clm と mlm の両方が同一ステップで定義される "both" でのみ意味が明確
        self._can_grad_sim = (self.mask_policy == "both")

        # RNG per rank (mask sampling)
        dev = self.model.device
        self._rng = torch.Generator(device=dev)
        base_seed = int(getattr(self.args, "seed", 42))
        local_rank = int(getattr(self.args, "local_rank", 0) or 0)
        self._rng.manual_seed(base_seed + local_rank)

        # param count for TFLOPs
        self._n_params = sum(p.numel() for p in self.model.parameters())

        # ---- interval accumulators (reset at each Trainer.log call) ----
        self._reset_interval()

        # ---- macro-step timing ----
        self._step_start_time: Optional[float] = None
        self._micro_in_step: int = 0

        # grad_sim batch cache (only set when needed)
        self._grad_sim_batch: Optional[Dict[str, torch.Tensor]] = None
        self._pending_grad_sim: Optional[Dict[str, float]] = None

        # world_size
        self._world_size = int(os.environ.get("WORLD_SIZE", "1"))

    def _reset_interval(self):
        self._int_micro = 0
        self._int_loss_total = 0.0
        self._int_loss_clm = 0.0
        self._int_loss_mlm = 0.0
        self._int_loss_clm_n = 0
        self._int_loss_mlm_n = 0

        self._int_active_tokens = 0
        self._int_tokens_base = 0  # sum(B*S*passes) over micros
        self._int_step_time = 0.0  # sum macro-step wall time over steps in interval

        self._int_grad_norm = 0.0
        self._int_upd2w = 0.0
        self._int_grad_n = 0

    def _mode_now(self) -> str:
        if self.mask_policy in ("clm", "mlm", "both"):
            return self.mask_policy
        # alternate by macro-step parity
        return "mlm" if (self.state.global_step % 2 == 1) else "clm"

    def _build_mlm_batch(self, inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, int]:
        tok = getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)
        special_ids = special_ids_from_tokenizer(tok)

        input_ids = inputs["input_ids"]
        sel = choose_mlm_positions_random(
            input_ids=input_ids,
            mask_ratio=self.mlm_mask_ratio,
            special_ids=special_ids,
            rng=self._rng,
        )
        x = input_ids.clone()
        x[sel] = int(self.mask_token_id)

        labels = torch.full_like(input_ids, -100)
        labels[sel] = input_ids[sel]
        nmask = int(sel.sum().item())
        return x, labels, nmask

    def _mlm_loss_from_logits(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        mask = labels != -100
        if not mask.any():
            return logits.sum() * 0.0
        V = logits.size(-1)
        logits_m = logits[mask].view(-1, V)
        targets = labels[mask].view(-1)
        return F.cross_entropy(logits_m, targets, reduction="mean")

    def _compute_lr(self) -> float:
        # best-effort
        try:
            return float(self._get_learning_rate())
        except Exception:
            opt = getattr(self, "optimizer", None)
            if opt is None or len(opt.param_groups) == 0:
                return float("nan")
            return float(opt.param_groups[0].get("lr", float("nan")))

    @torch.no_grad()
    def _note_step_start_if_needed(self):
        if self._step_start_time is None:
            self._step_start_time = time.time()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # accept Trainer>=4.44 kw
        _ = kwargs.get("num_items_in_batch", None)

        self._note_step_start_if_needed()

        mode = self._mode_now()
        B, S = inputs["input_ids"].shape

        # passes per micro: both => 2, else 1
        passes = 2 if mode == "both" else 1

        # ---- compute losses ----
        # CLM (built-in shifted loss)
        def run_clm() -> Tuple[torch.Tensor, int]:
            out = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None),
                labels=inputs["labels"],
                mask_function=(causal_mask_fn if self.is_flex else None),
            )
            active = int(B * max(0, S - 1))
            return out.loss, active

        # MLM (manual CE, no shift)
        def run_mlm() -> Tuple[torch.Tensor, int]:
            x_mlm, y_mlm, nmask = self._build_mlm_batch(inputs)
            out = model(
                input_ids=x_mlm,
                attention_mask=inputs.get("attention_mask", None),
                labels=None,
                mask_function=full_visible_mask_fn,
            )
            loss_mlm = self._mlm_loss_from_logits(out.logits, y_mlm)
            return loss_mlm, nmask

        loss_clm = None
        loss_mlm = None
        active = 0

        if mode == "clm":
            loss_clm, active = run_clm()
            loss_total = loss_clm
        elif mode == "mlm":
            loss_mlm, active = run_mlm()
            loss_total = loss_mlm
        else:  # both
            loss_clm, act_c = run_clm()
            loss_mlm, act_m = run_mlm()
            active = int(act_c + act_m)
            loss_total = self.w_clm * loss_clm + self.w_mlm * loss_mlm

        # ---- accumulate interval stats (NO logging here) ----
        self._int_micro += 1
        self._int_loss_total += float(loss_total.detach().float().cpu())
        if loss_clm is not None:
            self._int_loss_clm += float(loss_clm.detach().float().cpu())
            self._int_loss_clm_n += 1
        if loss_mlm is not None:
            self._int_loss_mlm += float(loss_mlm.detach().float().cpu())
            self._int_loss_mlm_n += 1

        self._int_active_tokens += int(active)
        self._int_tokens_base += int(B * S * passes)

        # ---- micro counter for step-end bookkeeping ----
        self._micro_in_step += 1

        # ---- cache batch for grad_sim only when needed ----
        # We want grad_sim at logging steps only, using the last micro of the macro-step.
        if (
            self.grad_sim_enabled
            and self._can_grad_sim
            and self.args.logging_steps > 0
            and self._micro_in_step == int(self.args.gradient_accumulation_steps)
        ):
            # this optimizer update will become global_step+1
            next_step = int(self.state.global_step) + 1
            if (next_step % int(self.args.logging_steps)) == 0:
                # store a lightweight batch (on CPU to avoid holding GPU mem)
                self._grad_sim_batch = {
                    "input_ids": inputs["input_ids"].detach().cpu(),
                    "attention_mask": inputs.get("attention_mask", None).detach().cpu() if inputs.get("attention_mask", None) is not None else None,
                    "labels": inputs["labels"].detach().cpu(),
                }

        return (loss_total, None) if return_outputs else loss_total

    def _compute_grad_sim_all_params(self) -> Dict[str, float]:
        """
        Compute cosine similarity between grad(loss_clm) and grad(loss_mlm) on the cached batch.
        Default: ALL parameters (as requested). Heavy for large models.
        Returns dict with grad_sim, grad_norm_clm, grad_norm_mlm, grad_sim_ms, grad_sim_oom.
        """
        out: Dict[str, float] = {}
        if self._grad_sim_batch is None:
            out["grad_sim"] = float("nan")
            out["grad_sim_ms"] = 0.0
            out["grad_sim_oom"] = 0.0
            return out

        # Move batch back to device
        device = self.model.device
        batch = {
            "input_ids": self._grad_sim_batch["input_ids"].to(device, non_blocking=True),
            "labels": self._grad_sim_batch["labels"].to(device, non_blocking=True),
        }
        if self._grad_sim_batch.get("attention_mask", None) is not None:
            batch["attention_mask"] = self._grad_sim_batch["attention_mask"].to(device, non_blocking=True)

        # IMPORTANT:
        # - default: all params (can be very heavy). If you need a quick fallback:
        #   params = [self.model.lm_head.weight]  # 1-line change
        params = [p for p in self.model.parameters() if p.requires_grad]

        t0 = time.time()
        oom = 0.0
        try:
            with self.compute_loss_context_manager():
                # CLM loss
                out_c = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask", None),
                    labels=batch["labels"],
                    mask_function=(causal_mask_fn if self.is_flex else None),
                )
                loss_clm = out_c.loss

                # MLM loss (manual, no shift)
                x_mlm, y_mlm, _ = self._build_mlm_batch(batch)
                out_m = self.model(
                    input_ids=x_mlm,
                    attention_mask=batch.get("attention_mask", None),
                    labels=None,
                    mask_function=full_visible_mask_fn,
                )
                loss_mlm = self._mlm_loss_from_logits(out_m.logits, y_mlm)

            # grads (do NOT touch .grad)
            grads_c = torch.autograd.grad(loss_clm, params, retain_graph=False, create_graph=False, allow_unused=True)
            grads_m = torch.autograd.grad(loss_mlm, params, retain_graph=False, create_graph=False, allow_unused=True)

            dot = torch.tensor(0.0, device=device)
            na = torch.tensor(0.0, device=device)
            nb = torch.tensor(0.0, device=device)
            for gc, gm in zip(grads_c, grads_m):
                if gc is None or gm is None:
                    continue
                gc = gc.detach().float()
                gm = gm.detach().float()
                dot += (gc * gm).sum()
                na += (gc * gc).sum()
                nb += (gm * gm).sum()

            if dist_is_init():
                dist.all_reduce(dot, op=dist.ReduceOp.SUM)
                dist.all_reduce(na, op=dist.ReduceOp.SUM)
                dist.all_reduce(nb, op=dist.ReduceOp.SUM)

            dotv = float(dot.item())
            nav = float(na.item())
            nbv = float(nb.item())
            cos = dotv / (math.sqrt(max(nav, 1e-30)) * math.sqrt(max(nbv, 1e-30)) + 1e-30)

            out["grad_sim"] = float(cos)
            out["grad_norm_clm"] = math.sqrt(max(nav, 1e-30))
            out["grad_norm_mlm"] = math.sqrt(max(nbv, 1e-30))

        except RuntimeError as e:
            # handle OOM gracefully
            msg = str(e).lower()
            if "out of memory" in msg or "cuda oom" in msg:
                oom = 1.0
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                out["grad_sim"] = float("nan")
                out["grad_norm_clm"] = float("nan")
                out["grad_norm_mlm"] = float("nan")
            else:
                raise

        out["grad_sim_ms"] = (time.time() - t0) * 1000.0
        out["grad_sim_oom"] = oom
        return out

    def optimizer_step(self, *args, **kwargs):
        """
        Called on each MACRO step. We use it to:
          - measure grad_norm / update_to_weight (before step)
          - compute grad_sim (only if enabled and at logging step)
          - measure step wall time (from first micro to here)
        """
        # ---- step wall time (includes micro fwd/bwd; includes grad_sim if we do it here) ----
        step_dt = 0.0
        if self._step_start_time is not None:
            step_dt = max(1e-9, time.time() - self._step_start_time)

        # ---- grad norm & update/weight (pre-step) ----
        gnorm = global_grad_norm(self.model)
        wnorm = global_weight_norm(self.model)
        lr = self._compute_lr()
        upd2w = float(lr) * float(gnorm) / float(max(wnorm, 1e-30))

        self._int_grad_norm += float(gnorm)
        self._int_upd2w += float(upd2w)
        self._int_grad_n += 1

        # ---- grad similarity only when: enabled AND this step will be logged ----
        if self.grad_sim_enabled and self._can_grad_sim and self.args.logging_steps > 0:
            next_step = int(self.state.global_step) + 1  # will become global_step after this update
            if (next_step % int(self.args.logging_steps)) == 0:
                # compute here so its time cost is included in step_dt and thus in tflops drop
                gs = self._compute_grad_sim_all_params()
                self._pending_grad_sim = gs
                # grad_sim itself adds time; update step_dt after computing it
                if self._step_start_time is not None:
                    step_dt = max(1e-9, time.time() - self._step_start_time)

        # add step time to interval and reset step timer bookkeeping
        self._int_step_time += float(step_dt)
        self._step_start_time = None
        self._micro_in_step = 0

        # do the actual optimizer step
        ret = super().optimizer_step(*args, **kwargs)
        return ret

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None):
        """
        Intercept Trainer's periodic logging, and replace with our macro-step aggregated logs.
        This avoids micro-batch spam and prints one compact dict at logging_steps.
        """
        # pass through eval logs etc.
        if any(k.startswith("eval_") for k in logs.keys()):
            return super().log(logs, start_time=start_time)

        # Build our aggregated metrics for the interval since last log call.
        # Trainer calls log() at logging_steps; so this is exactly our desired output.
        dt = max(1e-9, float(self._int_step_time))  # sum of macro-step times in interval

        # losses: average over micros
        micro = max(1, int(self._int_micro))
        loss_total = self._int_loss_total / micro
        loss_clm = (self._int_loss_clm / max(1, self._int_loss_clm_n)) if self._int_loss_clm_n > 0 else float("nan")
        loss_mlm = (self._int_loss_mlm / max(1, self._int_loss_mlm_n)) if self._int_loss_mlm_n > 0 else float("nan")

        # throughput
        tps_active = float(self._int_active_tokens) / dt

        # step-based TFLOPs (accounts passes via _int_tokens_base)
        # FLOPs ≈ 6 * N * (tokens_base * world_size) / time
        flops_est = 6.0 * float(self._n_params) * float(self._int_tokens_base) * float(self._world_size)
        tflops = (flops_est / dt) / 1e12

        # grad stats: average over macro steps in interval (usually 1 if logging_steps=1)
        grad_norm = (self._int_grad_norm / max(1, self._int_grad_n)) if self._int_grad_n > 0 else float("nan")
        upd2w = (self._int_upd2w / max(1, self._int_grad_n)) if self._int_grad_n > 0 else float("nan")

        # progress scalars
        step = int(self.state.global_step)
        epoch = float(self.state.epoch) if self.state.epoch is not None else float("nan")
        lr = self._compute_lr()

        out = {
            "step": step,
            "epoch": round(epoch, 4) if epoch == epoch else epoch,
            "lr": float(lr),
            "loss": round(float(loss_total), 4),              # keep key "loss" for Trainer-like behavior
            "loss_total": round(float(loss_total), 4),
            "loss_clm": round(float(loss_clm), 4) if loss_clm == loss_clm else loss_clm,
            "loss_mlm": round(float(loss_mlm), 4) if loss_mlm == loss_mlm else loss_mlm,
            "grad_norm": round(float(grad_norm), 4) if grad_norm == grad_norm else grad_norm,
            "update_to_weight": round(float(upd2w), 6) if upd2w == upd2w else upd2w,
            "tps_active": round(float(tps_active), 1),
            "tflops": round(float(tflops), 2),
        }

        if self._pending_grad_sim is not None:
            gs = self._pending_grad_sim
            out["grad_sim"] = round(float(gs.get("grad_sim", float("nan"))), 6) if gs.get("grad_sim", float("nan")) == gs.get("grad_sim", float("nan")) else gs.get("grad_sim", float("nan"))
            if "grad_norm_clm" in gs:
                out["grad_norm_clm"] = round(float(gs["grad_norm_clm"]), 4) if gs["grad_norm_clm"] == gs["grad_norm_clm"] else gs["grad_norm_clm"]
            if "grad_norm_mlm" in gs:
                out["grad_norm_mlm"] = round(float(gs["grad_norm_mlm"]), 4) if gs["grad_norm_mlm"] == gs["grad_norm_mlm"] else gs["grad_norm_mlm"]
            out["grad_sim_ms"] = round(float(gs.get("grad_sim_ms", 0.0)), 2)
            out["grad_sim_oom"] = float(gs.get("grad_sim_oom", 0.0))
            self._pending_grad_sim = None
            self._grad_sim_batch = None

        # reset interval accumulators AFTER producing log
        self._reset_interval()

        return super().log(out, start_time=start_time)

# -------------------------- CLI / main --------------------------
def parse_args():
    p = argparse.ArgumentParser()

    # model
    p.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-0.6B")
    p.add_argument("--attn_impl", type=str, default="flex_attention",
                   choices=["flex_attention", "sdpa", "flash_attention_2", "eager"])

    # data
    p.add_argument("--dataset_name", type=str, default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--dataset_config", type=str, default=None)
    p.add_argument("--dataset_split", type=str, default="train")
    p.add_argument("--text_column", type=str, default="text")
    p.add_argument("--streaming", action="store_true", default=True)
    p.add_argument("--shuffle_buffer", type=int, default=10_000)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--add_bos_at_chunk_start", action="store_true", default=False)

    # policy
    p.add_argument("--mask_policy", type=str, default="both",
                   choices=["clm", "mlm", "alternate", "both", "causal", "bidir"])
    p.add_argument("--mlm_mask_ratio", type=float, default=0.15)
    p.add_argument("--both_weights", type=str, default="1.0,1.0")
    p.add_argument("--ensure_mask_token", type=lambda x: x.lower() == "true", default=True)
    p.add_argument("--mask_token_str", type=str, default="<mask>")

    # logging / analysis
    p.add_argument("--grad_sim", type=lambda x: x.lower() == "true", default=False)

    # training
    p.add_argument("--output_dir", type=str, default="ckpt_out")
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--logging_steps", type=int, default=20)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bf16", type=lambda x: x.lower() == "true", default=True)
    p.add_argument("--fp16", type=lambda x: x.lower() == "true", default=False)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--report_to", type=str, default="none")
    p.add_argument("--ddp_find_unused_parameters", type=lambda x: x.lower() == "true", default=False)

    return p.parse_args()

def main():
    args = parse_args()

    tok = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=True,
        trust_remote_code=True,
    )

    dtype = torch.bfloat16 if args.bf16 and torch.cuda.is_available() else (
        torch.float16 if args.fp16 and torch.cuda.is_available() else torch.float32
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        attn_implementation=args.attn_impl,
        use_cache=False,
        trust_remote_code=True,
        torch_dtype=dtype,
    )

    # distributed info (for dataset sharding)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))

    alias = {"causal": "clm", "bidir": "mlm"}
    policy = alias.get(args.mask_policy, args.mask_policy)
    need_mlm = policy in ("mlm", "alternate", "both")

    mask_id = None
    if need_mlm:
        if args.ensure_mask_token:
            mask_id = ensure_mask_token(tok, model, mask_token=args.mask_token_str)
        else:
            mask_id = getattr(tok, "mask_token_id", None)
            if mask_id is None:
                raise RuntimeError("mask_token_id がありません。--ensure_mask_token True を使うか、tokenizer側で <mask> を用意してください。")

    train_ds = PackedStreamingIterable(
        tokenizer=tok,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        dataset_split=args.dataset_split,
        text_column=args.text_column,
        seqlen=args.seqlen,
        streaming=args.streaming,
        shuffle_buffer=args.shuffle_buffer,
        seed=args.seed,
        world_size=world_size,
        rank=rank,
        add_eos_between_docs=True,
        add_bos_at_chunk_start=args.add_bos_at_chunk_start,
        cache_dir=args.cache_dir,
    )

    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to=args.report_to,
        ddp_find_unused_parameters=args.ddp_find_unused_parameters,
    )

    trainer = CLMMLMTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        data_collator=clm_collate,
        processing_class=tok,  # avoid tokenizer deprecation warning
        mask_policy=args.mask_policy,
        mlm_mask_ratio=args.mlm_mask_ratio,
        both_weights=args.both_weights,
        attn_impl=args.attn_impl,
        mask_token_id=mask_id,
        grad_sim=args.grad_sim,
    )

    if trainer.is_world_process_zero():
        emb = model.get_input_embeddings().weight
        head = getattr(model, "lm_head", None).weight if getattr(model, "lm_head", None) is not None else None
        tied = (head is not None and emb.data_ptr() == head.data_ptr())
        print({
            "vocab_size": len(tok),
            "mask_token": getattr(tok, "mask_token", None),
            "mask_token_id": getattr(tok, "mask_token_id", None),
            "unk_token_id": getattr(tok, "unk_token_id", None),
            "eos_token_id": getattr(tok, "eos_token_id", None),
            "emb_rows": int(emb.shape[0]),
            "emb_vs_head_tied": bool(tied),
            "attn_impl": args.attn_impl,
            "mask_policy": args.mask_policy,
            "mlm_mask_ratio": args.mlm_mask_ratio,
            "both_weights": args.both_weights,
            "grad_sim": bool(args.grad_sim),
            "logging_steps": int(args.logging_steps),
        })

    trainer.train()

    if trainer.is_world_process_zero():
        trainer.save_model(args.output_dir)
        tok.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
