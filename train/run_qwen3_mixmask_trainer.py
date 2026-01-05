#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CLM/MLM (dual-mode via is_causal) trainer for decoder-only models using AutoModelForCausalLM.

Key idea:
- Load AutoModelForCausalLM (so saving/tie-weights are handled safely by HF).
- For CLM: call the full model (model(...)) and compute shifted CE from logits (same as your AutoModel version).
- For MLM (bidir-ish): DO NOT call the full model forward (it may enforce causal paths).
  Instead, call the backbone directly (e.g., model.model) to get last_hidden_state,
  then apply model's own lm_head / output embeddings to get logits, and compute MLM CE only on masked positions.
- Robustly toggle attention behavior using:
    * forward kw: is_causal=True/False (if accepted by backbone)
    * and robust override: module.is_causal=True/False (FlashAttention2 sometimes consults this)
- Keeps important training diagnostics:
  grad_norm, grad_norm_rms, update_to_weight, grad_sim (+ norms), JSONL/CSV logging
"""

import os
import time
import math
import argparse
import inspect
import json
import csv
from typing import Optional, Iterator, List, Dict, Tuple
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)


# -------------------------- is_causal robustness --------------------------
@contextmanager
def force_module_is_causal(model: torch.nn.Module, flag: bool):
    """
    Some attention backends (incl. FlashAttention2) consult module.is_causal.
    Force it for every submodule that has it, then restore.
    """
    touched = []
    for m in model.modules():
        if hasattr(m, "is_causal"):
            try:
                old = bool(getattr(m, "is_causal"))
                setattr(m, "is_causal", bool(flag))
                touched.append((m, old))
            except Exception:
                pass
    try:
        yield
    finally:
        for m, old in touched:
            try:
                setattr(m, "is_causal", old)
            except Exception:
                pass


def _forward_accepts_param(model, name: str) -> bool:
    """
    best-effort signature check:
    - forward に name がある or **kwargs がある => True
    """
    try:
        sig = inspect.signature(model.forward)
        if name in sig.parameters:
            return True
        for p in sig.parameters.values():
            if p.kind == inspect.Parameter.VAR_KEYWORD:
                return True
        return False
    except Exception:
        return True


def _unwrap_model(model):
    return getattr(model, "module", model)


def get_backbone_from_causallm(model: torch.nn.Module) -> torch.nn.Module:
    """
    For most HF causalLM classes:
      - model.model is the backbone (e.g., Qwen3ForCausalLM.model -> Qwen3Model)
      - some use base_model / transformer / gpt_neox, etc.
    We prefer .model if it exists and is nn.Module.
    """
    m = _unwrap_model(model)
    # common: .model
    if hasattr(m, "model") and isinstance(getattr(m, "model"), torch.nn.Module):
        return getattr(m, "model")
    # fallback: base_model
    if hasattr(m, "base_model") and isinstance(getattr(m, "base_model"), torch.nn.Module):
        return getattr(m, "base_model")
    # fallback: transformer
    if hasattr(m, "transformer") and isinstance(getattr(m, "transformer"), torch.nn.Module):
        return getattr(m, "transformer")
    # last resort: itself
    return m


def get_lm_head_module(model: torch.nn.Module) -> torch.nn.Module:
    """
    Use HF's official output embeddings if possible.
    """
    m = _unwrap_model(model)
    fn = getattr(m, "get_output_embeddings", None)
    if callable(fn):
        head = fn()
        if head is not None:
            return head
    if hasattr(m, "lm_head") and isinstance(getattr(m, "lm_head"), torch.nn.Module):
        return getattr(m, "lm_head")
    raise RuntimeError("Could not find lm_head / output embeddings on this model.")


# -------------------------- Tokenizer / mask token setup --------------------------
def ensure_mask_token(tokenizer, model, mask_token: str = "<mask>") -> int:
    """
    tokenizer に <mask> を追加して mask_token_id を確保。
    AutoModelForCausalLM 側の resize + tie を利用。
    """
    mask_id = getattr(tokenizer, "mask_token_id", None)
    if mask_id is None:
        num_added = tokenizer.add_special_tokens({"mask_token": mask_token})
        mask_id = tokenizer.mask_token_id
        if num_added > 0:
            model.resize_token_embeddings(len(tokenizer))
            if hasattr(model, "tie_weights"):
                try:
                    model.tie_weights()
                except Exception:
                    pass
    if mask_id is None:
        raise RuntimeError("mask_token_id を確保できませんでした。")
    return int(mask_id)


# -------------------------- Dataset: streaming fixed-length packing --------------------------
class PackedStreamingIterable(IterableDataset):
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


# -------------------------- Collator --------------------------
def clm_collate(examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    input_ids = torch.stack([ex["input_ids"] for ex in examples], dim=0)  # (B,S)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


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
    B, S = input_ids.shape
    device = input_ids.device
    if mask_ratio <= 0.0:
        return torch.zeros((B, S), dtype=torch.bool, device=device)

    cand = torch.ones((B, S), dtype=torch.bool, device=device)
    for sid in special_ids:
        cand &= (input_ids != sid)

    rnd = torch.rand((B, S), generator=rng, device=device)
    sel = (rnd < mask_ratio) & cand

    # ensure >=1 per row
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


# -------------------------- Norm helpers (per-GPU; no allreduce) --------------------------
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


# -------------------------- Logging callback: JSONL + CSV --------------------------
class JsonlCsvLoggerCallback(TrainerCallback):
    DEFAULT_FIELDS = [
        "step", "epoch", "lr",
        "loss", "loss_clm", "loss_mlm",
        "tps_active", "tflops",
        "grad_norm", "grad_norm_rms", "update_to_weight",
        "grad_sim", "grad_norm_clm", "grad_norm_mlm",
        "grad_norm_clm_rms", "grad_norm_mlm_rms",
        "grad_sim_ms", "grad_sim_oom",
    ]

    def __init__(self, jsonl_path: Optional[str], csv_path: Optional[str], fields: Optional[List[str]] = None):
        self.jsonl_path = jsonl_path
        self.csv_path = csv_path
        self.fields = fields or list(self.DEFAULT_FIELDS)
        self._csv_inited = False

    def _is_process_zero(self, args, state) -> bool:
        return getattr(state, "is_world_process_zero", True) if hasattr(state, "is_world_process_zero") else True

    def _ensure_parent(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def _init_csv_if_needed(self):
        if self.csv_path is None or self._csv_inited:
            return
        self._ensure_parent(self.csv_path)
        file_exists = os.path.exists(self.csv_path) and os.path.getsize(self.csv_path) > 0
        if not file_exists:
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=self.fields)
                w.writeheader()
        self._csv_inited = True

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        if not self._is_process_zero(args, state):
            return

        record = dict(logs)
        record["_time"] = time.time()

        if self.jsonl_path:
            self._ensure_parent(self.jsonl_path)
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        if self.csv_path:
            self._init_csv_if_needed()
            row = {k: record.get(k, "") for k in self.fields}
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=self.fields)
                w.writerow(row)


# -------------------------- Trainer --------------------------
class CLMMLMTrainer(Trainer):
    """
    - CLM: full model forward -> logits -> shifted CE (manual)
    - MLM: backbone forward (avoid causalLM wrapper) -> last_hidden_state -> lm_head -> masked CE
    - both/alternate supported
    - grad_sim: cosine similarity between grad(loss_clm) and grad(loss_mlm) (log step only)
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
            raise ValueError("--both_weights は 'w_clm,w_mlm' の2要素で指定してください。")
        self.w_clm, self.w_mlm = w

        need_mlm = self.mask_policy in ("mlm", "alternate", "both")
        self.mask_token_id = int(mask_token_id) if (need_mlm and mask_token_id is not None) else None
        if need_mlm and self.mask_token_id is None:
            raise RuntimeError("MLMを使うには mask_token_id が必要です。")

        self.grad_sim_enabled = bool(grad_sim)
        self._can_grad_sim = (self.mask_policy == "both")

        # ---- backbone + lm_head handle ----
        self.backbone = get_backbone_from_causallm(self.model)
        self.lm_head = get_lm_head_module(self.model)
        self._backbone_accepts_is_causal = _forward_accepts_param(self.backbone, "is_causal")

        # RNG（GPU上でMLMマスク生成）
        dev = next(self.model.parameters()).device
        self._rng = torch.Generator(device=dev)
        base_seed = int(getattr(self.args, "seed", 42))
        local_rank = int(getattr(self.args, "local_rank", 0) or 0)
        self._rng.manual_seed(base_seed + local_rank)

        self._n_params = int(sum(p.numel() for p in self.model.parameters()))
        self._reset_interval()

        self._step_start_time: Optional[float] = None
        self._micro_in_step: int = 0

        self._pending_grad_stats: Optional[Dict[str, float]] = None
        self._pending_grad_sim: Optional[Dict[str, float]] = None

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
        return x, labels, int(sel.sum().item())

    def _mlm_loss_from_logits(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        mask = labels != -100
        if not mask.any():
            return logits.sum() * 0.0
        V = logits.size(-1)
        return F.cross_entropy(logits[mask].view(-1, V), labels[mask].view(-1), reduction="mean")

    def _clm_loss_from_logits(self, logits: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        logits_s = logits[:, :-1, :].contiguous()
        labels_s = input_ids[:, 1:].contiguous()
        am_s = attention_mask[:, 1:].contiguous().to(torch.bool)
        V = logits_s.size(-1)
        if am_s.any():
            loss = F.cross_entropy(
                logits_s[am_s].view(-1, V),
                labels_s[am_s].view(-1),
                reduction="mean",
            )
        else:
            loss = logits.sum() * 0.0
        return loss

    def _compute_lr(self) -> float:
        try:
            return float(self._get_learning_rate())
        except Exception:
            opt = getattr(self, "optimizer", None)
            if opt is None or len(opt.param_groups) == 0:
                return float("nan")
            return float(opt.param_groups[0].get("lr", float("nan")))

    def _backbone_forward(self, input_ids, attention_mask, is_causal: Optional[bool]):
        """
        Call backbone forward. Pass is_causal if accepted; otherwise rely on module.is_causal override.
        """
        kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
            output_hidden_states=False,
        )
        if is_causal is not None and self._backbone_accepts_is_causal:
            try:
                return self.backbone(**kwargs, is_causal=bool(is_causal))
            except TypeError:
                return self.backbone(**kwargs)
        return self.backbone(**kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        mode = self._mode_now()
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
        B, S = input_ids.shape
        passes = 2 if mode == "both" else 1

        def run_clm():
            # CLM: full model forward is fine (causal is natural), but we still force backbone.is_causal=True for safety.
            with force_module_is_causal(self.backbone, True):
                out = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
            loss = self._clm_loss_from_logits(out.logits, input_ids, attention_mask)
            active = int(attention_mask[:, 1:].sum().item())
            return loss, active

        def run_mlm():
            x_mlm, y_mlm, nmask = self._build_mlm_batch(inputs)
            # MLM: call BACKBONE directly to avoid causalLM wrapper behavior.
            with force_module_is_causal(self.backbone, False):
                bo = self._backbone_forward(x_mlm, attention_mask, is_causal=False)
            hidden = bo.last_hidden_state
            logits = self.lm_head(hidden)
            return self._mlm_loss_from_logits(logits, y_mlm), nmask

        loss_clm = None
        loss_mlm = None
        active = 0

        if mode == "clm":
            loss_clm, active = run_clm()
            loss_total = loss_clm
        elif mode == "mlm":
            loss_mlm, active = run_mlm()
            loss_total = loss_mlm
        else:
            loss_clm, act_c = run_clm()
            loss_mlm, act_m = run_mlm()
            active = int(act_c + act_m)
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
        - autograd.grad を使う（通常学習の backward とは別の “追加” 勾配計算）
        - 集計は GPU 上で行い、最後にスカラーのみ取得
        """
        device = next(self.model.parameters()).device
        t0 = time.time()
        out: Dict[str, float] = {"grad_sim": float("nan"), "grad_sim_oom": 0.0}

        params = [p for p in self.model.parameters() if p.requires_grad]

        try:
            with self.compute_loss_context_manager():
                input_ids = inputs["input_ids"]
                attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))

                # CLM (full model)
                with force_module_is_causal(self.backbone, True):
                    o_clm = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        return_dict=True,
                    )
                loss_clm = self._clm_loss_from_logits(o_clm.logits, input_ids, attention_mask)

                # MLM (backbone + lm_head)
                x_mlm, y_mlm, _ = self._build_mlm_batch(inputs)
                with force_module_is_causal(self.backbone, False):
                    bo = self._backbone_forward(x_mlm, attention_mask, is_causal=False)
                logits_mlm = self.lm_head(bo.last_hidden_state)
                loss_mlm = self._mlm_loss_from_logits(logits_mlm, y_mlm)

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

    def _is_end_of_step(self) -> bool:
        if getattr(self, "accelerator", None) is not None:
            return bool(self.accelerator.sync_gradients)
        gas = int(getattr(self.args, "gradient_accumulation_steps", 1))
        return (self._micro_in_step % max(1, gas) == 0)

    def _get_grad_norm_and_rms(self) -> Tuple[float, float]:
        """
        - grad_norm: 可能なら deepspeed engine の global grad norm を優先（ZeRO-2 等）
        - grad_norm_rms: per-GPU の local RMS
        """
        rms_local = float(local_grad_rms(self.model))
        eng = getattr(self, "deepspeed", None)
        if eng is not None and hasattr(eng, "get_global_grad_norm"):
            try:
                g = eng.get_global_grad_norm()
                if isinstance(g, (float, int)):
                    return float(g), rms_local
                return float(getattr(g, "item", lambda: g)()), rms_local
            except Exception:
                pass
        return float(local_grad_norm(self.model)), rms_local

    def training_step(self, model, inputs, num_items_in_batch=None):
        self._micro_in_step += 1
        if self._step_start_time is None:
            self._step_start_time = time.time()

        try:
            loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
        except TypeError:
            loss = super().training_step(model, inputs)

        if self._is_end_of_step():
            next_step = int(self.state.global_step) + 1
            is_log_step = (self.args.logging_steps > 0) and (next_step % int(self.args.logging_steps) == 0)

            if is_log_step:
                gnorm, grms = self._get_grad_norm_and_rms()
                wnorm = float(local_weight_norm(self.model))
                lr = float(self._compute_lr())
                upd2w = lr * float(gnorm) / max(wnorm, 1e-30)
                self._pending_grad_stats = {
                    "grad_norm": float(gnorm),
                    "grad_norm_rms": float(grms),
                    "update_to_weight": float(upd2w),
                }

                if self.grad_sim_enabled and self._can_grad_sim:
                    self._pending_grad_sim = self._compute_grad_sim_all_params(inputs)

            step_dt = max(1e-9, time.time() - self._step_start_time)
            self._int_step_time += float(step_dt)
            self._step_start_time = None
            self._micro_in_step = 0

        return loss

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None):
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


# -------------------------- CLI / main --------------------------
def _str2bool(x: str) -> bool:
    return str(x).lower() in ("1", "true", "yes", "y", "t")


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-0.6B")
    p.add_argument("--attn_impl", type=str, default="flash_attention_2",
                   choices=["sdpa", "flash_attention_2", "eager", "flex_attention"])

    p.add_argument("--dataset_name", type=str, default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--dataset_config", type=str, default=None)
    p.add_argument("--dataset_split", type=str, default="train")
    p.add_argument("--text_column", type=str, default="text")
    p.add_argument("--streaming", action="store_true", default=True)
    p.add_argument("--shuffle_buffer", type=int, default=10_000)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--add_bos_at_chunk_start", action="store_true", default=False)

    p.add_argument("--mask_policy", type=str, default="both",
                   choices=["clm", "mlm", "alternate", "both", "causal", "bidir"])
    p.add_argument("--mlm_mask_ratio", type=float, default=0.15)
    p.add_argument("--both_weights", type=str, default="1.0,1.0")
    p.add_argument("--ensure_mask_token", type=_str2bool, default=True)
    p.add_argument("--mask_token_str", type=str, default="<mask>")

    p.add_argument("--grad_sim", type=_str2bool, default=False)

    p.add_argument("--output_dir", type=str, default="ckpt_out")

    p.add_argument("--logging_dir", type=str, default=None)
    p.add_argument("--report_to", type=str, default="none")
    p.add_argument("--log_jsonl", type=str, default=None)
    p.add_argument("--log_csv", type=str, default=None)

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
    p.add_argument("--bf16", type=_str2bool, default=True)
    p.add_argument("--fp16", type=_str2bool, default=False)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--ddp_find_unused_parameters", type=_str2bool, default=False)

    p.add_argument("--deepspeed", type=str, default=None)

    return p.parse_args()


def main():
    args = parse_args()

    if args.logging_dir is None:
        args.logging_dir = os.path.join(args.output_dir, "runs")
    if args.log_jsonl is None:
        args.log_jsonl = os.path.join(args.output_dir, "train_log.jsonl")
    if args.log_csv is None:
        args.log_csv = os.path.join(args.output_dir, "train_log.csv")

    tok = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True, trust_remote_code=True)

    use_cuda = torch.cuda.is_available()
    if args.attn_impl == "flash_attention_2" and not use_cuda:
        raise RuntimeError("flash_attention_2 は CUDA が必要です。")

    dtype = torch.bfloat16 if (args.bf16 and use_cuda) else (
        torch.float16 if (args.fp16 and use_cuda) else torch.float32
    )
    if args.attn_impl == "flash_attention_2" and dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError("flash_attention_2 は fp16/bf16 が必要です。（--bf16 True or --fp16 True）")

    # Prefer new arg name dtype if available; fall back to torch_dtype.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            attn_implementation=args.attn_impl,
            trust_remote_code=True,
            dtype=dtype,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            attn_implementation=args.attn_impl,
            trust_remote_code=True,
            torch_dtype=dtype,
        )

    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable") and args.gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass

    # need MLM?
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
                raise RuntimeError("mask_token_id がありません。--ensure_mask_token True を使ってください。")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))

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

    targs_kwargs = dict(
        output_dir=args.output_dir,
        logging_dir=args.logging_dir,
        report_to=args.report_to,
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
        ddp_find_unused_parameters=args.ddp_find_unused_parameters,
    )
    if args.deepspeed:
        targs_kwargs["deepspeed"] = args.deepspeed

    targs = TrainingArguments(**targs_kwargs)

    # tokenizer deprecation 回避（transformers バージョン差吸収）
    trainer_init_sig = inspect.signature(Trainer.__init__)
    trainer_tokenizer_kw = {}
    if "processing_class" in trainer_init_sig.parameters:
        trainer_tokenizer_kw["processing_class"] = tok
    else:
        trainer_tokenizer_kw["tokenizer"] = tok

    trainer = CLMMLMTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        data_collator=clm_collate,
        mask_policy=args.mask_policy,
        mlm_mask_ratio=args.mlm_mask_ratio,
        both_weights=args.both_weights,
        mask_token_id=mask_id,
        grad_sim=args.grad_sim,
        **trainer_tokenizer_kw,
    )

    trainer.add_callback(JsonlCsvLoggerCallback(jsonl_path=args.log_jsonl, csv_path=args.log_csv))

    if trainer.is_world_process_zero():
        backbone = get_backbone_from_causallm(model)
        print({
            "vocab_size": len(tok),
            "mask_token": getattr(tok, "mask_token", None),
            "mask_token_id": getattr(tok, "mask_token_id", None),
            "attn_impl": args.attn_impl,
            "dtype": str(dtype),
            "mask_policy": args.mask_policy,
            "grad_sim": bool(args.grad_sim),
            "backbone_type": str(type(backbone)),
            "backbone_accepts_is_causal": bool(_forward_accepts_param(backbone, "is_causal")),
            "logging_steps": int(args.logging_steps),
            "deepspeed": args.deepspeed,
            "logging_dir": args.logging_dir,
            "report_to": args.report_to,
            "log_jsonl": args.log_jsonl,
            "log_csv": args.log_csv,
        })

    trainer.train()

    if trainer.is_world_process_zero():
        trainer.save_model(args.output_dir)
        tok.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
