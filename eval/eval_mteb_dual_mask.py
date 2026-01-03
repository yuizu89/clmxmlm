#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate a CLM/MLM-capable decoder-only model as an embedding model on MTEB (v2 API).

Key features (keeps your original intent):
- Two inference masks (FlexAttention mask_function):
  (1) bidir/full-visible attention -> pooling
  (2) causal attention             -> pooling
- Combination:
  - concat: [emb_bidir ; emb_causal]
  - avg:    (emb_bidir + emb_causal)/2
- Pooling per mode:
  - mean / last / cls / eos

Important (MTEB v2):
- Custom models must implement EncoderProtocol:
  encode(inputs: DataLoader[BatchedInput], task_metadata, hf_split, hf_subset, prompt_type, **kwargs)
  (NOT encode(List[str]) anymore)

Results:
- mteb.evaluate() no longer accepts output_folder.
- Use ResultCache(cache_path=...) to store results on disk.
"""

import os
import re
import json
import time
import argparse
import inspect
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Literal

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from mteb.models import ModelMeta 

# -------------------- FlexAttention mask functions (same as training) --------------------
def causal_mask_fn(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
    return kv_idx <= q_idx

def full_visible_mask_fn(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
    return True


# -------------------- pooling --------------------
Pooling = Literal["mean", "last", "cls", "eos"]

def pool_hidden(
    last_hidden: torch.Tensor,          # (B, S, H)
    attention_mask: torch.Tensor,       # (B, S) 1=valid, 0=pad
    method: Pooling = "mean",
    *,
    input_ids: Optional[torch.Tensor] = None,  # (B,S) needed for eos
    eos_token_id: Optional[int] = None,
) -> torch.Tensor:
    B, S, H = last_hidden.shape
    am = attention_mask.to(last_hidden.device).to(torch.long)

    if method == "mean":
        denom = am.sum(dim=1, keepdim=True).clamp(min=1)  # (B,1)
        x = (last_hidden * am.unsqueeze(-1)).sum(dim=1) / denom
        return x

    if method == "last":
        idx = (am.sum(dim=1) - 1).clamp(min=0)  # (B,)
        return last_hidden[torch.arange(B, device=last_hidden.device), idx]

    if method == "cls":
        return last_hidden[:, 0, :]

    if method == "eos":
        if input_ids is None or eos_token_id is None:
            raise ValueError("pooling='eos' requires input_ids and eos_token_id.")
        out = torch.empty((B, H), device=last_hidden.device, dtype=last_hidden.dtype)
        for b in range(B):
            row = input_ids[b]
            eos_pos = torch.where(row == int(eos_token_id))[0]
            if eos_pos.numel() > 0:
                j = int(eos_pos[-1].item())
            else:
                j = int((am[b].sum() - 1).clamp(min=0).item())
            out[b] = last_hidden[b, j]
        return out

    raise ValueError(f"unknown pooling: {method}")


# -------------------- utils --------------------
def safe_name(x: str) -> str:
    x = x.rstrip("/").replace("\\", "/")
    x = x.split("/")[-1] if "/" in x else x
    x = re.sub(r"[^A-Za-z0-9._-]+", "_", x)
    return x or "model"

def _supports_mask_function(model) -> bool:
    # best-effort: check forward signature
    try:
        sig = inspect.signature(model.forward)
        return "mask_function" in sig.parameters
    except Exception:
        # fallback: many trust_remote_code models are hard to introspect
        return True


# -------------------- MTEB v2 EncoderProtocol implementation --------------------
@dataclass
class DualMaskConfig:
    model_name_or_path: str
    attn_impl: str = "flex_attention"
    device: str = "cuda"
    dtype: str = "bf16"                 # bf16/fp16/fp32
    max_length: int = 512
    normalize: bool = True

    mask_mode: str = "bidir"            # bidir/causal/concat/avg
    pooling_bidir: Pooling = "mean"
    pooling_causal: Pooling = "mean"

    hf_token: Optional[str] = None      # for gated/private models


class DualMaskHFEncoder:
    """
    Implements MTEB v2 EncoderProtocol:
      encode(inputs: DataLoader[BatchedInput], task_metadata, hf_split, hf_subset, prompt_type, **kwargs) -> np.ndarray
    """

    def __init__(self, cfg: DualMaskConfig):
        self.cfg = cfg

        # tokenizer
        tok_kwargs = dict(use_fast=True, trust_remote_code=True)
        if cfg.hf_token:
            tok_kwargs["token"] = cfg.hf_token
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, **tok_kwargs)

        # dtype
        torch_dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }[cfg.dtype]

        # model
        model_kwargs = dict(
            attn_implementation=cfg.attn_impl,
            trust_remote_code=True,
        )
        # transformers versions differ: some prefer dtype=..., some torch_dtype=...
        if cfg.hf_token:
            model_kwargs["token"] = cfg.hf_token

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                cfg.model_name_or_path,
                dtype=torch_dtype,
                **model_kwargs,
            )
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(
                cfg.model_name_or_path,
                torch_dtype=torch_dtype,
                **model_kwargs,
            )

        self.model.eval()
        self.device = torch.device(cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu")
        self.model.to(self.device)

        self.eos_token_id = getattr(self.tokenizer, "eos_token_id", None)

        mm = cfg.mask_mode
        if mm not in ("bidir", "causal", "concat", "avg"):
            raise ValueError("--mask_mode must be one of: bidir, causal, concat, avg")

        # bidir requires flex_attention + mask_function support
        need_bidir = mm in ("bidir", "concat", "avg")
        if need_bidir and cfg.attn_impl != "flex_attention":
            raise ValueError("bidir/concat/avg requires --attn_impl flex_attention.")
        if need_bidir and not _supports_mask_function(self.model):
            raise ValueError("This model forward() does not appear to accept mask_function; bidir is not available.")

        # give MTEB a stable name (avoids placeholder like no_model_name/available)
        self.model_name = cfg.model_name_or_path
        self.embedding_dim = getattr(self.model.config, "hidden_size", None)

    def get_sentence_embedding_dimension(self) -> int:
        if self.embedding_dim is None:
            raise RuntimeError("Could not infer hidden_size from model.config.")
        mm = self.cfg.mask_mode
        if mm == "concat":
            return int(self.embedding_dim) * 2
        return int(self.embedding_dim)

    @torch.no_grad()
    def _forward_hidden(self, input_ids, attention_mask, mode: Literal["bidir", "causal"]):
        # For causal mode, you can omit mask_function (model already causal),
        # but we keep it consistent if flex_attention is used.
        if self.cfg.attn_impl == "flex_attention":
            mask_fn = full_visible_mask_fn if mode == "bidir" else causal_mask_fn
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
                mask_function=mask_fn,
            )
        else:
            if mode == "bidir":
                raise ValueError("bidir requires flex_attention.")
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
        return out.hidden_states[-1]  # (B,S,H)

    @torch.no_grad()
    def _encode_texts_one_mode(self, texts: List[str], mode: Literal["bidir", "causal"], *, max_length: int) -> np.ndarray:
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            add_special_tokens=True,
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        h = self._forward_hidden(input_ids, attention_mask, mode)
        if mode == "bidir":
            pooled = pool_hidden(
                h, attention_mask, self.cfg.pooling_bidir,
                input_ids=input_ids, eos_token_id=self.eos_token_id
            )
        else:
            pooled = pool_hidden(
                h, attention_mask, self.cfg.pooling_causal,
                input_ids=input_ids, eos_token_id=self.eos_token_id
            )

        if self.cfg.normalize:
            pooled = F.normalize(pooled, p=2, dim=-1)

        return pooled.detach().float().cpu().numpy()

    @staticmethod
    def _coerce_text_list(x) -> List[str]:
        """
        BatchedInput["text"] can be list[str] or list[list[str]] depending on task.
        - if list[list[str]], join with whitespace.
        """
        if x is None:
            return []
        if isinstance(x, (tuple, list)) and len(x) > 0 and isinstance(x[0], (tuple, list)):
            return [" ".join(map(str, xi)) for xi in x]
        return [str(t) for t in x]

    # ---- MTEB v2 required signature ----
    def encode(
        self,
        inputs,                 # DataLoader[BatchedInput]
        task_metadata=None,
        hf_split: str = "",
        hf_subset: str = "",
        prompt_type=None,
        **kwargs,
    ) -> np.ndarray:
        # Allow overriding max_length/normalize at call-time (via encode_kwargs in mteb.evaluate)
        max_length = int(kwargs.get("max_length", self.cfg.max_length))

        all_vecs: List[np.ndarray] = []
        mm = self.cfg.mask_mode

        for batch in inputs:
            # Text tasks: BatchedInput should include "text"
            texts = self._coerce_text_list(batch.get("text", None))
            if not texts:
                # Fail fast with a clear message
                raise ValueError(f"BatchedInput has no usable 'text' field. keys={list(batch.keys())}")

            if mm in ("bidir", "causal"):
                v = self._encode_texts_one_mode(texts, mm, max_length=max_length)

            elif mm == "concat":
                a = self._encode_texts_one_mode(texts, "bidir", max_length=max_length)
                b = self._encode_texts_one_mode(texts, "causal", max_length=max_length)
                v = np.concatenate([a, b], axis=1)
                if self.cfg.normalize:
                    denom = np.linalg.norm(v, axis=1, keepdims=True)
                    v = v / np.maximum(denom, 1e-12)

            elif mm == "avg":
                a = self._encode_texts_one_mode(texts, "bidir", max_length=max_length)
                b = self._encode_texts_one_mode(texts, "causal", max_length=max_length)
                v = (a + b) * 0.5
                if self.cfg.normalize:
                    denom = np.linalg.norm(v, axis=1, keepdims=True)
                    v = v / np.maximum(denom, 1e-12)

            else:
                raise ValueError(f"unknown mask_mode: {mm}")

            all_vecs.append(v)

        if not all_vecs:
            return np.zeros((0, self.get_sentence_embedding_dimension()), dtype=np.float32)
        return np.concatenate(all_vecs, axis=0)


# -------------------- runner --------------------
def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--model_name_or_path", type=str, required=True)

    # embedding behavior
    p.add_argument("--mask_mode", type=str, default="bidir",
                   choices=["bidir", "causal", "concat", "avg"])
    p.add_argument("--pooling_bidir", type=str, default="mean", choices=["mean", "last", "cls", "eos"])
    p.add_argument("--pooling_causal", type=str, default="mean", choices=["mean", "last", "cls", "eos"])
    p.add_argument("--no_normalize", action="store_true", default=False)
    p.add_argument("--max_length", type=int, default=512)

    # runtime
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--attn_impl", type=str, default="flex_attention",
                   choices=["flex_attention", "sdpa", "flash_attention_2", "eager"])
    p.add_argument("--hf_token", type=str, default=None)

    # MTEB selection (v2)
    p.add_argument("--languages", type=str, default="eng",
                   help="comma-separated ISO639-3 (recommended): e.g., eng,jpn")
    p.add_argument("--tasks", type=str, default="",
                   help="comma-separated task names, e.g. STS22.v2,Banking77Classification.v2")
    p.add_argument("--benchmark", type=str, default="",
                   help="benchmark name, e.g. 'MTEB(eng, v2)' (preferred for full suite)")

    # results/cache
    p.add_argument("--output_dir", type=str, default="mteb_results")
    p.add_argument("--overwrite_strategy", type=str, default="only-missing",
                   choices=["only-missing", "always", "never"],
                   help="ResultCache overwrite strategy")

    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # quiet TF logs if TF exists
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    import mteb
    from mteb.cache import ResultCache

    langs = [x.strip() for x in args.languages.split(",") if x.strip()]

    # --- select tasks/benchmark ---
    if args.tasks.strip():
        task_names = [x.strip() for x in args.tasks.split(",") if x.strip()]
        tasks = mteb.get_tasks(tasks=task_names, languages=langs)
    else:
        # default to benchmark if tasks not specified
        bench_name = args.benchmark.strip() or f"MTEB({langs[0]}, v2)"
        tasks = mteb.get_benchmark(bench_name)

    # --- build model ---
    cfg = DualMaskConfig(
        model_name_or_path=args.model_name_or_path,
        attn_impl=args.attn_impl,
        device=args.device,
        dtype=args.dtype,
        max_length=args.max_length,
        normalize=(not args.no_normalize),
        mask_mode=args.mask_mode,
        pooling_bidir=args.pooling_bidir,
        pooling_causal=args.pooling_causal,
        hf_token=args.hf_token,
    )
    model = DualMaskHFEncoder(cfg)

    model.mteb_model_meta = ModelMeta(
        name=args.model_name_or_path,
        revision="no_revision_available",
        release_date=None,
        languages=langs,
    )

    run_name = (
        f"{safe_name(args.model_name_or_path)}__"
        f"{args.mask_mode}__"
        f"pb-{args.pooling_bidir}__pc-{args.pooling_causal}__"
        f"ml{args.max_length}"
    )
    out_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)

    # Result cache: this is where results are written to disk
    cache = ResultCache(cache_path=out_dir)

    t0 = time.time()
    results = mteb.evaluate(
        model,
        tasks,
        cache=cache,
        overwrite_strategy=args.overwrite_strategy,
        encode_kwargs={
            "max_length": args.max_length,
        },
    )

    # write a compact summary for convenience
    summary = {}
    for r in results:
        try:
            summary[r.task_name] = {
                "main_score": float(r.get_score()),
                "metric": str(r.task.metadata.main_score),
                "main_scores": r.only_main_score().to_dict(),
            }
        except Exception:
            summary[r.task_name] = {"raw": str(r)}

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[done] saved to: {out_dir}")
    print(f"[done] elapsed: {(time.time()-t0):.1f}s")


if __name__ == "__main__":
    main()
