#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate a CLM/MLM-capable decoder-only model as an embedding model on MTEB.

Key features:
- Two inference masks:
  (1) bidir/full-visible attention   -> pooling
  (2) causal attention              -> pooling
- Combination:
  - concat: [emb_bidir ; emb_causal]
  - avg:    (emb_bidir + emb_causal)/2
- Pooling:
  - mean (default): mean over non-pad tokens
  - last: last non-pad token (often reasonable for causal)
  - cls:  first token (only if you intentionally use BOS/CLS semantics)

This script assumes:
- transformers model forward can accept `mask_function=` like your training code
  (works when using FlexAttention integration).
If your runtime model doesn't accept mask_function, see the note at the bottom.
"""

import os
import json
import math
import argparse
from typing import List, Optional, Literal

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from mteb import MTEB


# -------------------- FlexAttention mask functions (same as training) --------------------
def causal_mask_fn(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
    return kv_idx <= q_idx

def full_visible_mask_fn(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
    return True


# -------------------- pooling --------------------
def pool_hidden(
    last_hidden: torch.Tensor,          # (B, S, H)
    attention_mask: torch.Tensor,       # (B, S) 1=valid, 0=pad
    method: Literal["mean", "last", "cls"] = "mean",
) -> torch.Tensor:
    B, S, H = last_hidden.shape
    am = attention_mask.to(last_hidden.device).to(torch.long)

    if method == "mean":
        denom = am.sum(dim=1, keepdim=True).clamp(min=1)  # (B,1)
        x = (last_hidden * am.unsqueeze(-1)).sum(dim=1) / denom
        return x

    if method == "last":
        # last non-pad token
        idx = (am.sum(dim=1) - 1).clamp(min=0)  # (B,)
        return last_hidden[torch.arange(B, device=last_hidden.device), idx]

    if method == "cls":
        return last_hidden[:, 0, :]

    raise ValueError(f"unknown pooling: {method}")


# -------------------- embedding wrapper for MTEB --------------------
class DualMaskHFEmbedder:
    """
    MTEB calls `encode(sentences, batch_size=..., **kwargs)`.

    We expose:
      - mask_mode: 'bidir' | 'causal' | 'concat' | 'avg'
      - pooling_bidir / pooling_causal
    """
    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cuda",
        dtype: str = "bf16",
        max_length: int = 512,
        mask_mode: str = "bidir",
        pooling_bidir: str = "mean",
        pooling_causal: str = "mean",
        normalize: bool = True,
        attn_impl: str = "flex_attention",
        trust_remote_code: bool = True,
    ):
        self.device = device
        self.max_length = int(max_length)
        self.mask_mode = mask_mode
        self.pooling_bidir = pooling_bidir
        self.pooling_causal = pooling_causal
        self.normalize = bool(normalize)
        self.attn_impl = attn_impl

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, use_fast=True, trust_remote_code=trust_remote_code
        )

        torch_dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }[dtype]

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            attn_implementation=attn_impl,
            trust_remote_code=trust_remote_code,
        ).to(device)

        self.model.eval()

    @torch.no_grad()
    def _forward_hidden(self, input_ids, attention_mask, mask_function):
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            mask_function=mask_function,
        )
        # last layer hidden states
        # transformers CausalLMOutputWithPast: hidden_states is tuple(layer0..last)
        last_hidden = out.hidden_states[-1]  # (B,S,H)
        return last_hidden

    @torch.no_grad()
    def _encode_one_mode(self, sentences: List[str], batch_size: int, mode: str) -> np.ndarray:
        all_vecs = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
                add_special_tokens=True,
            )
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)

            if mode == "bidir":
                h = self._forward_hidden(input_ids, attention_mask, full_visible_mask_fn)
                v = pool_hidden(h, attention_mask, self.pooling_bidir)
            elif mode == "causal":
                h = self._forward_hidden(input_ids, attention_mask, causal_mask_fn)
                v = pool_hidden(h, attention_mask, self.pooling_causal)
            else:
                raise ValueError(mode)

            if self.normalize:
                v = torch.nn.functional.normalize(v, p=2, dim=-1)

            all_vecs.append(v.detach().float().cpu().numpy())
        return np.concatenate(all_vecs, axis=0)

    def encode(
        self,
        sentences: List[str],
        batch_size: int = 32,
        **kwargs,
    ):
        if isinstance(sentences, str):
            sentences = [sentences]

        mm = self.mask_mode
        if mm in ("bidir", "causal"):
            return self._encode_one_mode(sentences, batch_size, mm)

        if mm == "concat":
            a = self._encode_one_mode(sentences, batch_size, "bidir")
            b = self._encode_one_mode(sentences, batch_size, "causal")
            return np.concatenate([a, b], axis=1)

        if mm == "avg":
            a = self._encode_one_mode(sentences, batch_size, "bidir")
            b = self._encode_one_mode(sentences, batch_size, "causal")
            v = (a + b) * 0.5
            if self.normalize:
                denom = np.linalg.norm(v, axis=1, keepdims=True)
                denom = np.maximum(denom, 1e-12)
                v = v / denom
            return v

        raise ValueError(f"unknown mask_mode: {mm}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", type=str, required=True)

    # embedding behavior
    p.add_argument("--mask_mode", type=str, default="bidir",
                   choices=["bidir", "causal", "concat", "avg"])
    p.add_argument("--pooling_bidir", type=str, default="mean", choices=["mean", "last", "cls"])
    p.add_argument("--pooling_causal", type=str, default="mean", choices=["mean", "last", "cls"])
    p.add_argument("--normalize", action="store_true", default=True)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=32)

    # runtime
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--attn_impl", type=str, default="flex_attention",
                   choices=["flex_attention", "sdpa", "flash_attention_2", "eager"])

    # MTEB selection
    p.add_argument("--task_langs", type=str, default="en",
                   help="comma-separated, e.g. en or en,ja")
    p.add_argument("--tasks", type=str, default="",
                   help="comma-separated task names. empty => MTEB(task_langs=...) default suite")
    p.add_argument("--output_dir", type=str, default="mteb_results")
    p.add_argument("--no_overwrite", action="store_true", default=False)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    task_langs = [x.strip() for x in args.task_langs.split(",") if x.strip()]
    if args.tasks.strip():
        tasks = [x.strip() for x in args.tasks.split(",") if x.strip()]
        evaluation = MTEB(tasks=tasks, task_langs=task_langs)
    else:
        evaluation = MTEB(task_langs=task_langs)

    model = DualMaskHFEmbedder(
        model_name_or_path=args.model_name_or_path,
        device=args.device,
        dtype=args.dtype,
        max_length=args.max_length,
        mask_mode=args.mask_mode,
        pooling_bidir=args.pooling_bidir,
        pooling_causal=args.pooling_causal,
        normalize=args.normalize,
        attn_impl=args.attn_impl,
        trust_remote_code=True,
    )

    run_name = f"{os.path.basename(args.model_name_or_path.rstrip('/'))}__{args.mask_mode}__pb-{args.pooling_bidir}__pc-{args.pooling_causal}"
    out_dir = os.path.join(args.output_dir, run_name)
    if args.no_overwrite and os.path.exists(out_dir):
        raise RuntimeError(f"output exists: {out_dir}")
    os.makedirs(out_dir, exist_ok=True)

    results = evaluation.run(
        model,
        output_folder=out_dir,
        batch_size=args.batch_size,
    )

    # Save a compact summary
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[done] saved to: {out_dir}")


if __name__ == "__main__":
    main()