#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import argparse
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Iterable

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# -------------------------- FlexAttention mask functions --------------------------
def causal_mask_fn(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
    return kv_idx <= q_idx

def full_visible_mask_fn(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
    return True


# -------------------------- Pooling --------------------------
def pool_hidden(
    hidden: torch.Tensor,            # (B,S,H)
    attention_mask: torch.Tensor,    # (B,S) 1 for valid
    pooling: str,
    eos_token_id: Optional[int],
    input_ids: Optional[torch.Tensor] = None,  # (B,S)
) -> torch.Tensor:
    pooling = pooling.lower()
    B, S, H = hidden.shape
    am = attention_mask.to(hidden.device)

    if pooling == "mean":
        denom = am.sum(dim=1, keepdim=True).clamp_min(1)
        x = (hidden * am.unsqueeze(-1)).sum(dim=1) / denom
        return x

    if pooling == "last":
        # last valid token
        idx = (am.sum(dim=1) - 1).clamp_min(0).long()  # (B,)
        return hidden[torch.arange(B, device=hidden.device), idx]

    if pooling == "cls":
        # first token
        return hidden[:, 0, :]

    if pooling == "eos":
        if eos_token_id is None or input_ids is None:
            raise ValueError("pooling=eos には eos_token_id と input_ids が必要です。")
        # last eos in each row, fallback to last valid
        out = torch.empty((B, H), device=hidden.device, dtype=hidden.dtype)
        for b in range(B):
            row = input_ids[b]
            eos_pos = torch.where(row == eos_token_id)[0]
            if eos_pos.numel() > 0:
                j = int(eos_pos[-1].item())
            else:
                j = int((am[b].sum() - 1).clamp_min(0).item())
            out[b] = hidden[b, j]
        return out

    raise ValueError(f"Unknown pooling: {pooling}")


# -------------------------- HF embedder wrapper for MTEB --------------------------
@dataclass
class HFEmbedderConfig:
    model_name_or_path: str
    attn_impl: str = "flex_attention"  # must be flex_attention for bidir
    mask_mode: str = "bidir"           # "bidir" or "causal"
    pooling: str = "mean"              # mean/last/eos/cls
    max_length: int = 512
    dtype: str = "bf16"                # bf16/fp16/fp32
    device: str = "cuda"               # cuda/cpu
    batch_size: int = 32
    normalize: bool = True


class HFEmbedder:
    """
    MTEBが期待する最小インタフェース:
      - encode(sentences: List[str], **kwargs) -> np.ndarray [N,D]
    """
    def __init__(self, cfg: HFEmbedderConfig):
        self.cfg = cfg

        self.tok = AutoTokenizer.from_pretrained(
            cfg.model_name_or_path,
            use_fast=True,
            trust_remote_code=True,
        )

        torch_dtype = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }[cfg.dtype]

        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name_or_path,
            attn_implementation=cfg.attn_impl,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )

        self.model.eval()
        self.device = torch.device(cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu")
        self.model.to(self.device)

        # for MTEB metadata (なくても大抵動くが、あると親切)
        self.model_name = cfg.model_name_or_path
        self.embedding_dim = getattr(self.model.config, "hidden_size", None)

        if cfg.mask_mode not in ("bidir", "causal"):
            raise ValueError("mask_mode must be 'bidir' or 'causal'")

        if cfg.mask_mode == "bidir" and cfg.attn_impl != "flex_attention":
            raise ValueError("bidir(full-visible) には --attn_impl flex_attention が必要です。")

    def get_sentence_embedding_dimension(self) -> int:
        if self.embedding_dim is None:
            raise RuntimeError("hidden_size を取得できませんでした。")
        return int(self.embedding_dim)

    @torch.no_grad()
    def encode(
        self,
        sentences: List[str],
        **kwargs,
    ) -> np.ndarray:
        bs = int(kwargs.get("batch_size", self.cfg.batch_size))
        max_length = int(kwargs.get("max_length", self.cfg.max_length))
        normalize = bool(kwargs.get("normalize", self.cfg.normalize))

        outs: List[np.ndarray] = []
        for i in range(0, len(sentences), bs):
            batch = sentences[i:i+bs]
            enc = self.tok(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
                add_special_tokens=True,
            )
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)

            # Qwen系 + flex_attention 前提：mask_function を渡して attention 可視性を切り替え
            mask_fn = full_visible_mask_fn if self.cfg.mask_mode == "bidir" else causal_mask_fn

            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
                mask_function=mask_fn if self.cfg.attn_impl == "flex_attention" else None,
            )

            hidden = out.hidden_states[-1]  # (B,S,H)
            pooled = pool_hidden(
                hidden=hidden,
                attention_mask=attention_mask,
                pooling=self.cfg.pooling,
                eos_token_id=getattr(self.tok, "eos_token_id", None),
                input_ids=input_ids,
            )  # (B,H)

            if normalize:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)

            outs.append(pooled.detach().float().cpu().numpy())

        return np.concatenate(outs, axis=0)


# -------------------------- MTEB runner (new API) --------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)

    parser.add_argument("--attn_impl", type=str, default="flex_attention")
    parser.add_argument("--mask_mode", type=str, default="bidir", choices=["bidir", "causal"])
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "last", "eos", "cls"])

    # MTEB selection
    parser.add_argument("--tasks", type=str, default="",
                        help="カンマ区切りタスク名 (例: Banking77Classification.v2,STSBenchmark)")
    parser.add_argument("--benchmark", type=str, default="",
                        help="ベンチマーク名 (例: MTEB(en) など。環境のmtebが提供する名前に依存)")
    parser.add_argument("--languages", type=str, default="en",
                        help="ISO 639-3 を推奨（例: en は曖昧になりうるので eng 推奨。カンマ区切り）")

    # encode config
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument("--no_normalize", action="store_true", default=False)

    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    parser.add_argument("--output_folder", type=str, default="mteb_results")
    parser.add_argument("--overwrite", action="store_true", default=False)

    args = parser.parse_args()
    if args.no_normalize:
        args.normalize = False

    # reduce noisy TF logs if tensorflow is installed somewhere
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    import mteb  # new API: get_tasks / evaluate

    langs = [x.strip() for x in args.languages.split(",") if x.strip()]
    # NOTE: mteb docs/issue では languages は ISO 639-3 を推奨していることがある
    # (例: 'eng', 'jpn' など)
    # ここはユーザ指定をそのまま渡す

    # --- select tasks ---
    if args.tasks.strip():
        task_names = [x.strip() for x in args.tasks.split(",") if x.strip()]
        tasks = mteb.get_tasks(tasks=task_names, languages=langs)
    elif args.benchmark.strip():
        tasks = mteb.get_tasks(benchmark=args.benchmark.strip(), languages=langs)
    else:
        raise ValueError("Either --tasks or --benchmark must be specified.")

    cfg = HFEmbedderConfig(
        model_name_or_path=args.model_name_or_path,
        attn_impl=args.attn_impl,
        mask_mode=args.mask_mode,
        pooling=args.pooling,
        max_length=args.max_length,
        dtype=args.dtype,
        device=args.device,
        batch_size=args.batch_size,
        normalize=args.normalize,
    )
    model = HFEmbedder(cfg)

    run_name = f"{safe_name(args.model_name_or_path)}__{args.mask_mode}__{args.pooling}"
    out_dir = os.path.join(args.output_folder, run_name)
    os.makedirs(out_dir, exist_ok=True)

    # --- evaluate ---
    t0 = time.time()
    results = mteb.evaluate(
        model,
        tasks=tasks,
        output_folder=out_dir,
        overwrite=args.overwrite,
        encode_kwargs={
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "normalize": args.normalize,
        },
    )

    # also dump a top-level json for convenience
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[done] output_folder={out_dir}  elapsed={(time.time()-t0):.1f}s")


def safe_name(x: str) -> str:
    return x.replace("/", "_").replace(":", "_").replace(" ", "_")


if __name__ == "__main__":
    main()
