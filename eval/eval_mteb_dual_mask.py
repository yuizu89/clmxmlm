#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MTEB v2-compatible evaluation script for a CLM/MLM-capable decoder-only model
used as an embedding model.

Restored original features:
- mask_mode: bidir / causal / concat / avg
- pooling_bidir and pooling_causal can be different
- normalization control
- works with FlexAttention mask_function (bidir) and plain causal attention

MTEB v2 changes:
- Use mteb.get_tasks / mteb.get_benchmark
- Use mteb.evaluate(..., cache=ResultCache(cache_path=...), overwrite_strategy=...)
  (NO output_folder argument to evaluate)
"""

import os
import json
import time
import argparse
from typing import List, Optional, Literal, Any, Dict, Iterable, Union

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


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
        idx = (am.sum(dim=1) - 1).clamp(min=0).to(torch.long)  # (B,)
        return last_hidden[torch.arange(B, device=last_hidden.device), idx]

    if method == "cls":
        return last_hidden[:, 0, :]

    raise ValueError(f"unknown pooling: {method}")


def safe_name(x: str) -> str:
    return x.replace("/", "_").replace(":", "_").replace(" ", "_")


def _to_jsonable(obj: Any) -> Any:
    """Best-effort conversion for ModelResult etc."""
    if hasattr(obj, "model_dump"):  # pydantic v2
        return obj.model_dump()
    if hasattr(obj, "dict"):        # pydantic v1
        return obj.dict()
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


# -------------------- embedding wrapper for MTEB (v2 EncoderProtocol-friendly) --------------------
class DualMaskHFEmbedder:
    """
    Supports both:
    - MTEB v2 style: encode(inputs: DataLoader[BatchedInput], task_metadata, hf_split, hf_subset, ...)
    - SentenceTransformers-like: encode(sentences: List[str], batch_size=..., ...)

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
        self.device = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        self.max_length = int(max_length)
        self.mask_mode = mask_mode
        self.pooling_bidir = pooling_bidir
        self.pooling_causal = pooling_causal
        self.normalize = bool(normalize)
        self.attn_impl = attn_impl
        self.trust_remote_code = trust_remote_code

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, use_fast=True, trust_remote_code=trust_remote_code
        )

        # Ensure pad_token exists for padding=True
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                # last resort: add pad token
                self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

        torch_dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }[dtype]

        # transformers newer versions prefer `dtype=`; keep compatibility
        model_kwargs = dict(
            attn_implementation=attn_impl,
            trust_remote_code=trust_remote_code,
        )
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                dtype=torch_dtype,
                **model_kwargs,
            )
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch_dtype,
                **model_kwargs,
            )

        # if we added pad token newly, resize embeddings
        if len(self.tokenizer) > self.model.get_input_embeddings().weight.shape[0]:
            self.model.resize_token_embeddings(len(self.tokenizer))
            if hasattr(self.model, "tie_weights"):
                self.model.tie_weights()

        self.model.to(self.device)
        self.model.eval()

        # bidir requires flex_attention mask_function to override visibility
        needs_bidir = self.mask_mode in ("bidir", "concat", "avg")
        if needs_bidir and self.attn_impl != "flex_attention":
            raise ValueError("mask_mode が bidir/concat/avg の場合、--attn_impl flex_attention が必要です。")

    @torch.no_grad()
    def _forward_last_hidden(self, input_ids, attention_mask, mask_function):
        kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        if self.attn_impl == "flex_attention":
            kwargs["mask_function"] = mask_function
        out = self.model(**kwargs)
        return out.hidden_states[-1]  # (B,S,H)

    @torch.no_grad()
    def _encode_texts_batch(self, texts: List[str]) -> torch.Tensor:
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            add_special_tokens=True,
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        mm = self.mask_mode

        if mm == "bidir":
            h = self._forward_last_hidden(input_ids, attention_mask, full_visible_mask_fn)
            v = pool_hidden(h, attention_mask, self.pooling_bidir)

            if self.normalize:
                v = torch.nn.functional.normalize(v, p=2, dim=-1)
            return v

        if mm == "causal":
            # For non-flex attention, this is just standard causal LM attention
            h = self._forward_last_hidden(input_ids, attention_mask, causal_mask_fn)
            v = pool_hidden(h, attention_mask, self.pooling_causal)

            if self.normalize:
                v = torch.nn.functional.normalize(v, p=2, dim=-1)
            return v

        # mm in ("concat", "avg") -> compute both in one tokenization
        h_b = self._forward_last_hidden(input_ids, attention_mask, full_visible_mask_fn)
        v_b = pool_hidden(h_b, attention_mask, self.pooling_bidir)

        h_c = self._forward_last_hidden(input_ids, attention_mask, causal_mask_fn)
        v_c = pool_hidden(h_c, attention_mask, self.pooling_causal)

        if self.normalize:
            v_b = torch.nn.functional.normalize(v_b, p=2, dim=-1)
            v_c = torch.nn.functional.normalize(v_c, p=2, dim=-1)

        if mm == "concat":
            # keep original semantics: concat normalized parts (not renormalizing the concatenated vector)
            return torch.cat([v_b, v_c], dim=1)

        if mm == "avg":
            v = (v_b + v_c) * 0.5
            if self.normalize:
                v = torch.nn.functional.normalize(v, p=2, dim=-1)
            return v

        raise ValueError(f"unknown mask_mode: {mm}")

    def _extract_texts_from_batch(self, batch: Any) -> List[str]:
        """
        MTEB v2 DataLoader[BatchedInput] typically yields a dict with key 'text' (list[str]). :contentReference[oaicite:6]{index=6}
        Be tolerant to a few variants.
        """
        if isinstance(batch, dict):
            if "text" in batch:
                x = batch["text"]
                if isinstance(x, str):
                    return [x]
                return list(x)
            # fallback: (title, body) style
            if "title" in batch and "body" in batch:
                titles = batch["title"]
                bodies = batch["body"]
                return [f"{t}\n{b}" for t, b in zip(titles, bodies)]
            # last resort: stringify whole dict
            return [str(batch)]
        if isinstance(batch, str):
            return [batch]
        if isinstance(batch, (list, tuple)):
            # list[str]
            if len(batch) == 0:
                return []
            if isinstance(batch[0], str):
                return list(batch)
            return [str(x) for x in batch]
        return [str(batch)]

    def encode(
        self,
        inputs: Any,
        task_metadata: Any = None,
        hf_split: Optional[str] = None,
        hf_subset: Optional[str] = None,
        prompt_type: Any = None,
        **kwargs,
    ) -> np.ndarray:
        """
        MTEB v2 will call this with inputs as a DataLoader[BatchedInput]. :contentReference[oaicite:7]{index=7}
        We also support list[str] for convenience.
        """
        # If SentenceTransformers-like call: encode(sentences: List[str], batch_size=...)
        if isinstance(inputs, (list, tuple)) and (len(inputs) == 0 or isinstance(inputs[0], str)):
            sentences = list(inputs)
            bs = int(kwargs.get("batch_size", 32))
            outs: List[np.ndarray] = []
            for i in range(0, len(sentences), bs):
                vec = self._encode_texts_batch(sentences[i:i+bs])
                outs.append(vec.detach().float().cpu().numpy())
            return np.concatenate(outs, axis=0) if outs else np.zeros((0, 0), dtype=np.float32)

        # MTEB v2: DataLoader or iterable of batches
        outs: List[np.ndarray] = []
        for batch in inputs:
            texts = self._extract_texts_from_batch(batch)
            if not texts:
                continue
            vec = self._encode_texts_batch(texts)
            outs.append(vec.detach().float().cpu().numpy())

        return np.concatenate(outs, axis=0) if outs else np.zeros((0, 0), dtype=np.float32)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", type=str, required=True)

    # embedding behavior
    p.add_argument("--mask_mode", type=str, default="bidir",
                   choices=["bidir", "causal", "concat", "avg"])
    p.add_argument("--pooling_bidir", type=str, default="mean", choices=["mean", "last", "cls"])
    p.add_argument("--pooling_causal", type=str, default="mean", choices=["mean", "last", "cls"])
    p.add_argument("--normalize", action="store_true", default=True)
    p.add_argument("--no_normalize", action="store_true", default=False)
    p.add_argument("--max_length", type=int, default=512)

    # runtime
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--attn_impl", type=str, default="flex_attention",
                   choices=["flex_attention", "sdpa", "flash_attention_2", "eager"])

    # MTEB selection
    p.add_argument("--languages", type=str, default="eng",
                   help="comma-separated language codes (MTEB recommends ISO 639-3 like 'eng','jpn').")
    p.add_argument("--tasks", type=str, default="",
                   help="comma-separated task names (e.g., Banking77Classification,STSBenchmark).")
    p.add_argument("--benchmark", type=str, default="",
                   help="benchmark name (e.g., 'MTEB(eng, v2)'). If empty and tasks empty, try to auto-pick.")
    p.add_argument("--output_dir", type=str, default="mteb_results")

    # overwrite strategy (MTEB v2)
    p.add_argument("--overwrite_strategy", type=str, default="only-missing",
                   choices=["always", "never", "only-missing", "only-cache"])
    p.add_argument("--no_overwrite", action="store_true", default=False,
                   help="legacy behavior: if run folder exists and non-empty, raise error.")

    # optional predictions
    p.add_argument("--save_predictions", action="store_true", default=False)
    p.add_argument("--show_progress_bar", action="store_true", default=True)
    p.add_argument("--no_progress_bar", action="store_true", default=False)

    # encode kwargs
    p.add_argument("--encode_batch_size", type=int, default=32,
                   help="passed via encode_kwargs; effective mainly for list[str] encode.")
    return p.parse_args()


def _auto_pick_benchmark(mteb_mod, langs: List[str]) -> Optional[str]:
    """
    Try to find a benchmark like 'MTEB(eng, v2)' for the given language.
    Falls back to None if not found.
    """
    try:
        get_benchmarks = getattr(mteb_mod, "get_benchmarks", None)
        if get_benchmarks is None:
            return None
        bms = get_benchmarks()
        # each benchmark has .name and .aliases (per docs)
        target = langs[0] if len(langs) == 1 else None

        # prefer v2
        candidates = []
        for bm in bms:
            name = getattr(bm, "name", "")
            aliases = getattr(bm, "aliases", []) or []
            keys = [name] + list(aliases)
            if target is not None:
                if any(f"({target}" in k for k in keys):
                    candidates.append(name)
            else:
                # multilingual heuristic
                if any(("multi" in k.lower() or "multilingual" in k.lower()) for k in keys):
                    candidates.append(name)

        if not candidates:
            return None

        v2 = [c for c in candidates if "v2" in c.lower()]
        return sorted(v2)[-1] if v2 else sorted(candidates)[-1]
    except Exception:
        return None


def main():
    args = parse_args()
    if args.no_normalize:
        args.normalize = False
    if args.no_progress_bar:
        args.show_progress_bar = False

    # reduce TF-related noise if TF is installed somewhere
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

    import mteb
    from mteb.cache import ResultCache

    langs = [x.strip() for x in args.languages.split(",") if x.strip()]
    if not langs:
        raise ValueError("--languages is empty.")

    # ---- select tasks ----
    tasks = None
    benchmark_name_used = None

    # get_tasks fn name differs in docs/examples; be tolerant
    get_tasks_fn = getattr(mteb, "get_tasks", None) or getattr(mteb, "get_task", None)
    if get_tasks_fn is None:
        raise RuntimeError("mteb.get_tasks / mteb.get_task が見つかりません。mteb のバージョンを確認してください。")

    if args.tasks.strip():
        task_names = [x.strip() for x in args.tasks.split(",") if x.strip()]
        tasks = get_tasks_fn(task_names, languages=langs)
    else:
        bm_name = args.benchmark.strip()
        if not bm_name:
            bm_name = _auto_pick_benchmark(mteb, langs)
        if bm_name:
            benchmark_name_used = bm_name
            bm = mteb.get_benchmark(bm_name)
            tasks = bm.tasks
        else:
            # fallback: all tasks matching languages (can be large)
            tasks = get_tasks_fn(tasks=None, languages=langs)

    # ---- model ----
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

    run_name = (
        f"{safe_name(args.model_name_or_path)}"
        f"__{args.mask_mode}"
        f"__pb-{args.pooling_bidir}"
        f"__pc-{args.pooling_causal}"
        f"__lang-{'-'.join(langs)}"
    )
    if benchmark_name_used:
        run_name += f"__bm-{safe_name(benchmark_name_used)}"

    out_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.no_overwrite and os.path.exists(out_dir) and os.listdir(out_dir):
        raise RuntimeError(f"output exists (non-empty) and --no_overwrite is set: {out_dir}")
    os.makedirs(out_dir, exist_ok=True)

    # ---- cache & prediction folder ----
    cache = ResultCache(cache_path=out_dir)
    pred_dir = os.path.join(out_dir, "predictions") if args.save_predictions else None

    # ---- run evaluate (MTEB v2) ----
    t0 = time.time()
    result = mteb.evaluate(
        model,
        tasks=tasks,
        cache=cache,
        overwrite_strategy=args.overwrite_strategy,
        prediction_folder=pred_dir,
        show_progress_bar=args.show_progress_bar,
        encode_kwargs={
            "batch_size": args.encode_batch_size,
            "show_progress_bar": args.show_progress_bar,
        },
    )

    # Save artifacts
    with open(os.path.join(out_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name_or_path": args.model_name_or_path,
                "mask_mode": args.mask_mode,
                "pooling_bidir": args.pooling_bidir,
                "pooling_causal": args.pooling_causal,
                "normalize": args.normalize,
                "max_length": args.max_length,
                "device": args.device,
                "dtype": args.dtype,
                "attn_impl": args.attn_impl,
                "languages": langs,
                "tasks": args.tasks,
                "benchmark": args.benchmark,
                "benchmark_used": benchmark_name_used,
                "overwrite_strategy": args.overwrite_strategy,
                "save_predictions": args.save_predictions,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(result), f, ensure_ascii=False, indent=2)

    print(f"[done] out_dir={out_dir}")
    print(f"       elapsed={(time.time() - t0):.1f}s")


if __name__ == "__main__":
    main()
