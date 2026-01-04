#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate a CLM/MLM-capable decoder-only model as an embedding model on MTEB (v2 API).

Patch:
- Add robust sanity-check that verifies bidir mask is EFFECTIVE (hidden differs from causal).
- For bidir, prefer `or_mask_function=full_visible_mask_fn` (Transformers mask utils path).
  Fallback to `mask_function=full_visible_mask_fn` if `or_mask_function` isn't accepted.
- Print which mask kwargs were actually used.
"""

import os
import re
import json
import time
import argparse
import inspect
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Literal, Union

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

def all_false_mask_fn(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
    return False


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

def _forward_accepts_any_kwargs(model) -> bool:
    try:
        sig = inspect.signature(model.forward)
        for p in sig.parameters.values():
            if p.kind == inspect.Parameter.VAR_KEYWORD:
                return True
        return False
    except Exception:
        return True

def _forward_accepts_param(model, name: str) -> bool:
    """
    Conservative check:
    - If forward explicitly has `name`, True.
    - Else if forward has **kwargs, True (might be accepted but still ignored internally).
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

def to_iso_lang_script(lang: str) -> str:
    if lang == "eng":
        return "eng-Latn"
    if lang == "jpn":
        return "jpn-Jpan"
    return lang

ArrayLike = Union[np.ndarray, torch.Tensor]

def _to_torch(x: ArrayLike) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    return x

def cosine_sim_matrix(a: ArrayLike, b: ArrayLike) -> ArrayLike:
    a_is_np = isinstance(a, np.ndarray)
    b_is_np = isinstance(b, np.ndarray)
    ta = _to_torch(a).float()
    tb = _to_torch(b).float()

    if ta.dim() == 1:
        ta = ta.unsqueeze(0)
    if tb.dim() == 1:
        tb = tb.unsqueeze(0)

    ta = F.normalize(ta, p=2, dim=-1)
    tb = F.normalize(tb, p=2, dim=-1)
    sim = ta @ tb.transpose(0, 1)

    if a_is_np or b_is_np:
        return sim.cpu().numpy()
    return sim

def cosine_sim_pairwise(a: ArrayLike, b: ArrayLike) -> ArrayLike:
    a_is_np = isinstance(a, np.ndarray)
    b_is_np = isinstance(b, np.ndarray)
    ta = _to_torch(a).float()
    tb = _to_torch(b).float()

    if ta.dim() == 1:
        ta = ta.unsqueeze(0)
    if tb.dim() == 1:
        tb = tb.unsqueeze(0)

    ta = F.normalize(ta, p=2, dim=-1)
    tb = F.normalize(tb, p=2, dim=-1)
    sim = (ta * tb).sum(dim=-1)

    if a_is_np or b_is_np:
        return sim.cpu().numpy()
    return sim


# -------------------- config --------------------
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

    hf_token: Optional[str] = None

    # sanity / debug
    sanity_check: bool = True
    sanity_eps: float = 1e-6
    sanity_fail: bool = True            # if True: raise on failure; else: warn and continue
    sanity_text: str = "Hello world. This is a sanity check sentence for bidirectional masking."


class DualMaskHFEncoder:
    def __init__(self, cfg: DualMaskConfig):
        self.cfg = cfg

        tok_kwargs = dict(use_fast=True, trust_remote_code=True)
        if cfg.hf_token:
            tok_kwargs["token"] = cfg.hf_token
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, **tok_kwargs)

        torch_dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }[cfg.dtype]

        model_kwargs = dict(attn_implementation=cfg.attn_impl, trust_remote_code=True)
        if cfg.hf_token:
            model_kwargs["token"] = cfg.hf_token

        # NOTE: some HF versions want torch_dtype, not dtype
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                cfg.model_name_or_path,
                torch_dtype=torch_dtype,
                **model_kwargs,
            )
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(
                cfg.model_name_or_path,
                dtype=torch_dtype,
                **model_kwargs,
            )

        self.model.eval()
        self.device = torch.device(cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu")
        self.model.to(self.device)

        self.eos_token_id = getattr(self.tokenizer, "eos_token_id", None)

        mm = cfg.mask_mode
        if mm not in ("bidir", "causal", "concat", "avg"):
            raise ValueError("--mask_mode must be one of: bidir, causal, concat, avg")

        # bidir requires flex_attention
        need_bidir = mm in ("bidir", "concat", "avg")
        if need_bidir and cfg.attn_impl != "flex_attention":
            raise ValueError("bidir/concat/avg requires --attn_impl flex_attention.")

        # detect forward kwargs (best-effort)
        self._accepts_mask_function = _forward_accepts_param(self.model, "mask_function")
        self._accepts_or_mask_function = _forward_accepts_param(self.model, "or_mask_function")
        self._accepts_and_mask_function = _forward_accepts_param(self.model, "and_mask_function")

        # name + dim
        self.model_name = cfg.model_name_or_path
        self.embedding_dim = getattr(self.model.config, "hidden_size", None)

        self._mteb_model_meta: Optional[ModelMeta] = None

        # keep last used mask kwargs for debug prints
        self._last_mask_kwargs: Dict[str, Any] = {}

        # sanity check (only when bidir is requested)
        if need_bidir and cfg.sanity_check:
            self._sanity_check_bidir_effect(eps=float(cfg.sanity_eps), fail=bool(cfg.sanity_fail))

    @property
    def mteb_model_meta(self) -> ModelMeta:
        if self._mteb_model_meta is None:
            raise RuntimeError("mteb_model_meta is not set. Please set model.mteb_model_meta in main().")
        return self._mteb_model_meta

    @mteb_model_meta.setter
    def mteb_model_meta(self, meta: ModelMeta) -> None:
        self._mteb_model_meta = meta

    def similarity(self, embeddings1: ArrayLike, embeddings2: ArrayLike) -> ArrayLike:
        return cosine_sim_matrix(embeddings1, embeddings2)

    def similarity_pairwise(self, embeddings1: ArrayLike, embeddings2: ArrayLike) -> ArrayLike:
        return cosine_sim_pairwise(embeddings1, embeddings2)

    def get_sentence_embedding_dimension(self) -> int:
        if self.embedding_dim is None:
            raise RuntimeError("Could not infer hidden_size from model.config.")
        if self.cfg.mask_mode == "concat":
            return int(self.embedding_dim) * 2
        return int(self.embedding_dim)

    def _attn_impl_actual(self) -> str:
        # best-effort: transformers stores this in config
        return str(getattr(self.model.config, "_attn_implementation", getattr(self.model.config, "attn_implementation", "unknown")))

    @torch.no_grad()
    def _forward_hidden(self, input_ids, attention_mask, mode: Literal["bidir", "causal"]):
        """
        Key patch:
        - For bidir: try `or_mask_function=full_visible_mask_fn` first (preferred path).
          Fallback to `mask_function=full_visible_mask_fn` if needed.
        - For causal: keep default causal behavior; optionally pass mask_function=causal_mask_fn if supported.
        """
        assert self.cfg.attn_impl == "flex_attention", "bidir path requires flex_attention"

        base_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

        mask_kwargs: Dict[str, Any] = {}
        if mode == "bidir":
            # Preferred: OR the causal mask with "all True" -> full visible (except padding).
            if self._accepts_or_mask_function:
                mask_kwargs["or_mask_function"] = full_visible_mask_fn
            # fallback (older/custom code paths)
            elif self._accepts_mask_function:
                mask_kwargs["mask_function"] = full_visible_mask_fn
            else:
                # No known hook, rely on default (but then bidir likely impossible)
                mask_kwargs = {}

        else:  # causal
            if self._accepts_mask_function:
                mask_kwargs["mask_function"] = causal_mask_fn
            # else: do nothing (default causal)

        self._last_mask_kwargs = dict(mask_kwargs)

        # Try forward; if TypeError (unexpected kwargs), progressively drop.
        try:
            out = self.model(**base_kwargs, **mask_kwargs)
        except TypeError as e:
            # Drop unknown hooks and retry as fallback.
            retry_kwargs = dict(base_kwargs)
            out = self.model(**retry_kwargs)

        return out.hidden_states[-1]

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
        if x is None:
            return []
        if isinstance(x, (tuple, list)) and len(x) > 0 and isinstance(x[0], (tuple, list)):
            return [" ".join(map(str, xi)) for xi in x]
        return [str(t) for t in x]

    def encode(
        self,
        inputs,
        *,
        task_metadata=None,
        hf_split: str = "",
        hf_subset: str = "",
        prompt_type=None,
        **kwargs,
    ) -> np.ndarray:
        max_length = int(kwargs.get("max_length", self.cfg.max_length))
        all_vecs: List[np.ndarray] = []
        mm = self.cfg.mask_mode

        for batch in inputs:
            texts = batch.get("text", None)

            if texts is None:
                if "query" in batch:
                    texts = batch["query"]
                elif "title" in batch and "body" in batch:
                    texts = [f"{t} {b}".strip() for t, b in zip(batch["title"], batch["body"])]
                elif "text1" in batch and "text2" in batch:
                    texts = [f"{a} {b}".strip() for a, b in zip(batch["text1"], batch["text2"])]

            texts = self._coerce_text_list(texts)
            if not texts:
                raise ValueError(f"BatchedInput has no usable text field. keys={list(batch.keys())}")

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

    @torch.no_grad()
    def _sanity_check_bidir_effect(self, eps: float = 1e-6, fail: bool = True):
        """
        Verify that bidir != causal in practice.
        Prints:
          - which kwargs were used for bidir/causal
          - max|diff| on hidden (all tokens, and excluding last token)
          - max|diff| on pooled embedding (for current pooling setup)
        """
        # Build a batch that will produce padding + multiple tokens
        texts = [
            self.cfg.sanity_text,
            self.cfg.sanity_text + " Extra tail to change length a bit."
        ]
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=min(128, self.cfg.max_length),
            return_tensors="pt",
            add_special_tokens=True,
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        # causal
        h_c = self._forward_hidden(input_ids, attention_mask, "causal")
        causal_kwargs = dict(self._last_mask_kwargs)

        # bidir
        h_b = self._forward_hidden(input_ids, attention_mask, "bidir")
        bidir_kwargs = dict(self._last_mask_kwargs)

        # diffs
        diff = (h_b - h_c).abs()
        max_hidden = float(diff.max().item())
        max_hidden_excl_last = float(diff[:, :-1].max().item()) if diff.size(1) > 1 else max_hidden

        # pooled diffs under current pooling config
        emb_c = pool_hidden(
            h_c, attention_mask, self.cfg.pooling_causal,
            input_ids=input_ids, eos_token_id=self.eos_token_id
        )
        emb_b = pool_hidden(
            h_b, attention_mask, self.cfg.pooling_bidir,
            input_ids=input_ids, eos_token_id=self.eos_token_id
        )
        if self.cfg.normalize:
            emb_c = F.normalize(emb_c, p=2, dim=-1)
            emb_b = F.normalize(emb_b, p=2, dim=-1)
        max_emb = float((emb_b - emb_c).abs().max().item())

        print(
            "[sanity] attn_impl(requested)={} attn_impl(actual)={}".format(
                self.cfg.attn_impl, self._attn_impl_actual()
            )
        )
        print("[sanity] forward accepts: mask_function={} or_mask_function={} and_mask_function={}".format(
            self._accepts_mask_function, self._accepts_or_mask_function, self._accepts_and_mask_function
        ))
        print("[sanity] causal kwargs used =", causal_kwargs)
        print("[sanity] bidir  kwargs used =", bidir_kwargs)
        print("[sanity] max|bidir-causal| hidden(all)={:.3e} hidden(excl_last)={:.3e} emb={:.3e} eps={:.3e}".format(
            max_hidden, max_hidden_excl_last, max_emb, eps
        ))

        if max_hidden_excl_last <= eps:
            msg = (
                "Sanity check failed: bidir seems identical to causal.\n"
                "This likely means:\n"
                "  - bidir hooks (or_mask_function/mask_function) are ignored, OR\n"
                "  - backend still enforces causal masking, OR\n"
                "  - flex_attention is not actually applying the custom mask.\n"
                "Next steps:\n"
                "  - Ensure torch/transformers versions support flex_attention masking for this model.\n"
                "  - If still identical, you will need a model-side patch to disable causal enforcement.\n"
            )
            if fail:
                raise RuntimeError(msg)
            else:
                print("[sanity][WARN]\n" + msg)


# -------------------- runner --------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", type=str, required=True)

    # embedding behavior
    p.add_argument("--mask_mode", type=str, default="bidir", choices=["bidir", "causal", "concat", "avg"])
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
    p.add_argument("--languages", type=str, default="eng", help="comma-separated ISO639-3: e.g., eng,jpn")
    p.add_argument("--tasks", type=str, default="", help="comma-separated task names, e.g. STS22.v2,Banking77Classification.v2")
    p.add_argument("--benchmark", type=str, default="", help="benchmark name, e.g. 'MTEB(eng, v2)'")

    # results/cache
    p.add_argument("--output_dir", type=str, default="mteb_results")
    p.add_argument("--overwrite_strategy", type=str, default="only-missing",
                   choices=["only-missing", "always", "never"])

    # Optional: pin model revision
    p.add_argument("--model_revision", type=str, default=None)

    # sanity check knobs
    p.add_argument("--no_sanity_check", action="store_true", default=False)
    p.add_argument("--sanity_eps", type=float, default=1e-6)
    p.add_argument("--sanity_fail", action="store_true", default=True)
    p.add_argument("--sanity_text", type=str, default="Hello world. This is a sanity check sentence for bidirectional masking.")

    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    import mteb
    from mteb.cache import ResultCache

    langs = [x.strip() for x in args.languages.split(",") if x.strip()]

    # --- select tasks/benchmark ---
    if args.tasks.strip():
        task_names = [x.strip() for x in args.tasks.split(",") if x.strip()]
        tasks = mteb.get_tasks(tasks=task_names, languages=langs)
    else:
        bench_name = args.benchmark.strip() or f"MTEB({langs[0]}, v2)"
        tasks = mteb.get_benchmark(bench_name)

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
        sanity_check=(not args.no_sanity_check),
        sanity_eps=float(args.sanity_eps),
        sanity_fail=bool(args.sanity_fail),
        sanity_text=args.sanity_text,
    )
    model = DualMaskHFEncoder(cfg)

    # --- fill ModelMeta ---
    n_params = sum(p.numel() for p in model.model.parameters())
    max_tokens = getattr(model.model.config, "max_position_embeddings", None) or args.max_length
    embed_dim = model.get_sentence_embedding_dimension()

    model.mteb_model_meta = ModelMeta(
        loader=None,
        loader_kwargs={},
        name=args.model_name_or_path,
        revision=args.model_revision,
        release_date=None,
        languages=[to_iso_lang_script(l) for l in langs],
        n_parameters=int(n_params),
        memory_usage_mb=None,
        max_tokens=float(max_tokens) if max_tokens is not None else None,
        embed_dim=int(embed_dim),
        license=None,
        open_weights=None,
        public_training_code=None,
        public_training_data=None,
        framework=["PyTorch"],
        reference=None,
        similarity_fn_name="cosine",
        use_instructions=False,
        training_datasets=None,
    )

    run_name = (
        f"{safe_name(args.model_name_or_path)}__"
        f"{args.mask_mode}__"
        f"pb-{args.pooling_bidir}__pc-{args.pooling_causal}__"
        f"ml{args.max_length}"
    )
    out_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)

    cache = ResultCache(cache_path=out_dir)

    t0 = time.time()
    results = mteb.evaluate(
        model,
        tasks,
        cache=cache,
        overwrite_strategy=args.overwrite_strategy,
        encode_kwargs={"max_length": args.max_length},
    )

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
