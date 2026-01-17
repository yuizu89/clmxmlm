from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Literal, Union

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# training-side masking utils
from ..masking import MaskController, get_backbone_from_causallm


Pooling = Literal["mean", "last", "cls", "eos"]
ArrayLike = Union[np.ndarray, torch.Tensor]


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


def _forward_accepts_param(model, name: str) -> bool:
    """Best-effort signature check."""
    try:
        sig = inspect.signature(model.forward)
        if name in sig.parameters:
            return True
        return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    except Exception:
        return True


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


def _find_subseq(haystack: List[int], needle: List[int]) -> Optional[int]:
    """Return first index where needle occurs in haystack (naive search)."""
    if not needle or not haystack or len(needle) > len(haystack):
        return None
    n = len(needle)
    for i in range(len(haystack) - n + 1):
        if haystack[i : i + n] == needle:
            return i
    return None


@dataclass
class DualMaskConfig:
    model_name_or_path: str
    attn_impl: str = "flash_attention_2"
    device: str = "cuda"
    dtype: str = "bf16"                  # bf16/fp16/fp32
    max_length: int = 512
    normalize: bool = True

    mask_mode: str = "bidir"             # bidir/causal/concat/avg
    pooling_bidir: Pooling = "mean"
    pooling_causal: Pooling = "mean"

    hf_token: Optional[str] = None

    # sanity / debug
    sanity_check: bool = True
    sanity_eps: float = 1e-6
    sanity_fail: bool = True
    sanity_text: str = "Hello world. This is a sanity check sentence for bidirectional masking."


class DualMaskHFEncoder:
    """
    MTEB v2 compatible embedding model wrapper.

    Key masking idea (training-aligned):
      - toggle module.is_causal via MaskController (cached is_causal modules)
      - pass is_causal kw if backbone accepts it (best-effort)
      - keep attention_mask as 2D (B,S)
    """

    def __init__(self, cfg: DualMaskConfig):
        self.cfg = cfg

        tok_kwargs = dict(use_fast=True, trust_remote_code=True)
        if cfg.hf_token:
            tok_kwargs["token"] = cfg.hf_token
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, **tok_kwargs)

        # decoder-only models sometimes lack pad token; make padding safe
        if getattr(self.tokenizer, "pad_token_id", None) is None:
            if getattr(self.tokenizer, "eos_token", None) is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        torch_dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }[cfg.dtype]

        model_kwargs = dict(attn_implementation=cfg.attn_impl, trust_remote_code=True)
        if cfg.hf_token:
            model_kwargs["token"] = cfg.hf_token

        try:
            self.model = AutoModel.from_pretrained(
                cfg.model_name_or_path,
                torch_dtype=torch_dtype,
                **model_kwargs,
            )
        except TypeError:
            self.model = AutoModel.from_pretrained(
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

        self.backbone = get_backbone_from_causallm(self.model)
        self.controller = MaskController(self.backbone)

        self.model_name = cfg.model_name_or_path
        self.embedding_dim = getattr(self.model.config, "hidden_size", None)

        if cfg.sanity_check:
            self._sanity_check_suffix_effect_hidden(eps=float(cfg.sanity_eps), fail=bool(cfg.sanity_fail))

    # --- MTEB-required methods ---
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

    # --- helpers ---
    def _attn_impl_actual(self) -> str:
        return str(getattr(self.model.config, "_attn_implementation",
                           getattr(self.model.config, "attn_implementation", "unknown")))

    def _pick_last_hidden(self, out) -> torch.Tensor:
        if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
            return out.last_hidden_state
        if hasattr(out, "hidden_states") and out.hidden_states is not None:
            return out.hidden_states[-1]
        raise RuntimeError("Model output has neither last_hidden_state nor hidden_states.")

    @torch.no_grad()
    def _backbone_forward_hidden(self, input_ids, attention_mask, is_causal: bool):
        base_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
            output_hidden_states=True,
        )

        used_is_causal_kw = False
        with self.controller.set(is_causal):
            if self.controller.accepts_is_causal:
                try:
                    out = self.controller.backbone_forward(is_causal=is_causal, **base_kwargs)
                    used_is_causal_kw = True
                except TypeError:
                    out = self.controller.backbone_forward(is_causal=None, **base_kwargs)
            else:
                out = self.controller.backbone_forward(is_causal=None, **base_kwargs)

        h = self._pick_last_hidden(out)
        return h, used_is_causal_kw

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

        want_causal = (mode == "causal")
        h, _ = self._backbone_forward_hidden(input_ids, attention_mask, is_causal=want_causal)

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
    def _sanity_check_suffix_effect_hidden(self, eps: float = 1e-6, fail: bool = True):
        """
        Training-style sanity check ("suffix effect") using hidden states only.

        Two sequences share same prefix but different suffix.
        Compare hidden at a prefix position between seqs:
          causal => diff ~ 0
          bidir  => diff >> 0
        """
        prefix = "Hello world. This is the SAME prefix for both sequences. "
        s1 = prefix + "SUFFIX_A: cats are wonderful and playful."
        s2 = prefix + "SUFFIX_B: quantum field theory is quite abstract."

        enc = self.tokenizer(
            [s1, s2],
            padding=True,
            truncation=True,
            max_length=min(256, self.cfg.max_length),
            return_tensors="pt",
            add_special_tokens=True,
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        pref_ids = self.tokenizer(prefix, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
        full0 = input_ids[0].tolist()
        start = _find_subseq(full0, pref_ids)
        if start is None:
            start = 1 if len(full0) > 1 else 0

        rel = max(1, min(len(pref_ids) - 2, 16))
        pos = start + rel

        valid_len0 = int(attention_mask[0].sum().item())
        pos = max(0, min(pos, max(0, valid_len0 - 2)))

        h_c, used_kw_c = self._backbone_forward_hidden(input_ids, attention_mask, is_causal=True)
        h_b, used_kw_b = self._backbone_forward_hidden(input_ids, attention_mask, is_causal=False)

        diff_c = float((h_c[0, pos, :] - h_c[1, pos, :]).abs().max().item())
        diff_b = float((h_b[0, pos, :] - h_b[1, pos, :]).abs().max().item())
        diff_toggle = float((h_b[0, pos, :] - h_c[0, pos, :]).abs().max().item())

        print("[mask sanity] attn_impl(requested)={} attn_impl(actual)={}".format(
            self.cfg.attn_impl, self._attn_impl_actual()
        ))
        print("[mask sanity] accepts is_causal={} (kw used: causal={}, bidir={})".format(
            bool(self.controller.accepts_is_causal), used_kw_c, used_kw_b
        ))
        print("[mask sanity] n_is_causal_modules={} backbone_type={}".format(
            int(len(self.controller.mods)), str(type(self.backbone))
        ))
        print("[mask sanity] prefix_len_tokens(no_special)={} start={} pos={}".format(
            len(pref_ids), start, pos
        ))
        print("[mask sanity] diff_causal(prefix-pos between seqs)={:.6g} diff_bidir={:.6g} diff_toggle(seq0)={:.6g} eps={:.3e}".format(
            diff_c, diff_b, diff_toggle, eps
        ))

        problems = []
        if diff_c > eps:
            problems.append(
                f"causal seems affected by suffix (diff_causal={diff_c:.6g} > eps={eps:.3e}). "
                "This suggests the backend/mask is not truly causal in this path."
            )
        if diff_b <= eps:
            problems.append(
                f"bidir seems identical to causal (diff_bidir={diff_b:.6g} <= eps={eps:.3e}). "
                "This suggests bidir toggling is not taking effect (still causal)."
            )

        if problems:
            msg = (
                "Sanity check failed (suffix effect):\n"
                + "\n".join([f"  - {p}" for p in problems])
                + "\n\nNext steps:\n"
                "  - Try --attn_impl sdpa (often respects is_causal more consistently).\n"
                "  - Ensure you are passing 2D attention_mask (this script does).\n"
                "  - If still failing, a model-side patch may be required (e.g., disabling internal causal mask enforcement).\n"
            )
            if fail:
                raise RuntimeError(msg)
            else:
                print("[mask sanity][WARN]\n" + msg)
