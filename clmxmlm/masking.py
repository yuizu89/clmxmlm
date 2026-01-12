from __future__ import annotations

import inspect
from contextlib import contextmanager
from typing import List, Optional, Dict, Any

import torch


def unwrap_model(m: torch.nn.Module) -> torch.nn.Module:
    return getattr(m, "module", m)


def get_backbone_from_causallm(model: torch.nn.Module) -> torch.nn.Module:
    """
    Typical HF CausalLM structure:
      - model.model is backbone (e.g., Qwen3ForCausalLM.model -> Qwen3Model)
    Also try base_model / transformer fallbacks.
    """
    m = unwrap_model(model)
    if hasattr(m, "model") and isinstance(getattr(m, "model"), torch.nn.Module):
        return getattr(m, "model")
    if hasattr(m, "base_model") and isinstance(getattr(m, "base_model"), torch.nn.Module):
        return getattr(m, "base_model")
    if hasattr(m, "transformer") and isinstance(getattr(m, "transformer"), torch.nn.Module):
        return getattr(m, "transformer")
    return m


def forward_accepts_param(mod: torch.nn.Module, name: str) -> bool:
    """
    best-effort: forward signature contains name OR has **kwargs
    """
    try:
        sig = inspect.signature(mod.forward)
        if name in sig.parameters:
            return True
        return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    except Exception:
        return True


def cache_is_causal_modules(root: torch.nn.Module) -> List[torch.nn.Module]:
    """
    Cache all submodules that have attribute `is_causal`.
    """
    return [m for m in root.modules() if hasattr(m, "is_causal")]


@contextmanager
def force_is_causal_cached(mods: List[torch.nn.Module], flag: bool):
    """
    Fast toggle of module.is_causal for cached modules.
    """
    old = []
    for m in mods:
        try:
            old.append(bool(getattr(m, "is_causal")))
            setattr(m, "is_causal", bool(flag))
        except Exception:
            old.append(None)
    try:
        yield
    finally:
        for m, v in zip(mods, old):
            if v is None:
                continue
            try:
                setattr(m, "is_causal", v)
            except Exception:
                pass


class MaskController:
    """
    A reusable controller to toggle causal/bidir behavior for decoder-only backbones.

    Key idea (works for Qwen3 + FlashAttention2 as you verified):
      - set module.is_causal = True/False (some backends consult it)
      - pass forward kw: is_causal=True/False if backbone accepts it
      - keep attention_mask as 2D (B,S). (FA2-friendly)
    """

    def __init__(self, backbone: torch.nn.Module):
        self.backbone = backbone
        self.mods = cache_is_causal_modules(backbone)
        self.accepts_is_causal = forward_accepts_param(backbone, "is_causal")

    @contextmanager
    def set(self, is_causal: bool):
        """
        Context manager: force is_causal for all relevant modules during forward.
        """
        with force_is_causal_cached(self.mods, is_causal):
            yield

    def backbone_forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        is_causal: Optional[bool],
        use_cache: bool = False,
        return_dict: bool = True,
        output_hidden_states: bool = False,
        **extra,
    ):
        """
        Call backbone forward with best-effort is_causal propagation.
        """
        kwargs: Dict[str, Any] = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            return_dict=return_dict,
            output_hidden_states=output_hidden_states,
        )
        kwargs.update(extra)

        if is_causal is not None and self.accepts_is_causal:
            try:
                return self.backbone(**kwargs, is_causal=bool(is_causal))
            except TypeError:
                # Some implementations accept it via **kwargs but still error; fallback.
                return self.backbone(**kwargs)

        return self.backbone(**kwargs)


@torch.no_grad()
def sanity_check_suffix_effect(
    *,
    tokenizer,
    backbone: torch.nn.Module,
    lm_head: torch.nn.Module,
    device: torch.device,
    controller: Optional[MaskController] = None,
    attn_mask_2d: bool = True,
):
    """
    A diagnostic identical in spirit to the successful Colab test:
      - Two sequences share same prefix, differ in suffix
      - Compare logits at a prefix position:
          causal   => diff ~ 0
          noncausal=> diff >> 0
    """
    if controller is None:
        controller = MaskController(backbone)

    prefix = "Hello world. This is the SAME prefix for both sequences. "
    s1 = prefix + "SUFFIX_A: cats are wonderful and playful."
    s2 = prefix + "SUFFIX_B: quantum field theory is quite abstract."
    batch = tokenizer([s1, s2], return_tensors="pt", padding=True, truncation=True).to(device)

    pref_ids = tokenizer([prefix], return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    pref_len = int(pref_ids.numel())
    pos = max(1, min(pref_len - 2, 16))

    def run(is_causal_flag: bool):
        with controller.set(is_causal_flag):
