# encodeval_ft/masking.py
from __future__ import annotations

import inspect
from contextlib import contextmanager
from typing import Optional

import torch


def forward_accepts_param(model: torch.nn.Module, name: str) -> bool:
    """Best-effort signature check: has 'name' or **kwargs."""
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


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "module", model)


def get_backbone_from_causallm(model: torch.nn.Module) -> torch.nn.Module:
    """
    Most HF CausalLM classes keep backbone at .model.
    Fallback to base_model/transformer/itself.
    """
    m = unwrap_model(model)
    for attr in ("model", "base_model", "transformer"):
        if hasattr(m, attr) and isinstance(getattr(m, attr), torch.nn.Module):
            return getattr(m, attr)
    return m


@contextmanager
def force_module_is_causal(module: torch.nn.Module, flag: bool):
    """
    Some backends consult module.is_causal internally (e.g., FlashAttention2 paths).
    Force .is_causal recursively, then restore.
    """
    touched = []
    for m in module.modules():
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


@torch.no_grad()
def backbone_forward(
    backbone: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    want_causal: bool,
    use_cache: bool = False,
    output_hidden_states: bool = False,
):
    """
    Call backbone with (optional) is_causal kw if accepted, plus module.is_causal override.
    """
    accepts = forward_accepts_param(backbone, "is_causal")
    kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=use_cache,
        return_dict=True,
        output_hidden_states=output_hidden_states,
    )

    with force_module_is_causal(backbone, want_causal):
        if accepts:
            try:
                out = backbone(**kwargs, is_causal=bool(want_causal))
            except TypeError:
                out = backbone(**kwargs)
        else:
            out = backbone(**kwargs)
    return out
