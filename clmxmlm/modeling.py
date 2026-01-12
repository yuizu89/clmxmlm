from __future__ import annotations

from typing import Optional, List

import torch


def get_lm_head_module(model: torch.nn.Module) -> torch.nn.Module:
    """
    Prefer HF official output embeddings if possible.
    """
    m = getattr(model, "module", model)
    fn = getattr(m, "get_output_embeddings", None)
    if callable(fn):
        head = fn()
        if head is not None:
            return head
    if hasattr(m, "lm_head") and isinstance(getattr(m, "lm_head"), torch.nn.Module):
        return getattr(m, "lm_head")
    raise RuntimeError("Could not find lm_head / output embeddings on this model.")


def ensure_mask_token(tokenizer, model, mask_token: str = "<mask>") -> int:
    """
    Ensure tokenizer has a mask token and return its id.
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
        raise RuntimeError("Failed to ensure mask_token_id.")
    return int(mask_id)


def special_ids_from_tokenizer(tok) -> List[int]:
    ids = []
    for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
        v = getattr(tok, name, None)
        if v is not None:
            ids.append(int(v))
    return sorted(set(ids))
