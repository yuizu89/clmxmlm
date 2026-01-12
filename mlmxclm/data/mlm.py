√from __future__ import annotations

from typing import List, Tuple

import torch


def choose_mlm_positions_random(
    input_ids: torch.LongTensor,
    mask_ratio: float,
    special_ids: List[int],
    rng: torch.Generator,
) -> torch.BoolTensor:
    """
    Select positions to mask (True=mask) excluding special tokens.
    Ensures >=1 masked per row.
    """
    B, S = input_ids.shape
    device = input_ids.device
    if mask_ratio <= 0.0:
        return torch.zeros((B, S), dtype=torch.bool, device=device)

    cand = torch.ones((B, S), dtype=torch.bool, device=device)
    for sid in special_ids:
        cand &= (input_ids != sid)

    rnd = torch.rand((B, S), generator=rng, device=device)
    sel = (rnd < mask_ratio) & cand

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


def build_mlm_batch(
    *,
    input_ids: torch.LongTensor,
    mask_token_id: int,
    mask_ratio: float,
    special_ids: List[int],
    rng: torch.Generator,
) -> Tuple[torch.LongTensor, torch.LongTensor, int]:
    """
    Returns:
      x_mlm: input_ids with masked positions replaced by mask_token_id
      y_mlm: labels with only masked positions filled, others=-100
      nmask: number of masked tokens
    """
    sel = choose_mlm_positions_random(input_ids, mask_ratio, special_ids, rng)
    x = input_ids.clone()
    x[sel] = int(mask_token_id)

    y = torch.full_like(input_ids, -100)
    y[sel] = input_ids[sel]
    return x, y, int(sel.sum().item())
