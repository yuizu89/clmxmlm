from __future__ import annotations

from typing import List, Optional, Tuple

import torch


def choose_mntp_positions_random(
    input_ids: torch.LongTensor,
    mask_ratio: float,
    special_ids: List[int],
    rng: torch.Generator,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.BoolTensor:
    """
    MNTP: Select positions to MASK in the *target token* space (positions k+1).
    Returns:
      sel: BoolTensor (B, S) where True means "this token position is replaced by <mask>".

    Constraints:
      - Never mask position 0 (because MNTP predicts masked token at previous position).
      - Exclude special tokens (pad/bos/eos/etc.) from being masked.
      - If attention_mask is provided, do not mask padding (attention_mask==0).
      - Ensures >= 1 masked token per row when possible.
    """
    B, S = input_ids.shape
    device = input_ids.device

    sel = torch.zeros((B, S), dtype=torch.bool, device=device)

    # MNTP needs at least 2 tokens to have a "previous position"
    if mask_ratio <= 0.0 or S < 2:
        return sel

    # candidate mask: start with all True, then apply constraints
    cand = torch.ones((B, S), dtype=torch.bool, device=device)

    # cannot mask position 0 for MNTP (no previous token to predict from)
    cand[:, 0] = False

    # avoid masking padding positions if attention_mask is given
    if attention_mask is not None:
        cand &= attention_mask.to(torch.bool)

    # avoid masking special tokens
    for sid in special_ids:
        cand &= (input_ids != int(sid))

    # random selection
    rnd = torch.rand((B, S), generator=rng, device=device)
    sel = (rnd < float(mask_ratio)) & cand

    # ensure >= 1 masked per row when possible
    any_row = sel.any(dim=1)
    if (~any_row).any():
        bad_rows = torch.where(~any_row)[0]
        for b in bad_rows.tolist():
            idxs = torch.where(cand[b])[0]
            if idxs.numel() == 0:
                # no valid candidate in this row -> cannot ensure masking
                continue
            j = idxs[torch.randint(0, idxs.numel(), (1,), generator=rng, device=device)]
            sel[b, j] = True

    return sel


def build_mntp_batch(
    *,
    input_ids: torch.LongTensor,
    mask_token_id: int,
    mask_ratio: float,
    special_ids: List[int],
    rng: torch.Generator,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.LongTensor, torch.LongTensor, int]:
    """
    Build an MNTP batch.

    MNTP (Masked Next-Token Prediction):
      - choose positions to mask in input (these correspond to tokens at position k+1)
      - replace those tokens by <mask> in x
      - compute loss at the previous positions k:
            labels[k] = original_token[k+1]   if (k+1) was masked
            labels[others] = -100

    Returns:
      x_mntp: (B,S) input_ids with masked positions replaced by mask_token_id
      y_mntp: (B,S) labels with MNTP-supervised positions filled, others=-100
      nmask:  number of supervised positions (== number of masked tokens, excluding pos0)
    """
    B, S = input_ids.shape
    device = input_ids.device

    sel = choose_mntp_positions_random(
        input_ids=input_ids,
        mask_ratio=float(mask_ratio),
        special_ids=special_ids,
        rng=rng,
        attention_mask=attention_mask,
    )

    x = input_ids.clone()
    x[sel] = int(mask_token_id)

    y = torch.full_like(input_ids, -100)

    if S >= 2:
        # If position (k+1) is masked, we supervise prediction at position k.
        shift_sel = sel[:, 1:]  # (B, S-1) indicates masked target positions
        # fill labels at previous positions
        y[:, :-1][shift_sel] = input_ids[:, 1:][shift_sel]
        nmask = int(shift_sel.sum().item())
    else:
        nmask = 0

    return x, y, nmask


# ---- Backward-compatible aliases (optional but handy) ----
# If other code still imports build_mlm_batch / choose_mlm_positions_random,
# they will now behave as MNTP.
choose_mlm_positions_random = choose_mntp_positions_random
build_mlm_batch = build_mntp_batch
