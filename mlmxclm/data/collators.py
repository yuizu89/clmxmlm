from __future__ import annotations

from typing import List, Dict
import torch


def clm_collate(examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Packed fixed-length blocks => no padding needed.
    """
    input_ids = torch.stack([ex["input_ids"] for ex in examples], dim=0)  # (B,S)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)         # (B,S)
    return {"input_ids": input_ids, "attention_mask": attention_mask}
