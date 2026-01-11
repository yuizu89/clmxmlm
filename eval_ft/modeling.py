# encodeval_ft/modeling.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .masking import backbone_forward


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # last_hidden: (B,S,H), attention_mask: (B,S)
    am = attention_mask.to(last_hidden.device).to(torch.float32)
    denom = am.sum(dim=1, keepdim=True).clamp_min(1.0)
    x = (last_hidden * am.unsqueeze(-1)).sum(dim=1) / denom
    return x


@dataclass
class FTMaskCfg:
    """
    bidir: want_causal=False (as in paper's finetuning setup)
    causal: want_causal=True
    """
    ft_mask: str = "bidir"  # "bidir" or "causal"

    def want_causal(self) -> bool:
        if self.ft_mask == "bidir":
            return False
        if self.ft_mask == "causal":
            return True
        raise ValueError(f"unknown ft_mask: {self.ft_mask}")


class DecoderOnlyForSequenceClassification(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_size: int, num_labels: int, mask_cfg: FTMaskCfg, dropout: float = 0.1):
        super().__init__()
        self.backbone = backbone
        self.mask_cfg = mask_cfg
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        out = backbone_forward(
            self.backbone, input_ids, attention_mask,
            want_causal=self.mask_cfg.want_causal(),
            use_cache=False,
            output_hidden_states=False,
        )
        h = out.last_hidden_state
        pooled = mean_pool(h, attention_mask)
        logits = self.classifier(self.dropout(pooled))

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return {"loss": loss, "logits": logits}


class DecoderOnlyForTokenClassification(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_size: int, num_labels: int, mask_cfg: FTMaskCfg, dropout: float = 0.1):
        super().__init__()
        self.backbone = backbone
        self.mask_cfg = mask_cfg
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        out = backbone_forward(
            self.backbone, input_ids, attention_mask,
            want_causal=self.mask_cfg.want_causal(),
            use_cache=False,
            output_hidden_states=False,
        )
        h = out.last_hidden_state
        logits = self.classifier(self.dropout(h))

        loss = None
        if labels is not None:
            # labels: (B,S) with -100 for ignored
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)

        return {"loss": loss, "logits": logits}


class DecoderOnlyForExtractiveQA(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_size: int, mask_cfg: FTMaskCfg, dropout: float = 0.1):
        super().__init__()
        self.backbone = backbone
        self.mask_cfg = mask_cfg
        self.dropout = nn.Dropout(dropout)
        self.qa_outputs = nn.Linear(hidden_size, 2)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        start_positions=None,
        end_positions=None,
        **kwargs
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        out = backbone_forward(
            self.backbone, input_ids, attention_mask,
            want_causal=self.mask_cfg.want_causal(),
            use_cache=False,
            output_hidden_states=False,
        )
        h = self.dropout(out.last_hidden_state)
        logits = self.qa_outputs(h)  # (B,S,2)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        loss = None
        if start_positions is not None and end_positions is not None:
            # clamp to sequence length
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            loss = (loss_fct(start_logits, start_positions) + loss_fct(end_logits, end_positions)) * 0.5

        return {"loss": loss, "start_logits": start_logits, "end_logits": end_logits}


class DPRDualEncoder(nn.Module):
    """
    In-batch negatives DPR loss:
      sim = q @ d^T / temperature
      labels = arange(B)
    """
    def __init__(self, backbone: nn.Module, hidden_size: int, mask_cfg: FTMaskCfg, temperature: float = 0.05, normalize: bool = True):
        super().__init__()
        self.backbone = backbone
        self.mask_cfg = mask_cfg
        self.temperature = float(temperature)
        self.normalize = bool(normalize)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = backbone_forward(
            self.backbone, input_ids, attention_mask,
            want_causal=self.mask_cfg.want_causal(),
            use_cache=False,
            output_hidden_states=False,
        )
        h = out.last_hidden_state
        x = mean_pool(h, attention_mask)
        if self.normalize:
            x = F.normalize(x, p=2, dim=-1)
        return x

    def forward(self, q_input_ids=None, q_attention_mask=None, d_input_ids=None, d_attention_mask=None, **kwargs):
        q = self.encode(q_input_ids, q_attention_mask)
        d = self.encode(d_input_ids, d_attention_mask)

        logits = (q @ d.transpose(0, 1)) / max(self.temperature, 1e-6)
        labels = torch.arange(logits.size(0), device=logits.device)
        loss = F.cross_entropy(logits, labels)
        return {"loss": loss, "logits": logits}
