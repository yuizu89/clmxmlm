# clmxmlm/clmxmlm/eval_ft/heads.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..masking import MaskController


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    last_hidden: (B, S, H)
    attention_mask: (B, S) with 1 for valid tokens
    """
    am = attention_mask.to(device=last_hidden.device, dtype=torch.float32)
    denom = am.sum(dim=1, keepdim=True).clamp_min(1.0)
    x = (last_hidden * am.unsqueeze(-1)).sum(dim=1) / denom
    return x


@dataclass
class FTMaskCfg:
    """
    Finetuning mask mode:

    - bidir: want_causal=False  (prefix can attend to suffix; "bidir-ish")
    - causal: want_causal=True  (standard decoder causal masking)
    """
    ft_mask: str = "bidir"  # "bidir" or "causal"

    def want_causal(self) -> bool:
        if self.ft_mask == "bidir":
            return False
        if self.ft_mask == "causal":
            return True
        raise ValueError(f"unknown ft_mask: {self.ft_mask}")


class _BackboneMixin:
    """
    Shared helper: run backbone with MaskController so that is_causal reliably reaches attention backend.
    """

    backbone: nn.Module
    mask_cfg: FTMaskCfg
    controller: MaskController

    def _ensure_attention_mask(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if attention_mask is None:
            return torch.ones_like(input_ids, dtype=torch.long)
        return attention_mask

    def _run_backbone(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        is_causal = bool(self.mask_cfg.want_causal())
        # controller.set(...) may also flip module.is_causal etc (depending on your implementation)
        with self.controller.set(is_causal):
            out = self.controller.backbone_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                is_causal=is_causal,
                use_cache=False,
                return_dict=True,
                output_hidden_states=False,
            )
        return out


class DecoderOnlyForSequenceClassification(nn.Module, _BackboneMixin):
    def __init__(
        self,
        backbone: nn.Module,
        hidden_size: int,
        num_labels: int,
        mask_cfg: FTMaskCfg,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = backbone
        self.mask_cfg = mask_cfg
        self.controller = MaskController(self.backbone)

        self.dropout = nn.Dropout(float(dropout))
        self.classifier = nn.Linear(int(hidden_size), int(num_labels))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        attention_mask = self._ensure_attention_mask(input_ids, attention_mask)

        out = self._run_backbone(input_ids, attention_mask)
        h = out.last_hidden_state  # (B,S,H)

        pooled = mean_pool(h, attention_mask)
        logits = self.classifier(self.dropout(pooled))

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return {"loss": loss, "logits": logits}


class DecoderOnlyForTokenClassification(nn.Module, _BackboneMixin):
    def __init__(
        self,
        backbone: nn.Module,
        hidden_size: int,
        num_labels: int,
        mask_cfg: FTMaskCfg,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = backbone
        self.mask_cfg = mask_cfg
        self.controller = MaskController(self.backbone)

        self.dropout = nn.Dropout(float(dropout))
        self.classifier = nn.Linear(int(hidden_size), int(num_labels))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        attention_mask = self._ensure_attention_mask(input_ids, attention_mask)

        out = self._run_backbone(input_ids, attention_mask)
        h = out.last_hidden_state  # (B,S,H)

        logits = self.classifier(self.dropout(h))  # (B,S,C)

        loss = None
        if labels is not None:
            # labels: (B,S), with -100 ignored (HF convention)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )

        return {"loss": loss, "logits": logits}


class DecoderOnlyForExtractiveQA(nn.Module, _BackboneMixin):
    def __init__(
        self,
        backbone: nn.Module,
        hidden_size: int,
        mask_cfg: FTMaskCfg,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = backbone
        self.mask_cfg = mask_cfg
        self.controller = MaskController(self.backbone)

        self.dropout = nn.Dropout(float(dropout))
        self.qa_outputs = nn.Linear(int(hidden_size), 2)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        attention_mask = self._ensure_attention_mask(input_ids, attention_mask)

        out = self._run_backbone(input_ids, attention_mask)
        h = self.dropout(out.last_hidden_state)  # (B,S,H)

        logits = self.qa_outputs(h)  # (B,S,2)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)  # (B,S)
        end_logits = end_logits.squeeze(-1)      # (B,S)

        loss = None
        if start_positions is not None and end_positions is not None:
            # clamp to sequence length
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            loss = 0.5 * (loss_fct(start_logits, start_positions) + loss_fct(end_logits, end_positions))

        return {"loss": loss, "start_logits": start_logits, "end_logits": end_logits}


class DPRDualEncoder(nn.Module, _BackboneMixin):
    """
    DPR-style dual encoder with in-batch negatives:
      sim = q @ d^T / temperature
      labels = arange(B)
    """
    def __init__(
        self,
        backbone: nn.Module,
        hidden_size: int,
        mask_cfg: FTMaskCfg,
        temperature: float = 0.05,
        normalize: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.mask_cfg = mask_cfg
        self.controller = MaskController(self.backbone)

        self.temperature = float(temperature)
        self.normalize = bool(normalize)

        # (optional) projection head if you ever want it; for now identity
        self._hidden_size = int(hidden_size)

    @torch.no_grad()
    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        attention_mask = self._ensure_attention_mask(input_ids, attention_mask)
        out = self._run_backbone(input_ids, attention_mask)
        h = out.last_hidden_state
        x = mean_pool(h, attention_mask)
        if self.normalize:
            x = F.normalize(x, p=2, dim=-1)
        return x

    def forward(
        self,
        q_input_ids: Optional[torch.Tensor] = None,
        q_attention_mask: Optional[torch.Tensor] = None,
        d_input_ids: Optional[torch.Tensor] = None,
        d_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        if q_input_ids is None or d_input_ids is None:
            raise ValueError("DPRDualEncoder.forward requires q_input_ids and d_input_ids")

        q_attention_mask = self._ensure_attention_mask(q_input_ids, q_attention_mask)
        d_attention_mask = self._ensure_attention_mask(d_input_ids, d_attention_mask)

        # NOTE: encode() is @torch.no_grad; for training DPR you might want gradients.
        # For evaluation/encoding it is fine. If you want trainable DPR later, remove @torch.no_grad above.
        q = self.encode(q_input_ids, q_attention_mask)
        d = self.encode(d_input_ids, d_attention_mask)

        logits = (q @ d.transpose(0, 1)) / max(self.temperature, 1e-6)
        labels = torch.arange(logits.size(0), device=logits.device)
        loss = F.cross_entropy(logits, labels)
        return {"loss": loss, "logits": logits}
