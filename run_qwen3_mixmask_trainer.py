#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Qwen3 CLM + MLM (same batch) trainer
- Same data: build CLM inputs (original) + MLM inputs (random <mask> replacement) from the same batch
- MLM loss is computed manually (no shift) on masked positions only
- Uses FlexAttention mask_function for MLM (full visible). CLM can be SDPA/Flash/Eager or FlexAttention.
- Adds <mask> token to tokenizer if missing (and resizes model embeddings)

Run examples:

# CLM+MLM simultaneously (recommended)
python run_qwen3_clm_mlm_trainer.py \
  --model_name_or_path Qwen/Qwen3-0.6B \
  --dataset_name HuggingFaceFW/fineweb-edu --dataset_split train --text_column text --streaming \
  --seqlen 2048 --per_device_train_batch_size 2 --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 --warmup_steps 100 --max_steps 1000 \
  --attn_impl flex_attention \
  --mask_policy both --both_weights 1.0,1.0 \
  --mlm_mask_ratio 0.15 \
  --bf16 True --report_to none --logging_steps 20

# CLM only (SDPA ok)
python run_qwen3_clm_mlm_trainer.py \
  --attn_impl sdpa --mask_policy clm ...

# MLM only (FlexAttention required)
python run_qwen3_clm_mlm_trainer.py \
  --attn_impl flex_attention --mask_policy mlm --mlm_mask_ratio 0.15 ...
"""

import os
import time
import math
import argparse
from typing import Optional, Iterator, List, Dict, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)

# -------------------------- FlexAttention mask functions --------------------------
def causal_mask_fn(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
    """Allow attention to <= current position (causal)."""
    return kv_idx <= q_idx

def full_visible_mask_fn(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
    """Allow attention to all positions (bidirectional / full visible)."""
    return True

# -------------------------- Tokenizer / mask token setup --------------------------
def ensure_mask_token(tokenizer, model, mask_token: str = "<mask>") -> int:
    """
    Ensure tokenizer has mask_token_id. If missing, add special token <mask> and resize model embeddings.
    Return mask_token_id.
    """
    mask_id = getattr(tokenizer, "mask_token_id", None)
    if mask_id is None:
        # Add as special token and register as mask_token
        num_added = tokenizer.add_special_tokens({"mask_token": mask_token})
        mask_id = tokenizer.mask_token_id

        # If vocabulary grew, resize embeddings and keep tying
        if num_added > 0:
            model.resize_token_embeddings(len(tokenizer))
            if hasattr(model, "tie_weights"):
                model.tie_weights()

    if mask_id is None:
        raise RuntimeError("mask_token_id を確保できませんでした。tokenizer / model の状態を確認してください。")
    return int(mask_id)

# -------------------------- Dataset: streaming fixed-length packing --------------------------
class PackedStreamingIterable(IterableDataset):
    """
    streaming -> tokenize(add_special_tokens=False) -> append EOS per document -> pack into fixed seqlen chunks.
    """
    def __init__(
        self,
        *,
        tokenizer,
        dataset_name: str,
        dataset_config: Optional[str],
        dataset_split: str,
        text_column: str,
        seqlen: int,
        streaming: bool,
        shuffle_buffer: int,
        seed: int,
        world_size: int,
        rank: int,
        add_eos_between_docs: bool = True,
        add_bos_at_chunk_start: bool = False,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        self.tok = tokenizer
        self.name = dataset_name
        self.config = dataset_config
        self.split = dataset_split
        self.text_key = text_column
        self.seqlen = int(seqlen)
        self.streaming = bool(streaming)
        self.shuffle_buffer = int(shuffle_buffer)
        self.seed = int(seed)
        self.world_size = max(1, int(world_size))
        self.rank = max(0, int(rank))
        self.add_eos_between_docs = bool(add_eos_between_docs)
        self.add_bos_at_chunk_start = bool(add_bos_at_chunk_start)
        self.cache_dir = cache_dir

        assert self.tok.eos_token_id is not None, "tokenizer に eos_token_id が必要です。"

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        ds = load_dataset(
            self.name,
            self.config if self.config else None,
            split=self.split,
            streaming=self.streaming,
            cache_dir=self.cache_dir,
        )

        # shard per rank if available
        if hasattr(ds, "shard"):
            ds = ds.shard(num_shards=self.world_size, index=self.rank)

        # shuffle (streaming shuffle uses buffer)
        if hasattr(ds, "shuffle") and self.shuffle_buffer > 0:
            ds = ds.shuffle(buffer_size=self.shuffle_buffer, seed=self.seed)

        eos = int(self.tok.eos_token_id)
        buf: List[int] = []

        for ex in ds:
            txt = ex.get(self.text_key, "")
            if not isinstance(txt, str) or not txt:
                continue

            ids = self.tok(
                txt,
                add_special_tokens=False,
                return_attention_mask=False,
            )["input_ids"]

            if self.add_eos_between_docs:
                ids = ids + [eos]

            buf.extend(ids)

            while len(buf) >= self.seqlen:
                chunk = buf[: self.seqlen]
                buf = buf[self.seqlen :]

                if self.add_bos_at_chunk_start and getattr(self.tok, "bos_token_id", None) is not None:
                    chunk[0] = int(self.tok.bos_token_id)

                yield {"input_ids": torch.tensor(chunk, dtype=torch.long)}

# -------------------------- Collator (simple) --------------------------
def clm_collate(examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    # examples: [{"input_ids": (S,)}...]
    input_ids = torch.stack([ex["input_ids"] for ex in examples], dim=0)  # (B,S)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    labels = input_ids.clone()  # CLM: model will shift internally when using built-in loss
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

# -------------------------- MLM helper --------------------------
def choose_mlm_positions_random(
    input_ids: torch.LongTensor,
    mask_ratio: float,
    special_ids: List[int],
    rng: torch.Generator,
) -> torch.BoolTensor:
    """
    Bernoulli(mask_ratio) per position; exclude special token ids; ensure at least one masked position per sample.
    """
    B, S = input_ids.shape
    device = input_ids.device
    sel = torch.zeros((B, S), dtype=torch.bool, device=device)
    if mask_ratio <= 0.0:
        return sel

    # candidate mask (exclude specials)
    cand = torch.ones((B, S), dtype=torch.bool, device=device)
    for sid in special_ids:
        cand &= (input_ids != sid)

    rnd = torch.rand((B, S), generator=rng, device=device)
    sel = (rnd < mask_ratio) & cand

    # ensure >=1 per row
    any_row = sel.any(dim=1)
    if (~any_row).any():
        bad = torch.where(~any_row)[0].tolist()
        for b in bad:
            # pick a random candidate index
            idxs = torch.where(cand[b])[0]
            if idxs.numel() == 0:
                idxs = torch.arange(S, device=device)
            j = idxs[torch.randint(0, idxs.numel(), (1,), generator=rng, device=device)]
            sel[b, j] = True

    return sel

# -------------------------- Trainer --------------------------
class CLMMLMTrainer(Trainer):
    """
    mask_policy:
      - clm       : CLM only
      - mlm       : MLM only
      - alternate : alternate clm/mlm by step
      - both      : compute both losses from the same batch and sum with weights
    """

    def __init__(
        self,
        *args,
        mask_policy: str = "both",
        mlm_mask_ratio: float = 0.15,
        both_weights: str = "1.0,1.0",   # "w_clm,w_mlm"
        attn_impl: str = "flex_attention",
        mask_token_id: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # backward compat aliases
        alias = {"causal": "clm", "bidir": "mlm"}
        self.mask_policy = alias.get(mask_policy, mask_policy)
        self.mlm_mask_ratio = float(mlm_mask_ratio)
        self.attn_impl = attn_impl
        self.is_flex = (attn_impl == "flex_attention")

        if self.mask_policy in ("mlm", "alternate", "both") and not self.is_flex:
            raise RuntimeError("MLM を含む学習には --attn_impl flex_attention が必要です。")

        w = [float(x.strip()) for x in both_weights.split(",")]
        if len(w) != 2:
            raise ValueError("--both_weights は 'w_clm,w_mlm' の2要素で指定してください（例: 1.0,1.0）")
        self.w_clm, self.w_mlm = w

        self.mask_token_id = mask_token_id  # must be set if MLM is used

        # RNG per rank
        dev = self.model.device
        self._rng = torch.Generator(device=dev)
        base_seed = int(getattr(self.args, "seed", 42))
        local_rank = int(getattr(self.args, "local_rank", 0) or 0)
        self._rng.manual_seed(base_seed + local_rank)

        # for speed metrics
        self._n_params = sum(p.numel() for p in self.model.parameters())
        self._tok_total = 0
        self._tok_last = 0
        self._t_last = time.time()

    @staticmethod
    def _special_ids_from_tokenizer(tok) -> List[int]:
        ids = []
        for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
            v = getattr(tok, name, None)
            if v is not None:
                ids.append(int(v))
        return sorted(set(ids))

    def _mode_now(self) -> str:
        if self.mask_policy in ("clm", "mlm", "both"):
            return self.mask_policy
        # alternate
        return "mlm" if (self.state.global_step % 2 == 1) else "clm"

    def _build_mlm_batch(self, inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        From the same input_ids:
          - replace selected positions with mask_token_id
          - labels valid only on selected positions; others -100
        Returns (masked_input_ids, labels, num_masked)
        """
        if self.mask_token_id is None:
            raise RuntimeError("MLM を使うには mask_token_id が必要です（<mask>追加/resizeが必要）。")

        tok = getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)
        special_ids = self._special_ids_from_tokenizer(tok)

        input_ids = inputs["input_ids"]
        B, S = input_ids.shape

        sel = choose_mlm_positions_random(
            input_ids=input_ids,
            mask_ratio=self.mlm_mask_ratio,
            special_ids=special_ids,
            rng=self._rng,
        )

        x = input_ids.clone()
        x[sel] = int(self.mask_token_id)

        labels = torch.full_like(input_ids, -100)
        labels[sel] = input_ids[sel]

        num_masked = int(sel.sum().item())
        return x, labels, num_masked

    def _mlm_loss_from_logits(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        logits: (B,S,V), labels: (B,S) with -100 ignored
        compute CE over masked positions only (no shift)
        """
        # flatten masked positions
        mask = labels != -100
        if not mask.any():
            # should not happen due to fallback, but be safe
            return logits.sum() * 0.0

        V = logits.size(-1)
        logits_m = logits[mask].view(-1, V)
        targets = labels[mask].view(-1)
        return F.cross_entropy(logits_m, targets, reduction="mean")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Trainer>=4.44 passes num_items_in_batch; accept it
        _ = kwargs.get("num_items_in_batch", None)

        mode = self._mode_now()

        # ---------- CLM loss (built-in, shifted internally) ----------
        def run_clm_loss() -> Tuple[torch.Tensor, int, torch.Tensor]:
            out = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None),
                labels=inputs["labels"],
                mask_function=(causal_mask_fn if self.is_flex else None),
            )
            # active CLM tokens approx B*(S-1) because of shift
            B, S = inputs["input_ids"].shape
            active = int(B * max(0, S - 1))
            return out.loss, active, out.logits

        # ---------- MLM loss (manual CE, no shift) ----------
        def run_mlm_loss() -> Tuple[torch.Tensor, int]:
            x_mlm, y_mlm, nmask = self._build_mlm_batch(inputs)
            out = model(
                input_ids=x_mlm,
                attention_mask=inputs.get("attention_mask", None),
                labels=None,  # IMPORTANT: no built-in shifted loss
                mask_function=full_visible_mask_fn,  # bidirectional
            )
            loss_mlm = self._mlm_loss_from_logits(out.logits, y_mlm)
            return loss_mlm, nmask

        # ---------- choose policy ----------
        if mode == "clm":
            loss_clm, act, _ = run_clm_loss()
            loss = loss_clm
            self.log({"loss_clm": round(float(loss_clm.detach().cpu()), 4)})
            active = act

        elif mode == "mlm":
            loss_mlm, nmask = run_mlm_loss()
            loss = loss_mlm
            self.log({"loss_mlm": round(float(loss_mlm.detach().cpu()), 4)})
            active = nmask

        else:  # both
            loss_clm, act_clm, _ = run_clm_loss()
            loss_mlm, nmask = run_mlm_loss()
            loss = self.w_clm * loss_clm + self.w_mlm * loss_mlm
            self.log({
                "loss_clm": round(float(loss_clm.detach().cpu()), 4),
                "loss_mlm": round(float(loss_mlm.detach().cpu()), 4),
            })
            active = int(act_clm + nmask)

        # ---------- speed metrics ----------
        with torch.no_grad():
            t_now = time.time()
            dt = max(1e-6, t_now - self._t_last)
            self._tok_total += int(active)
            dTok = self._tok_total - self._tok_last
            tps = dTok / dt
            tflops = (6.0 * self._n_params * tps) / 1e12
            self._t_last = t_now
            self._tok_last = self._tok_total
            self.log({"tps_active": round(float(tps), 1), "tflops": round(float(tflops), 2)})

        return (loss, None) if return_outputs else loss

# -------------------------- CLI / main --------------------------
def parse_args():
    p = argparse.ArgumentParser()

    # model
    p.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-0.6B")
    p.add_argument("--attn_impl", type=str, default="flex_attention",
                   choices=["flex_attention", "sdpa", "flash_attention_2", "eager"])

    # data
    p.add_argument("--dataset_name", type=str, default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--dataset_config", type=str, default=None)
    p.add_argument("--dataset_split", type=str, default="train")
    p.add_argument("--text_column", type=str, default="text")
    p.add_argument("--streaming", action="store_true", default=True)
    p.add_argument("--shuffle_buffer", type=int, default=10_000)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--add_bos_at_chunk_start", action="store_true", default=False)

    # policy
    p.add_argument("--mask_policy", type=str, default="both",
                   choices=["clm", "mlm", "alternate", "both", "causal", "bidir"])
    p.add_argument("--mlm_mask_ratio", type=float, default=0.15)
    p.add_argument("--both_weights", type=str, default="1.0,1.0")
    p.add_argument("--ensure_mask_token", type=lambda x: x.lower() == "true", default=True)
    p.add_argument("--mask_token_str", type=str, default="<mask>")

    # training
    p.add_argument("--output_dir", type=str, default="ckpt_out")
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--logging_steps", type=int, default=20)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bf16", type=lambda x: x.lower() == "true", default=True)
    p.add_argument("--fp16", type=lambda x: x.lower() == "true", default=False)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--report_to", type=str, default="none")
    p.add_argument("--ddp_find_unused_parameters", type=lambda x: x.lower() == "true", default=False)

    return p.parse_args()

def main():
    args = parse_args()

    tok = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=True,
        trust_remote_code=True,
    )

    # load model
    dtype = torch.bfloat16 if args.bf16 and torch.cuda.is_available() else (torch.float16 if args.fp16 and torch.cuda.is_available() else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        attn_implementation=args.attn_impl,
        use_cache=False,
        trust_remote_code=True,
        torch_dtype=dtype,
    )

    # distributed info (for dataset sharding)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))

    # if MLM is used, ensure mask token
    alias = {"causal": "clm", "bidir": "mlm"}
    policy = alias.get(args.mask_policy, args.mask_policy)
    need_mlm = policy in ("mlm", "alternate", "both")

    mask_id = None
    if need_mlm:
        if args.ensure_mask_token:
            mask_id = ensure_mask_token(tok, model, mask_token=args.mask_token_str)
        else:
            mask_id = getattr(tok, "mask_token_id", None)
            if mask_id is None:
                raise RuntimeError("mask_token_id がありません。--ensure_mask_token True を使うか、tokenizer側で <mask> を用意してください。")

    # dataset
    train_ds = PackedStreamingIterable(
        tokenizer=tok,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        dataset_split=args.dataset_split,
        text_column=args.text_column,
        seqlen=args.seqlen,
        streaming=args.streaming,
        shuffle_buffer=args.shuffle_buffer,
        seed=args.seed,
        world_size=world_size,
        rank=rank,
        add_eos_between_docs=True,
        add_bos_at_chunk_start=args.add_bos_at_chunk_start,
        cache_dir=args.cache_dir,
    )

    # training args
    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to=args.report_to,
        ddp_find_unused_parameters=args.ddp_find_unused_parameters,
    )

    trainer = CLMMLMTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        data_collator=clm_collate,
        processing_class=tok,  # avoid tokenizer deprecation warning
        mask_policy=args.mask_policy,
        mlm_mask_ratio=args.mlm_mask_ratio,
        both_weights=args.both_weights,
        attn_impl=args.attn_impl,
        mask_token_id=mask_id,
    )

    if trainer.is_world_process_zero():
        emb = model.get_input_embeddings().weight
        head = getattr(model, "lm_head", None).weight if getattr(model, "lm_head", None) is not None else None
        tied = (head is not None and emb.data_ptr() == head.data_ptr())
        print({
            "vocab_size": len(tok),
            "mask_token": getattr(tok, "mask_token", None),
            "mask_token_id": getattr(tok, "mask_token_id", None),
            "unk_token_id": getattr(tok, "unk_token_id", None),
            "eos_token_id": getattr(tok, "eos_token_id", None),
            "emb_rows": int(emb.shape[0]),
            "emb_vs_head_tied": bool(tied),
            "attn_impl": args.attn_impl,
            "mask_policy": args.mask_policy,
            "mlm_mask_ratio": args.mlm_mask_ratio,
            "both_weights": args.both_weights,
        })

    trainer.train()

    if trainer.is_world_process_zero():
        trainer.save_model(args.output_dir)
        tok.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
