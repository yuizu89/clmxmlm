#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3 0.6B〜8B: CLM+MLM 同時学習トレーナ
- 同一バッチの input_ids から、CLM 用（そのまま）と MLM 用（ランダム [MASK] 置換）を生成
- CLM と MLM を同一ステップで forward し、損失を重み付き合算（--both_weights）
- MLM 時は FlexAttention で「全可視（双方向）」にするため mask_function を使用
- 語彙は増やさない（既存の mask_token_id が無い場合は unk_token_id を使用）

依存:
  pip install -U torch accelerate transformers datasets

例:
  # CLM + MLM を同時学習（推奨）
  python run_qwen3_clm_mlm_trainer.py \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --dataset_name HuggingFaceFW/fineweb-edu --dataset_split train --text_column text --streaming \
    --seqlen 2048 --per_device_train_batch_size 2 --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 --warmup_steps 100 --max_steps 1000 \
    --attn_impl flex_attention \
    --mask_policy both --both_weights 1.0,1.0 \
    --mlm_mask_ratio 0.15 \
    --bf16 True --report_to none --logging_steps 20

  # CLM のみ（SDPAでOK）
  python run_qwen3_clm_mlm_trainer.py \
    --attn_impl sdpa --mask_policy clm ...

  # MLM のみ（FlexAttention必須）
  python run_qwen3_clm_mlm_trainer.py \
    --attn_impl flex_attention --mask_policy mlm --mlm_mask_ratio 0.15 ...
"""
import os, math, time, argparse
from typing import Optional, Iterator, List, Dict, Tuple

import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    Qwen3ForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

# ------------------------ Dataset: streaming fixed-length packing ------------------------
class PackedStreamingIterable(IterableDataset):
    """
    streaming で読み込み → tokenize(add_special_tokens=False) → 文末に EOS を1つだけ付与して連結 →
    seqlen で固定長に切り出し（padしない）
    """
    def __init__(
        self,
        *,
        tokenizer: AutoTokenizer,
        dataset_name: str = "HuggingFaceFW/fineweb-edu",
        dataset_config: Optional[str] = None,
        dataset_split: str = "train",
        text_column: str = "text",
        seqlen: int = 2048,
        streaming: bool = True,
        shuffle_buffer: int = 10_000,
        seed: int = 42,
        world_size: int = 1,
        rank: int = 0,
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
        self.seqlen = seqlen
        self.streaming = streaming
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.world_size = max(1, world_size)
        self.rank = max(0, rank)
        self.add_eos_between_docs = add_eos_between_docs
        self.add_bos_at_chunk_start = add_bos_at_chunk_start
        self.cache_dir = cache_dir
        assert tokenizer.eos_token_id is not None, "tokenizer に eos_token が必要です。"

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        ds = load_dataset(
            self.name,
            self.config if self.config else None,
            split=self.split,
            streaming=self.streaming,
            cache_dir=self.cache_dir,
        )
        if hasattr(ds, "shard"):
            ds = ds.shard(num_shards=self.world_size, index=self.rank)
        if hasattr(ds, "shuffle") and self.shuffle_buffer > 0:
            ds = ds.shuffle(buffer_size=self.shuffle_buffer, seed=self.seed)

        eos = self.tok.eos_token_id
        buf: List[int] = []

        for ex in ds:
            txt = ex.get(self.text_key, "")
            if not isinstance(txt, str) or not txt:
                continue
            ids = self.tok(txt, add_special_tokens=False, return_attention_mask=False)["input_ids"]
            if self.add_eos_between_docs:
                ids = ids + [eos]
            buf.extend(ids)

            while len(buf) >= self.seqlen:
                chunk = buf[:self.seqlen]
                buf = buf[self.seqlen:]
                if self.add_bos_at_chunk_start and self.tok.bos_token_id is not None:
                    chunk[0] = self.tok.bos_token_id
                yield {"input_ids": torch.tensor(chunk, dtype=torch.long)}

# ------------------------ FlexAttention mask functions ------------------------
def full_visible_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
    """MLM 用：全トークン可視（双方向）。"""
    return True

# ------------------------ MLM helpers ------------------------
def choose_mlm_positions_random(
    input_ids: torch.LongTensor,
    mask_ratio: float,
    special_ids: Optional[torch.Tensor] = None,
    rng: Optional[torch.Generator] = None,
) -> torch.BoolTensor:
    """
    Bernoulli(mask_ratio) でランダムにマスク位置を選ぶ。
    - 特殊トークン（EOS/BOS/PAD 等）は除外。
    - 全Falseになるのを避けるため、最低1個は有効化する。
    """
    B, S = input_ids.shape
    device = input_ids.device
    sel = torch.zeros((B, S), dtype=torch.bool, device=device)
    if mask_ratio <= 0.0:
        return sel
    # 候補（特殊トークン除外）
    cand = torch.ones((B, S), dtype=torch.bool, device=device)
    if special_ids is not None and special_ids.numel() > 0:
        for sid in special_ids.tolist():
            cand &= (input_ids != sid)

    # Bernoulli
    rnd = torch.rand((B, S), generator=rng, device=device)
    sel = (rnd < mask_ratio) & cand

    # 各行最低1個
    any_row = sel.any(dim=1)
    if (~any_row).any():
        for b in torch.where(~any_row)[0].tolist():
            # 先頭/末尾より中域を優先（適当に 1..S-2 から探す）
            start, end = (1, max(1, S - 1))
            fallback = torch.arange(start, end, device=device)
            if special_ids is not None and special_ids.numel() > 0:
                mask_ok = torch.ones_like(fallback, dtype=torch.bool, device=device)
                for sid in special_ids.tolist():
                    mask_ok &= (input_ids[b, start:end] != sid)
                fallback = fallback[mask_ok]
            if fallback.numel() == 0:
                fallback = torch.arange(0, S, device=device)
            idx = fallback[torch.randint(0, fallback.numel(), (1,), device=device)]
            sel[b, idx] = True
    return sel

# ------------------------ Trainer subclass ------------------------
class MixTaskTrainer(Trainer):
    """
    mask_policy:
      - "clm"       : CLM のみ
      - "mlm"       : MLM のみ（FlexAttention 必須）
      - "alternate" : ステップごとに clm / mlm を交互
      - "both"      : 同一ステップで clm と mlm を両方 forward し、重み付き合算
    """
    def __init__(
        self,
        *args,
        mask_policy: str = "both",
        mlm_mask_ratio: float = 0.15,
        both_weights: str = "1.0,1.0",   # "w_clm,w_mlm"
        attn_impl: str = "flex_attention",
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        # 後方互換（causal/bidir を受け付ける）
        alias = {"causal": "clm", "bidir": "mlm"}
        mask_policy = alias.get(mask_policy, mask_policy)

        self.mask_policy = mask_policy
        self.mlm_mask_ratio = float(mlm_mask_ratio)
        self.attn_impl = attn_impl
        self.is_flex = (attn_impl == "flex_attention")

        if self.mask_policy in ("mlm", "alternate", "both") and not self.is_flex:
            raise RuntimeError("MLM を含む学習（mlm/alternate/both）には --attn_impl flex_attention が必要です。")

        w = [float(x.strip()) for x in both_weights.split(",")]
        if len(w) != 2:
            raise ValueError("--both_weights は 'w_clm,w_mlm' の2要素で指定してください（例: 1.0,1.0）")
        self.w_clm, self.w_mlm = w

        # RNG（再現性）。Trainer の seed/locale を活用
        dev = self.model.device if hasattr(self, "model") else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self._rng = torch.Generator(device=dev)
        try:
            base_seed = int(getattr(self.args, "seed", 42))
        except Exception:
            base_seed = 42
        local_rank = int(getattr(self.args, "local_rank", 0) or 0)
        self._rng.manual_seed(base_seed + local_rank)

        # ログ用
        self._tok_total = 0
        self._tok_last = 0
        self._t_last = time.time()
        self._n_params = sum(p.numel() for p in self.model.parameters())

    def _mode_now(self) -> str:
        if self.mask_policy in ("clm", "mlm", "both"):
            return self.mask_policy
        # alternate
        return "mlm" if (self.state.global_step % 2 == 1) else "clm"

    @staticmethod
    def _get_processor(trainer) -> AutoTokenizer:
        proc = getattr(trainer, "processing_class", None)
        if proc is None:
            proc = getattr(trainer, "tokenizer", None)  # 互換用
        return proc

    def _build_mlm_batch(self, inputs: Dict[str, torch.Tensor], processor: AutoTokenizer) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        同じ input_ids から MLM 用 (input, labels) を作る。
        - 入力の選択位置を mask_token_id（無ければ unk）に置換
        - labels は選択位置のみ元トークン、それ以外は -100
        - 返り値: (masked_input_ids, labels, 有効ラベル数)
        """
        input_ids = inputs["input_ids"]
        B, S = input_ids.shape
        labels = torch.full_like(input_ids, -100)

        mask_id = getattr(processor, "mask_token_id", None)
        if mask_id is None:
            mask_id = processor.unk_token_id
        assert mask_id is not None, "tokenizer に unk_token が必要です。"

        # 特殊トークン（BOS/EOS/PAD）を除外
        special_ids = []
        for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
            val = getattr(processor, name, None)
            if val is not None:
                special_ids.append(val)
        special_ids = torch.tensor(special_ids, device=input_ids.device, dtype=torch.long) if len(special_ids) else None

        sel = choose_mlm_positions_random(input_ids, self.mlm_mask_ratio, special_ids, self._rng)

        x = input_ids.clone()
        x[sel] = mask_id
        labels[sel] = input_ids[sel]
        active = int(sel.sum().item())
        return x, labels, active

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Trainer>=4.44 互換
        _ = kwargs.get("num_items_in_batch", None)
        processor = self._get_processor(self)
        mode = self._mode_now()

        # CLM: そのまま（labels は DataCollatorForLanguageModeling(mlm=False) で未シフトの複製）
        if mode == "clm":
            out = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None),
                labels=inputs["labels"],  # 未シフト
            )
            loss = out.loss
            act = (inputs["labels"] != -100).sum().item()
            self.log({"loss_clm": round(float(loss.detach().cpu()), 4)})

        # MLM: FlexAttention + 全可視（full_visible_mask）
        elif mode == "mlm":
            x_mlm, y_mlm, act = self._build_mlm_batch(inputs, processor)
            out = model(
                input_ids=x_mlm,
                attention_mask=inputs.get("attention_mask", None),
                labels=y_mlm,
                mask_function=full_visible_mask,  # 全可視
            )
            loss = out.loss
            self.log({"loss_mlm": round(float(loss.detach().cpu()), 4)})

        # BOTH: 同一ステップで CLM+MLM を forward → 重み付き合算
        else:  # "both"
            # CLM
            out_c = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None),
                labels=inputs["labels"],
            )
            loss_c = out_c.loss
            act_c = (inputs["labels"] != -100).sum().item()

            # MLM（FlexAttention）
            x_mlm, y_mlm, act_m = self._build_mlm_batch(inputs, processor)
            out_m = model(
                input_ids=x_mlm,
                attention_mask=inputs.get("attention_mask", None),
                labels=y_mlm,
                mask_function=full_visible_mask,
            )
            loss_m = out_m.loss
            act = int(act_c + act_m)

            # 重み付き合算
            loss = self.w_clm * loss_c + self.w_mlm * loss_m

            self.log({
                "loss_clm": round(float(loss_c.detach().cpu()), 4),
                "loss_mlm": round(float(loss_m.detach().cpu()), 4),
            })

        # 共通: 速度/TFLOPs 概算
        with torch.no_grad():
            t_now = time.time()
            dt = max(1e-6, t_now - self._t_last)
            self._tok_total += int(act)
            dTok = self._tok_total - self._tok_last
            tps = dTok / dt
            tflops = (6.0 * self._n_params * tps) / 1e12
            self._t_last = t_now
            self._tok_last = self._tok_total
            self.log({"tps_active": round(float(tps), 1),
                      "tflops": round(float(tflops), 2)})

        return (loss, out) if return_outputs else loss

# ------------------------ CLI ------------------------
def parse_args():
    p = argparse.ArgumentParser()
    # モデル/注意実装
    p.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-0.6B")
    p.add_argument("--attn_impl", type=str, default="flex_attention",
                   choices=["flex_attention", "sdpa", "flash_attention_2", "eager"])

    # データ
    p.add_argument("--dataset_name", type=str, default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--dataset_config", type=str, default=None)
    p.add_argument("--dataset_split", type=str, default="train")
    p.add_argument("--text_column", type=str, default="text")
    p.add_argument("--streaming", action="store_true", default=True)
    p.add_argument("--shuffle_buffer", type=int, default=10000)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--add_bos_at_chunk_start", action="store_true", default=False)

    # 学習方針
    p.add_argument("--mask_policy", type=str, default="both",
                   choices=["clm", "mlm", "alternate", "both", "causal", "bidir"])  # 後方互換名を含む
    p.add_argument("--mlm_mask_ratio", type=float, default=0.15)
    p.add_argument("--both_weights", type=str, default="1.0,1.0")  # "w_clm,w_mlm"

    # TrainingArguments（主要）
    p.add_argument("--output_dir", type=str, default="ckpt_out")
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--per_device_eval_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--logging_steps", type=int, default=20)
    p.add_argument("--save_steps", type=int, default=0)
    p.add_argument("--bf16", type=lambda x: x.lower() == "true", default=True)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--report_to", type=str, default="none")
    p.add_argument("--ddp_find_unused_parameters", action="store_true", default=False)
    return p.parse_args()

def main():
    args = parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True, trust_remote_code=True)

    model = Qwen3ForCausalLM.from_pretrained(
        args.model_name_or_path,
        attn_implementation=args.attn_impl,
        use_cache=False,
        trust_remote_code=True,
        torch_dtype=(torch.bfloat16 if args.bf16 and torch.cuda.is_available() else torch.float32),
    )

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    train_ds = PackedStreamingIterable(
        tokenizer=tok,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        dataset_split=args.dataset_split,
        text_column=args.text_column,
        seqlen=args.seqlen,
        streaming=args.streaming,
        shuffle_buffer=args.shuffle_buffer,
        seed=1337,
        world_size=world_size,
        rank=rank,
        add_eos_between_docs=True,
        add_bos_at_chunk_start=args.add_bos_at_chunk_start,
        cache_dir=args.cache_dir,
    )

    # CLM ラベル生成は collator に任せる（未シフト、-100 マスク）
    collator = DataCollatorForLanguageModeling(tok, mlm=False)
    train_dl = DataLoader(train_ds, batch_size=args.per_device_train_batch_size,
                          shuffle=False, drop_last=True, num_workers=0, collate_fn=collator)

    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to=args.report_to,
        ddp_find_unused_parameters=args.ddp_find_unused_parameters,
    )

    trainer = MixTaskTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,          # IterableDataset を渡す（内部で DL 構築）
        processing_class=tok,            # ★ tokenizer 警告の公式解
        data_collator=collator,
        mask_policy=args.mask_policy,
        mlm_mask_ratio=args.mlm_mask_ratio,
        both_weights=args.both_weights,
        attn_impl=args.attn_impl,
    )

    if trainer.is_world_process_zero():
        emb = model.get_input_embeddings().weight
        head = model.lm_head.weight
        print({
            "vocab_size": len(tok),
            "emb_vs_head_tied": (emb.data_ptr() == head.data_ptr()),
            "attn_impl": args.attn_impl,
            "mask_policy": args.mask_policy,
            "mlm_mask_ratio": args.mlm_mask_ratio,
        })

    trainer.train()

    if trainer.is_world_process_zero():
        trainer.save_model(args.output_dir)
        tok.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
