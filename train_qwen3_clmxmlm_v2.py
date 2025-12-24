#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3 (0.6B〜8B) 継続学習: run_clm.py ベース拡張
- CLM（causal）と bidir（FlexAttention + マスク関数）を切替
- FineWeb-Edu streaming を固定長でパック
- 語彙は増やさない（mask_token は既存のものを使用；無ければ UNK）

依存:
  pip install -U torch accelerate transformers datasets

実行例:
  # CLMのみ（SDPAでOK）
  python run_qwen3_mixmask_trainer.py \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --dataset_name HuggingFaceFW/fineweb-edu --dataset_split train --text_column text --streaming \
    --seqlen 2048 --per_device_train_batch_size 2 --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 --warmup_steps 100 --max_steps 1000 \
    --attn_impl sdpa --mask_policy causal \
    --bf16 True --report_to none --logging_steps 20

  # CLMとbidirを交互（FlexAttention必須）
  python run_qwen3_mixmask_trainer.py \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --dataset_name HuggingFaceFW/fineweb-edu --dataset_split train --text_column text --streaming \
    --seqlen 2048 --per_device_train_batch_size 2 --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 --warmup_steps 100 --max_steps 1000 \
    --attn_impl flex_attention --mask_policy alternate --bidir_ratio 0.2 \
    --bf16 True --report_to none --logging_steps 20
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

# ====================== FineWeb streaming fixed-length packer ======================
class FineWebPackedIterable(IterableDataset):
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

# ====================== Attention mask functions (FlexAttention用) ======================
def bidir_mask_function(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
    """
    情報リーク防止：クエリ位置 q は「直右のトークン（q+1）」を見ない。
    これにより「(q+1) を予測対象」にしても、q がそのトークン自体を参照せずに
    左右文脈（q+1 以外）から復元するよう学習できる。
    """
    return (kv_idx != q_idx + 1)

# ====================== Helper: non-random selection for bidir ======================
def deterministic_select_mask_positions(seq_len: int, ratio: float) -> torch.BoolTensor:
    """
    完全ランダムではなく「等間隔サンプリング」に近い選択（ユーザ希望に配慮）
    先頭(0)は除外、選択範囲は [1, seq_len-1] （= (q+1) 側の学習対象）
    """
    sel = torch.zeros(seq_len, dtype=torch.bool)
    if seq_len <= 1 or ratio <= 0.0:
        return sel
    num_targets = seq_len - 1
    k = max(1, int(round(ratio * num_targets)))
    stride = max(1, num_targets // k)
    idx = torch.arange(1, seq_len, dtype=torch.long)[::stride][:k]
    sel[idx] = True
    return sel

# ====================== Trainer subclass: CLM / bidir 切替 ======================
class MixMaskTrainer(Trainer):
    def __init__(self, *args, mask_policy: str = "causal", bidir_ratio: float = 0.2, attn_impl: str = "sdpa", **kwargs):
        """
        mask_policy: "causal" | "bidir" | "alternate"
        attn_impl  : "flex_attention" | "sdpa" | "flash_attention_2" | "eager"
        """
        super().__init__(*args, **kwargs)
        self.mask_policy = mask_policy
        self.bidir_ratio = bidir_ratio
        self.attn_impl = attn_impl
        # FlexAttention が必要
        self.is_flex = (attn_impl == "flex_attention")
        if self.mask_policy in ("bidir", "alternate") and not self.is_flex:
            raise RuntimeError("bidir/alternate を使うには --attn_impl flex_attention が必要です。")

        # ログ用のトークンカウンタ
        self._tok_total = 0
        self._tok_last = 0
        self._t_last = time.time()
        self._n_params = sum(p.numel() for p in self.model.parameters())

    def _mode_now(self) -> str:
        if self.mask_policy in ("causal", "bidir"):
            return self.mask_policy
        # alternate: ステップごとに交互
        return "bidir" if (self.state.global_step % 2 == 1) else "causal"

    def _build_bidir_batch(self, inputs: Dict[str, torch.Tensor], tokenizer: AutoTokenizer) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        入力の (q+1) 側を一部 mask_id で置換し、その位置だけ labels を残す。それ以外は -100。
        モデル側の ForCausalLMLoss は内部シフトするので、labels は「未シフト」でOK。
        """
        input_ids = inputs["input_ids"]
        B, S = input_ids.shape
        labels = input_ids.clone()
        labels[:] = -100  # デフォルトは無視

        # 置換に使うトークン（mask_token_id があればそれを、なければ unk）
        mask_id = getattr(tokenizer, "mask_token_id", None)
        if mask_id is None:
            mask_id = tokenizer.unk_token_id
        assert mask_id is not None, "tokenizer に unk_token が必要です。"

        x = input_ids.clone()
        active = 0
        for b in range(B):
            sel = deterministic_select_mask_positions(S, self.bidir_ratio)  # True の列が (q+1) 側
            x[b][sel] = mask_id
            labels[b][sel] = input_ids[b][sel]
            active += int(sel.sum().item())

        return x, labels

    def compute_loss(self, model, inputs, return_outputs=False):
        # DataCollatorForLanguageModeling(mlm=False) で labels=input_ids が入ってくる想定
        tokenizer = self.tokenizer
        mode = self._mode_now()

        if mode == "bidir":
            # FlexAttention のときだけ mask_function を渡す
            x, y = self._build_bidir_batch(inputs, tokenizer)
            outputs = model(input_ids=x,
                            attention_mask=inputs.get("attention_mask", None),
                            labels=y,
                            mask_function=bidir_mask_function)
            loss = outputs.loss
        else:
            # causal: SDPA/Flash/Eager もOK。mask_function は渡さない
            outputs = model(input_ids=inputs["input_ids"],
                            attention_mask=inputs.get("attention_mask", None),
                            labels=inputs["labels"])
            loss = outputs.loss

        # ログ（tokens/sec, TFLOPsの概算）
        with torch.no_grad():
            # 今バッチの有効トークン数
            if mode == "bidir":
                active = (y != -100).sum().item()
            else:
                # CLM: 内部シフトで実質 S-1 トークンが対象（paddingがあれば collator 側で -100）
                active = (inputs["labels"] != -100).sum().item()
            self._tok_total += active
            t_now = time.time()
            dt = max(1e-6, t_now - self._t_last)
            dTok = self._tok_total - self._tok_last
            tps = dTok / dt
            tflops = (6.0 * self._n_params * tps) / 1e12
            # 次回用
            self._t_last = t_now
            self._tok_last = self._tok_total

            # 代表的なログ値
            if mode == "bidir":
                self.log({"loss_bidir": float(loss.detach().cpu()),
                          "tps_active": float(tps), "tflops": float(tflops)})
            else:
                self.log({"loss_causal": float(loss.detach().cpu()),
                          "tps_active": float(tps), "tflops": float(tflops)})

        return (loss, outputs) if return_outputs else loss

# ====================== main ======================
def parse_args():
    p = argparse.ArgumentParser()
    # モデル/注意実装
    p.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-0.6B")
    p.add_argument("--attn_impl", type=str, default="sdpa",
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

    # 目的切替
    p.add_argument("--mask_policy", type=str, default="causal",
                   choices=["causal", "bidir", "alternate"])
    p.add_argument("--bidir_ratio", type=float, default=0.2)

    # TrainingArguments 相当（主要どころ）
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

    # Tokenizer（語彙は増やさない。mask_token は既存/UNK）
    tok = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True, trust_remote_code=True)
    # pad を使わない（固定長パック）ので pad_token 設定は不要
    # 参考: 既存 mask_token が無ければ後で UNK を代用する

    # Model
    model = Qwen3ForCausalLM.from_pretrained(
        args.model_name_or_path,
        attn_implementation=args.attn_impl,
        use_cache=False,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.bf16 and torch.cuda.is_available() else torch.float32,
    )

    # Data: streaming fixed-length packing
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))

    train_ds = FineWebPackedIterable(
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

    # DataLoader + collator（CLM: 未シフト labels）
    collator = DataCollatorForLanguageModeling(tok, mlm=False)
    train_dl = DataLoader(train_ds, batch_size=args.per_device_train_batch_size,
                          shuffle=False, drop_last=True, num_workers=0,
                          collate_fn=collator)

    # TrainingArguments
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

    # Trainer（CLM/bidir 切替）
    trainer = MixMaskTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,   # IterableDataset を直渡し（Trainer内部でDL構築）
        tokenizer=tok,
        data_collator=collator,
        mask_policy=args.mask_policy,
        bidir_ratio=args.bidir_ratio,
        attn_impl=args.attn_impl,
    )

    # ちょいサニティ表示
    if trainer.is_world_process_zero():
        emb = model.get_input_embeddings().weight
        head = model.lm_head.weight
        print({
            "vocab_size": len(tok),
            "emb_vs_head_tied": (emb.data_ptr() == head.data_ptr()),
            "attn_impl": args.attn_impl,
            "mask_policy": args.mask_policy,
            "bidir_ratio": args.bidir_ratio,
        })

    trainer.train()

    if trainer.is_world_process_zero():
        trainer.save_model(args.output_dir)   # 最終チェックポイント保存
        tok.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
