from __future__ import annotations

import os
import argparse
import json
import inspect

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments

from ..utils.seed import set_seed
from ..masking import get_backbone_from_causallm, sanity_check_suffix_effect
from ..modeling import ensure_mask_token, get_lm_head_module
from ..data.streaming import PackedStreamingIterable
from ..data.collators import clm_collate
from .trainer_dual import DualCLMMLMTrainer
from .callbacks import JsonlCsvLoggerCallback


def _str2bool(x: str) -> bool:
    return str(x).lower() in ("1", "true", "yes", "y", "t")


def parse_args():
    p = argparse.ArgumentParser()

    # model
    p.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-0.6B")
    p.add_argument("--attn_impl", type=str, default="flash_attention_2",
                   choices=["sdpa", "flash_attention_2", "eager", "flex_attention"])
    p.add_argument("--bf16", type=_str2bool, default=True)
    p.add_argument("--fp16", type=_str2bool, default=False)
    p.add_argument("--gradient_checkpointing", action="store_true")

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

    # dual loss
    p.add_argument("--mask_policy", type=str, default="both",
                   choices=["clm", "mlm", "alternate", "both", "causal", "bidir"])
    p.add_argument("--mlm_mask_ratio", type=float, default=0.15)
    p.add_argument("--both_weights", type=str, default="1.0,1.0")

    p.add_argument("--ensure_mask_token", type=_str2bool, default=True)
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
    p.add_argument("--ddp_find_unused_parameters", type=_str2bool, default=False)
    p.add_argument("--deepspeed", type=str, default=None)

    # logging files
    p.add_argument("--log_jsonl", type=str, default=None)
    p.add_argument("--log_csv", type=str, default=None)

    # optional: sanity check before train
    p.add_argument("--sanity_check", type=_str2bool, default=True)

    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    if args.log_jsonl is None:
        args.log_jsonl = os.path.join(args.output_dir, "train_log.jsonl")
    if args.log_csv is None:
        args.log_csv = os.path.join(args.output_dir, "train_log.csv")

    tok = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True, trust_remote_code=True)

    use_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if (args.bf16 and use_cuda) else (
        torch.float16 if (args.fp16 and use_cuda) else torch.float32
    )
    if args.attn_impl == "flash_attention_2":
        if not use_cuda:
            raise RuntimeError("flash_attention_2 requires CUDA.")
        if dtype not in (torch.float16, torch.bfloat16):
            raise RuntimeError("flash_attention_2 requires fp16/bf16 (set --bf16 True or --fp16 True).")

    # load model
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            attn_implementation=args.attn_impl,
            trust_remote_code=True,
            torch_dtype=dtype,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            attn_implementation=args.attn_impl,
            trust_remote_code=True,
            dtype=dtype,
        )

    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable") and args.gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass

    # MLM needs mask token
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
                raise RuntimeError("mask_token_id not found. Use --ensure_mask_token True.")
            mask_id = int(mask_id)

    # dataset (streaming + packing)
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
        seed=args.seed,
        world_size=world_size,
        rank=rank,
        add_eos_between_docs=True,
        add_bos_at_chunk_start=args.add_bos_at_chunk_start,
        cache_dir=args.cache_dir,
    )

    # TrainingArguments
    targs_kwargs = dict(
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
        bf16=(args.bf16 and use_cuda),
        fp16=(args.fp16 and use_cuda),
        gradient_checkpointing=args.gradient_checkpointing,
        ddp_find_unused_parameters=args.ddp_find_unused_parameters,
        report_to="none",
        remove_unused_columns=False,
    )
    if args.deepspeed:
        targs_kwargs["deepspeed"] = args.deepspeed

    targs = TrainingArguments(**targs_kwargs)

    # tokenizer kw compatibility
    trainer_init_sig = inspect.signature(DualCLMMLMTrainer.__init__)
    trainer_tokenizer_kw = {}
    if "processing_class" in trainer_init_sig.parameters:
        trainer_tokenizer_kw["processing_class"] = tok
    else:
        trainer_tokenizer_kw["tokenizer"] = tok

    trainer = DualCLMMLMTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        data_collator=clm_collate,
        mask_policy=args.mask_policy,
        mlm_mask_ratio=args.mlm_mask_ratio,
        both_weights=args.both_weights,
        mask_token_id=mask_id,
        **trainer_tokenizer_kw,
    )
    trainer.add_callback(JsonlCsvLoggerCallback(jsonl_path=args.log_jsonl, csv_path=args.log_csv))

    # Sanity check (uses same backbone path as training)
    if args.sanity_check and trainer.is_world_process_zero():
        backbone = get_backbone_from_causallm(model)
        lm_head = get_lm_head_module(model)
        device = next(model.parameters()).device
        out = sanity_check_suffix_effect(
            tokenizer=tok,
            backbone=backbone,
            lm_head=lm_head,
            device=device,
        )
        print("[sanity_check]", json.dumps(out, indent=2))
        if out["diff_causal"] > 1e-3 or out["diff_bidir"] < 1e-2:
            print("[warn] sanity_check suggests toggle may be wrong in this environment.")

    if trainer.is_world_process_zero():
        backbone = get_backbone_from_causallm(model)
        print({
            "vocab_size": len(tok),
            "mask_token": getattr(tok, "mask_token", None),
            "mask_token_id": getattr(tok, "mask_token_id", None),
            "attn_impl": args.attn_impl,
            "dtype": str(next(model.parameters()).dtype),
            "mask_policy": args.mask_policy,
            "backbone_type": str(type(backbone)),
            "backbone_accepts_is_causal": True,  # Qwen3 is True in your test
            "logging_steps": int(args.logging_steps),
            "deepspeed": args.deepspeed,
            "log_jsonl": args.log_jsonl,
            "log_csv": args.log_csv,
        })

    trainer.train()

    if trainer.is_world_process_zero():
        trainer.save_model(args.output_dir)
        tok.save_pretrained(args.output_dir)
        print("[done] saved to", args.output_dir)


if __name__ == "__main__":
    main()
