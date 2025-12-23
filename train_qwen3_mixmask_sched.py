#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3 (0.6B〜8B) 継続学習: CLM + bidir 切替学習（FineWeb対応・固定長パック）

修正ポイント（重要）:
- CLM の labels は「非シフト」のまま model に渡す（HF 側で自動シフト）。← 二重シフト防止
- bidir(MNTP等価): 「(t+1) をマスクした位置のみ labels の列 (t+1) に x_{t+1} を置く」
  => HF の shift 規約 (labels[...,1:]) と完全整合
- 語彙は増やさない（<mask> 追加/resize 禁止）。bidir の置換は UNK を代用。

依存:
  pip install -U torch accelerate transformers datasets
"""

import os, time, math, argparse, random
from typing import List, Dict, Tuple, Iterator, Optional

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, IterableDataset

from accelerate import Accelerator
from accelerate.utils import DistributedType
from accelerate.logging import get_logger

from transformers import AutoTokenizer, Qwen3ForCausalLM, get_cosine_schedule_with_warmup

# ===================== Attention mask callbacks =====================
def causal_mask(b, h, q_idx, kv_idx):
    # 標準の因果マスク（下三角）
    return q_idx >= kv_idx

def full_visible_mask(b, h, q_idx, kv_idx):
    # 双方向（全可視）
    return torch.ones_like(q_idx >= 0, dtype=torch.bool)

# ===================== Toy dataset (簡易動作確認用) =====================
class ToyIDs(Dataset):
    def __init__(self, vocab_size=32000, n=10000, seqlen=2048, seed=7):
        g = torch.Generator().manual_seed(seed)
        self.data = torch.randint(10, vocab_size, (n, seqlen), generator=g)
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return {"input_ids": self.data[i]}

# ===================== FineWeb: streaming packer =====================
class FineWebPackedIterable(IterableDataset):
    """
    HF datasets を streaming 読み→rank shard→buffer shuffle→tokenize(add_special_tokens=False)
    → ドキュメント末尾に EOS を1つ付与 → 連結 → seqlen 固定長でパック
    add_bos_at_chunk_start=True なら、各チャンク先頭トークンを BOS に差し替え（長さ不変）
    """
    def __init__(
        self,
        *,
        hf_name: str,
        hf_config: Optional[str],
        hf_split: str,
        tokenizer: AutoTokenizer,
        seqlen: int,
        world_size: int,
        rank: int,
        text_key: str = "text",
        streaming: bool = True,
        shuffle_buffer: int = 10_000,
        seed: int = 42,
        cache_dir: Optional[str] = None,
        add_eos_between_docs: bool = True,
        add_bos_at_chunk_start: bool = False,
    ):
        super().__init__()
        self.hf_name = hf_name
        self.hf_config = hf_config
        self.hf_split = hf_split
        self.tok = tokenizer
        self.seqlen = seqlen
        self.world_size = max(1, world_size)
        self.rank = max(0, rank)
        self.text_key = text_key
        self.streaming = streaming
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.cache_dir = cache_dir
        self.add_eos_between_docs = add_eos_between_docs
        self.add_bos_at_chunk_start = add_bos_at_chunk_start
        assert tokenizer.eos_token_id is not None, "tokenizer に eos_token が必要です。"

    def _hf_iter(self):
        from datasets import load_dataset
        ds = load_dataset(
            self.hf_name,
            self.hf_config if self.hf_config else None,
            split=self.hf_split,
            streaming=self.streaming,
            cache_dir=self.cache_dir,
        )
        if hasattr(ds, "shard"):
            ds = ds.shard(num_shards=self.world_size, index=self.rank)
        if hasattr(ds, "shuffle") and self.shuffle_buffer > 0:
            ds = ds.shuffle(buffer_size=self.shuffle_buffer, seed=self.seed)
        return ds

    def _tokenize_text(self, txt: str) -> List[int]:
        ids = self.tok(txt, add_special_tokens=False, return_attention_mask=False)["input_ids"]
        if not isinstance(ids, list): ids = list(ids)
        return ids

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        eos = self.tok.eos_token_id
        buf: List[int] = []
        for ex in self._hf_iter():
            txt = ex.get(self.text_key, "")
            if not isinstance(txt, str) or len(txt) == 0:
                continue
            ids = self._tokenize_text(txt)
            if self.add_eos_between_docs:
                ids = ids + [eos]
            buf.extend(ids)
            while len(buf) >= self.seqlen:
                chunk = buf[:self.seqlen]
                buf = buf[self.seqlen:]
                if self.add_bos_at_chunk_start:
                    bos = self.tok.bos_token_id
                    if bos is not None:
                        chunk[0] = bos
                yield {"input_ids": torch.tensor(chunk, dtype=torch.long)}

# ===================== collate: ★非シフト★ =====================
def clm_collate(batch):
    """
    HF CausalLM は内部で shift するので、ここでは labels=ids をそのまま渡す
    """
    ids = torch.stack([ex["input_ids"] for ex in batch], dim=0)
    labels = ids.clone()  # -100 は不要（内部で labels[...,1:] を参照）
    return {"input_ids": ids, "labels": labels}

# ===================== distributed-safe utils =====================
def allreduce_sum_scalar(x: float, device=None) -> float:
    t = torch.tensor([x], device=device or ("cuda" if torch.cuda.is_available() else "cpu"), dtype=torch.float32)
    if dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item())

def allreduce_max_scalar(x: float, device=None) -> float:
    t = torch.tensor([x], device=device or ("cuda" if torch.cuda.is_available() else "cpu"), dtype=torch.float32)
    if dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())

def global_grad_norm(model) -> float:
    sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            sq += p.grad.detach().float().pow(2).sum().item()
    return math.sqrt(max(allreduce_sum_scalar(sq), 1e-30))

def global_weight_norm(model) -> float:
    sq = 0.0
    for p in model.parameters():
        sq += p.detach().float().pow(2).sum().item()
    return math.sqrt(max(allreduce_sum_scalar(sq), 1e-30))

# ===================== save (Accelerate/FSDP safe) =====================
def save_hf_checkpoint(accelerator: Accelerator, model, tokenizer, outdir: str, tag: str):
    accelerator.wait_for_everyone()
    state_dict = accelerator.get_state_dict(model)
    unwrapped = accelerator.unwrap_model(model)
    save_dir = os.path.join(outdir, tag)
    if accelerator.is_main_process:
        os.makedirs(save_dir, exist_ok=True)
        unwrapped.save_pretrained(save_dir, state_dict=state_dict, safe_serialization=True)
        tokenizer.save_pretrained(save_dir)
    accelerator.wait_for_everyone()

# ===================== bidir views (MNTP等価, 100/0/0) =====================
def make_bidir_views(
    input_ids: torch.Tensor,
    *,
    mask_id: int,
    mask_ratio: float = 0.2,
    exclude_special: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    HFシフト規約に整合:
    - to_mask: (t+1) をマスクするブール
    - labels は「(t+1) が選ばれた位置の列 (t+1) に x_{t+1} を置く」（それ以外は -100）
    - 入力 x は to_mask の位置を mask_id に置換
    """
    B, S = input_ids.shape
    labels = torch.full_like(input_ids, -100)
    to_mask = (torch.rand_like(input_ids.float()) < mask_ratio)

    active_count = 0
    if S > 1:
        sel = to_mask[:, 1:]                # 列 1..S-1（= t+1）
        labels[:, 1:][sel] = input_ids[:, 1:][sel]
        active_count = int(sel.sum().item())

    x = input_ids.clone()
    x[to_mask] = mask_id
    return x, labels, active_count

# ===================== main =====================
def parse_list(s: str) -> List[str]:
    return [t.strip() for t in s.split(",") if t.strip()]

def main():
    ap = argparse.ArgumentParser()
    # 基本
    ap.add_argument("--model_id", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--attn_impl", default="sdpa", choices=["flex_attention", "sdpa", "flash_attention_2", "eager"])
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup_steps", type=int, default=100)
    ap.add_argument("--max_steps", type=int, default=1000)
    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--max_grad_norm", type=float, default=None)

    # 目的切替
    ap.add_argument("--mask_policy", choices=["both", "alternate"], default="alternate")
    ap.add_argument("--both_masks", default="causal,bidir")
    ap.add_argument("--both_weights", default=None)  # 例 "1.0,1.0"
    ap.add_argument("--alternate_order", default="causal,bidir")
    ap.add_argument("--alt_unit", choices=["micro", "step"], default="micro")

    # bidir 指定
    ap.add_argument("--bidir_ratio", type=float, default=0.2)

    # データ
    ap.add_argument("--data_source", choices=["toy", "fineweb"], default="fineweb")
    ap.add_argument("--hf_name", type=str, default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--hf_config", type=str, default=None)
    ap.add_argument("--hf_split", type=str, default="train")
    ap.add_argument("--hf_text_key", type=str, default="text")
    ap.add_argument("--hf_streaming", action="store_true", default=True)
    ap.add_argument("--hf_shuffle_buffer", type=int, default=10000)
    ap.add_argument("--hf_cache_dir", type=str, default=None)
    ap.add_argument("--add_bos_at_chunk_start", action="store_true", default=False)

    # 保存/計測
    ap.add_argument("--output_dir", type=str, default="ckpt_out")
    ap.add_argument("--save_every", type=int, default=0)
    ap.add_argument("--report_every", type=int, default=20)
    ap.add_argument("--gpu_peak_tflops", type=float, default=None)
    ap.add_argument("--profile_timing", action="store_true")

    args = ap.parse_args()

    # Accelerator（CPUならAMPオフ）
    mp = "bf16" if (args.bf16 and torch.cuda.is_available()) else "no"
    accelerator = Accelerator(
        log_with=None,
        gradient_accumulation_steps=args.grad_accum,
        mixed_precision=mp,
    )
    logger = get_logger(__name__, log_level="INFO")
    is_main = accelerator.is_main_process
    world_size = accelerator.num_processes
    rank = accelerator.process_index

    # Tokenizer（語彙は増やさない。<mask> は追加しない）
    tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=True, trust_remote_code=True)
    # bidir の置換トークンは既存の UNK を代用
    mask_id = tok.unk_token_id
    assert mask_id is not None, "tokenizer に unk_token が必要です。"

    # Model
    dtype = torch.bfloat16 if (args.bf16 and torch.cuda.is_available()) else torch.float32
    attn_impl = args.attn_impl
    flex_available = False
    if attn_impl == "flex_attention":
        try:
            import torch.nn.attention.flex_attention as _fa  # noqa: F401
            flex_available = True
        except Exception:
            flex_available = False
            attn_impl = "sdpa"
            if is_main:
                print("[warn] flex_attention が見つからないため sdpa にフォールバックします。")

    base_model = Qwen3ForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        attn_implementation=attn_impl,
        use_cache=False,
        trust_remote_code=True,
    )
    n_params_total = sum(p.numel() for p in base_model.parameters())

    # Optim/Sched
    opt = torch.optim.AdamW(base_model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    sch = get_cosine_schedule_with_warmup(opt, args.warmup_steps, args.max_steps)

    # Dataset & DataLoader
    if args.data_source == "toy":
        ds = ToyIDs(seqlen=args.seqlen)
        dl = DataLoader(
            ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
            collate_fn=clm_collate, num_workers=4, persistent_workers=True,
            pin_memory=torch.cuda.is_available()
        )
    else:
        ds = FineWebPackedIterable(
            hf_name=args.hf_name,
            hf_config=args.hf_config,
            hf_split=args.hf_split,
            tokenizer=tok,
            seqlen=args.seqlen,
            world_size=world_size,
            rank=rank,
            text_key=args.hf_text_key,
            streaming=args.hf_streaming,
            shuffle_buffer=args.hf_shuffle_buffer,
            seed=1337,
            cache_dir=args.hf_cache_dir,
            add_eos_between_docs=True,
            add_bos_at_chunk_start=args.add_bos_at_chunk_start,
        )
        dl = DataLoader(
            ds, batch_size=args.batch_size, shuffle=False, drop_last=True,
            collate_fn=clm_collate, num_workers=0,
            pin_memory=torch.cuda.is_available()
        )

    # Prepare (DDP/FSDP)
    model, opt, dl, sch = accelerator.prepare(base_model, opt, dl, sch)
    unwrapped = accelerator.unwrap_model(model)

    if is_main:
        print(f"#params: {n_params_total/1e6:.1f} M, attn_impl={attn_impl}, flex_ok={flex_available}, unk_mask_id={mask_id}")
        if args.data_source == "fineweb":
            print(f"FineWeb: name={args.hf_name}, config={args.hf_config}, split={args.hf_split}, text_key={args.hf_text_key}, streaming={args.hf_streaming}")

    # --- 軽い sanity: tying/語彙/ln|V| と最初の eval_probe ---
    if is_main:
        emb = unwrapped.get_input_embeddings().weight
        head_w = unwrapped.lm_head.weight
        print({
            "vocab_size": len(tok),
            "emb_vs_head_tied": emb.data_ptr() == head_w.data_ptr(),
            "ln_vocab": round(math.log(len(tok)), 4),
        })
        # eval probe（学習なし）
        try:
            model.eval(); tot=0.0; cnt=0
            with torch.no_grad():
                for i, bb in zip(range(4), dl):
                    bb = {k: v.to(accelerator.device) for k, v in bb.items()}
                    out = model(bb["input_ids"], labels=bb["labels"], mask_function=causal_mask)
                    tot += float(out.loss.item()); cnt += 1
            model.train()
            print({"eval_probe_loss": round(tot/max(1,cnt), 4)})
        except Exception as e:
            print({"eval_probe_error": repr(e)})

    # 計画
    allowed_modes = {"causal", "bidir"}
    if args.mask_policy == "both":
        both = parse_list(args.both_masks); assert all(m in allowed_modes for m in both) and len(both) >= 2
        if args.both_weights:
            w = [float(x) for x in parse_list(args.both_weights)]; assert len(w) == len(both)
        else:
            w = [1.0] * len(both)
        mask_plan_static: List[Tuple[str, float]] = list(zip(both, w))
    else:
        cycle = parse_list(args.alternate_order); assert all(m in allowed_modes for m in cycle) and len(cycle) >= 1

    wants_bidir = (
        (args.mask_policy == "both" and any(m == "bidir" for m, _ in (mask_plan_static if 'mask_plan_static' in locals() else [])))
        or (args.mask_policy == "alternate" and any(m == "bidir" for m in cycle))
    )
    if wants_bidir and not flex_available:
        raise RuntimeError("bidir を使うには --attn_impl flex_attention が必要です。CLMのみなら --alternate_order causal --both_masks causal を指定。")

    opt.zero_grad(set_to_none=True)

    ema_alpha = 0.1
    ema_loss = None; ema_tps_act = 0.0; ema_tflops = 0.0
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    last_step_end = time.perf_counter()

    micro_step = 0; opt_step = 0; t0 = time.perf_counter()
    model.train()

    while opt_step < args.max_steps:
        for batch in dl:
            if opt_step >= args.max_steps: break

            batch_start = time.perf_counter()
            host_gap_ms = (batch_start - last_step_end) * 1000.0

            device = accelerator.device
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            if args.mask_policy == "both":
                plan = mask_plan_static
            else:
                idx = (micro_step if args.alt_unit == "micro" else opt_step) % len(cycle)
                plan = [(cycle[idx], 1.0)]

            use_cuda_ev = args.profile_timing and torch.cuda.is_available()
            if use_cuda_ev:
                torch.cuda.synchronize()
                e_f0 = torch.cuda.Event(True); e_f1 = torch.cuda.Event(True)
                e_b0 = torch.cuda.Event(True); e_b1 = torch.cuda.Event(True)
                e_o0 = torch.cuda.Event(True); e_o1 = torch.cuda.Event(True)

            if use_cuda_ev: e_f0.record()
            loss_items: Dict[str, float] = {}
            bidir_active_this_step = 0

            with accelerator.accumulate(model):
                loss_total = 0.0; weight_sum = 0.0

                for mode, weight in plan:
                    if mode == "causal":
                        out = model(batch["input_ids"], labels=batch["labels"], mask_function=causal_mask)
                        loss_items["causal"] = float(out.loss.detach().item())
                        loss_total += weight * out.loss; weight_sum += weight

                    elif mode == "bidir":
                        xb, yb, act_cnt = make_bidir_views(
                            batch["input_ids"], mask_id=mask_id, mask_ratio=args.bidir_ratio
                        )
                        bidir_active_this_step += act_cnt
                        out = model(xb, labels=yb, mask_function=full_visible_mask)
                        loss_items["bidir"] = float(out.loss.detach().item())
                        loss_total += weight * out.loss; weight_sum += weight

                    else:
                        raise ValueError(f"unknown mode: {mode}")

                loss = loss_total / max(1e-9, weight_sum)
                if use_cuda_ev: e_f1.record()

                if use_cuda_ev: e_b0.record()
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    if args.max_grad_norm is not None:
                        accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    opt.step()
                    if sch is not None: sch.step()
                    opt.zero_grad(set_to_none=True)
                if use_cuda_ev:
                    e_b1.record(); e_o0.record(); torch.cuda.synchronize(); e_o1.record()

            # active tokens
            clm_active_local = int((batch["labels"] != -100).sum().item()) if any(m=="causal" for m,_ in plan) else 0
            effective_active_local = clm_active_local + bidir_active_this_step

            # 経過時間（全rank max）
            t1 = time.perf_counter()
            dt_local = max(1e-9, t1 - t0); t0 = t1

            effective_active_global = allreduce_sum_scalar(effective_active_local, device=device)
            dt_global = allreduce_max_scalar(dt_local, device=device)

            tps_active = effective_active_global / dt_global
            tflops = (6.0 * n_params_total * tps_active) / 1e12

            # EMA
            ema_tps_act = (1 - ema_alpha) * ema_tps_act + ema_alpha * tps_active
            ema_tflops  = (1 - ema_alpha) * ema_tflops  + ema_alpha * tflops
            ema_loss    = loss.item() if ema_loss is None else (0.9 * ema_loss + 0.1 * loss.item())

            # lr / grad_norm / update-to-weight
            lr_now = sch.get_last_lr()[0]
            g_norm = global_grad_norm(model)
            wt_norm = global_weight_norm(model)
            utw = (lr_now * g_norm / wt_norm) if wt_norm else float("nan")

            # メモリ
            if torch.cuda.is_available():
                peak_alloc_gb = torch.cuda.max_memory_allocated() / 1e9
                peak_res_gb   = torch.cuda.max_memory_reserved() / 1e9
                torch.cuda.reset_peak_memory_stats()
            else:
                peak_alloc_gb = peak_res_gb = 0.0

            # 時間内訳
            if use_cuda_ev:
                fwd_ms  = e_f0.elapsed_time(e_f1)
                bwd_ms  = e_b0.elapsed_time(e_b1)
                optim_ms= e_o0.elapsed_time(e_o1)
            else:
                fwd_ms = bwd_ms = optim_ms = 0.0

            # ログ
            if is_main and (micro_step % args.report_every == 0):
                log = dict(
                    micro=micro_step, step=opt_step, plan=[m for m, _ in plan],
                    loss=float(loss.item()), loss_ema=round(ema_loss, 4),
                    loss_causal=round(loss_items.get("causal", float("nan")), 4),
                    loss_bidir=round(loss_items.get("bidir", float("nan")), 4),
                    lr=lr_now, grad_norm=round(g_norm, 3),
                    update_to_weight=round(utw, 6),
                    tps_active=round(tps_active, 1), tps_active_ema=round(ema_tps_act, 1),
                    tflops=round(tflops, 2), tflops_ema=round(ema_tflops, 2),
                    host_gap_ms=round(host_gap_ms, 2),
                    fwd_ms=round(fwd_ms, 2), bwd_ms=round(bwd_ms, 2), optim_ms=round(optim_ms, 2),
                    peak_alloc_gb=round(peak_alloc_gb, 2), peak_reserved_gb=round(peak_res_gb, 2),
                )
                print(log, flush=True)

            # 定期保存
            if (args.save_every > 0) and accelerator.sync_gradients and (opt_step % args.save_every == 0) and (opt_step > 0):
                save_hf_checkpoint(accelerator, model, tok, args.output_dir, f"step{opt_step:06d}")

            micro_step += 1
            if accelerator.sync_gradients:
                opt_step += 1
                last_step_end = time.perf_counter()

    if is_main: print("Saving final checkpoint...")
    save_hf_checkpoint(accelerator, model, tok, args.output_dir, "final")

    if accelerator.state.distributed_type != DistributedType.NO:
        dist.barrier()
    if is_main: print("Done.")

if __name__ == "__main__":
    main()
