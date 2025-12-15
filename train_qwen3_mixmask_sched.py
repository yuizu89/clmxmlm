#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3 (0.6B〜8B) 継続学習: CLM + bidir（MNTP等価, 100/0/0）
- 目的切替:
    (A) --mask_policy both       : 同一バッチで causal と bidir を実行し平均損失で更新
    (B) --mask_policy alternate  : 指定順で目的を循環（micro or step 単位）
- bidir: 選択トークンは必ず [MASK]（100/0/0）。損失は「右の真値 x_i を左 i-1 で予測」。
- 注意: bidir（=双方向注意）は **PyTorch FlexAttention** 経由でのみ安全に実現。
        → attn_impl が flex_attention 以外のときに bidir を使おうとしたらエラーで止めます。
- 計測: loss / loss_causal / loss_bidir / tokens/sec(active, global) / TFLOPs(global) /
        lr / grad_norm(global) / update_to_weight / 勾配類似度(任意) /
        GPUメモリ / fwd,bwd,optim 時間 / host待ち時間
- 保存: save_pretrained（Accelerate 経由; FSDP/ZeRO対応）
依存: torch>=2.6, transformers>=4.57, accelerate>=0.33
"""

import os, time, math, argparse
from typing import List, Dict, Tuple
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from accelerate.utils import DistributedType
from accelerate.logging import get_logger
from transformers import AutoTokenizer, Qwen3ForCausalLM, get_cosine_schedule_with_warmup

# ---- attention masks ----
def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx  # 因果（下三角）

def full_visible_mask(b, h, q_idx, kv_idx):
    return torch.ones_like(q_idx >= 0, dtype=torch.bool)  # 双方向（全可視）

# ---- toy dataset / collator (CLM 用の右シフト) ----
class ToyIDs(Dataset):
    def __init__(self, vocab_size=32000, n=10000, seqlen=2048, seed=7):
        g = torch.Generator().manual_seed(seed)
        self.data = torch.randint(10, vocab_size, (n, seqlen), generator=g)
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return {"input_ids": self.data[i]}

def simple_collate(batch, pad_id=0):
    ids = torch.stack([ex["input_ids"] for ex in batch], dim=0)
    labels = ids.clone()
    labels[:, :-1] = ids[:, 1:]
    labels[:, -1] = -100
    return {"input_ids": ids, "labels": labels}

# ---- distributed-safe helpers ----
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

@torch.no_grad()
def _dot_norms_from_two_grad_lists(grads_a, grads_b, device):
    dot = torch.tensor(0.0, device=device); na = torch.tensor(0.0, device=device); nb = torch.tensor(0.0, device=device)
    for ga, gb in zip(grads_a, grads_b):
        if ga is None or gb is None: continue
        ga = ga.float(); gb = gb.float()
        dot += (ga * gb).sum(); na += (ga * ga).sum(); nb += (gb * gb).sum()
    if dist.is_initialized():
        dist.all_reduce(dot); dist.all_reduce(na); dist.all_reduce(nb)
    return float(dot.item()), float(na.item()), float(nb.item())

def grad_cosine_similarity_from_losses(model, loss_a, loss_b) -> float:
    params = [p for p in model.parameters() if p.requires_grad]
    ga = torch.autograd.grad(loss_a, params, retain_graph=False, create_graph=False, allow_unused=True)
    gb = torch.autograd.grad(loss_b, params, retain_graph=False, create_graph=False, allow_unused=True)
    device = loss_a.device
    dot, na, nb = _dot_norms_from_two_grad_lists(ga, gb, device=device)
    if na <= 0.0 or nb <= 0.0: return float("nan")
    return dot / (math.sqrt(na) * math.sqrt(nb) + 1e-12)

def parse_list(s: str) -> List[str]:
    return [t.strip() for t in s.split(",") if t.strip()]

# ---- save (Accelerate/FSDP safe) ----
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

# ---- bidir views (MNTP等価, 100/0/0) ----
def make_bidir_views(
    input_ids: torch.Tensor,
    *,
    tokenizer: AutoTokenizer,
    mask_ratio: float = 0.2,
    exclude_special: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    bidir（MNTP等価, 100/0/0）: 選ばれた i は必ず [MASK]。
    labels[:, t] は (t+1) が選ばれている場合のみ x_{t+1}、それ以外は -100。
    戻り値: (masked_input_ids, labels, active_count)
    """
    device = input_ids.device
    B, S = input_ids.shape
    labels = torch.full_like(input_ids, -100)

    mask_id = tokenizer.mask_token_id
    assert mask_id is not None, "mask_token_id がありません。'<mask>' を追加し model を resize 済みである必要があります。"

    eligible = torch.ones_like(input_ids, dtype=torch.bool)
    eligible[:, 0] = False  # i-1 が存在しない先頭は除外
    if exclude_special:
        for tid in [tokenizer.pad_token_id, tokenizer.eos_token_id,
                    tokenizer.bos_token_id, tokenizer.unk_token_id, mask_id]:
            if tid is not None:
                eligible &= (input_ids != tid)

    to_mask = (torch.rand_like(input_ids.float()) < mask_ratio) & eligible  # (B,S)

    active_count = 0
    if S > 1:
        m_right = to_mask[:, 1:]
        tgt = input_ids[:, 1:]
        labels[:, :-1][m_right] = tgt[m_right]
        active_count = int(m_right.sum().item())

    x = input_ids.clone()
    if to_mask.any():
        idx = to_mask.nonzero(as_tuple=False)
        x[idx[:, 0], idx[:, 1]] = mask_id

    return x, labels, active_count

# ---- main ----
def main():
    ap = argparse.ArgumentParser()
    # 基本
    ap.add_argument("--model_id", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--attn_impl", default="flex_attention",
                    choices=["flex_attention", "sdpa", "flash_attention_2", "eager"])
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

    # bidir
    ap.add_argument("--bidir_ratio", type=float, default=0.2,
                    help="入力から“予測対象”に選ぶ割合（先頭位置は除外）")
    ap.add_argument("--bidir_exclude_special", action="store_true", default=True)

    # 保存
    ap.add_argument("--output_dir", type=str, default="ckpt_out")
    ap.add_argument("--save_every", type=int, default=0)

    # 測定
    ap.add_argument("--report_every", type=int, default=20)
    ap.add_argument("--gpu_peak_tflops", type=float, default=None)
    ap.add_argument("--profile_timing", action="store_true")

    # 勾配類似度（任意）
    ap.add_argument("--grad_sim_every", type=int, default=0)
    ap.add_argument("--grad_sim_pair", default="causal,bidir")

    # データ（デモ）
    ap.add_argument("--dataset", default="toy", choices=["toy"])

    args = ap.parse_args()

    # ★ grad_accum を Accelerator に反映（以前は未適用だった）
    accelerator = Accelerator(log_with=None, gradient_accumulation_steps=args.grad_accum)
    logger = get_logger(__name__, log_level="INFO")
    is_main = accelerator.is_main_process

    # tokenizer（[MASK] を保証）
    tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    added_mask = False
    if tok.mask_token is None:
        tok.add_special_tokens({"mask_token": "<mask>"})
        added_mask = True

    # model 準備と attn_impl の可用性チェック
    dtype = torch.bfloat16 if args.bf16 and torch.cuda.is_available() else torch.float32
    attn_impl = args.attn_impl
    flex_available = False
    if attn_impl == "flex_attention":
        try:
            import torch.nn.attention.flex_attention as _fa  # noqa: F401
            flex_available = True
        except Exception:
            flex_available = False

    # bidir を使う計画かどうか（CLIから早期判定）
    wants_bidir = ("bidir" in parse_list(args.both_masks)) or ("bidir" in parse_list(args.alternate_order))

    # flex が必要なのに無い場合は、ここで明確に停止（サイレント劣化を防ぐ）
    if wants_bidir and not flex_available:
        raise RuntimeError(
            "bidir（双方向注意）は flex_attention が必要ですが、環境で見つかりませんでした。\n"
            "PyTorch>=2.6 の FlexAttention を有効にするか、--alternate_order を causal のみへ変更してください。"
        )

    # 実際の attn_impl を決定（bidir を使わないならフォールバック可）
    if attn_impl == "flex_attention" and not flex_available:
        # bidir 不要なら sdpa にフォールバックして続行
        attn_impl = "sdpa"
        if is_main:
            print("[warn] flex_attention が見つからないため sdpa にフォールバックします（bidir は無効化してください）。")

    base_model = Qwen3ForCausalLM.from_pretrained(
        args.model_id, torch_dtype=dtype, attn_implementation=attn_impl, use_cache=False
    )
    if added_mask:
        base_model.resize_token_embeddings(len(tok))
    n_params_total = sum(p.numel() for p in base_model.parameters())

    # optim/sched
    opt = torch.optim.AdamW(base_model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    sch = get_cosine_schedule_with_warmup(opt, args.warmup_steps, args.max_steps)

    # data
    if args.dataset == "toy":
        ds = ToyIDs(seqlen=args.seqlen)
    else:
        raise NotImplementedError("実データへ差し替えてください。")

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=lambda b: simple_collate(b, pad_id=tok.pad_token_id or 0),
        pin_memory=torch.cuda.is_available(),
        num_workers=4,
        persistent_workers=True,
    )

    # prepare
    model, opt, dl, sch = accelerator.prepare(base_model, opt, dl, sch)
    if is_main:
        print(f"#params: {n_params_total/1e6:.1f} M, attn_impl={attn_impl}, flex_ok={flex_available}, mask_token_id={tok.mask_token_id}")

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

    opt.zero_grad(set_to_none=True)

    ema_alpha = 0.1
    ema_loss = None; ema_tps_act = 0.0; ema_tflops = 0.0
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    last_step_end = time.perf_counter()

    micro_step = 0; opt_step = 0; t0 = time.perf_counter()
    model.train()

    pair = parse_list(args.grad_sim_pair)
    if len(pair) != 2 or not all(m in allowed_modes for m in pair):
        if is_main: print(f"[warn] --grad_sim_pair={args.grad_sim_pair} を causal,bidir に修正します")
        pair = ["causal", "bidir"]

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

            # bidir を要求しているのに flex が無い計画になっていないか再確認
            if any(m == "bidir" for m, _ in plan) and not flex_available:
                raise RuntimeError("bidir は flex_attention 必須です。--attn_impl flex_attention にしてください。")

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
                            batch["input_ids"], tokenizer=tok,
                            mask_ratio=args.bidir_ratio, exclude_special=args.bidir_exclude_special,
                        )
                        bidir_active_this_step += act_cnt
                        out = model(xb, labels=yb, mask_function=full_visible_mask)  # 双方向（FlexAttention前提）
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

            # 経過時間（全rankの max）
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

            # lr / grad_norm / update-to-weight（global）
            lr_now = sch.get_last_lr()[0]
            g_norm = global_grad_norm(model)
            wt_norm = global_weight_norm(model)
            utw = (lr_now * g_norm / wt_norm) if wt_norm else float("nan")

            # 勾配コサイン類似度（指定間隔）
            grad_sim = float("nan")
            if (args.grad_sim_every > 0) and accelerator.sync_gradients and (opt_step % args.grad_sim_every == 0):
                def fwd_loss(mode_name: str):
                    if mode_name == "causal":
                        return model(batch["input_ids"], labels=batch["labels"], mask_function=causal_mask).loss
                    elif mode_name == "bidir":
                        xi, xl, _ = make_bidir_views(
                            batch["input_ids"], tokenizer=tok,
                            mask_ratio=args.bidir_ratio, exclude_special=args.bidir_exclude_special,
                        )
                        return model(xi, labels=xl, mask_function=full_visible_mask).loss
                    else:
                        raise ValueError(mode_name)
                a, b = parse_list(args.grad_sim_pair)
                loss_a = fwd_loss(a); loss_b = fwd_loss(b)
                grad_sim = grad_cosine_similarity_from_losses(model, loss_a, loss_b)

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
                    grad_sim=round(grad_sim, 4),
                    tps_active=round(tps_active, 1), tps_active_ema=round(ema_tps_act, 1),
                    tflops=round(tflops, 2), tflops_ema=round(ema_tflops, 2),
                    host_gap_ms=round(host_gap_ms, 2),
                    fwd_ms=round(fwd_ms, 2), bwd_ms=round(bwd_ms, 2), optim_ms=round(optim_ms, 2),
                    peak_alloc_gb=round(peak_alloc_gb, 2), peak_reserved_gb=round(peak_res_gb, 2),
                )
                if args.gpu_peak_tflops:
                    log["mfu_%"] = round(100.0 * ema_tflops / args.gpu_peak_tflops, 1)
                print(log, flush=True)

            # 定期保存（最終マイクロでのみ）
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
