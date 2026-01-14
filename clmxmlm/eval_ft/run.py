# clmxmlm/eval_ft/run.py
from __future__ import annotations

import os
import json
import argparse
import inspect
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np
from tqdm import tqdm

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    DataCollatorForTokenClassification,
)
from datasets import Dataset
import evaluate

# ---- current package imports ----
from ..masking import (
    get_backbone_from_causallm,
    MaskController,
    sanity_check_suffix_effect,
)
from ..modeling import get_lm_head_module

# eval_ft local modules
from .heads import (
    FTMaskCfg,
    DecoderOnlyForSequenceClassification,
    DecoderOnlyForTokenClassification,
    DecoderOnlyForExtractiveQA,
    DPRDualEncoder,
)
from . import tasks as T


# ---------------------- utils ----------------------
def set_seed(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_name(x: str) -> str:
    x = x.rstrip("/").replace("\\", "/")
    x = x.split("/")[-1]
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in x) or "model"


def infer_hidden_size(backbone) -> int:
    cfg = getattr(backbone, "config", None)
    hs = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None)
    if hs is None:
        raise RuntimeError("Could not infer hidden size from backbone.config.")
    return int(hs)


def _trainer_tokenizer_kw(tok):
    """
    transformers のバージョンで Trainer が tokenizer を嫌うことがあるため吸収。
    """
    sig = inspect.signature(Trainer.__init__)
    if "processing_class" in sig.parameters:
        return {"processing_class": tok}
    return {"tokenizer": tok}


def _make_training_args(args, run_dir: str, use_cuda: bool):
    """
    TrainingArguments のバージョン差（evaluation_strategy / eval_strategy 等）を吸収。
    """
    sig = inspect.signature(TrainingArguments.__init__).parameters

    kw = dict(
        output_dir=run_dir,
        learning_rate=float(args.learning_rate),
        lr_scheduler_type="linear",
        max_steps=int(args.max_steps),
        num_train_epochs=float(args.num_train_epochs),
        per_device_train_batch_size=int(args.per_device_train_batch_size),
        per_device_eval_batch_size=int(args.per_device_eval_batch_size),
        gradient_accumulation_steps=int(args.gradient_accumulation_steps),
        weight_decay=float(args.weight_decay),
        logging_steps=int(args.logging_steps),
        report_to="none",
        remove_unused_columns=False,  # important for QA/IR
        seed=int(args.seed),
        bf16=(args.dtype == "bf16" and use_cuda),
        fp16=(args.dtype == "fp16" and use_cuda),
    )

    # warmup: warmup_ratio or warmup_steps
    if "warmup_ratio" in sig:
        kw["warmup_ratio"] = float(args.warmup_ratio)
    else:
        kw["warmup_steps"] = max(0, int(float(args.warmup_ratio) * max(1, int(args.max_steps))))

    # save/eval strategy naming differences
    if "save_strategy" in sig:
        kw["save_strategy"] = args.save_strategy

    if "evaluation_strategy" in sig:
        kw["evaluation_strategy"] = args.eval_strategy
    elif "eval_strategy" in sig:
        kw["eval_strategy"] = args.eval_strategy

    if "eval_steps" in sig:
        kw["eval_steps"] = int(args.eval_steps)

    return TrainingArguments(**kw)


# ---------------------- SC ----------------------
def build_sc(tokenizer, train_ds, eval_ds, task: str, max_length: int):
    k1, k2 = T.SC_KEYS[task]

    def tok_fn(ex):
        if k2 is None:
            return tokenizer(ex[k1], truncation=True, max_length=max_length)
        return tokenizer(ex[k1], ex[k2], truncation=True, max_length=max_length)

    train_tok = train_ds.map(
        tok_fn,
        batched=True,
        remove_columns=[c for c in train_ds.column_names if c not in ("label",)],
    )
    eval_tok = eval_ds.map(
        tok_fn,
        batched=True,
        remove_columns=[c for c in eval_ds.column_names if c not in ("label",)],
    )

    train_tok = train_tok.rename_column("label", "labels")
    eval_tok = eval_tok.rename_column("label", "labels")

    collator = DataCollatorWithPadding(tokenizer)

    if task in ("sc_sst2", "sc_mnli"):
        metric = evaluate.load("accuracy")

        def compute_metrics(p):
            preds = np.argmax(p.predictions, axis=-1)
            return metric.compute(predictions=preds, references=p.label_ids)

    elif task == "sc_qqp":
        glue = evaluate.load("glue", "qqp")

        def compute_metrics(p):
            preds = np.argmax(p.predictions, axis=-1)
            out = glue.compute(predictions=preds, references=p.label_ids)
            out["acc_f1_avg"] = float(out["accuracy"] + out["f1"]) * 0.5
            return out

    else:
        metric = evaluate.load("accuracy")

        def compute_metrics(p):
            preds = np.argmax(p.predictions, axis=-1)
            return metric.compute(predictions=preds, references=p.label_ids)

    return train_tok, eval_tok, collator, compute_metrics


# ---------------------- TC ----------------------
def build_tc(tokenizer, train_ds, eval_ds, label_list: List[str], tok_col: str, lab_col: str, max_length: int):
    def tokenize_and_align(examples):
        tokenized = tokenizer(
            examples[tok_col],
            truncation=True,
            is_split_into_words=True,
            max_length=max_length,
        )
        all_labels = examples[lab_col]
        new_labels = []
        for i, labels in enumerate(all_labels):
            word_ids = tokenized.word_ids(batch_index=i)
            prev = None
            aligned = []
            for w in word_ids:
                if w is None:
                    aligned.append(-100)
                elif w != prev:
                    aligned.append(int(labels[w]))
                else:
                    aligned.append(-100)
                prev = w
            new_labels.append(aligned)
        tokenized["labels"] = new_labels
        return tokenized

    train_tok = train_ds.map(tokenize_and_align, batched=True, remove_columns=train_ds.column_names)
    eval_tok = eval_ds.map(tokenize_and_align, batched=True, remove_columns=eval_ds.column_names)
    collator = DataCollatorForTokenClassification(tokenizer)

    seqeval = evaluate.load("seqeval")

    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=-1)
        labels = p.label_ids
        true_preds = []
        true_labels = []
        for pr, lb in zip(preds, labels):
            tp = []
            tl = []
            for pi, li in zip(pr, lb):
                if li == -100:
                    continue
                tp.append(label_list[int(pi)])
                tl.append(label_list[int(li)])
            true_preds.append(tp)
            true_labels.append(tl)
        return seqeval.compute(predictions=true_preds, references=true_labels)

    return train_tok, eval_tok, collator, compute_metrics


# ---------------------- QA ----------------------
def build_qa_features(tokenizer, train_ds: Dataset, eval_ds: Dataset, task: str, max_length: int, doc_stride: int = 128):
    pad_on_right = True
    if task == "qa_record":
        q_col, c_col = "query", "passage"
    else:
        q_col, c_col = "question", "context"

    def _get_answers(ex):
        if task == "qa_record":
            ans = ex.get("answers", [])
            if isinstance(ans, dict) and "text" in ans:
                ans = ans["text"]
            return [str(a) for a in (ans or [])]
        ans = ex.get("answers", None)
        if ans is None:
            return []
        if isinstance(ans, dict) and "text" in ans:
            return [str(t) for t in ans["text"]]
        return [str(a) for a in ans]

    def prepare_train_features(examples):
        questions = [q.lstrip() for q in examples[q_col]]
        contexts = examples[c_col]

        tokenized = tokenizer(
            questions if pad_on_right else contexts,
            contexts if pad_on_right else questions,
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=False,
        )

        sample_mapping = tokenized.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized.pop("offset_mapping")

        start_positions = []
        end_positions = []

        for i, offsets in enumerate(offset_mapping):
            sample_idx = sample_mapping[i]
            context = contexts[sample_idx]
            answers = _get_answers({k: examples[k][sample_idx] for k in examples.keys()})

            if task == "qa_squad_v2" and len(answers) == 0:
                start_positions.append(0)
                end_positions.append(0)
                continue

            answer_text = answers[0] if answers else ""
            if not answer_text:
                start_positions.append(0)
                end_positions.append(0)
                continue

            # ---- robust answer_start handling for batched datasets ----
            if task == "qa_record":
                start_char = context.find(answer_text)
            else:
                # SQuAD/SQuADv2 batched map: examples["answers"] is List[Dict[text, answer_start]]
                ans_entry = examples.get("answers", None)
                start_char = -1
                if isinstance(ans_entry, list) and sample_idx < len(ans_entry):
                    ae = ans_entry[sample_idx]
                    if isinstance(ae, dict) and "answer_start" in ae and ae["answer_start"]:
                        start_char = int(ae["answer_start"][0])
                if start_char < 0:
                    start_char = context.find(answer_text)

            if start_char < 0:
                start_positions.append(0)
                end_positions.append(0)
                continue

            end_char = start_char + len(answer_text)

            sequence_ids = tokenized.sequence_ids(i)
            ctx_index = 1 if pad_on_right else 0

            # Find context span in tokens
            token_start = 0
            while token_start < len(sequence_ids) and sequence_ids[token_start] != ctx_index:
                token_start += 1
            token_end = len(sequence_ids) - 1
            while token_end >= 0 and sequence_ids[token_end] != ctx_index:
                token_end -= 1

            # Guard: if no context span found
            if token_start >= len(sequence_ids) or token_end < 0 or token_start > token_end:
                start_positions.append(0)
                end_positions.append(0)
                continue

            # If answer not fully inside this window, label as (0,0)
            if not (offsets[token_start][0] <= start_char and offsets[token_end][1] >= end_char):
                start_positions.append(0)
                end_positions.append(0)
            else:
                # IMPORTANT: keep original context span bounds
                ctx_start = token_start
                ctx_end = token_end

                start = ctx_start
                while start <= ctx_end and offsets[start][0] <= start_char:
                    start += 1
                sp = max(ctx_start, start - 1)

                end = ctx_end
                while end >= ctx_start and offsets[end][1] >= end_char:
                    end -= 1
                ep = min(ctx_end, end + 1)

                start_positions.append(int(sp))
                end_positions.append(int(ep))

        tokenized["start_positions"] = start_positions
        tokenized["end_positions"] = end_positions
        return tokenized

    def prepare_eval_features(examples):
        questions = [q.lstrip() for q in examples[q_col]]
        contexts = examples[c_col]

        tokenized = tokenizer(
            questions if pad_on_right else contexts,
            contexts if pad_on_right else questions,
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=False,
        )

        sample_mapping = tokenized.pop("overflow_to_sample_mapping")
        tokenized["example_id"] = []

        for i in range(len(tokenized["input_ids"])):
            sample_idx = sample_mapping[i]
            ex_id = examples.get("id", None)
            if ex_id is None and task == "qa_record":
                ex_id = examples.get("idx", None)
            tokenized["example_id"].append(ex_id[sample_idx] if ex_id is not None else str(sample_idx))

            sequence_ids = tokenized.sequence_ids(i)
            ctx_index = 1 if pad_on_right else 0
            offsets = tokenized["offset_mapping"][i]
            tokenized["offset_mapping"][i] = [o if sequence_ids[k] == ctx_index else None for k, o in enumerate(offsets)]

        return tokenized

    train_features = train_ds.map(prepare_train_features, batched=True, remove_columns=train_ds.column_names)
    eval_features = eval_ds.map(prepare_eval_features, batched=True, remove_columns=eval_ds.column_names)
    return train_features, eval_features


def postprocess_qa_predictions(examples: Dataset, features: Dataset, raw_predictions, n_best_size=20, max_answer_length=30):
    start_logits, end_logits = raw_predictions
    preds = {}

    feats_by_eid: Dict[Any, List[int]] = {}
    for i, eid in enumerate(features["example_id"]):
        feats_by_eid.setdefault(eid, []).append(i)

    contexts = examples["context"] if "context" in examples.column_names else examples["passage"]
    ex_ids = examples["id"] if "id" in examples.column_names else examples["idx"]

    for ex_i, ex_id in enumerate(ex_ids):
        eid = ex_id
        if eid not in feats_by_eid:
            continue
        best_score = -1e30
        best_text = ""
        context = contexts[ex_i]

        for fi in feats_by_eid[eid]:
            offsets = features["offset_mapping"][fi]
            s_log = start_logits[fi]
            e_log = end_logits[fi]

            start_indexes = np.argsort(s_log)[-n_best_size:][::-1]
            end_indexes = np.argsort(e_log)[-n_best_size:][::-1]

            for s in start_indexes:
                for e in end_indexes:
                    if s >= len(offsets) or e >= len(offsets) or offsets[s] is None or offsets[e] is None:
                        continue
                    if e < s or (e - s + 1) > max_answer_length:
                        continue
                    start_char, _ = offsets[s]
                    _, end_char = offsets[e]
                    score = float(s_log[s]) + float(e_log[e])
                    if score > best_score:
                        best_score = score
                        best_text = context[start_char:end_char]

        preds[eid] = best_text

    return preds


def _qa_sanitize_logits(logits) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Trainer/Accelerate が pad_across_processes する logits に None が混ざると落ちる。
    ここで (start_logits, end_logits) だけに正規化する。
    """
    # dict / ModelOutput 形式
    if isinstance(logits, dict):
        sl = logits.get("start_logits", None)
        el = logits.get("end_logits", None)
        if sl is not None and el is not None:
            return sl, el
        # それ以外は、tensorだけ抽出して先頭2つを使う
        ts = [v for v in logits.values() if torch.is_tensor(v)]
        if len(ts) >= 2:
            return ts[0], ts[1]
        raise RuntimeError(f"Cannot extract QA logits from dict keys={list(logits.keys())}")

    # tuple/list 形式（ここが一番多い：(..., None, None) を含む）
    if isinstance(logits, (tuple, list)):
        ts = [x for x in logits if torch.is_tensor(x)]
        if len(ts) >= 2:
            return ts[0], ts[1]
        raise RuntimeError(f"Cannot extract QA logits from tuple/list: {type(logits)} len={len(logits)}")

    # tensor単体（想定外だが一応）
    if torch.is_tensor(logits):
        raise RuntimeError("QA logits must be (start_logits, end_logits) but got a single tensor.")

    raise RuntimeError(f"Unsupported logits type for QA: {type(logits)}")


class QATrainer(Trainer):
    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        loss, logits, labels = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
        )

        if logits is not None:
            sl, el = _qa_sanitize_logits(logits)
            logits = (sl, el)

            # ★ここが追加：eval_datasetにlabelsが無いと compute_metrics が呼ばれないのでダミーを作る
            if labels is None:
                # shape は (batch, 1) などなら何でもOK（compute_metrics側で参照しない）
                labels = torch.zeros((sl.shape[0], 1), dtype=torch.long, device=sl.device)

        return loss, logits, labels



# ---------------------- IR (optional) ----------------------
def run_ir_optional(args, tokenizer, backbone, mask_cfg: FTMaskCfg):
    try:
        import faiss  # noqa
        from beir.retrieval.evaluation import EvaluateRetrieval
    except Exception as e:
        raise RuntimeError("IR requires beir + faiss + pytrec_eval. Please install them.") from e

    ir = T.load_ir(args.task)
    corpus = ir["corpus"]
    queries = ir["queries"]
    qrels_ds = ir["qrels"]
    if qrels_ds is None:
        raise RuntimeError(f"No qrels found for {ir['path']}. Can't evaluate retrieval properly.")

    qrels: Dict[str, Dict[str, int]] = {}
    for row in qrels_ds:
        qid = str(row.get("query-id", row.get("qid", row.get("query_id", row.get("queryid", "")))))
        did = str(row.get("corpus-id", row.get("docid", row.get("doc_id", row.get("docid", "")))))
        score = int(row.get("score", row.get("relevance", row.get("label", 1))))
        if qid and did:
            qrels.setdefault(qid, {})[did] = score

    dpr = DPRDualEncoder(backbone=backbone, hidden_size=infer_hidden_size(backbone), mask_cfg=mask_cfg)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    dpr.to(device)
    dpr.eval()

    def doc_text(r):
        title = (r.get("title") or "").strip()
        text = (r.get("text") or "").strip()
        return (title + " " + text).strip() if title else text

    doc_ids = []
    doc_texts = []
    for r in corpus:
        doc_ids.append(str(r.get("_id", r.get("docid", r.get("id", "")))))
        doc_texts.append(doc_text(r))

    doc_embs = []
    bs = args.ir_encode_batch
    for i in tqdm(range(0, len(doc_texts), bs), desc="encode corpus"):
        batch = doc_texts[i : i + bs]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=args.max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            emb = dpr.encode(enc["input_ids"], enc["attention_mask"]).detach().float().cpu().numpy()
        doc_embs.append(emb)
    doc_embs = np.concatenate(doc_embs, axis=0).astype("float32")

    dim = doc_embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(doc_embs)

    def q_text(r):
        return (r.get("text") or "").strip()

    results: Dict[str, Dict[str, float]] = {}
    q_ids = []
    q_texts = []
    for r in queries:
        qid = str(r.get("_id", r.get("id", r.get("qid", ""))))
        qt = q_text(r)
        if qid and qt:
            q_ids.append(qid)
            q_texts.append(qt)

    topk = int(args.ir_topk)
    for i in tqdm(range(0, len(q_texts), bs), desc="encode queries"):
        batch = q_texts[i : i + bs]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=args.max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            qemb = dpr.encode(enc["input_ids"], enc["attention_mask"]).detach().float().cpu().numpy().astype("float32")
        scores, idxs = index.search(qemb, topk)
        for j in range(len(batch)):
            qid = q_ids[i + j]
            results[qid] = {}
            for s, di in zip(scores[j].tolist(), idxs[j].tolist()):
                if di < 0:
                    continue
                results[qid][doc_ids[di]] = float(s)

    evaluator = EvaluateRetrieval()
    ndcg, _map, recall, precision = evaluator.evaluate(qrels, results, k_values=[10])
    return {
        "ndcg@10": float(ndcg["NDCG@10"]),
        "recall@10": float(recall["Recall@10"]),
        "precision@10": float(precision["P@10"]),
    }


# ---------------------- main ----------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument(
        "--task",
        type=str,
        required=True,
        choices=list(T.SC_TASKS.keys()) + list(T.TC_TASKS.keys()) + list(T.QA_TASKS.keys()) + list(T.IR_TASKS.keys()),
    )
    p.add_argument("--output_dir", type=str, default="encodeval_ft_results")

    p.add_argument("--ft_mask", type=str, default="bidir", choices=["bidir", "causal"])
    p.add_argument("--mask_sanity", action="store_true", help="run bidir/causal sanity check and exit")
    p.add_argument("--mask_sanity_strict", action="store_true", help="fail if bidir does not differ from causal")

    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--attn_impl", type=str, default="sdpa", choices=["sdpa", "flash_attention_2", "eager", "flex_attention"])
    p.add_argument("--max_length", type=int, default=512)

    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--per_device_train_batch_size", type=int, default=8)
    p.add_argument("--per_device_eval_batch_size", type=int, default=8)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--save_strategy", type=str, default="no", choices=["no", "steps", "epoch"])
    p.add_argument("--eval_strategy", type=str, default="steps", choices=["no", "steps", "epoch"])
    p.add_argument("--eval_steps", type=int, default=200)

    p.add_argument("--doc_stride", type=int, default=128)

    p.add_argument("--ir_topk", type=int, default=100)
    p.add_argument("--ir_encode_batch", type=int, default=64)

    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    run_tag = f"{safe_name(args.model_name_or_path)}__{args.task}__{args.ft_mask}__lr{args.learning_rate:g}__s{args.seed}"
    run_dir = os.path.join(args.output_dir, run_tag)
    os.makedirs(run_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True, trust_remote_code=True)

    use_cuda = torch.cuda.is_available() and args.device == "cuda"
    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    if args.attn_impl == "flash_attention_2" and (not use_cuda or torch_dtype not in (torch.float16, torch.bfloat16)):
        raise RuntimeError("flash_attention_2 requires CUDA and fp16/bf16.")

    # load CausalLM
    load_kw = dict(attn_implementation=args.attn_impl, trust_remote_code=True)
    try:
        m = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **load_kw, dtype=torch_dtype)
    except TypeError:
        m = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **load_kw, torch_dtype=torch_dtype)

    m.config.use_cache = False

    device = torch.device(args.device if (args.device == "cpu" or use_cuda) else "cpu")
    m.to(device)

    backbone = get_backbone_from_causallm(m)
    hidden = infer_hidden_size(backbone)
    mask_cfg = FTMaskCfg(ft_mask=args.ft_mask)

    # ---- sanity check (bidir vs causal) at backbone level ----
    lm_head = get_lm_head_module(m)
    controller = MaskController(backbone)
    sc = sanity_check_suffix_effect(
        tokenizer=tok,
        backbone=backbone,
        lm_head=lm_head,
        device=device,
        controller=controller,
        attn_mask_2d=True,
    )
    print("[mask sanity]", sc)

    if args.mask_sanity:
        return

    if args.mask_sanity_strict:
        if not (sc["diff_causal"] < 1e-4 and sc["diff_bidir"] > 1e-3):
            raise RuntimeError(f"Mask sanity failed: {sc}")

    targs = _make_training_args(args, run_dir=run_dir, use_cuda=use_cuda)

    results: Dict[str, Any] = {"args": vars(args), "mask_sanity": sc}

    # ---- dispatch ----
    if args.task in T.SC_TASKS:
        train_ds, eval_ds, num_labels = T.load_sc(args.task)
        model = DecoderOnlyForSequenceClassification(backbone, hidden, num_labels, mask_cfg).to(device)

        train_tok, eval_tok, collator, compute_metrics = build_sc(tok, train_ds, eval_ds, args.task, args.max_length)

        trainer = Trainer(
            model=model,
            args=targs,
            train_dataset=train_tok,
            eval_dataset=eval_tok,
            data_collator=collator,
            compute_metrics=compute_metrics,
            **_trainer_tokenizer_kw(tok),
        )

        trainer.train()
        results["metrics"] = trainer.evaluate()

    elif args.task in T.TC_TASKS:
        train_ds, eval_ds, label_list, (tok_col, lab_col) = T.load_tc(args.task)
        num_labels = len(label_list)

        model = DecoderOnlyForTokenClassification(backbone, hidden, num_labels, mask_cfg).to(device)

        train_tok, eval_tok, collator, compute_metrics = build_tc(
            tok, train_ds, eval_ds, label_list, tok_col, lab_col, args.max_length
        )

        trainer = Trainer(
            model=model,
            args=targs,
            train_dataset=train_tok,
            eval_dataset=eval_tok,
            data_collator=collator,
            compute_metrics=compute_metrics,
            **_trainer_tokenizer_kw(tok),
        )

        trainer.train()
        results["metrics"] = trainer.evaluate()

    elif args.task in T.QA_TASKS:
        train_ds, eval_ds, metric = T.load_qa(args.task)

        model = DecoderOnlyForExtractiveQA(backbone, hidden, mask_cfg).to(device)

        train_feat, eval_feat = build_qa_features(
            tok, train_ds, eval_ds, args.task, args.max_length, doc_stride=args.doc_stride
        )

        # Trainer が tensor 化できない列を eval_dataset から除去（postprocess 用の eval_feat は保持）
        drop_cols = [c for c in ("offset_mapping", "example_id") if c in eval_feat.column_names]
        eval_dataset_for_trainer = eval_feat.remove_columns(drop_cols) if drop_cols else eval_feat

        # 可変長 input_ids を padding してバッチ化
        qa_collator = DataCollatorWithPadding(tok)

        def compute_metrics_for_qa(p):
            # p.predictions は (start_logits, end_logits) を期待（QATrainerが保証）
            preds = postprocess_qa_predictions(eval_ds, eval_feat, p.predictions)
            if args.task == "qa_record":
                predictions = []
                references = []
                for ex in eval_ds:
                    idx = ex["idx"]
                    answers = ex.get("answers", [])
                    if isinstance(answers, dict) and "text" in answers:
                        answers = answers["text"]
                    predictions.append({"idx": idx, "prediction_text": preds.get(idx, "")})
                    references.append({"idx": idx, "answers": [str(a) for a in (answers or [])]})
                return metric.compute(predictions=predictions, references=references)
            else:
                formatted_preds = [{"id": ex["id"], "prediction_text": preds.get(ex["id"], "")} for ex in eval_ds]
                refs = [{"id": ex["id"], "answers": ex["answers"]} for ex in eval_ds]
                return metric.compute(predictions=formatted_preds, references=refs)

        # ★ここがポイント：QATrainer を使う（logits から None を除去して (start,end) に固定）
        trainer = QATrainer(
            model=model,
            args=targs,
            train_dataset=train_feat,
            eval_dataset=eval_dataset_for_trainer,
            data_collator=qa_collator,
            compute_metrics=compute_metrics_for_qa,
            **_trainer_tokenizer_kw(tok),
        )

        trainer.train()
        results["metrics"] = trainer.evaluate()

    elif args.task in T.IR_TASKS:
        out = run_ir_optional(args, tok, backbone, mask_cfg)
        results["metrics"] = out

    else:
        raise ValueError(f"Unknown task: {args.task}")

    with open(os.path.join(run_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[done] {run_dir}")
    print(json.dumps(results.get("metrics", {}), indent=2))


if __name__ == "__main__":
    main()
