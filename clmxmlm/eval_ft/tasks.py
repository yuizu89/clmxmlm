# clmxmlm/eval_ft/tasks.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
from datasets import load_dataset, Dataset, DatasetDict, get_dataset_config_names
import evaluate


# ---------- Sequence Classification (GLUE) ----------
SC_TASKS = {
    "sc_sst2": {"path": "glue", "name": "sst2"},
    "sc_mnli": {"path": "glue", "name": "mnli"},
    "sc_qqp":  {"path": "glue", "name": "qqp"},
}

# GLUE key mapping
SC_KEYS = {
    "sc_sst2": ("sentence", None),
    "sc_mnli": ("premise", "hypothesis"),
    "sc_qqp":  ("question1", "question2"),
}


# ---------- Token Classification (NER) ----------
# CoNLL2003 is standard on HF
# OntoNotes NER: tner/ontonotes5 is commonly used as NER-ready subset
# UNER: universalner/universal_ner; pick English configs dynamically
TC_TASKS = {
    "tc_conll2003": {"path": "conll2003", "name": None},
    "tc_ontonotes5": {"path": "tner/ontonotes5", "name": None},
    "tc_uner_en": {"path": "universalner/universal_ner", "name": None},  # we'll auto-pick an English config
}

# Token/label field candidates (dataset-dependent)
TC_FIELD_CANDIDATES = [
    ("tokens", "ner_tags"),
    ("tokens", "tags"),
    ("words", "ner_tags"),
    ("sentence", "labels"),
]


# ---------- QA ----------
QA_TASKS = {
    "qa_squad": {"path": "squad", "name": None, "metric": ("squad", None)},
    "qa_squad_v2": {"path": "squad_v2", "name": None, "metric": ("squad_v2", None)},
    "qa_record": {"path": "super_glue", "name": "record", "metric": ("super_glue", "record")},
}


# ---------- IR ----------
# BEIR via datasets: BeIR/nq, BeIR/msmarco (for eval)
# MLDR: Shitao/MLDR (has corpus + queries + qrels style fields; but structure differs)
IR_TASKS = {
    "ir_msmarco": {"path": "BeIR/msmarco", "name": None},
    "ir_nq": {"path": "BeIR/nq", "name": None},
    "ir_mldr_en": {"path": "Shitao/MLDR", "name": None, "lang": "en"},
}


def _pick_split(dsd: DatasetDict, preferred: List[str]) -> Dataset:
    for k in preferred:
        if k in dsd:
            return dsd[k]
    # fallback: first split
    return dsd[list(dsd.keys())[0]]


def load_sc(task: str) -> Tuple[Dataset, Dataset, int]:
    spec = SC_TASKS[task]
    dsd = load_dataset(spec["path"], spec["name"])
    # Use GLUE official validation splits
    if task == "sc_mnli":
        train = dsd["train"]
        eval_ds = dsd["validation_matched"]
    else:
        train = dsd["train"]
        eval_ds = dsd["validation"]
    num_labels = int(train.features["label"].num_classes)
    return train, eval_ds, num_labels


def load_tc(task: str) -> Tuple[Dataset, Dataset, List[str], Tuple[str, str]]:
    spec = TC_TASKS[task]

    if task == "tc_uner_en":
        # auto-pick an English config (UNER_English-*)
        cfgs = get_dataset_config_names(spec["path"])
        eng = [c for c in cfgs if re.search(r"English", c, re.IGNORECASE)]
        if not eng:
            raise RuntimeError(f"Could not find English config in {spec['path']} configs: {cfgs[:20]}...")
        # pick the first English config deterministically
        cfg = sorted(eng)[0]
        dsd = load_dataset(spec["path"], cfg)
    else:
        dsd = load_dataset(spec["path"])

    train = _pick_split(dsd, ["train", "training"])
    eval_ds = _pick_split(dsd, ["validation", "valid", "dev"])

    # find token/label columns
    tok_col, lab_col = None, None
    for a, b in TC_FIELD_CANDIDATES:
        if a in train.column_names and b in train.column_names:
            tok_col, lab_col = a, b
            break
    if tok_col is None:
        raise RuntimeError(f"Could not find token/label columns in {task}. columns={train.column_names}")

    # label names
    feat = train.features[lab_col]
    if hasattr(feat, "feature") and hasattr(feat.feature, "names"):
        # e.g., Sequence(ClassLabel(...))
        label_list = list(feat.feature.names)
    elif hasattr(feat, "names"):
        # e.g., ClassLabel(...)
        label_list = list(feat.names)
    else:
        # fallback (robust): train[lab_col] is usually List[List[int]]
        max_id = -1
        for seq in train[lab_col]:
            if not seq:
                continue
            m = max(seq)
            if m > max_id:
                max_id = m
        if max_id < 0:
            max_id = 0
        label_list = [str(i) for i in range(int(max_id) + 1)]

    return train, eval_ds, label_list, (tok_col, lab_col)


def load_qa(task: str) -> Tuple[Dataset, Dataset, Any]:
    spec = QA_TASKS[task]
    dsd = load_dataset(spec["path"], spec["name"]) if spec["name"] else load_dataset(spec["path"])
    train = _pick_split(dsd, ["train"])
    eval_ds = _pick_split(dsd, ["validation", "dev"])

    metric_name, metric_cfg = spec["metric"]
    metric = evaluate.load(metric_name, metric_cfg) if metric_cfg else evaluate.load(metric_name)
    return train, eval_ds, metric


def load_ir(task: str) -> Dict[str, Any]:
    """
    Returns dict with:
      - corpus: Dataset
      - queries: Dataset
      - qrels: Dataset (if available)
      - split hints
    For BeIR/*: subsets are usually "corpus" and "queries" (and sometimes "qrels").
    We'll try to load them robustly.
    """
    spec = IR_TASKS[task]
    path = spec["path"]

    def load_subset(subset: str):
        try:
            dd = load_dataset(path, subset)
            return dd
        except Exception:
            return None

    out: Dict[str, Any] = {"path": path, "task": task}

    dd_corpus = load_subset("corpus")
    if dd_corpus is None:
        raise RuntimeError(f"Failed to load IR corpus subset for {path}. Try checking HF dataset card.")
    out["corpus"] = _pick_split(dd_corpus, ["corpus", "train"])

    dd_queries = load_subset("queries")
    if dd_queries is None:
        raise RuntimeError(f"Failed to load IR queries subset for {path}.")
    # prefer "test" if present else "queries"/"train"
    out["queries"] = _pick_split(dd_queries, ["test", "dev", "validation", "queries", "train"])

    dd_qrels = load_subset("qrels")
    if dd_qrels is not None:
        out["qrels"] = _pick_split(dd_qrels, ["test", "dev", "validation", "qrels", "train"])
    else:
        out["qrels"] = None

    # MLDR is different; user can extend later if needed.
    if task == "ir_mldr_en":
        out["lang"] = spec.get("lang", "en")

    return out
