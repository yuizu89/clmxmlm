# encodeval_ft/tasks.py
from __future__ import annotations

import re
from typing import Dict, Any, List, Tuple

import numpy as np
from datasets import load_dataset, Dataset, DatasetDict, get_dataset_config_names
import evaluate


# ---------- helpers ----------
def _load_dataset(*args, **kwargs):
    """
    HF datasets の一部は custom code を含み、datasets>= の設定で
    trust_remote_code=True を明示しないとロードできないことがあります（例: conll2003）。
    非対話環境（Colabのbash実行）でプロンプトが出ると落ちるので、ここで統一的に許可します。
    """
    kwargs.setdefault("trust_remote_code", True)
    return load_dataset(*args, **kwargs)


def _pick_split(dsd: DatasetDict, preferred: List[str]) -> Dataset:
    for k in preferred:
        if k in dsd:
            return dsd[k]
    return dsd[list(dsd.keys())[0]]


def _pick_split_with_name(dsd: DatasetDict, preferred: List[str]) -> Tuple[Dataset, str]:
    for k in preferred:
        if k in dsd:
            return dsd[k], k
    k = list(dsd.keys())[0]
    return dsd[k], k


# ---------- Sequence Classification (GLUE) ----------
SC_TASKS = {
    "sc_sst2": {"path": "glue", "name": "sst2"},
    "sc_mnli": {"path": "glue", "name": "mnli"},
    "sc_qqp":  {"path": "glue", "name": "qqp"},
}

SC_KEYS = {
    "sc_sst2": ("sentence", None),
    "sc_mnli": ("premise", "hypothesis"),
    "sc_qqp":  ("question1", "question2"),
}


def load_sc(task: str) -> Tuple[Dataset, Dataset, int]:
    spec = SC_TASKS[task]
    dsd = _load_dataset(spec["path"], spec["name"])
    if task == "sc_mnli":
        train = dsd["train"]
        eval_ds = dsd["validation_matched"]
    else:
        train = dsd["train"]
        eval_ds = dsd["validation"]
    num_labels = int(train.features["label"].num_classes)
    return train, eval_ds, num_labels


# ---------- Token Classification (NER) ----------
TC_TASKS = {
    "tc_conll2003": {"path": "conll2003", "name": None},
    "tc_ontonotes5": {"path": "tner/ontonotes5", "name": None},
    "tc_uner_en": {"path": "universalner/universal_ner", "name": None},  # auto-pick English config
}

TC_FIELD_CANDIDATES = [
    ("tokens", "ner_tags"),
    ("tokens", "tags"),
    ("words", "ner_tags"),
    ("sentence", "labels"),
]


def load_tc(task: str) -> Tuple[Dataset, Dataset, List[str], Tuple[str, str]]:
    spec = TC_TASKS[task]

    if task == "tc_uner_en":
        cfgs = get_dataset_config_names(spec["path"])
        eng = [c for c in cfgs if re.search(r"English", c, re.IGNORECASE)]
        if not eng:
            raise RuntimeError(f"Could not find English config in {spec['path']} configs: {cfgs[:20]}...")
        cfg = sorted(eng)[0]
        dsd = _load_dataset(spec["path"], cfg)
    else:
        # conll2003 などは trust_remote_code=True が必要になるケースがある
        dsd = _load_dataset(spec["path"])

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

    # label names: prefer dataset feature metadata
    feat = train.features[lab_col]
    label_list = None
    try:
        if hasattr(feat, "feature") and hasattr(feat.feature, "names"):
            label_list = list(feat.feature.names)
        elif hasattr(feat, "names"):
            label_list = list(feat.names)
    except Exception:
        label_list = None

    if label_list is None:
        # fallback: scan ragged sequences safely (avoid np.max on list-of-lists)
        mx = 0
        for seq in train[lab_col]:
            if not seq:
                continue
            mx = max(mx, max(int(x) for x in seq))
        label_list = [str(i) for i in range(mx + 1)]

    return train, eval_ds, label_list, (tok_col, lab_col)


# ---------- QA ----------
QA_TASKS = {
    "qa_squad": {"path": "squad", "name": None, "metric": ("squad", None)},
    "qa_squad_v2": {"path": "squad_v2", "name": None, "metric": ("squad_v2", None)},
    "qa_record": {"path": "super_glue", "name": "record", "metric": ("super_glue", "record")},
}


def load_qa(task: str) -> Tuple[Dataset, Dataset, Any]:
    spec = QA_TASKS[task]
    dsd = _load_dataset(spec["path"], spec["name"]) if spec["name"] else _load_dataset(spec["path"])
    train = _pick_split(dsd, ["train"])
    eval_ds = _pick_split(dsd, ["validation", "dev"])

    metric_name, metric_cfg = spec["metric"]
    metric = evaluate.load(metric_name, metric_cfg) if metric_cfg else evaluate.load(metric_name)
    return train, eval_ds, metric


# ---------- IR ----------
IR_TASKS = {
    "ir_msmarco": {"path": "BeIR/msmarco", "name": None},
    "ir_nq": {"path": "BeIR/nq", "name": None},
    "ir_mldr_en": {"path": "Shitao/MLDR", "name": None, "lang": "en"},
}


def load_ir(task: str) -> Dict[str, Any]:
    """
    BEIR の一部は qrels が別データセット "<path>-qrels" になっています（例: BeIR/msmarco-qrels）。
    """
    spec = IR_TASKS[task]
    path = spec["path"]

    def try_load_subset(ds_path: str, subset: str):
        try:
            return _load_dataset(ds_path, subset)
        except Exception:
            return None

    out: Dict[str, Any] = {"path": path, "task": task}

    # corpus
    dd_corpus = try_load_subset(path, "corpus")
    if dd_corpus is None:
        raise RuntimeError(f"Failed to load IR corpus subset for {path}.")
    out["corpus"] = _pick_split(dd_corpus, ["corpus", "train"])

    # queries
    dd_queries = try_load_subset(path, "queries")
    if dd_queries is None:
        raise RuntimeError(f"Failed to load IR queries subset for {path}.")
    queries_ds, queries_split = _pick_split_with_name(dd_queries, ["test", "dev", "validation", "queries", "train"])
    out["queries"] = queries_ds
    out["queries_split"] = queries_split

    # qrels: 1) same dataset subset "qrels" 2) fallback to "<path>-qrels"
    dd_qrels = try_load_subset(path, "qrels")

    if dd_qrels is None:
        qrels_path = path if path.endswith("-qrels") else (path + "-qrels")
        try:
            dd_qrels = _load_dataset(qrels_path)
            out["qrels_path"] = qrels_path
        except Exception:
            dd_qrels = None

    if dd_qrels is not None:
        if isinstance(dd_qrels, DatasetDict) and queries_split in dd_qrels:
            out["qrels"] = dd_qrels[queries_split]
        else:
            out["qrels"] = _pick_split(dd_qrels, ["test", "dev", "validation", "qrels", "train"])
    else:
        out["qrels"] = None

    if task == "ir_mldr_en":
        out["lang"] = spec.get("lang", "en")

    return out
