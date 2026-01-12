from __future__ import annotations

import os
import json
import csv
import time
from typing import Optional, List, Dict

from transformers import TrainerCallback


class JsonlCsvLoggerCallback(TrainerCallback):
    """
    Save Trainer logs to JSONL + CSV (process zero only).
    """

    DEFAULT_FIELDS = [
        "step", "epoch", "lr",
        "loss", "loss_clm", "loss_mlm",
        "tps_active", "tflops",
        "grad_sim", "grad_norm", "grad_norm_rms",
        "update_to_weight",
    ]

    def __init__(self, jsonl_path: Optional[str], csv_path: Optional[str], fields: Optional[List[str]] = None):
        self.jsonl_path = jsonl_path
        self.csv_path = csv_path
        self.fields = fields or list(self.DEFAULT_FIELDS)
        self._csv_inited = False

    def _is_process_zero(self, state) -> bool:
        return getattr(state, "is_world_process_zero", True)

    def _ensure_parent(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def _init_csv_if_needed(self):
        if self.csv_path is None or self._csv_inited:
            return
        self._ensure_parent(self.csv_path)
        file_exists = os.path.exists(self.csv_path) and os.path.getsize(self.csv_path) > 0
        if not file_exists:
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=self.fields)
                w.writeheader()
        self._csv_inited = True

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or not self._is_process_zero(state):
            return

        record = dict(logs)
        record["_time"] = time.time()

        if self.jsonl_path:
            self._ensure_parent(self.jsonl_path)
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        if self.csv_path:
            self._init_csv_if_needed()
            row = {k: record.get(k, "") for k in self.fields}
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=self.fields)
                w.writerow(row)
