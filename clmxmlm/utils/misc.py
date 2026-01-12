from __future__ import annotations

def safe_name(x: str) -> str:
    x = x.rstrip("/").replace("\\", "/")
    x = x.split("/")[-1]
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in x) or "model"
