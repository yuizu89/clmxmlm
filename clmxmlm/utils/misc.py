from __future__ import annotations

def safe_name(x: str) -> str:
    x = x.rstrip("/").replace("\\", "/")
    x = x.split("/")[-1]
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in x) or "model"

def to_iso_lang_script(lang: str) -> str:
    # MTEB v2 recommended language tags (common cases)
    if lang == "eng":
        return "eng-Latn"
    if lang == "jpn":
        return "jpn-Jpan"
    return lang
