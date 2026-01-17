from __future__ import annotations

import os
import json
import time
import argparse
from typing import Any, Dict, List, Optional

from mteb.models import ModelMeta

from .encoder_dual_mask import DualMaskConfig, DualMaskHFEncoder
from ..utils.misc import safe_name, to_iso_lang_script
from ..utils.seed import set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", type=str, required=True)

    # embedding behavior
    p.add_argument("--mask_mode", type=str, default="bidir", choices=["bidir", "causal", "concat", "avg"])
    p.add_argument("--pooling_bidir", type=str, default="mean", choices=["mean", "last", "cls", "eos"])
    p.add_argument("--pooling_causal", type=str, default="mean", choices=["mean", "last", "cls", "eos"])
    p.add_argument("--no_normalize", action="store_true", default=False)
    p.add_argument("--max_length", type=int, default=512)

    # runtime
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--attn_impl", type=str, default="flash_attention_2",
                   choices=["sdpa", "flash_attention_2", "eager", "flex_attention"])
    p.add_argument("--hf_token", type=str, default=None)

    # MTEB selection (v2)
    p.add_argument("--languages", type=str, default="eng", help="comma-separated ISO639-3: e.g., eng,jpn")
    p.add_argument("--tasks", type=str, default="", help="comma-separated task names, e.g. STS22.v2,Banking77Classification.v2")
    p.add_argument("--benchmark", type=str, default="", help="benchmark name, e.g. 'MTEB(eng, v2)'")

    # results/cache
    p.add_argument("--output_dir", type=str, default="mteb_results")
    p.add_argument("--overwrite_strategy", type=str, default="only-missing",
                   choices=["only-missing", "always", "never"])

    # Optional: pin model revision
    p.add_argument("--model_revision", type=str, default=None)

    # sanity check knobs
    p.add_argument("--no_sanity_check", action="store_true", default=False)
    p.add_argument("--sanity_eps", type=float, default=1e-6)
    p.add_argument("--sanity_fail", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--sanity_text", type=str, default="Hello world. This is a sanity check sentence for bidirectional masking.")

    # misc
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def build_model_meta(
    *,
    model: DualMaskHFEncoder,
    model_name_or_path: str,
    langs: List[str],
    revision: Optional[str],
    max_length: int,
) -> ModelMeta:
    n_params = sum(p.numel() for p in model.model.parameters())
    max_tokens = getattr(model.model.config, "max_position_embeddings", None) or max_length
    embed_dim = model.get_sentence_embedding_dimension()

    return ModelMeta(
        loader=None,
        loader_kwargs={},
        name=model_name_or_path,
        revision=revision,
        release_date=None,
        languages=[to_iso_lang_script(l) for l in langs],
        n_parameters=int(n_params),
        memory_usage_mb=None,
        max_tokens=float(max_tokens) if max_tokens is not None else None,
        embed_dim=int(embed_dim),
        license=None,
        open_weights=None,
        public_training_code=None,
        public_training_data=None,
        framework=["PyTorch"],
        reference=None,
        similarity_fn_name="cosine",
        use_instructions=False,
        training_datasets=None,
    )


def main():
    args = parse_args()
    set_seed(int(args.seed))

    os.makedirs(args.output_dir, exist_ok=True)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    import mteb
    from mteb.cache import ResultCache

    langs = [x.strip() for x in args.languages.split(",") if x.strip()]

    # --- select tasks/benchmark ---
    if args.tasks.strip():
        task_names = [x.strip() for x in args.tasks.split(",") if x.strip()]
        tasks = mteb.get_tasks(tasks=task_names, languages=langs)
    else:
        bench_name = args.benchmark.strip() or f"MTEB({langs[0]}, v2)"
        tasks = mteb.get_benchmark(bench_name)

    cfg = DualMaskConfig(
        model_name_or_path=args.model_name_or_path,
        attn_impl=args.attn_impl,
        device=args.device,
        dtype=args.dtype,
        max_length=args.max_length,
        normalize=(not args.no_normalize),
        mask_mode=args.mask_mode,
        pooling_bidir=args.pooling_bidir,
        pooling_causal=args.pooling_causal,
        hf_token=args.hf_token,
        sanity_check=(not args.no_sanity_check),
        sanity_eps=float(args.sanity_eps),
        sanity_fail=bool(args.sanity_fail),
        sanity_text=args.sanity_text,
    )
    model = DualMaskHFEncoder(cfg)
    model.mteb_model_meta = build_model_meta(
        model=model,
        model_name_or_path=args.model_name_or_path,
        langs=langs,
        revision=args.model_revision,
        max_length=args.max_length,
    )

    run_name = (
        f"{safe_name(args.model_name_or_path)}__"
        f"{args.mask_mode}__"
        f"pb-{args.pooling_bidir}__pc-{args.pooling_causal}__"
        f"ml{args.max_length}"
    )
    out_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)
    cache = ResultCache(cache_path=out_dir)

    t0 = time.time()
    results = mteb.evaluate(
        model,
        tasks,
        cache=cache,
        overwrite_strategy=args.overwrite_strategy,
        encode_kwargs={"max_length": args.max_length},
    )

    summary: Dict[str, Any] = {}
    for r in results:
        try:
            summary[r.task_name] = {
                "main_score": float(r.get_score()),
                "metric": str(r.task.metadata.main_score),
                "main_scores": r.only_main_score().to_dict(),
            }
        except Exception:
            summary[r.task_name] = {"raw": str(r)}

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[done] saved to: {out_dir}")
    print(f"[done] elapsed: {(time.time()-t0):.1f}s")


if __name__ == "__main__":
    main()
