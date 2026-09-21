"""Independent GPT-5.6 Luna sensitivity benchmark.

This reuses the frozen inputs, cached embeddings and baseline artifacts from
the release benchmark, but writes a separate 20-fit cloud-only batch. It is
never appended to the stopped mixed cloud/local run.
"""
from __future__ import annotations

import argparse
import fcntl
import importlib.metadata
import json
from pathlib import Path
import shutil
import time

from hybrid_topic.generation_budget import freeze_model_catalog, verify_frozen_model_catalog
from experiments.data_io import sha256, write_json
from experiments.release_benchmark import (
    CLOUD_MODEL as PREVIOUS_CLOUD_MODEL,
    DATASETS,
    EMBEDDING,
    SEEDS,
    CachedEmbedder,
    OpenAITopicGenerator,
    SentenceTransformerEmbedder,
    atomic_json,
    prepare_reuse,
    run_one,
    validate_cached_embeddings,
)

LUNA_MODEL = "gpt-5.6-luna"
LUNA_CONTEXT = 1_050_000
LUNA_REASONING_EFFORT = "low"
LUNA_TOKENIZER = "o200k_base"
LUNA_OUTPUT_TOKENS = 16_384
PLANNED_FITS = len(DATASETS) * len(SEEDS)


def execute_luna(data_root: Path, cache_root: Path, out: Path, api_key_file: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with (out / "execution.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out / "preflight.json").exists():
            raise RuntimeError("This Luna batch has already started; inspect it before resuming")
        data, reuse = prepare_reuse(data_root, cache_root, out)
        sources = {
            str(p): sha256(p)
            for folder in ["hybrid_topic", "experiments"]
            for p in Path(folder).glob("*.py")
        }
        model_catalog = freeze_model_catalog(out / "source_snapshot/model_catalog.json")
        preflight = {
            "model_catalog": model_catalog,
            "protocol": "release-20260907-luna-catalog-v2",
            "model": LUNA_MODEL,
            "provider": "openai",
            "reasoning_effort": LUNA_REASONING_EFFORT,
            "tokenizer_encoding": LUNA_TOKENIZER,
            "context_window": LUNA_CONTEXT,
            "max_output_tokens": LUNA_OUTPUT_TOKENS,
            "input_context_ratio": 0.75,
            "safety_margin_tokens": 1024,
            "max_format_retries": 1,
            "seeds": SEEDS,
            "datasets": DATASETS,
            "planned_fits": PLANNED_FITS,
            "input_manifest_sha256": sha256(data_root / "inputs/manifest.json"),
            "protocol_sha256": sha256("docs/EXPERIMENT_PROTOCOL_RELEASE.md"),
            "data_root": str(data_root),
            "cache_root": str(cache_root),
            "baseline_reuse": reuse,
            "sources": sources,
            "previous_primary_cloud_model": PREVIOUS_CLOUD_MODEL,
            "no_fee_ceiling_user_authorized": True,
            "packages": {
                p: importlib.metadata.version(p)
                for p in ["numpy", "pandas", "scikit-learn", "openai", "tiktoken", "sentence-transformers"]
            },
        }
        write_json(out / "preflight.json", preflight)
        for source in sources:
            destination = out / "source_snapshot" / source
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

        api_key = api_key_file.read_text().strip()
        embedding = SentenceTransformerEmbedder(EMBEDDING, device="mps", local_files_only=True)
        for dataset in DATASETS:
            d = data[dataset]
            train_vectors, test_vectors = validate_cached_embeddings(cache_root / "embeddings", dataset, d)
            for seed in SEEDS:
                for source, expected in sources.items():
                    if sha256(source) != expected:
                        raise RuntimeError(f"frozen execution source changed: {source}")
                verify_frozen_model_catalog(model_catalog)
                backend = OpenAITopicGenerator(
                    catalog_path=model_catalog["path"],
                    model=LUNA_MODEL,
                    provider="openai",
                    api_key=api_key,
                    context_window=LUNA_CONTEXT,
                    max_output_tokens=LUNA_OUTPUT_TOKENS,
                    input_context_ratio=0.75,
                    safety_margin_tokens=1024,
                    max_format_retries=1,
                    sampling_seed=seed,
                    reasoning_effort=LUNA_REASONING_EFFORT,
                    tokenizer_encoding=LUNA_TOKENIZER,
                )
                if backend.generation_config["catalog_sha256"] != model_catalog["sha256"]:
                    raise RuntimeError("frozen model catalog changed during generator configuration")
                backend.client = backend.client.with_options(timeout=300, max_retries=0)
                original_request = backend._request
                try:
                    run_one(
                        out / "luna" / f"{dataset}_seed{seed}", "luna", dataset, seed,
                        d, train_vectors, test_vectors, backend, embedding, cache_root,
                    )
                finally:
                    backend._request = original_request
                atomic_json(out / "status.json", {
                    "completed": len(list(out.glob("*/*/complete.json"))),
                    "failed": len(list(out.glob("*/*/failure.json"))),
                    "planned": PLANNED_FITS,
                })

        completed = {
            str(p.parent.relative_to(out)): sha256(p)
            for p in out.glob("*/*/complete.json")
        }
        failures = {
            str(p.parent.relative_to(out)): sha256(p)
            for p in out.glob("*/*/failure.json")
        }
        if len(completed) + len(failures) != PLANNED_FITS:
            raise RuntimeError("planned Luna run accounting mismatch")
        write_json(out / "predictions_frozen.json", {
            "preflight_sha256": sha256(out / "preflight.json"),
            "runs": completed,
            "failures": failures,
            "planned": PLANNED_FITS,
        })
        requests = [json.loads(p.read_text()) for p in out.glob("*/*/requests/request_*.json")]
        usage = [r.get("usage") for r in requests if r.get("usage")]
        write_json(out / "usage_summary.json", {
            "provider": "openai",
            "model": LUNA_MODEL,
            "cloud_requests": len(requests),
            "prompt_tokens": sum(r.get("prompt_tokens", 0) for r in usage),
            "completion_tokens": sum(r.get("completion_tokens", 0) for r in usage),
            "requests_without_usage": len(requests) - len(usage),
            "standard_uncached_usd_estimate": sum(
                (r.get("prompt_tokens", 0) * 0.2 + r.get("completion_tokens", 0) * 1.2) / 1_000_000
                for r in usage
            ),
            "rate_note": "GPT-5.6 Luna standard uncached rates; estimate excludes cache discounts and unknown usage",
        })
        print(f"FROZEN {len(completed)} completed {len(failures)} failed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--api-key-file", type=Path, default=Path("api.txt"))
    args = parser.parse_args()
    execute_luna(args.data, args.cache, args.out, args.api_key_file)
