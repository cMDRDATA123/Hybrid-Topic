"""Create an immutable cloud-only freeze from the completed cloud half.

The mixed queue was intentionally stopped before its combined freeze. This
command copies only the already completed cloud runs, writes a new cloud-only
preflight and prediction manifest, and leaves the original mixed directory
untouched. It does not call a provider or rerun a fit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time

from experiments.data_io import sha256, write_json


DATASETS = ("bills", "sib200", "massive", "ag_news")
SEEDS = (11, 23, 37, 53, 71)


def freeze(source: Path, output: Path) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"cloud freeze output is not empty: {output}")
    preflight_source = source / "preflight.json"
    if not preflight_source.is_file():
        raise ValueError("source run has no preflight.json")
    preflight = json.loads(preflight_source.read_text())
    keys = [f"cloud/{dataset}_seed{seed}" for dataset in DATASETS for seed in SEEDS]
    for key in keys:
        run = source / key
        if not (run / "complete.json").is_file():
            raise ValueError(f"cloud run is incomplete: {key}")

    output.mkdir(parents=True, exist_ok=True)
    cloud_preflight = dict(preflight)
    cloud_preflight.update({
        "release_scope": "cloud_only",
        "source_mixed_run": str(source),
        "source_mixed_preflight_sha256": sha256(preflight_source),
        "backends": ["cloud"],
        "planned_fits": len(keys),
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    write_json(output / "preflight.json", cloud_preflight)
    if (source / "source_snapshot").is_dir():
        shutil.copytree(source / "source_snapshot", output / "source_snapshot")
    copied = {}
    for key in keys:
        source_run = source / key
        output_run = output / key
        shutil.copytree(source_run, output_run)
        copied[key] = sha256(output_run / "complete.json")
    write_json(output / "predictions_frozen.json", {
        "preflight_sha256": sha256(output / "preflight.json"),
        "runs": copied,
        "failures": {},
        "planned": len(keys),
        "scope": "cloud_only",
        "source_mixed_run": str(source),
    })

    requests = []
    for key in keys:
        requests.extend(json.loads(path.read_text()) for path in (output / key / "requests").glob("request_*.json"))
    usage = [r.get("usage") for r in requests if isinstance(r.get("usage"), dict)]
    prompt_tokens = sum(int(u.get("prompt_tokens", 0)) for u in usage)
    completion_tokens = sum(int(u.get("completion_tokens", 0)) for u in usage)
    write_json(output / "usage_summary.json", {
        "cloud_requests": len(requests),
        "cloud_prompt_tokens": prompt_tokens,
        "cloud_completion_tokens": completion_tokens,
        "cloud_requests_without_usage": len(requests) - len(usage),
        "cloud_usd_at_standard_no_discount_rates": (prompt_tokens * 0.4 + completion_tokens * 1.6) / 1_000_000,
        "rate_note": "GPT-4.1 mini standard uncached rates; estimate excludes cache discounts and unknown-usage requests",
        "source_mixed_run": str(source),
    })
    write_json(output / "status.json", {"completed": len(keys), "failed": 0, "planned": len(keys)})
    return {"completed": len(keys), "failed": 0, "planned": len(keys), "output": str(output)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.source, args.output), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
