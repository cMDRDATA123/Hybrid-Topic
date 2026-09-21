"""Refresh marked manuscript tables from the versioned TopicGPT-aligned results."""
import csv
import json
import math
from pathlib import Path


def main():
    source = Path("paper/results/topicgpt_aligned_20260911/all_methods.csv")
    manifest = json.loads(source.with_name('manifest.json').read_text())
    if manifest['metric_version'] != 'topicgpt-weighted-best-f1-v1':
        raise ValueError('unexpected primary metric version')
    rows = list(csv.DictReader(source.open(newline="", encoding="utf-8")))
    keyed = {(r["generator"], r["dataset"], r["method"]): r for r in rows}
    if len(rows) != 48 or len(keyed) != 48:
        raise ValueError("expected 48 unique frozen aggregate rows")

    def cell(generator, dataset, method, metric):
        row = keyed[(generator, dataset, method)]
        mean, sd = float(row[f"{metric}_mean"]), float(row[f"{metric}_std"])
        if row["runs"] != "5" or not all(map(math.isfinite, [mean, sd])):
            raise ValueError("invalid aggregate row")
        return f"{mean:.4f} ± {sd:.4f}"

    datasets = [("bills", "Bills"), ("sib200", "SIB-200"),
                ("massive", "MASSIVE"), ("ag_news", "AG News")]
    main_rows = ["| Collection | Configuration | HMP | Coverage |",
                 "|---|---|---:|---:|"]
    configs = [("GPT-4.1 mini", "hybrid", "Hybrid / GPT-4.1 mini"),
               ("Luna", "hybrid", "Hybrid / Luna"),
               ("GPT-4.1 mini", "bertopic", "BERTopic (shared)"),
               ("GPT-4.1 mini", "kmeans", "KMeans / GPT-4.1 mini K"),
               ("Luna", "kmeans", "KMeans / Luna K")]
    for dataset, title in datasets:
        for gen, method, label in configs:
            main_rows.append(f"| {title} | {label} | {cell(gen, dataset, method, 'harmonic_purity')} | {cell(gen, dataset, method, 'coverage')} |")
    component_rows = ["| Generator | Collection | Hybrid initial | Hybrid final | Direct initial | Direct final |",
                      "|---|---|---:|---:|---:|---:|"]
    for gen in ("GPT-4.1 mini", "Luna"):
        for dataset, title in datasets:
            values = [cell(gen, dataset, method, "harmonic_purity")
                      for method in ("initial", "hybrid", "direct_initial", "direct_hybrid")]
            component_rows.append(f"| {gen} | {title} | " + " | ".join(values) + " |")
    targets = [(Path("paper/manuscript.md"), "main-table", main_rows),
               (Path("paper/supplementary_material.md"), "component-table", component_rows)]
    pending = []
    for path, name, table in targets:
        draft = path.read_text(encoding="utf-8")
        start, end = f"<!-- {name}:start -->", f"<!-- {name}:end -->"
        if draft.count(start) != 1 or draft.count(end) != 1:
            raise ValueError(f"missing or duplicate markers for {name}")
        prefix, rest = draft.split(start)
        _, suffix = rest.split(end)
        draft = prefix + start + "\n" + "\n".join(table) + "\n" + end + suffix
        pending.append((path, draft))
    for path, draft in pending:
        path.write_text(draft, encoding="utf-8")
    print("Updated 20 main rows and 8 supplementary component rows from frozen aggregates.")


if __name__ == "__main__":
    main()
