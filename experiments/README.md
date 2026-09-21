# Release experiments

See [the reproduction guide](../docs/REPRODUCIBILITY.md) for commands and required inputs.

- `paper_tables.py`: build all paper tables from distributed numeric metric rows; no model/API/data text required.
- `score_release.py`: independently score a complete frozen prediction bundle against its matching gold sidecars.
- `luna_benchmark.py`: new cloud sensitivity execution, requiring the text pack and verified baseline/vector cache.
- `release_benchmark.py`: preserved mixed cloud/local execution path; the original local phase was stopped and is not part of this release's results.

New runs write separate directories and freeze the selected catalog. Source code,
data and cache hashes establish lineage. The public table-reproduction command is
independent of the private research cache. Do not infer that full raw-data replay is
available merely because the table script succeeds.
