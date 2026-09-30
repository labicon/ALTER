# Artifacts and integrity

Simulation artifacts are public at [ALTER-models](https://huggingface.co/Berkeley-ICON-Lab/ALTER-models)
and [ALTER-data](https://huggingface.co/datasets/Berkeley-ICON-Lab/ALTER-data).
`release/manifest.json` pins immutable Hub revisions and all file checksums.
Hardware artifacts are deferred.

The selected simulation files are split into model, pretraining-data,
adaptation-data, and provenance bundles. They preserve distinct storage
namespaces, the 400-demonstration base selection, nested 5/10/15-per-mode
adaptation membership, grouping and order. Do not regenerate selection from
relocated filenames: the historical selection ranks included absolute paths.

Download all four bundles for the materialization command below:

```bash
python scripts/download_release.py --manifest release/manifest.json \
  --bundles simulation-models simulation-pretraining-data \
  simulation-adaptation-data simulation-provenance \
  --destination /path/to/ALTER-artifacts
```

For file inspection only, `--bundles` can select a subset. Materialization currently
requires all four bundles, including pretraining data. Each bundle must name
a full 40-character Hub commit. Files are verified with full SHA-256 checksums;
valid existing files are reused and conflicts are rejected. The destination
cannot contain symlink path components. `--local-source /path/to/reviewed-bundles`
performs the same integrity/overwrite checks for an offline bundle.

Portable metadata uses `artifact://` references. The downloader installs the two control files automatically. The reviewed bundle also needs
`export-receipts.json` and `release-manifest.json` at its root; the manifest seals
the receipt checksum. These control files must accompany any published bundle.
Materialize a fresh runtime tree before using path-bearing statistics/manifests:

```bash
python scripts/materialize_release.py --bundle-root /path/to/ALTER-artifacts \
  --output "$PWD/public-validation/inputs"
```

Materialization currently requires the complete staged inventory; selection of
partial runtime dependency closures is not implemented. The downloader supports
individual bundle selection for inspection. Download all four bundles before
materialization.

The materializer hashes all supplied files before writing, preserves data-list
order, and derives new local checkpoint contracts only from verified original
artifact identities. It retains original file/artifact hashes in receipts and
`release_derivation`, recomputes runtime hashes, and checks the derived contract
with the existing strict validator. A derived contract is a new attestation;
it does not turn a historical mismatch or missing record into a passing result.
Historical selection/results remain evidence records, not newly run evaluations.
Only load trusted, reviewed pickle/PyTorch artifacts.

[workflow-artifacts.json](workflow-artifacts.json) maps selected paper runs to files. Model weights and
normalization arrays must remain paired. Large artifacts are ignored by Git.
No three-arm experiment bundle is included.
