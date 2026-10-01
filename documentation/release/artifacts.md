# Download simulation artifacts

Use this guide when you want to run the released simulation models or workflows.
It explains how to download the required files and prepare them for local use.
The README links directly to the public models and data; you only need this
process when using the repository’s download and materialization tools.

`release/manifest.json` pins the exact Hub revisions and file checksums. The
release is split into four bundles. Download all four to prepare a runnable
local tree; selecting fewer bundles is useful only for inspecting files.

From the repository root, run:

```bash
python scripts/download_release.py --manifest release/manifest.json \
  --bundles simulation-models simulation-pretraining-data \
  simulation-adaptation-data simulation-provenance \
  --destination /path/to/ALTER-artifacts

python scripts/materialize_release.py --bundle-root /path/to/ALTER-artifacts \
  --output "$PWD/public-validation/inputs"
```

The downloader checks files against the pinned manifest, reuses valid downloads,
and refuses conflicting files. Materialization checks the complete bundle and
creates a local runtime tree with usable paths. It needs all four bundles.
Only materialize files from the reviewed release manifest.
Keep downloads and generated files outside Git.

See [simulation workflows](simulation.md) for what to do with the prepared files.
For the separately versioned hardware downloads, use the [hardware guide](hardware.md).
