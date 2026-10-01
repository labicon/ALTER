# Hugging Face publication and verification

Simulation models are published at [ALTER-models](https://huggingface.co/Berkeley-ICON-Lab/ALTER-models)
(Apache-2.0); data are at [ALTER-data](https://huggingface.co/datasets/Berkeley-ICON-Lab/ALTER-data)
(CC BY 4.0). The author authorized public simulation and hardware uploads.
Hardware v1 is a separate version under `hardware/v1/`; see the [hardware guide](hardware.md).

The release manifests pin immutable artifact revisions and SHA-256 checksums:
[simulation](../../release/manifest.json) and
[hardware](../../release/hardware-manifest.json). They and the Hub copies identify
exactly which files were published. The structured records contain repository
revisions, inventory totals, and publication verification scope:
[simulation record](../../release/publication.json) and
[hardware record](../../release/hardware-publication.json).

## What was verified

For simulation, remote file sizes and SHA-256 identities were checked across the
full inventory. All selected model and provenance files, and one demonstration
from each data bundle, were downloaded anonymously. Remaining demonstrations were
copied from verified local exports for materialization checks; a complete network
redownload was not performed. Downloaded policies passed finite inference, one
CPU update, and save/reload. The base policy remained frozen during coordination
updates. Materialization and short simulation rollouts passed. These checks do
not reproduce full training or establish paper success rates.

For hardware, small files were downloaded and hashed; large-file checksums were
checked through Hub metadata. A representative file from each data bundle and
the model-only bundle were downloaded anonymously. The model-only bundle was fetched to a fresh directory and materialized for
offline inference. The full 69.75 GB training validation used local independent
copies, not a full network download. All six selected workflows passed short CPU
training checks; fourteen fixed-input outputs matched the originals byte-for-byte
in the same environment. See the [hardware model card](../../release/hardware-model-card-draft.md)
and [dataset card](../../release/hardware-dataset-card-draft.md) for artifact
contents and limitations.

These publication checks verify artifacts and basic offline function; they do
not substitute for paper reproduction or real-robot validation.

For future uploads, stage and inspect files locally, upload only a reviewed
inventory, and test anonymous downloads. Never store access tokens in Git or
chat. Retain prior artifact versions and update pinned revisions and checksums
together. See the [Hugging Face upload guide](https://huggingface.co/docs/huggingface_hub/en/guides/upload).
