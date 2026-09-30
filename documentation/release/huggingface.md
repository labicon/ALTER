# Published simulation artifacts

Models: https://huggingface.co/Berkeley-ICON-Lab/ALTER-models (Apache-2.0).
Data: https://huggingface.co/datasets/Berkeley-ICON-Lab/ALTER-data (CC BY 4.0).
The author authorized public simulation upload. Hardware artifacts are deferred.

[release/manifest.json](../../release/manifest.json) pins the artifact commits.
The Hub repositories also contain the same manifest, LICENSE, NOTICE, usage
instructions, per-repository file inventories and SHA256SUMS. The manifest pins
artifact commits that precede the final documentation commit, avoiding circular
self-references. Original exported payload bytes and checksums are unchanged.

All 706 payload/control files passed pre-upload checks. Anonymous remote file
sizes and SHA-256 identities were checked across the complete inventory; small
non-LFS files were downloaded to compute their hashes. All 19 selected model
pairs, all provenance files and one demonstration from each data bundle were
downloaded anonymously into an isolated directory. Remaining demonstrations were
copied from the verified local exports for materialization testing, not claimed
as a full network redownload. Existing valid downloads were reused successfully.

All 19 downloaded policies passed finite inference, one CPU update and save/reload;
coordination updates preserved the frozen base. Strict materialization and two-step
coordination/source simulations passed. These functional checks do not establish
paper success rates or reproduce complete training. Full paper reproduction is
outside the agreed release validation scope.

For future versions, stage locally, inspect metadata and content, use narrowly
scoped local authentication, and upload only the reviewed inventory. Never put
tokens in Git or chat. Retain prior artifact versions and update pinned revisions
and checksums together. Test anonymous downloads before publishing dependent code.
See the [official upload guide](https://huggingface.co/docs/huggingface_hub/en/guides/upload).
