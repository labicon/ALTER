# Guided artifact publication (pending approval)

No repository was created, no authentication was changed, and nothing was
uploaded during local preparation. Suggested names are
`Berkeley-ICON-Lab/ALTER-models` and `Berkeley-ICON-Lab/ALTER-data`; these names
and the publishing account still need confirmation.

1. Confirm the publishing account and organization membership.
2. Approve the separate model/dataset names, contents, cards and licenses.
3. After explicit authorization, create **private** staging repositories using
   the [official repository guide](https://huggingface.co/docs/huggingface_hub/en/guides/repository).
4. Authenticate locally, following [user access token guidance](https://huggingface.co/docs/hub/security-tokens).
   Use narrowly scoped upload access. Never paste tokens into chat or put them
   in Git. Do not reuse the legacy endpoint without verification.
5. Upload only the reviewed export inventory and control manifests. Private
   originals, migration scripts/logs, internal Git data and unsanitized records
   must remain outside the upload set.
6. Record immutable Hub commit IDs. Fill `repo_id`, `repo_type`, and `revision`
   for every bundle in `release/manifest.json`, including the control-file hashes.
7. Download into a fresh validation directory and check every checksum and
   materialization result. Do not change existing published artifact versions.
8. Obtain publication approval, make the intended repositories public, and test
   anonymous access before publishing dependent code/download instructions.

The [model card](https://huggingface.co/docs/hub/model-cards) and
[dataset card](https://huggingface.co/docs/hub/datasets-cards) should state the
contents, verified identities, training/evaluation protocol, selection caveats,
limitations and approved licenses. Draft card text is staged locally for review.
