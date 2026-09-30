---
license: apache-2.0
---

# ALTER simulation model card

Authors: Dayi Dong, Maulik Bhatt, Aayushi Shrivastava, Lasse Peters, Negar Mehr.

Contains the selected frozen base and 18 distinct adapted policies for the
reported two-arm budgets and coordination-head capacity comparison. Base and
normalization statistics are paired in the workflow artifact matrix. Model and
EMA state tensors are preserved; exported path-bearing metadata is sanitized
with original/export checksum receipts. Existing module/checkpoint formats are
retained. ResNet18 encoder parameters are contained in the policy files.

See USAGE.md for downloads and the status of the accompanying code release.
The fromscratch inference route also loads full-policy FT checkpoints.
These are research policies for their recorded observation/action conventions;
transfer to another robot or scene has not been validated by this release work.
No hardware checkpoints are included in this release.

Preparation checks cover loading all selected pairs, fixed-input source/public
parity for the representative base/head, one CPU training update, frozen-base
behavior, save/reload, and short simulation rollouts. They do not re-establish
paper success rates. Original selection/evaluation records are included as
historical evidence. FT-mixed selection-panel wording remains under review.
The base's original training contract is missing from the inspected archive.

## License and attribution

Original ALTER checkpoints and accompanying original records are released under
Apache-2.0; see LICENSE and NOTICE. This license does not replace terms for any
separately obtained third-party dependencies, assets or the paper.

Paper: [Residual Denoising Enables Sample-Efficient Multi-Agent Coordination on Demand](https://arxiv.org/abs/2609.32129).
