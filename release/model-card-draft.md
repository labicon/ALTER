# ALTER simulation model card — staging draft

Authors: Dayi Dong, Maulik Bhatt, Aayushi Shrivastava, Lasse Peters, Negar Mehr.

Contains the selected frozen base and 18 distinct adapted policies for the
reported two-arm budgets and coordination-head capacity comparison. Base and
normalization statistics are paired in the workflow artifact matrix. Model and
EMA state tensors are preserved; exported path-bearing metadata is sanitized
with original/export checksum receipts. Existing module/checkpoint formats are
retained. ResNet18 encoder parameters are contained in the policy files.

Use the public repository's installation, materialization and evaluation guides.
The fromscratch inference route also loads full-policy FT checkpoints.
These are research policies for their recorded observation/action conventions;
transfer to another robot or scene has not been validated by this release work.
No hardware checkpoints are included in this staging set.

Preparation checks cover loading all selected pairs, fixed-input source/public
parity for the representative base/head, one CPU training update, frozen-base
behavior, save/reload, and short simulation rollouts. They do not re-establish
paper success rates. Original selection/evaluation records are included as
historical evidence. FT-mixed selection-panel wording remains under review.
The base's original training contract is missing from the inspected archive.

License and publication permission: pending team approval. Do not upload this
staging draft or infer a license from the paper or another dependency.
