# Licensing and attribution review

No project license has been selected. This staging branch is not approved for
publication. A public repository alone does not establish redistribution rights.
The authors/team must approve terms after confirming ownership and obligations.

| Material | Evidence and decision still needed |
| --- | --- |
| ALTER source | Preserve the existing author attribution: Dayi Dong, Maulik Bhatt, Aayushi Shrivastava, Lasse Peters, Negar Mehr. Confirm contributor ownership and any upstream-derived implementations. No top-level source license was found. |
| Policy checkpoints | Simulation state dictionaries include trained ResNet18 encoders. Confirm permission for weights and included metadata; encoder architecture imports torchvision. |
| Demonstrations/replay | Confirm recording/simulator asset rights and permission to distribute demonstrations and policy-derived replay. Hardware files remain unavailable. |
| Website/paper/media | Existing public website retained byte-for-byte. Existing publication is not proof of permission to relicense paper figures, clips or logos. |
| ICON_Arm | Inspected package has a placeholder license; public availability and rights need confirmation. |
| External dependencies | Installed separately, not vendored. Review installed package notices and exact versions before distribution of any environment or assets. |

Apache-2.0 is a candidate for code: it includes a contributor patent grant and
requires preserving relevant notices. MIT is a shorter permissive alternative.
This is a proposed team choice, not a license declaration.
[Apache terms](https://www.apache.org/licenses/LICENSE-2.0),
[MIT terms](https://opensource.org/license/mit).
Weights, datasets and media need explicit compatible terms of their own; do not
apply the code choice to them automatically. Hugging Face cards should declare
only approved terms, using its [license metadata](https://huggingface.co/docs/hub/repositories-licenses).

The [installed dependency metadata inventory](../../release/dependency-license-inventory.json) records versions and available license notices. It is evidence for review, not a complete legal clearance of upstream-derived source or assets.
