# Licensing and attribution

The author selected Apache-2.0 for original ALTER code and checkpoints, and
CC BY 4.0 for original simulation demonstrations and replay datasets, and
explicitly authorized public simulation artifact upload on 2026-09-30.

[LICENSE](../../LICENSE) and [NOTICE](../../NOTICE) identify the source scope.
The standard texts are also available in [release/licenses](../../release/licenses/).
The paper, website assets, logos and third-party components retain their own
terms. The arXiv distribution license for the paper is unchanged. This technical
release record does not assert a separately verified institutional approval or
comprehensive legal clearance of upstream material.

| Material | Release terms and attribution |
| --- | --- |
| Original ALTER source | Apache-2.0. Authors: Dayi Dong, Maulik Bhatt, Aayushi Shrivastava, Lasse Peters, Negar Mehr. Existing upstream notices remain applicable. |
| Original simulation checkpoints | Apache-2.0. Selected ResNet18 encoder weights are included; the released encoder implementation constructs torchvision ResNet18 with weights=None. No separate DINO weight files are included. |
| Original simulation demonstrations/replay and records | CC BY 4.0. Simulator software and standalone scene/robot asset packages are installed separately; their terms are not replaced. |
| Website/paper/media | Retained unchanged; not relicensed by the code or artifact declarations. |
| Hardware artifacts | Published separately under `hardware/v1/` in the existing model/data repositories. This addition leaves existing repository license files unchanged. |
| ICON_Arm | Private external dependency, not bundled or relicensed by ALTER; its inspected package has a placeholder license. |
| External dependencies | Installed separately, not bundled as an environment. Preserve their applicable notices and terms. |

The [installed dependency metadata inventory](../../release/dependency-license-inventory.json)
records inspected versions and available license notices. It does not establish
ownership or replace the licenses of those packages.
