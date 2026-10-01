# Hardware guide

This guide is for using the released hardware checkpoints and prepared data
offline. Physical robot operation requires the separate, private [ICON_Arm
control project](https://github.com/labicon/ICON_Arm); this repository does not
include the camera setup, robot services, or an installation path for live control.

## Choose what to download

The hardware artifacts use a separate manifest and live under `hardware/v1/` in
the public [model repository](https://huggingface.co/Berkeley-ICON-Lab/ALTER-models/tree/main/hardware/v1)
and [dataset repository](https://huggingface.co/datasets/Berkeley-ICON-Lab/ALTER-data/tree/main/hardware/v1).

| What you need | Bundles |
| --- | --- |
| Load and evaluate released checkpoints offline | `hardware-models` |
| Adapt a policy using the prepared training caches | `hardware-models hardware-training-data` |
| Also inspect selected demonstrations and replay | Add `hardware-demonstrations-data` |

## Download and prepare

Run commands from the repository root. Downloads are checked against the immutable
revisions and SHA-256 checksums in [`release/hardware-manifest.json`](../../release/hardware-manifest.json).
The downloader reuses valid files and rejects conflicting ones. Keep hardware and
simulation downloads in separate directories.

Download the model bundle:

```bash
python scripts/download_release.py --manifest release/hardware-manifest.json \
  --bundles hardware-models --destination /path/to/ALTER-hardware-artifacts
```

To prepare an offline inference tree, materialize and validate it:

```bash
python scripts/materialize_hardware_release.py \
  --bundle-root /path/to/ALTER-hardware-artifacts \
  --output "$PWD/public-validation/hardware-local" --bundles hardware-models

python scripts/validate_hardware_models.py \
  --root "$PWD/public-validation/hardware-local/hardware" \
  --output "$PWD/public-validation/hardware-models.json"
```

To prepare for training, also download `hardware-training-data`, then include it
in both `--bundles` lists above. Add `hardware-demonstrations-data` to both lists
if you need the selected source demonstrations and replay. Materialization needs
each bundle you name to be downloaded first. Use a fresh output directory.

The published model-only bundle supports offline model use. Training needs its
matching prepared caches. The released caches restore selected training inputs;
they do not reconstruct missing raw recordings.

## Train and evaluate

Start with the [installation guide](installation.md) to create the hardware
Python environment. The hardware entry points live in
`hardware_training/`; use `--help` for each command’s required inputs and options.
The selected workflows have passed short CPU training checks, but those checks
do not reproduce the paper success rates. The release includes example prepared
cache manifests under `hardware/manifests/`.

The released model/data cards and verification details are in
[Hugging Face publication](huggingface.md#hardware-v1-verification). Offline
validation does not run physical trials.

## Physical robot setup

Live operation needs the private ICON_Arm project, ROS 2 Humble, xArm drivers,
and RealSense camera drivers. Its operator-managed setup is documented in its
[bootstrap guide](https://github.com/labicon/ICON_Arm/blob/bcd023f2700cdc67a58981dbf965c14d50dec544/BOOTSTRAP.md).
Hardware machine identities and camera addresses are not part of this release.
Coordinate any physical setup with an operator; no physical robot execution was
performed for this release.
