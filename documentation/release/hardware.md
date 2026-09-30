# Hardware preparation and offline validation

The hardware profile covers current-frame inference and training. The recovered
local archive contains the original 100k base and six 25k models (H69, S69,
FT_mixed, H30, S30, FT-mixed-30), plus their statistics and selected source data.
On 2026-09-30, the author confirmed that these six 25k checkpoints were used for
the paper hardware results. These hardware artifacts have not been published to
the Hub. Real robot validation of the public port remains pending; offline tests
do not establish hardware success rates.

## Recovered artifact checks

The archive contains 332 base-training and 32 validation records, 69 paired
demonstrations, and 68 replay records. The low-data subset contains 30 pairs and
30 replay records. Shared arm records refer to 501 unique data files. Base split
membership and ordering match the saved base statistics. All 574 archive files
passed inventory, size and SHA-256 checks.

In the separate Python 3.10 hardware environment, validate the recovered archive:

```bash
python scripts/validate_hardware_models.py --root /path/to/hardware \
  --output "$PWD/public-validation/hardware-models.json"
```

The validator uses the archive-relative model catalog and two saved camera
frames. It enables process-local NumPy 2 pickle-name compatibility for NumPy
1.26, without modifying datasets or the installed NumPy package. All seven
model/statistics pairs passed CPU loading and 50-step anchored sampling. The
14 fixed-input outputs matched an isolated archived source copy byte-for-byte
in the same local environment. This is not a cross-version numerical guarantee.
Only load trusted, checksum-verified pickle/PyTorch artifacts.

## Recovered training caches

The separate prepared-cache supplement contains all 206 original cache directories.
Its 774 files pass the pinned manifest, complete file-set, size and SHA-256 checks.
All action, image and timestamp array hashes match the previously recorded
training inputs. The high-data manifest has 206 records and the low-data manifest
has 90 records drawn from the same cache pool. Relocation changes only cache
paths, preserving record order and all other metadata.

The public loader reproduces the sample counts, eligibility exclusions and
sampling probabilities recorded in all six selected model statistics:

| Manifest | Two-arm samples | Single-arm replay samples |
| --- | ---: | ---: |
| High | 93,164 | 44,224 |
| Low | 37,319 | 19,921 |

All six public training workflows passed short CPU checks on the recovered
inputs: one update for each coordination head and five updates for each full
policy, with batch size 2 and zero loader workers. Losses and gradients were
finite; frozen-base and encoder/decoder update assertions passed. Each saved
checkpoint reloaded and produced a finite 20-by-7 action sample. A one-update
from-scratch check was too short to satisfy the existing encoder-update assertion
because of zero-initialized decoder layers; the assertion was retained.
These checks establish functional loading and training, not reproduced success rates.

The local manifests and archive metadata retain historical private paths; they
are validation inputs, not sanitized public exports. Prepared caches restore the
selected training inputs without reconstructing missing raw recordings. Hardware
artifact download instructions will be added when sanitized bundles are published.

For an existing verified local archive, use a fresh output directory and explicit
artifact paths. This one-step CPU check exercises coordination training with the
recorded replay and augmentation settings; it is not a paper reproduction run:

```bash
HW=/path/to/hardware
MANIFEST=/path/to/materialized/high.json
BASE="$HW/checkpoints/joint_bird_cardboardfb_aug24fwd_aug16bwd_pruned_uw"
python -m hardware_training.run_numpy_pickle_compat hardware_training.train_coordination_ab \
  --manifest "$MANIFEST" --reference-stats "$HW/checkpoints/sep14_H69/mixed_coord_head_placewipe_hardware_stats.pkl" \
  --base-checkpoint "$BASE/singlearm_mixedfront_e2e_shoulder_step100000.pt" \
  --base-stats "$BASE/singlearm_mixedfront_e2e_shoulder_stats.pkl" \
  --variant A --light-augmentation --grasp-transition-fraction 0 \
  --device cpu --steps 1 --batch-size 2 --microbatch 2 --workers 0 --save-every 1 \
  --output "$PWD/public-validation/hardware-H69"
```

The full-policy trainer also accepts `--device cpu` (default: `cuda`). For the
selected full-policy runs, supply `--include-singlearm`,
`--grasp-transition-fraction 0`, `--expected-episodes 69` (or `30` for the low
manifest), and the matching `--reference-stats`. FT-mixed additionally requires
`--init-checkpoint` and `--init-stats` pointing to the original base model and
statistics. From-scratch runs omit these initialization options. Use the NumPy
compatibility wrapper above when loading the recovered statistics under NumPy 1.26.

## External hardware setup

Physical robot operation uses the separate general robot-control project
[ICON_Arm](https://github.com/labicon/ICON_Arm), revision
`bcd023f2700cdc67a58981dbf965c14d50dec544`. It documents Ubuntu 22.04 / ROS 2 Humble,
the xArm ROS2 vendor `humble` branch, `xarm_msgs`, `cv_bridge`, and RealSense
`realsense2_camera`. See its [bootstrap guide](https://github.com/labicon/ICON_Arm/blob/bcd023f2700cdc67a58981dbf965c14d50dec544/BOOTSTRAP.md).
The author confirms that ICON_Arm is private and unavailable to external users.
Its inspected `package.xml` declares `TODO: License declaration`; ALTER does not
assign a license to this separate project. Exact deployed driver/camera commits
were not available in this checkout. These gaps affect the physical robot setup; they do
not block offline model validation. ICON_Arm remains an external dependency and
is not bundled with ALTER or covered by its license. The current public release
does not provide a complete installation path for physical robot operation.
Do not replace those components with untested implementations.

The two arm profiles retain the names `bird` and `cardboard`. Configure local
camera identities, addresses and ROS domains in the external hardware workspace
with an operator. This code does not ship the lab's device identities.

| Stage | Entry points |
| --- | --- |
| Raw conversion | `hardware_training/convert_npz_to_pkl.py`, `convert_npz_to_pkl_twoarm.py` |
| Audited preparation | `hardware_training/prepare_coordination_ab.py` (explicit data, original recording and selection roots) |
| Base training | `hardware_training/train_hardware_singlearm.py`, `train_singlearm_expert_lightaug.py` |
| ALTER | `hardware_training/train_coordination_ab.py` |
| FS / full-policy FT | `hardware_training/train_twoarm_standalone.py` |
| Offline inference | `hardware_training/inference_offline_eval.py`, `inference_offline_eval_coord.py` |
| ROS inference / execution | `scripts/run_sep14.sh`, `scripts/infer_campaign_sep14.sh` |

Keep the original action units and order: `[x_mm, y_mm, z_mm, roll, pitch, yaw,
gripper]`. Preserve preprocessing, phase weighting, augmentation, frame
eligibility and normalization. Offline tests do not reproduce scored physical
trials or establish permission to release camera recordings.

To run offline executor tests in the hardware environment:

```bash
PYTHONPATH=. python tests/run_hardware_mocks.py -o addopts=''
```

This harness installs inert ROS types; it does not start ROS or call a service.
It checks sequencing and gripper behavior. The Python 3.10 offline profile was
tested separately from simulation. Actual ROS imports/builds and camera streams
still need operator-coordinated validation on the hardware machine.

Set `PY` (or `PYTHON_BIN`) and `CHECKPOINT_ROOT` explicitly. The launcher accepts
`--dry-run` to print a command; direct executor `--dry-run` exercises more of the
ROS loop and should not be confused with printing only. Selected model folders
must match the historical layout documented by the recovered hardware artifacts.
The executor still enables servo mode and the gripper before its confirmation
prompt. This public port preserves that behavior; any change requires author
review and separate testing. No robot execution was launched during the port.
