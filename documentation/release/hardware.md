# Hardware preparation and offline validation

The hardware profile covers current-frame inference and training. The recovered
local archive contains the original 100k base and six documented 25k comparison
models (H69, S69, FT_mixed, H30, S30, FT-mixed-30), plus their statistics and
selected source data. These hardware artifacts have not been published to the
Hub. Real robot validation remains pending; offline tests do not establish
hardware success rates or identify the checkpoints used for scored paper trials.

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

## Training inputs still needed

The archived source data are available, but the selected trainers consume prepared
`images.npy` and `trajectory.npz` caches referenced by 206 manifest records.
Those caches are absent, and some original raw recordings needed for exact
reconstruction are absent too. Recover the matching caches and verify their
recorded array hashes before claiming that the archive supports training.
Do not substitute guessed timestamps, phases, eligibility or sampling weights.
Original metadata contains private paths and must be sanitized in independent
exports before artifact publication.

## External hardware setup

The inspected external dependency is
[ICON_Arm](https://github.com/labicon/ICON_Arm), revision
`bcd023f2700cdc67a58981dbf965c14d50dec544`. It documents Ubuntu 22.04 / ROS 2 Humble,
the xArm ROS2 vendor `humble` branch, `xarm_msgs`, `cv_bridge`, and RealSense
`realsense2_camera`. See its [bootstrap guide](https://github.com/labicon/ICON_Arm/blob/bcd023f2700cdc67a58981dbf965c14d50dec544/BOOTSTRAP.md).
Public access and its license have not been established; `package.xml` declares
`TODO: License declaration`. Exact deployed driver/camera commits were not
available in this checkout. These are live installation/release blockers.
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
eligibility and normalization. The staged tests do not certify the missing
paper hardware data or its release permission.

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
