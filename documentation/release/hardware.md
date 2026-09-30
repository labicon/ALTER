# Hardware preparation and offline validation

The hardware profile covers current-frame inference and training. Hardware
checkpoint/data recovery and real robot validation remain pending. Do not infer
hardware success from mock tests or simulation results.

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
