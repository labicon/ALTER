# Installation

Run commands from the repository root. The public preparation was tested with
Python 3.12.4 for simulation imports/model tests and Python 3.10.21 for hardware
offline tests. The archived Python 3.9 specification is not the release profile:
some runtime annotations require Python 3.10 or newer.

Create an independent environment. For the tested CPU configuration:

```bash
python3.12 -m venv .venv-simulation
source .venv-simulation/bin/activate
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements/simulation.txt
export PYTHONPATH="$PWD"
export PYTHONDONTWRITEBYTECODE=1
```

The simulation profile pins MuJoCo 3.3.0 and robosuite 1.5.1. Short CPU simulation rollouts passed using OSMesa (libosmesa6
25.1.7 on the validation host) with `MUJOCO_GL=osmesa` and
`PYOPENGL_PLATFORM=osmesa`. EGL failed on that host. Install the appropriate
OSMesa package for your system; these checks do not reproduce paper success rates. GPU
training requires a PyTorch build compatible with the host driver; a fresh GPU
installation has not been validated for this release preparation.

For hardware offline inference, create a separate Python 3.10 environment and
install the same PyTorch pair and `requirements/hardware.txt`. Live ROS Humble
requires Ubuntu 22.04 and its matching Python 3.10 message extensions. An offline
venv by itself does not provide ROS, camera drivers, or robot services.

```bash
python3.10 -m venv .venv-hardware
source .venv-hardware/bin/activate
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements/hardware.txt
```

Optional W&B logging requires installing `wandb` and supplying your own account.
The paper-selected ResNet18 encoder is contained in policy checkpoints and does
not fetch ImageNet weights. DINOv2 is an optional code path, not a verified paper
artifact: it additionally needs `timm` and verified weights through
`CODIFF_DINO_WEIGHTS`. Network fetching requires an explicit
`CODIFF_ALLOW_BACKBONE_DOWNLOAD=1`; its upstream revision is not pinned here.

Keep downloads outside Git. Set `CODIFF_DATA_ROOT`, `CHECKPOINT_ROOT`, and
`ROLLOUT_ROOT` for your installation; legacy `CODIFF_*` names remain supported.
Use explicit fresh output paths under `public-validation/` for tests. The
`.codiff.local.env.example` file and `scripts/env.sh` describe shell overrides.
Direct Python entry points require exported variables or explicit arguments.
