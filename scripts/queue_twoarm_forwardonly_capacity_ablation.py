#!/usr/bin/env python3
"""Immutable, user-gated queue for the two-arm forward-only capacity ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCHEMA = "twoarm_forwardonly_capacity_ablation.v1"
PRESETS = {
    "L": (256, 4, 1024),
    "M": (192, 3, 768),
    "S": (128, 3, 512),
    "XS": (64, 2, 256),
}
STEPS = tuple(range(100000, 900001, 100000))
XS_COORD_JOB = "capacity-xs-coord"
DEFAULT_XS_COORD_GPU = 0
SMOKE_SEEDS = ",".join(str(seed) for seed in range(2000, 2010))
FULL_SEEDS = ",".join(str(seed) for seed in range(1000, 1050))


def stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(path: Path, value: dict[str, Any], immutable: bool = False) -> None:
    if immutable and path.exists():
        raise FileExistsError("immutable artifact exists: {}".format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".{}.{}".format(path.name, os.getpid()))
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def read(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("expected object: {}".format(path))
    return value


def job_file(root: Path, job_id: str) -> Path:
    return root / "specs" / "jobs" / "{}.json".format(job_id)


def state_file(root: Path, job_id: str) -> Path:
    return root / "state" / "jobs" / "{}.json".format(job_id)


def queue_state_file(root: Path) -> Path:
    return root / "state" / "queue.json"


def transition(state: dict[str, Any], status: str) -> None:
    state["status"] = status
    state.setdefault("history", []).append({"time": stamp(), "status": status})


def get_option(argv: list[str], name: str) -> str:
    return argv[argv.index(name) + 1]


def select_fs_architecture(
    target: int,
    count: Callable[[int, int], int],
    d_models=range(64, 769, 4),
    ffns=range(128, 4097, 4),
) -> dict[str, Any]:
    """Deterministically find the closest valid four-head, depth-three FS F."""
    if target <= 0:
        raise ValueError("FS parameter target must be positive")
    candidates = []
    for d_model in d_models:
        if d_model < 4 or d_model % 4:
            continue
        for dim_feedforward in ffns:
            params = int(count(d_model, dim_feedforward))
            candidates.append((abs(params - target), d_model, dim_feedforward, params))
    if not candidates:
        raise ValueError("no valid four-head FS candidates")
    error, d_model, dim_feedforward, params = min(candidates)
    relative_delta = error / target
    if relative_delta > 0.001:
        raise ValueError("closest FS architecture exceeds 0.1 percent tolerance")
    return {
        "d_model": d_model,
        "n_heads": 4,
        "depth": 3,
        "dim_feedforward": dim_feedforward,
        "f_params": params,
        "target_f_params": target,
        "delta_f_params": params - target,
        "relative_delta": relative_delta,
    }


def validate_forwardonly_inputs(config: dict[str, Any]) -> dict[str, Any]:
    """Validate the exact 5/mode two-arm plus distilled-single-arm protocol."""
    from envs.arm.train.exact_threearm_manifest import (
        PLACEWIPE_ALL_EXPERT_SCHEMA,
        load_exact_manifest,
    )

    required = [
        config["manifest"],
        config["base_model"],
        config["base_stats"],
        config["twoarm_dir"],
        *config["singlearm_dirs"],
    ]
    for item in required:
        if not Path(item).exists():
            raise FileNotFoundError(item)
    if len(config["singlearm_dirs"]) != 2:
        raise ValueError("two-arm forward-only capacity runs require place-return and wipe single-arm roots")

    manifest_path = Path(config["manifest"]).resolve()
    raw_bytes = manifest_path.read_bytes()
    raw = json.loads(raw_bytes)
    if raw.get("schema") != PLACEWIPE_ALL_EXPERT_SCHEMA:
        raise ValueError("requires the exact two-arm forward-only all-expert manifest")
    if raw.get("direction") != "forward_only":
        raise ValueError("requires direction=forward_only")
    if any(int(raw.get(key, -1)) != 5 for key in (
        "per_mode_demo_budget",
        "twoarm_per_mode_demo_budget",
        "singlearm_per_mode_demo_budget",
    )):
        raise ValueError("requires exact 5/mode two-arm and distilled-single-arm inputs")

    loaded = load_exact_manifest(
        str(manifest_path),
        multiarm_root=str(Path(config["twoarm_dir"]).resolve()),
        singlearm_roots=[str(Path(item).resolve()) for item in config["singlearm_dirs"]],
    )
    if len(loaded["multiarm_train_files"]) != 20 or len(loaded["singlearm_train_files"]) != 20:
        raise ValueError("exact 5/mode forward-only manifest has unexpected rollout counts")
    return {
        "path": str(manifest_path),
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "schema": raw["schema"],
        "direction": raw["direction"],
        "per_mode_demo_budget": 5,
        "twoarm_per_mode_demo_budget": 5,
        "singlearm_per_mode_demo_budget": 5,
        "twoarm_train_file_count": len(loaded["multiarm_train_files"]),
        "singlearm_train_file_count": len(loaded["singlearm_train_files"]),
    }


def capacities(base_stats: Path, *, include_fs: bool = True) -> dict[str, Any]:
    """Measure Coord and capacity-matched FS models from the frozen base stats."""
    import torch

    from src.image_coordination import (
        ALL_BLOCKS_DECODER_EXECUTION,
        DECODER_CONDITIONING_CROSS_ATTN,
        SIDE_NET_FUSION_CONCAT,
        ImageCoordinationHead,
    )
    from src.image_diffusion import ImageConditional_ODE

    with base_stats.open("rb") as stream:
        stats = pickle.load(stream)
    if (
        str(stats.get("backbone", "resnet18")) != "resnet18"
        or int(stats.get("num_cameras", 1)) != 1
        or list(stats.get("frame_offsets", [0])) != [0]
    ):
        raise ValueError("requires a frozen forward-only shoulder-only ResNet-18 frame-0 base")

    common = {
        "x_dim": 7,
        "sigma_data": float(stats.get("sigma_data", 1.0)),
        "horizon": int(stats.get("horizon", 20)),
        "device": torch.device("cpu"),
        "num_cameras": 1,
        "backbone": "resnet18",
        "frame_offsets": (0,),
    }
    base = ImageConditional_ODE(
        d_model=int(stats.get("d_model", 256)),
        n_heads=int(stats.get("n_heads", 4)),
        depth=int(stats.get("depth", 3)),
        dim_feedforward=int(stats.get("dim_feedforward", 1024)),
        **common,
    )
    frozen = sum(parameter.numel() for parameter in base.F.parameters())
    coord: dict[str, dict[str, int]] = {}
    for label, (d_model, depth, dim_feedforward) in PRESETS.items():
        head = ImageCoordinationHead(
            x_dim=7,
            base_d_model=int(stats.get("d_model", 256)),
            d_model=d_model,
            n_heads=4,
            depth=depth,
            dim_feedforward=dim_feedforward,
            horizon=int(stats.get("horizon", 20)),
            num_cameras=1,
            tokens_per_camera=base.F.tokens_per_camera,
            use_side_net=True,
            side_net_fusion=SIDE_NET_FUSION_CONCAT,
            side_net_tokens_per_camera=64,
            decoder_execution=ALL_BLOCKS_DECODER_EXECUTION,
            decoder_conditioning=DECODER_CONDITIONING_CROSS_ATTN,
            frame_offsets=(0,),
        )
        trainable = sum(parameter.numel() for parameter in head.F.parameters())
        coord[label] = {
            "frozen_base_F": frozen,
            "trainable_head_F": trainable,
            "deployed_F": frozen + trainable,
        }

    if not include_fs:
        return {"coord": coord, "fs": {}}

    at_ff4: dict[int, int] = {}

    def fs_count(d_model: int, dim_feedforward: int) -> int:
        if d_model not in at_ff4:
            model = ImageConditional_ODE(
                d_model=d_model,
                n_heads=4,
                depth=3,
                dim_feedforward=4,
                **common,
            )
            at_ff4[d_model] = sum(parameter.numel() for parameter in model.F.parameters())
        return at_ff4[d_model] + (dim_feedforward - 4) * (12 * d_model + 6)

    fs = {
        label: select_fs_architecture(row["deployed_F"], fs_count)
        for label, row in coord.items()
    }
    for row in fs.values():
        model = ImageConditional_ODE(
            d_model=row["d_model"],
            n_heads=4,
            depth=3,
            dim_feedforward=row["dim_feedforward"],
            **common,
        )
        if sum(parameter.numel() for parameter in model.F.parameters()) != row["f_params"]:
            raise RuntimeError("FS parameter count formula mismatch")
    return {"coord": coord, "fs": fs}


def train_argv(
    method: str,
    label: str,
    config: dict[str, Any],
    counts: dict[str, Any],
    spec_path: Path,
) -> list[str]:
    prefix = "twoarm_forwardonly_capacity_{}_{}".format(method, label.lower())
    output_dir = Path(config["checkpoint_root"]) / "{}_5permode_steps900000_seed0".format(prefix)
    common = [
        "--singlearm-dirs", *config["singlearm_dirs"],
        "--twoarm-modes", "0", "1", "2", "3",
        "--manifest-path", config["manifest"],
        "--batch-size", "256",
        "--sampling-policy", "hierarchical",
        "--lr", "0.0002",
        "--frame-offsets", "0",
        "--checkpoint-dir", str(output_dir),
        "--checkpoint-prefix", prefix,
        "--max-train-steps", "900000",
        "--checkpoint-every-steps", "100000",
        "--val-every-steps", "100000",
        "--training-seed", "0",
        "--device", "cuda:0",
        "--capacity-job-contract", str(spec_path),
    ]
    if method == "coord":
        d_model, depth, dim_feedforward = PRESETS[label]
        return [
            "envs/arm/train/train_mixed_coordination_e2e_shoulder.py",
            "--twoarm-dir", config["twoarm_dir"],
            "--base-model-path", config["base_model"],
            "--base-stats-path", config["base_stats"],
            "--num-arms", "2",
            "--head-d-model", str(d_model),
            "--head-n-heads", "4",
            "--head-depth", str(depth),
            "--head-dim-feedforward", str(dim_feedforward),
            "--d-base-drop-prob", "0.1",
            "--use-side-net",
            "--side-net-fusion", "concat",
            "--side-net-tokens-per-camera", "64",
            "--decoder-execution", "all_blocks_repeat_last",
            "--decoder-conditioning", "cross_attn",
            "--singlearm-weight", "1.0",
            "--weight-decay", str(config["coord_weight_decay"]),
            *common,
        ]

    fs = counts["fs"][label]
    return [
        "envs/arm/train/train_combined_e2e_shoulder.py",
        "--shoulder-only",
        "--num-arms", "2",
        "--twoarm-dirs", config["twoarm_dir"],
        "--domain-balanced-batches",
        "--d-model", str(fs["d_model"]),
        "--n-heads", "4",
        "--depth", "3",
        "--dim-feedforward", str(fs["dim_feedforward"]),
        "--expected-model-params", str(fs["f_params"]),
        "--model-param-tolerance", "0",
        "--backbone", "resnet18",
        "--no-augment",
        "--val-holdout-fraction", "0",
        "--val-batches", "0",
        "--stats-path", str(output_dir / "{}_stats.pkl".format(prefix)),
        *common,
    ]


def initialize(config: dict[str, Any]) -> None:
    root = Path(config["state_root"]).resolve()
    if (root / "specs").exists():
        raise FileExistsError("refusing to replace immutable queue specs")

    normalized = {
        "state_root": str(root),
        "checkpoint_root": str(Path(config["checkpoint_root"]).resolve()),
        "manifest": str(Path(config["manifest"]).resolve()),
        "base_model": str(Path(config["base_model"]).resolve()),
        "base_stats": str(Path(config["base_stats"]).resolve()),
        "twoarm_dir": str(Path(config["twoarm_dir"]).resolve()),
        "singlearm_dirs": [str(Path(item).resolve()) for item in config["singlearm_dirs"]],
        "methods": list(config.get("methods", ("coord", "fs"))),
        "coord_weight_decay": float(config.get("coord_weight_decay", 1e-2)),
    }
    if not normalized["methods"] or any(method not in {"coord", "fs"} for method in normalized["methods"]):
        raise ValueError("methods must be a nonempty subset of {'coord', 'fs'}")
    if normalized["coord_weight_decay"] < 0:
        raise ValueError("coord_weight_decay must be nonnegative")
    manifest = validate_forwardonly_inputs(normalized)
    if "fs" in normalized["methods"]:
        counts = capacities(Path(normalized["base_stats"]))
    else:
        counts = capacities(Path(normalized["base_stats"]), include_fs=False)
    base = {
        "model_path": normalized["base_model"],
        "stats_path": normalized["base_stats"],
        "base_model_sha256": sha256(normalized["base_model"]),
        "base_stats_sha256": sha256(normalized["base_stats"]),
        "backbone": "resnet18",
        "num_cameras": 1,
        "frame_offsets": [0],
    }

    output_dirs: set[str] = set()
    job_ids: list[str] = []
    for label in PRESETS:
        for method in normalized["methods"]:
            job_id = "capacity-{}-{}".format(label.lower(), method)
            spec_path = job_file(root, job_id)
            argv = train_argv(method, label, normalized, counts, spec_path)
            output_dir = get_option(argv, "--checkpoint-dir")
            if output_dir in output_dirs:
                raise ValueError("duplicate output path")
            output_dirs.add(output_dir)
            model_counts = counts["coord"][label] if method == "coord" else counts["fs"][label]
            spec = {
                "schema": SCHEMA,
                "job_id": job_id,
                "method": method,
                "label": label,
                "capacity_preset": {
                    "d_model": PRESETS[label][0],
                    "depth": PRESETS[label][1],
                    "dim_feedforward": PRESETS[label][2],
                },
                "model_counts": model_counts,
                "optimizer": {
                    "name": "AdamW",
                    "lr": 2e-4,
                    "weight_decay": (
                        normalized["coord_weight_decay"] if method == "coord" else 1e-4
                    ),
                },
                "manifest": manifest,
                "frozen_base": base,
                "output_dir": output_dir,
                "training_argv": argv,
                "command_sha256": canonical_hash(argv),
                "checkpoint_steps": list(STEPS),
                "training_contract": {
                    "checkpoint_contract_suffix": ".contract.json",
                    "job_spec_path": str(spec_path),
                    "required_job_spec_hash": "written immutable spec hash",
                },
                "created_utc": stamp(),
            }
            write(spec_path, spec, immutable=True)
            write(
                state_file(root, job_id),
                {
                    "job_id": job_id,
                    "status": "PENDING_TRAIN",
                    "history": [{"time": stamp(), "status": "PENDING_TRAIN"}],
                    "checkpoint_steps": list(STEPS),
                    "attempts": 0,
                },
            )
            job_ids.append(job_id)

    write(
        root / "specs" / "experiment.json",
        {
            "schema": SCHEMA,
            "config": normalized,
            "manifest": manifest,
            "frozen_base": base,
            "capacity": counts,
            "jobs": job_ids,
            "training_contract": "every checkpoint must have a sealed contract sidecar",
        },
        immutable=True,
    )
    write(
        queue_state_file(root),
        {
            "schema": SCHEMA,
            "history": [{"time": stamp(), "status": "INITIALIZED"}],
        },
    )


def gpu(index: int) -> dict[str, str]:
    text = subprocess.check_output(
        [
            "nvidia-smi", "--id", str(index),
            "--query-gpu=uuid,name,driver_version,power.limit,clocks.current.graphics,clocks.current.memory,mig.mode.current",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    values = [value.strip() for value in text.split(",")]
    return dict(zip(("uuid", "sku", "driver", "power", "graphics_clock", "memory_clock", "mig"), values))


def processes(index: int) -> str:
    return subprocess.check_output(
        [
            "nvidia-smi", "--id", str(index),
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()


def snapshot(index: int, workers: int, paths: list[str]) -> dict[str, Any]:
    return {
        "time": stamp(),
        "gpu": gpu(index),
        "processes": processes(index),
        "nvidia_smi": subprocess.check_output(["nvidia-smi", "--id", str(index)], text=True),
        "host": {
            "cpu_count": os.cpu_count(),
            "loadavg": list(os.getloadavg()),
            "cpu_pressure": Path("/proc/pressure/cpu").read_text(),
            "memory_pressure": Path("/proc/pressure/memory").read_text(),
            "io_pressure": Path("/proc/pressure/io").read_text(),
        },
        "worker_count": workers,
        "storage_source": [str(Path(path).resolve()) for path in paths],
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "environment": {
            key: os.environ.get(key)
            for key in ("CONDA_PREFIX", "PYTHONPATH", "CUDA_VISIBLE_DEVICES")
        },
    }


def evaluation_gpu_lock(root: Path, uuid: str) -> Path:
    """Serialize evaluation releases without constraining training placement."""
    path = root / "state" / "evaluation_locks" / "{}.lock".format(uuid)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError("one evaluation release already owns GPU UUID {}".format(uuid)) from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump({"pid": os.getpid(), "uuid": uuid, "time": stamp()}, stream)
    return path


def run_process(command_line: list[str], env: dict[str, str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    if log.exists():
        raise FileExistsError("refusing to append to log {}".format(log))
    with log.open("x", encoding="utf-8") as stream:
        stream.write("COMMAND {}\n".format(" ".join(command_line)))
        stream.flush()
        return subprocess.run(
            command_line,
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
        ).returncode


def _training_env(gpu_index: int) -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu_index),
        "MUJOCO_EGL_DEVICE_ID": str(gpu_index),
        "EGL_VISIBLE_DEVICES": str(gpu_index),
        "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl",
        "PYTHONUNBUFFERED": "1",
    })
    return env


def _checkpoint_contracts(spec: dict[str, Any]) -> dict[str, dict[str, str]]:
    prefix = get_option(spec["training_argv"], "--checkpoint-prefix")
    output_dir = Path(spec["output_dir"])
    contracts: dict[str, dict[str, str]] = {}
    for step in spec["checkpoint_steps"]:
        checkpoint = output_dir / "{}_step{}.pt".format(prefix, step)
        contract = Path(str(checkpoint) + ".contract.json")
        if not checkpoint.is_file() or not contract.is_file():
            raise FileNotFoundError("missing checkpoint or sealed contract at step {}".format(step))
        contracts[str(step)] = {
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "contract_path": str(contract),
            "contract_sha256": sha256(contract),
        }
    return contracts


def run_training(root: Path, job_id: str, gpu_index: int, python: str) -> None:
    spec = read(job_file(root, job_id))
    state = read(state_file(root, job_id))
    if state["status"] != "PENDING_TRAIN":
        raise RuntimeError("job is not pending training")
    identity = gpu(gpu_index)
    active = root / "state" / "active_training" / "{}.json".format(job_id)
    try:
        workers = 8 if spec["method"] == "coord" else 4
        evidence_paths = [
            spec["output_dir"],
            spec["manifest"]["path"],
            spec["frozen_base"]["model_path"],
            spec["frozen_base"]["stats_path"],
        ]
        state["attempts"] = int(state.get("attempts", 0)) + 1
        state["training_gpu_uuid"] = identity["uuid"]
        state["training_gpu_index"] = gpu_index
        state["pre_train_hardware"] = snapshot(gpu_index, workers, evidence_paths)
        transition(state, "RUNNING_TRAIN")
        write(state_file(root, job_id), state)
        write(active, {
            "job_id": job_id,
            "gpu_uuid": identity["uuid"],
            "started_utc": stamp(),
            "command_sha256": spec["command_sha256"],
        })
        log = root / "logs" / "{}.train.attempt{}.log".format(job_id, state["attempts"])
        returncode = run_process([python, "-u", *spec["training_argv"]], _training_env(gpu_index), log)
        state["post_train_hardware"] = snapshot(gpu_index, workers, evidence_paths)
        state["train_returncode"] = returncode
        state["train_log"] = str(log)
        if returncode != 0:
            state["retryable_failure"] = "training_returncode_{}".format(returncode)
            transition(state, "PENDING_TRAIN")
            write(state_file(root, job_id), state)
            return
        try:
            state["checkpoint_contracts"] = _checkpoint_contracts(spec)
        except FileNotFoundError as error:
            state["retryable_failure"] = str(error)
            transition(state, "PENDING_TRAIN")
            write(state_file(root, job_id), state)
            return
        state.pop("retryable_failure", None)
        transition(state, "PENDING_SMOKE")
        write(state_file(root, job_id), state)
    finally:
        try:
            active.unlink()
        except FileNotFoundError:
            pass


def assert_evaluation_gpu(training_gpu_uuid: str, evaluator_gpu_uuid: str) -> None:
    if training_gpu_uuid == evaluator_gpu_uuid:
        raise RuntimeError("evaluation GPU UUID must differ from training GPU UUID")


def select_checkpoint(scores: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the frozen nine-scan rule: total, min-mode, earliest step."""
    if len(scores) != len(STEPS):
        raise ValueError("selection requires nine completed checkpoint scans")
    return sorted(
        scores,
        key=lambda row: (
            -int(row["total_success"]),
            -int(row["minimum_per_mode_success"]),
            int(row["step"]),
        ),
    )[0]


def evaluation_command(spec: dict[str, Any], step: int, output_dir: Path, full: bool) -> list[str]:
    prefix = get_option(spec["training_argv"], "--checkpoint-prefix")
    checkpoint = Path(spec["output_dir"]) / "{}_step{}.pt".format(prefix, step)
    stats = Path(spec["output_dir"]) / "{}_stats.pkl".format(prefix)
    command_line = [
        "scripts/run_eval.py",
        "--mode", "coord-shoulderonly" if spec["method"] == "coord" else "fromscratch-shoulderonly",
        "--task", "placewipe",
        "--env-variant", "hard",
        "--seeds", FULL_SEEDS if full else SMOKE_SEEDS,
        "--repeats", "1",
        "--mode-list", "0,1,2,3",
        "--replan-freq", "15",
        # The 200-episode headline panel must match the established two-arm
        # PlaceWipe horizon. Checkpoint smokes remain deliberately longer
        # diagnostic rollouts, but must never define the headline comparison.
        "--max-steps", "1800" if full else "3000",
        "--no-video",
        "--device", "cuda:0",
        "--checkpoint-path", str(checkpoint),
        "--adapter-stats-path", str(stats),
        "--strict-experiment-contract",
        "--experiment-profile", "headline_hard" if full else "diagnostic_smoke",
        "--eval-purpose", "twoarm_capacity_full200" if full else "twoarm_capacity_smoke40",
        "--output-dir", str(output_dir),
    ]
    if spec["method"] == "coord":
        command_line.extend([
            "--base-model-path", spec["frozen_base"]["model_path"],
            "--base-stats-path", spec["frozen_base"]["stats_path"],
            "--selector-mode", "coord",
        ])
    return command_line


def _one_eval_result(output_dir: Path) -> dict[str, Any]:
    results = sorted(output_dir.glob("eval_*.json"))
    if len(results) != 1:
        raise RuntimeError("expected one evaluation result in {}".format(output_dir))
    return read(results[0])


def _claim_evaluation_gpu(root: Path, state: dict[str, Any], gpu_index: int) -> tuple[dict[str, str], Path]:
    identity = gpu(gpu_index)
    assert_evaluation_gpu(state["training_gpu_uuid"], identity["uuid"])
    if processes(gpu_index):
        raise RuntimeError("evaluation GPU is not idle")
    return identity, evaluation_gpu_lock(root, identity["uuid"])


def release_smoke(root: Path, job_id: str, gpu_index: int, python: str) -> None:
    spec = read(job_file(root, job_id))
    state = read(state_file(root, job_id))
    if state["status"] != "PENDING_SMOKE":
        raise RuntimeError("smoke release requires PENDING_SMOKE")
    identity, lock = _claim_evaluation_gpu(root, state, gpu_index)
    try:
        attempt = int(state.get("smoke_attempts", 0)) + 1
        state["smoke_attempts"] = attempt
        scores = []
        for step in STEPS:
            output_dir = root / "evaluations" / job_id / "smoke" / "step{}".format(step)
            log = root / "logs" / "{}.smoke.step{}.attempt{}.log".format(job_id, step, attempt)
            returncode = run_process(
                [python, "-u", *evaluation_command(spec, step, output_dir, False)],
                _training_env(gpu_index),
                log,
            )
            if returncode != 0:
                state["retryable_failure"] = "smoke_step_{}_returncode_{}".format(step, returncode)
                write(state_file(root, job_id), state)
                return
            result = _one_eval_result(output_dir)
            modes = result.get("success_by_mode", {})
            successes = [int(modes[str(mode)]["success"]) for mode in range(4)]
            scores.append({
                "step": step,
                "total_success": sum(successes),
                "minimum_per_mode_success": min(successes),
                "per_mode_success": successes,
            })
        selection = {
            "schema": SCHEMA,
            "job_id": job_id,
            "scores": scores,
            "selected": select_checkpoint(scores),
            "selection_rule": "highest total smoke success, then highest minimum per-mode success, then earliest checkpoint",
            "evaluation_gpu_uuid": identity["uuid"],
        }
        selection_path = root / "selections" / "{}.json".format(job_id)
        write(selection_path, selection, immutable=True)
        state["selection_path"] = str(selection_path)
        state["smoke_gpu_uuid"] = identity["uuid"]
        state.pop("retryable_failure", None)
        transition(state, "PENDING_FULL")
        write(state_file(root, job_id), state)
    finally:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


def release_full(root: Path, job_id: str, gpu_index: int, python: str) -> None:
    spec = read(job_file(root, job_id))
    state = read(state_file(root, job_id))
    if state["status"] != "PENDING_FULL":
        raise RuntimeError("full release requires PENDING_FULL")
    selection = read(Path(state["selection_path"]))
    step = int(selection["selected"]["step"])
    identity, lock = _claim_evaluation_gpu(root, state, gpu_index)
    try:
        attempt = int(state.get("full_attempts", 0)) + 1
        state["full_attempts"] = attempt
        output_dir = root / "evaluations" / job_id / "full" / "step{}".format(step)
        log = root / "logs" / "{}.full.step{}.attempt{}.log".format(job_id, step, attempt)
        returncode = run_process(
            [python, "-u", *evaluation_command(spec, step, output_dir, True)],
            _training_env(gpu_index),
            log,
        )
        if returncode != 0:
            state["retryable_failure"] = "full_step_{}_returncode_{}".format(step, returncode)
            write(state_file(root, job_id), state)
            return
        result = _one_eval_result(output_dir)
        record_path = root / "evaluations" / job_id / "full" / "release.json"
        write(record_path, {
            "schema": SCHEMA,
            "job_id": job_id,
            "step": step,
            "evaluation_gpu_uuid": identity["uuid"],
            "result_path": str(next(output_dir.glob("eval_*.json"))),
            "result_sha256": sha256(next(output_dir.glob("eval_*.json"))),
            "selection_path": state["selection_path"],
            "total_success": result.get("total_success"),
            "created_utc": stamp(),
        }, immutable=True)
        state["full_gpu_uuid"] = identity["uuid"]
        state["full_result_path"] = str(record_path)
        state.pop("retryable_failure", None)
        transition(state, "COMPLETED")
        write(state_file(root, job_id), state)
    finally:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    initialize_parser = subparsers.add_parser("initialize")
    initialize_parser.add_argument("--state-root", required=True)
    initialize_parser.add_argument("--checkpoint-root", required=True)
    initialize_parser.add_argument("--manifest", required=True)
    initialize_parser.add_argument("--base-model", required=True)
    initialize_parser.add_argument("--base-stats", required=True)
    initialize_parser.add_argument("--twoarm-dir", required=True)
    initialize_parser.add_argument("--singlearm-dirs", nargs=2, required=True)
    initialize_parser.add_argument(
        "--methods", nargs="+", choices=("coord", "fs"), default=("coord", "fs"),
        help="Methods to seal into this queue; use 'coord' for an optimizer-control rerun.",
    )
    initialize_parser.add_argument(
        "--coord-weight-decay", type=float, default=1e-2,
        help="Explicit AdamW weight decay for Coord jobs (legacy default: 1e-2).",
    )

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--state-root", required=True)
    run_parser.add_argument("--job-id", required=True)
    run_parser.add_argument("--gpu-index", type=int, required=True)
    run_parser.add_argument("--python", default=sys.executable)

    first_launch = subparsers.add_parser("launch-xs-coord")
    first_launch.add_argument("--state-root", required=True)
    first_launch.add_argument("--gpu-index", type=int, default=DEFAULT_XS_COORD_GPU)
    first_launch.add_argument("--python", default=sys.executable)

    for action in ("release-smoke", "release-full"):
        release = subparsers.add_parser(action)
        release.add_argument("--state-root", required=True)
        release.add_argument("--job-id", required=True)
        release.add_argument("--gpu-index", type=int, required=True)
        release.add_argument("--python", default=sys.executable)

    status = subparsers.add_parser("status")
    status.add_argument("--state-root", required=True)
    args = parser.parse_args()
    root = Path(args.state_root)
    if args.action == "initialize":
        initialize({
            key: getattr(args, key)
            for key in (
                "state_root", "checkpoint_root", "manifest", "base_model", "base_stats",
                "twoarm_dir", "singlearm_dirs", "methods", "coord_weight_decay",
            )
        })
    elif args.action == "run":
        run_training(root, args.job_id, args.gpu_index, args.python)
    elif args.action == "launch-xs-coord":
        run_training(root, XS_COORD_JOB, args.gpu_index, args.python)
    elif args.action == "release-smoke":
        release_smoke(root, args.job_id, args.gpu_index, args.python)
    elif args.action == "release-full":
        release_full(root, args.job_id, args.gpu_index, args.python)
    else:
        jobs = [read(path) for path in sorted((root / "state" / "jobs").glob("*.json"))]
        print(json.dumps(jobs, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
