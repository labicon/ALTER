"""Experiment contract helpers for training and evaluation.

Contracts are JSON sidecars stored beside checkpoints.  They are intentionally
plain dictionaries so training scripts can add them without changing model
loading, while eval can validate the parts that are known.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import socket
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "experiment_contract.v1"
CONTRACT_FILENAME = "experiment_contract.json"
CONTRACT_SUFFIX = ".contract.json"
METRIC_SCHEMA_VERSION = "eval_metrics.v1"
PURE_FINETUNE_JOB_SCHEMA = "pure_finetune_full_policy_job.v1"
# V2 is intentionally a separate immutable campaign schema.  The trainer
# accepts v1 only to preserve already-sealed historical jobs.
PURE_FINETUNE_V2_JOB_SCHEMA = "pure_finetune_full_policy_job.v2"
HEADLINE_PLACEWIPE_SEEDS = list(range(1000, 1050))
# Process-local cache for immutable training inputs. The fingerprint prevents
# reuse after a checkpoint or stats file is rewritten.
_HASH_CACHE: dict[tuple[str, int | None], tuple[tuple[int, int, int, int], str]] = {}

# A profile may override or deliberately omit only the protocol fields stated
# below; artifact, task, and environment checks always remain active. Each
# diagnostic explicitly documents any protocol deviations.
EVALUATION_PROFILES: dict[str, dict[str, Any]] = {
    "headline_hard": {"purpose": "headline_hard", "overrides": {}, "ignore": ()},
    # Standard strict three-arm primary evaluation: no protocol deviations.
    # This profile is deliberately distinct from the hard-environment headline
    # protocol used by PlaceWipe experiments.
    "primary_three_arm": {"purpose": "primary_three_arm", "overrides": {}, "ignore": ()},
    "diagnostic_base_routing": {
        "purpose": "diagnostic_base_routing",
        "overrides": {"selector_mode": "base"},
        "ignore": (),
    },
    "diagnostic_selector_routing": {
        "purpose": "diagnostic_selector_routing",
        "overrides": {"selector_mode": "auto"},
        # The selector study owns its short seed batch, max-step setting, and
        # optional videos; they are not headline-comparable quantities.
        "ignore": ("seeds", "max_steps", "video_policy"),
    },
    "diagnostic_smoke": {
        "purpose": "diagnostic_smoke",
        # Smokes may reduce only the seed, mode, and step budget. Artifact,
        # task, environment, RF, routing, repeat, and video checks remain strict.
        "overrides": {},
        "ignore": ("seeds", "modes", "max_steps"),
    },
    "diagnostic_forensic_video": {
        "purpose": "diagnostic_forensic_video",
        "overrides": {"video_policy": "enabled"},
        # Exact forensic examples are selected by mode and seed. They retain
        # the primary max-step and RF15 protocol.
        "ignore": ("seeds", "modes"),
    },
    "diagnostic_rf5": {
        "purpose": "diagnostic_rf5",
        "overrides": {"replan_freq": 5},
        "ignore": (),
    },
    "diagnostic_rf10": {
        "purpose": "diagnostic_rf10",
        "overrides": {"replan_freq": 10},
        "ignore": (),
    },
    "diagnostic_short_video": {
        "purpose": "diagnostic_short_video",
        "overrides": {"max_steps": 1400, "video_policy": "enabled"},
        # Video batches are selected by the caller and are diagnostic only.
        "ignore": ("seeds",),
    },
}

_MODE_RE = re.compile(r"mode(-?\d+)")


def _json_safe(value: Any) -> Any:
    """Convert common scientific-Python values into JSON-safe values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    return str(value)


def _abs_or_none(path: str | os.PathLike[str] | None) -> str | None:
    if not path:
        return None
    return os.path.abspath(os.fspath(path))


def _file_fingerprint(path: Path) -> tuple[int, int, int, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def sha256_file(path: str | os.PathLike[str] | None, *, max_bytes: int | None = None) -> str | None:
    """Hash a file, reusing a digest only while its stat fingerprint is stable."""
    if not path:
        return None
    p = Path(path)
    fingerprint = _file_fingerprint(p)
    if fingerprint is None or not p.is_file():
        return None
    cache_key = (str(p.resolve()), max_bytes)
    cached = _HASH_CACHE.get(cache_key)
    if cached is not None and cached[0] == fingerprint:
        return cached[1]

    h = hashlib.sha256()
    read = 0
    with p.open("rb") as f:
        while True:
            if max_bytes is None:
                chunk = f.read(1024 * 1024)
            else:
                remaining = max_bytes - read
                if remaining <= 0:
                    break
                chunk = f.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            h.update(chunk)
            read += len(chunk)
    digest = h.hexdigest()
    # Do not cache a digest observed while the file was changing.
    if _file_fingerprint(p) == fingerprint:
        _HASH_CACHE[cache_key] = (fingerprint, digest)
    return digest


def _run_git(args: list[str], repo_root: str | None) -> str | None:
    if not repo_root:
        return None
    try:
        out = subprocess.check_output(
            ["git", "-C", repo_root, *args],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    return out.strip()


def git_summary(repo_root: str | None = None) -> dict[str, Any]:
    repo_root = repo_root or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    status = _run_git(["status", "--short"], repo_root)
    return {
        "commit": _run_git(["rev-parse", "HEAD"], repo_root),
        "dirty": bool(status),
        "dirty_status": status.splitlines()[:50] if status else [],
    }


def _stable_hash(payload: dict[str, Any]) -> str:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def finalize_contract(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize and seal a contract after any provenance annotations."""
    payload = _json_safe(payload)
    identity_payload = json.loads(json.dumps(payload, sort_keys=True))
    identity_payload.setdefault("contract", {}).pop("contract_id", None)
    identity_payload["contract"].pop("content_sha256", None)
    content_hash = _stable_hash(identity_payload)
    payload.setdefault("contract", {})["content_sha256"] = content_hash
    payload["contract"]["contract_id"] = content_hash[:16]
    return payload


def contract_path_for_checkpoint(checkpoint_path: str | os.PathLike[str]) -> Path:
    """Return the sidecar path uniquely associated with one checkpoint."""
    checkpoint = Path(checkpoint_path).resolve()
    return checkpoint.with_name(checkpoint.name + CONTRACT_SUFFIX)

def load_capacity_ablation_job_contract(path: str | None, *, method: str) -> dict[str, Any] | None:
    """Load and seal an immutable two-arm capacity-ablation job spec."""
    if not path:
        return None
    spec_path = Path(path).resolve()
    with spec_path.open(encoding="utf-8") as handle:
        spec = json.load(handle)
    if spec.get("schema") != "twoarm_forwardonly_capacity_ablation.v1":
        raise ValueError("unsupported capacity-ablation job schema")
    if spec.get("method") != method:
        raise ValueError("capacity-ablation job method does not match trainer")
    expected = spec.get("training_argv")
    if not isinstance(expected, list) or not expected or not all(isinstance(value, str) for value in expected):
        raise ValueError("capacity-ablation job has no immutable training command")
    expected_script = Path(expected[0]).resolve()
    runtime_script = Path(sys.argv[0]).resolve()
    if expected_script != runtime_script or expected[1:] != list(sys.argv[1:]):
        raise ValueError("runtime command differs from immutable capacity-ablation job spec")
    if spec.get("command_sha256") != _stable_hash(expected):
        raise ValueError("capacity-ablation command hash mismatch")
    for key in ("job_id", "output_dir", "manifest", "frozen_base", "model_counts"):
        if key not in spec:
            raise ValueError("capacity-ablation job is missing {}".format(key))
    immutable_inputs = (
        ("manifest", spec["manifest"], "path", "sha256"),
        ("frozen base model", spec["frozen_base"], "model_path", "base_model_sha256"),
        ("frozen base stats", spec["frozen_base"], "stats_path", "base_stats_sha256"),
    )
    for description, record, path_key, hash_key in immutable_inputs:
        if not isinstance(record, dict) or not record.get(path_key) or not record.get(hash_key):
            raise ValueError("capacity-ablation job is missing {} identity".format(description))
        if sha256_file(record[path_key]) != record[hash_key]:
            raise ValueError("{} no longer matches its immutable capacity-ablation hash".format(description))
    sealed = dict(spec)
    sealed["job_spec_path"] = str(spec_path)
    sealed["job_spec_sha256"] = sha256_file(spec_path)
    return sealed

def _contains_capacity_matching_controls(value: Any) -> bool:
    """Reject parameter/capacity matching controls anywhere in a pure spec."""
    prohibited = {
        "capacity_ablation", "expected_model_params", "model_param_tolerance",
        "capacity_job_contract", "parameter_matching", "capacity_matching",
    }
    if isinstance(value, dict):
        for key, nested in value.items():
            normalized = str(key).replace("-", "_").lower()
            if normalized in prohibited or _contains_capacity_matching_controls(nested):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_contains_capacity_matching_controls(item) for item in value)
    return False


def _validate_pure_finetune_v2_spec(spec: dict[str, Any]) -> None:
    """Validate v2's treatment, phase, and fixed-GPU campaign invariants."""
    required = ("campaign", "phase", "data_treatment", "gpu_assignment", "smoke", "watcher_argv")
    missing = [key for key in required if key not in spec]
    if missing:
        raise ValueError("pure-finetune v2 job is missing {}".format(", ".join(missing)))
    treatment = spec["data_treatment"]
    if treatment not in ("mixed_domain", "multiarm_only"):
        raise ValueError("pure-finetune v2 has an invalid data treatment")
    regime = spec["regime"]
    manifest = spec["manifest"]
    contract = spec["training_contract"]
    if regime.get("data_treatment") != treatment or contract.get("data_treatment") != treatment:
        raise ValueError("pure-finetune v2 treatment metadata disagrees")
    expected_identity = (
        ("placewipe", 2) if regime.get("name") == "twoarm_placewipe"
        else (("wipe", 3) if regime.get("name") == "threearm_highvar_tied_return_v3" else None)
    )
    if expected_identity is None or (regime.get("task_family"), regime.get("num_arms")) != expected_identity:
        raise ValueError("pure-finetune v2 has the wrong task family or arm count")
    if (manifest.get("task_family"), manifest.get("num_arms")) != expected_identity:
        raise ValueError("pure-finetune v2 manifest has the wrong task family or arm count")
    if (manifest.get("multiarm_train_file_count"), manifest.get("singlearm_train_file_count")) != (
        regime.get("multiarm_train_file_count"), regime.get("singlearm_train_file_count")
    ):
        raise ValueError("pure-finetune v2 manifest counts disagree with its regime")
    if regime.get("no_augmentation") is not True or contract.get("augmentation") is not False:
        raise ValueError("pure-finetune v2 must seal no augmentation")
    if regime.get("frame_offsets") != [0] or contract.get("frame_offsets") != [0]:
        raise ValueError("pure-finetune v2 must seal frame offset zero")
    if contract.get("added_modules") != [] or contract.get("frozen_modules_after_initialization") != []:
        raise ValueError("pure-finetune v2 must not add or freeze modules")
    expected_effective_batch = 512 if treatment == "mixed_domain" else 256
    expected_consumed = {"multiarm": 256, "singlearm": 256 if treatment == "mixed_domain" else 0}
    if contract.get("effective_batch_size") != expected_effective_batch:
        raise ValueError("pure-finetune v2 has the wrong effective batch size")
    if contract.get("training_consumed_samples_per_update") != expected_consumed:
        raise ValueError("pure-finetune v2 has the wrong consumed sample counts")
    available = regime.get("manifest_available_rollout_counts")
    if available != {
        "multiarm": regime.get("multiarm_train_file_count"),
        "singlearm": regime.get("singlearm_train_file_count"),
    }:
        raise ValueError("pure-finetune v2 manifest-available counts disagree")
    if treatment == "multiarm_only" and int(available["singlearm"]) <= 0:
        raise ValueError("pure-finetune v2 multiarm-only jobs must still validate a complete manifest")
    gpu = spec["gpu_assignment"]
    if not isinstance(gpu, dict) or not isinstance(gpu.get("training_gpu_index"), int):
        raise ValueError("pure-finetune v2 has no sealed training GPU")
    if gpu["training_gpu_index"] not in (0, 1, 2, 3):
        raise ValueError("pure-finetune v2 training GPU is outside the campaign allocation")
    if gpu.get("smoke_watcher_gpu_index") != gpu["training_gpu_index"]:
        raise ValueError("pure-finetune v2 watcher must share the sealed training GPU")
    if spec["phase"].get("name") not in ("A", "B"):
        raise ValueError("pure-finetune v2 has an invalid phase")
    if not isinstance(spec["watcher_argv"], list) or not all(isinstance(x, str) for x in spec["watcher_argv"]):
        raise ValueError("pure-finetune v2 has no immutable watcher command")
    if spec.get("watcher_command_sha256") != _stable_hash(spec["watcher_argv"]):
        raise ValueError("pure-finetune v2 watcher command hash mismatch")


def load_pure_finetune_job_contract(path: str | None) -> dict[str, Any] | None:
    """Load and verify one immutable pure full-policy fine-tuning job spec."""
    if not path:
        raise ValueError("pure fine-tuning requires --pure-finetune-job-contract")
    spec_path = Path(path).resolve()
    with spec_path.open(encoding="utf-8") as handle:
        spec = json.load(handle)
    if spec.get("schema") not in (PURE_FINETUNE_JOB_SCHEMA, PURE_FINETUNE_V2_JOB_SCHEMA):
        raise ValueError("unsupported pure-finetune job schema")
    expected = spec.get("training_argv")
    if not isinstance(expected, list) or not expected or not all(isinstance(value, str) for value in expected):
        raise ValueError("pure-finetune job has no immutable training command")
    expected_script = Path(expected[0]).resolve()
    runtime_script = Path(sys.argv[0]).resolve()
    if expected_script != runtime_script or expected[1:] != list(sys.argv[1:]):
        raise ValueError("runtime command differs from immutable pure-finetune job spec")
    if spec.get("command_sha256") != _stable_hash(expected):
        raise ValueError("pure-finetune command hash mismatch")

    required = (
        "job_id", "output_dir", "manifest", "base_policy", "regime",
        "training_contract",
    )
    missing = [key for key in required if key not in spec]
    if missing:
        raise ValueError("pure-finetune job is missing {}".format(", ".join(missing)))
    if spec.get("training_contract", {}).get("kind") != "pure_finetune_full_policy":
        raise ValueError("pure-finetune job has the wrong training contract kind")
    if _contains_capacity_matching_controls(spec) or any(
        any(flag in value for flag in ("expected-model-params", "model-param-tolerance", "capacity-job-contract"))
        for value in expected
    ):
        raise ValueError("pure-finetune job must not contain parameter-matching controls")
    if spec.get("schema") == PURE_FINETUNE_V2_JOB_SCHEMA:
        _validate_pure_finetune_v2_spec(spec)

    immutable_inputs = (
        ("manifest", spec["manifest"], "path", "sha256"),
        ("base policy model", spec["base_policy"], "model_path", "model_sha256"),
        ("base policy stats", spec["base_policy"], "stats_path", "stats_sha256"),
    )
    for description, record, path_key, hash_key in immutable_inputs:
        if not isinstance(record, dict) or not record.get(path_key) or not record.get(hash_key):
            raise ValueError("pure-finetune job is missing {} identity".format(description))
        if sha256_file(record[path_key]) != record[hash_key]:
            raise ValueError("{} no longer matches its immutable pure-finetune hash".format(description))

    sealed = dict(spec)
    sealed["job_spec_path"] = str(spec_path)
    sealed["job_spec_sha256"] = sha256_file(spec_path)
    return sealed

def _mode_key(path: str) -> str:
    match = _MODE_RE.search(str(path))
    return match.group(0) if match else "mode_unknown"


def _per_mode_counts(files: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(_mode_key(f) for f in files).items()))


def _infer_task_family(stats: dict[str, Any], args: argparse.Namespace | None = None) -> str | None:
    for source in (
        getattr(args, "task_family", None) if args is not None else None,
        getattr(args, "task", None) if args is not None else None,
        stats.get("task_family"),
        stats.get("task"),
        stats.get("pipeline"),
        stats.get("rollout_dir"),
        stats.get("base_model_path"),
    ):
        text = str(source or "").lower()
        if "placewipe" in text or "place_wipe" in text:
            return "placewipe"
        if "threearm" in text or ("wipe" in text and "placewipe" not in text):
            return "wipe"
        if "cube" in text:
            return "cube"
        if "pot" in text:
            return "pot"
    return None


def _infer_method(stats: dict[str, Any], args: argparse.Namespace | None = None) -> str | None:
    text = " ".join(
        str(v or "") for v in (
            getattr(args, "checkpoint_prefix", None) if args is not None else None,
            stats.get("pipeline"),
            stats.get("checkpoint_prefix"),
        )
    ).lower()
    if "coord" in text:
        return "coord"
    if "pure_finetune_full_policy" in text or "pure_full_policy" in text:
        return "pure_finetune"
    if "fromscratch" in text or "from_scratch" in text:
        return "fromscratch"
    if "lora" in text or "oft" in text or "finetune" in text:
        return "finetune"
    return None


def _infer_env_variant(stats: dict[str, Any], args: argparse.Namespace | None = None) -> str | None:
    explicit = (
        getattr(args, "expected_env_variant", None) if args is not None else None
    ) or stats.get("env_variant") or stats.get("expected_env_variant")
    if explicit and explicit != "auto":
        return str(explicit)
    text = " ".join(
        str(v or "") for v in (
            stats.get("rollout_dir"),
            stats.get("base_model_path"),
            stats.get("base_stats_path"),
            getattr(args, "rollout_dir", None) if args is not None else None,
            getattr(args, "checkpoint_dir", None) if args is not None else None,
        )
    ).lower()
    if "glass" in text:
        return "glass"
    if "hard" in text or "smalltable" in text or "spongein" in text:
        return "hard"
    if "placewipe" in text:
        return "standard"
    return None


def _infer_num_arms(stats: dict[str, Any], task_family: str | None, args: argparse.Namespace | None = None) -> int | None:
    for value in (getattr(args, "num_arms", None) if args is not None else None, stats.get("num_arms")):
        if value is not None:
            return int(value)
    if task_family == "wipe":
        return 3
    if task_family in ("placewipe", "cube", "pot"):
        return 2
    return None


def _selected_data(stats: dict[str, Any]) -> dict[str, Any]:
    manifest = stats.get("dataset_manifest")
    if isinstance(manifest, dict) and manifest:
        twoarm_train = [str(x) for x in manifest.get("twoarm_train_files", [])]
        singlearm_train = [str(x) for x in manifest.get("singlearm_train_files", [])]
        train_files = [str(x) for x in manifest.get("train_files", [])]
        if not train_files:
            train_files = twoarm_train + singlearm_train
        val_files = [
            *[str(x) for x in manifest.get("validation_files", [])],
            *[str(x) for x in manifest.get("val_files", [])],
            *[str(x) for x in manifest.get("twoarm_validation_files", [])],
            *[str(x) for x in manifest.get("singlearm_validation_files", [])],
        ]
        source = "dataset_manifest"
    else:
        twoarm_train = [str(x) for x in stats.get("twoarm_selected_rollout_files", [])]
        singlearm_train = [str(x) for x in stats.get("singlearm_selected_rollout_files", [])]
        train_files = [str(x) for x in stats.get("selected_rollout_files", [])]
        if not train_files:
            train_files = twoarm_train + singlearm_train
        val_files = [str(x) for x in stats.get("validation_rollout_files", [])]
        source = "stats_fields"

    rollout_dir = stats.get("rollout_dir")
    if rollout_dir and train_files and all(not os.path.dirname(f) for f in train_files):
        root = os.path.abspath(str(rollout_dir))
        train_files = [os.path.join(root, f) for f in train_files]
        val_files = [os.path.join(root, f) for f in val_files]
        twoarm_train = [os.path.join(root, f) for f in twoarm_train]
        singlearm_train = [os.path.join(root, f) for f in singlearm_train]

    fraction = stats.get("singlearm_data_fraction")
    data_policy = stats.get("data_policy")
    if data_policy is None and isinstance(manifest, dict):
        data_policy = manifest.get("data_policy")
        if (
            data_policy is None
            and manifest.get("schema") == "threearm_realign_v2_exact_manifest.v1"
        ):
            data_policy = "sa_matched_density"
    if data_policy is not None:
        data_policy = str(data_policy)
    elif singlearm_train:
        if fraction is not None and float(fraction) < 1.0:
            data_policy = "sa_fraction"
        elif len(twoarm_train) > 0 and len(singlearm_train) == int(round(1.5 * len(twoarm_train))):
            data_policy = "sa_matched_density"
        else:
            data_policy = "sa_full"
    elif fraction == 0.0:
        data_policy = "twoarm_only"

    if fraction is None and singlearm_train and data_policy in {"sa_full", "sa_matched_density"}:
        fraction = 1.0

    return {
        "source": source,
        "rollout_roots": sorted({os.path.dirname(f) for f in train_files if os.path.dirname(f)}),
        "selected_train_files": train_files,
        "selected_val_files": sorted(set(val_files)),
        "per_mode_counts": _per_mode_counts(train_files),
        "validation_per_mode_counts": _per_mode_counts(val_files),
        "twoarm_train_files": twoarm_train,
        "singlearm_train_files": singlearm_train,
        "twoarm_train_count": len(twoarm_train) if twoarm_train else None,
        "singlearm_train_count": len(singlearm_train) if singlearm_train else None,
        "singlearm_data_fraction": None if fraction is None else float(fraction),
        "singlearm_data_policy": data_policy,
        "data_policy": data_policy or ("twoarm_only" if train_files else None),
        "loaded_rollout_count": len(train_files),
        "loaded_sample_count": stats.get("dataset_size"),
    }


def _sample_file_hashes(files: list[str], *, limit: int = 32) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in files[:limit]:
        digest = sha256_file(path, max_bytes=1024 * 1024)
        if digest:
            hashes[path] = digest
    return hashes


def _expected_eval_policy(
    *,
    task_family: str | None,
    method: str | None,
    env_variant: str | None,
    data: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Return the evaluation protocol that strict placewipe runs must use."""
    modes = sorted({int(m.group(1)) for f in data["selected_train_files"] for m in [_MODE_RE.search(f)] if m})
    if task_family != "placewipe":
        return {
            "task": task_family, "env_variant": None, "env_class": None,
            "seeds": None, "repeats": None, "modes": modes,
            "max_steps": None, "replan_freq": None, "selector_mode": None,
            "video_policy": None,
            "sponge_scale": getattr(args, "expected_eval_sponge_scale", None),
        }

    env_classes = {
        "hard": "TwoArmPlaceWipeHard",
        "glass": "TwoArmGlassEnv",
        "standard": "TwoArmPlaceWipe",
    }
    return {
        "task": "placewipe",
        "env_variant": env_variant,
        "env_class": env_classes.get(env_variant),
        # Official headline placewipe evaluation uses this fixed 50-seed batch.
        "seeds": HEADLINE_PLACEWIPE_SEEDS,
        "repeats": int(getattr(args, "expected_eval_repeats", 1)),
        "modes": modes or [0, 1, 2, 3],
        "max_steps": int(getattr(args, "expected_eval_max_steps", 1800)),
        "replan_freq": int(getattr(args, "expected_eval_replan_freq", 15)),
        "selector_mode": "coord" if method == "coord" else None,
        "video_policy": "disabled",
        "sponge_scale": None,
    }


def build_training_contract(
    *,
    args: argparse.Namespace,
    stats: dict[str, Any],
    checkpoint_path: str,
    stats_path: str,
    producer_script: str,
    repo_root: str | None = None,
    task_family: str | None = None,
) -> dict[str, Any]:
    repo_root = repo_root or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    task_family = task_family or _infer_task_family(stats, args)
    method = _infer_method(stats, args)
    num_arms = _infer_num_arms(stats, task_family, args)
    data = _selected_data(stats)
    env_variant = _infer_env_variant(stats, args)
    checkpoint_path_abs = _abs_or_none(checkpoint_path)
    stats_path_abs = _abs_or_none(stats_path)
    base_model_path = _abs_or_none(stats.get("base_model_path"))
    base_stats_path = _abs_or_none(stats.get("base_stats_path"))

    argv = list(sys.argv)
    payload: dict[str, Any] = {
        "contract": {
            "schema_version": SCHEMA_VERSION,
            "created_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "producer_script": producer_script,
            "argv": argv,
            "cwd": os.getcwd(),
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "git": git_summary(repo_root),
        },
        "artifacts": {
            "checkpoint_path": checkpoint_path_abs,
            "stats_path": stats_path_abs,
            "config_path": None,
            "checkpoint_sha256": sha256_file(checkpoint_path_abs),
            "stats_sha256": sha256_file(stats_path_abs),
            "base_model_path": base_model_path,
            "base_model_sha256": sha256_file(base_model_path),
            "base_stats_path": base_stats_path,
            "base_stats_sha256": sha256_file(base_stats_path),
            "launcher_script": os.environ.get("LAUNCHER_SCRIPT"),
        },
        "model": {
            "method": method,
            "task_family": task_family,
            "num_arms": num_arms,
            "backbone": stats.get("backbone"),
            "cameras": stats.get("camera_views") or (
                ["shoulder"] if stats.get("num_cameras") == 1 else ["eye_in_hand", "shoulder"]
            ),
            "horizon": stats.get("horizon"),
            "frame_offsets": list(stats.get("frame_offsets", [0])),
            "architecture": {
                key: stats.get(key)
                for key in (
                    "d_model", "n_heads", "depth", "dim_feedforward",
                    "base_d_model", "base_n_heads", "base_depth", "base_dim_feedforward",
                    "head_d_model", "head_n_heads", "head_depth", "head_dim_feedforward",
                    "rank", "alpha", "lora_scope",
                )
                if key in stats
            },
            "decoder": {
                key: stats.get(key)
                for key in (
                    "decoder_execution", "decoder_conditioning", "use_side_net",
                    "side_net_input_size", "side_net_fusion", "side_net_tokens_per_camera",
                    "d_base_drop_prob",
                )
                if key in stats
            },
            "base_flavor": stats.get("base_flavor"),
            "parameters": {
                "total": stats.get("n_params"),
                "trainable": stats.get("n_params_trainable"),
            },
            "ema": {
                "ema_decay": stats.get("ema_decay"),
                "training_metric_ema_decay": stats.get("training_metric_ema_decay"),
            },
        },
        "training": {
            "training_seed": stats.get("training_seed") if stats.get("training_seed") is not None else getattr(args, "training_seed", None),
            "max_steps": stats.get("max_train_steps") or getattr(args, "max_train_steps", None),
            "epochs": stats.get("epochs") or getattr(args, "epochs", None),
            "checkpoint_every_steps": stats.get("checkpoint_every_steps") or getattr(args, "checkpoint_every_steps", None),
            "optimizer": {"lr": stats.get("lr") or getattr(args, "lr", None)},
            "batch_size": getattr(args, "batch_size", None),
            "augmentation": stats.get("augment"),
            "data_subset_seed": stats.get("data_subset_seed") or getattr(args, "data_subset_seed", None),
            "train_val_split": {
                "val_max_rollout_files": stats.get("val_max_rollout_files"),
                "val_batches": stats.get("val_batches"),
                "val_subset_seed": stats.get("val_subset_seed"),
            },
            "workers": {
                "num_workers": getattr(args, "num_workers", None),
                "prefetch_factor": getattr(args, "prefetch_factor", None),
            },
        },
        "data": data,
        "expected_eval": _expected_eval_policy(
            task_family=task_family, method=method, env_variant=env_variant,
            data=data, args=args,
        ),
        "task_extension": {
            "placewipe": {
                "env_variant": env_variant,
                "mode_mapping": {"0": "dest0_sponge0", "1": "dest0_sponge1", "2": "dest1_sponge0", "3": "dest1_sponge1"},
                "success_threshold_version": "utils.eval_metrics.PlaceWipeMetricsCollectorMixin.v1",
            } if task_family == "placewipe" else None,
            "three_arm_wipe": {
                "num_arms": num_arms,
                "mode_mapping": {"0": "tray0_sponge0", "1": "tray0_sponge1", "2": "tray1_sponge0", "3": "tray1_sponge1"},
                "sponge_scale": stats.get("sponge_scale", 1.0),
                "env_class": "ThreeArmWipe",
                "player_class": "ThreeArmWipeMPCPlayer",
                "success_threshold_version": "utils.eval_metrics.ThreeArmWipeMetricsCollectorMixin.v1",
            } if task_family == "wipe" else None,
        },
        "links": {
            "base_contract_id": stats.get("base_contract_id"),
            "training_stats_hash": sha256_file(stats_path_abs),
            "checkpoint_hash": sha256_file(checkpoint_path_abs),
            "wandb_run_id": os.environ.get("WANDB_RUN_ID") or getattr(args, "wandb_run_id", None),
        },
        "capacity_ablation": stats.get("capacity_ablation"),
        "inferred_fields": [],
    }
    if data["selected_train_files"]:
        payload["data"]["selected_file_hashes"] = _sample_file_hashes(data["selected_train_files"])
    if stats.get("pipeline") == "pure_finetune_full_policy":
        manifest = stats.get("dataset_manifest")
        if not isinstance(manifest, dict):
            manifest = {}
        payload["pure_finetune_full_policy"] = {
            "kind": "pure_finetune_full_policy",
            "architecture_treatment": stats.get("architecture_treatment"),
            "added_modules": stats.get("added_modules"),
            "frozen_modules_after_initialization": stats.get("frozen_modules_after_initialization"),
            "all_original_policy_parameters_trainable": stats.get("policy_parameters", {}).get("all_original_F_parameters_trainable"),
            "parameter_counts": stats.get("policy_parameters"),
            "optimizer": stats.get("optimizer"),
            "augmentation_treatment": stats.get("augmentation_treatment"),
            "data_treatment": stats.get("data_treatment"),
            "domain_batching": stats.get("domain_batching"),
            "rollout_counts": stats.get("rollout_counts"),
            "manifest": {
                key: manifest.get(key)
                for key in ("source_manifest_path", "source_manifest_sha256", "schema", "geometry_profile", "per_mode_demo_budget")
                if key in manifest
            },
            "immutable_job": stats.get("immutable_job"),
            "initial_provenance_hashes": stats.get("initial_provenance_hashes"),
            "checkpoint_provenance_hashes": stats.get("checkpoint_provenance_hashes"),
            "final_provenance_hashes": stats.get("final_provenance_hashes"),
        }

    return finalize_contract(payload)


def write_training_contract(
    *,
    args: argparse.Namespace,
    stats: dict[str, Any],
    checkpoint_path: str,
    stats_path: str,
    producer_script: str,
    repo_root: str | None = None,
    task_family: str | None = None,
) -> str:
    contract = build_training_contract(
        args=args,
        stats=stats,
        checkpoint_path=checkpoint_path,
        stats_path=stats_path,
        producer_script=producer_script,
        repo_root=repo_root,
        task_family=task_family,
    )
    out_path = contract_path_for_checkpoint(checkpoint_path)
    with out_path.open("w") as f:
        json.dump(contract, f, indent=2, sort_keys=True)
    return str(out_path)


def load_contract(path: str | os.PathLike[str]) -> dict[str, Any]:
    with Path(path).open() as f:
        return json.load(f)


def discover_contract(
    *,
    checkpoint_path: str | None = None,
    stats_path: str | None = None,
) -> tuple[str | None, dict[str, Any] | None]:
    candidates: list[Path] = []
    # A checkpoint-specific sidecar takes precedence. The directory-level
    # filename remains a read-only fallback for contracts written before v1.
    if checkpoint_path:
        candidates.append(contract_path_for_checkpoint(checkpoint_path))
    for raw in (checkpoint_path, stats_path):
        if not raw:
            continue
        p = Path(raw)
        parent = p if p.is_dir() else p.parent
        candidates.append(parent / CONTRACT_FILENAME)
    seen: set[Path] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if path.is_file():
            return str(path), load_contract(path)
    if stats_path:
        normalized_stats_path = _normalized_path(stats_path)
        for path in Path(stats_path).resolve().parent.glob(f"*{CONTRACT_SUFFIX}"):
            contract = load_contract(path)
            if _same_path(contract.get("artifacts", {}).get("stats_path"), normalized_stats_path):
                return str(path), contract
    return None, None


def _normalized_path(path: str | None) -> str | None:
    return os.path.abspath(os.path.expanduser(path)) if path else None


def _same_path(expected: str | None, actual: str | None) -> bool:
    if not expected or not actual:
        return True
    return _normalized_path(expected) == _normalized_path(actual)


def _same_stats_artifact(contract: dict[str, Any], actual_path: str | None) -> bool:
    """Accept a relocated stats file only when its sealed bytes are identical.

    Checkpoint-local copies are preferable to mutable repository-global stats
    paths. A pre-existing contract may name the original shared path, so exact
    pathname equality alone would reject an evaluation that used an
    independently stored copy of the same statistics. This exception is
    deliberately limited to adapter statistics and requires the training
    contract's SHA-256.
    """
    expected_hash = contract.get("artifacts", {}).get("stats_sha256")
    return bool(expected_hash and actual_path and sha256_file(actual_path) == expected_hash)


def contract_integrity_mismatches(
    contract: dict[str, Any], eval_context: dict[str, Any]
) -> list[str]:
    """Return contract-content or artifact-hash mismatches."""
    mismatches: list[str] = []
    saved_content_hash = contract.get("contract", {}).get("content_sha256")
    if saved_content_hash:
        identity_payload = json.loads(json.dumps(contract, sort_keys=True))
        identity_payload.get("contract", {}).pop("contract_id", None)
        identity_payload.get("contract", {}).pop("content_sha256", None)
        if _stable_hash(identity_payload) != saved_content_hash:
            mismatches.append("contract_content_sha256")

    artifacts = contract.get("artifacts", {})
    artifact_pairs = (
        ("checkpoint", "checkpoint_sha256", "checkpoint_path"),
        ("stats", "stats_sha256", "adapter_stats_path"),
        ("base_model", "base_model_sha256", "base_model_path"),
        ("base_stats", "base_stats_sha256", "base_stats_path"),
    )
    for artifact_name, hash_key, context_path_key in artifact_pairs:
        expected_hash = artifacts.get(hash_key)
        actual_path = eval_context.get(context_path_key)
        if not expected_hash or not actual_path:
            continue
        actual_hash = sha256_file(actual_path)
        if actual_hash is None:
            mismatches.append(f"{artifact_name}_missing")
        elif actual_hash != expected_hash:
            mismatches.append(f"{artifact_name}_sha256")
    return mismatches


def _sorted_ints(values: Any) -> list[int] | None:
    if values is None:
        return None
    try:
        return sorted(int(v) for v in values)
    except Exception:
        return None


def eval_context_from_args(args: argparse.Namespace, *, env_class: str | None, replan_freq: int, modes: list[Any]) -> dict[str, Any]:
    seeds = [int(s.strip()) for s in str(args.seeds).split(",") if s.strip()]
    return {
        "task": args.task,
        "mode": args.mode,
        "env_variant": getattr(args, "env_variant", None),
        "env_class": env_class,
        "seeds": seeds,
        "repeats": int(args.repeats),
        "modes": [int(m) for m in modes if m is not None],
        "max_steps": int(args.max_steps),
        "replan_freq": int(replan_freq),
        "selector_mode": getattr(args, "selector_mode", None),
        "sponge_scale": getattr(args, "sponge_scale", None),
        "video_policy": "disabled" if getattr(args, "no_video", False) else "enabled",
        "checkpoint_path": _abs_or_none(getattr(args, "checkpoint_path", None)),
        "adapter_stats_path": _abs_or_none(getattr(args, "adapter_stats_path", None)),
        "base_model_path": _abs_or_none(getattr(args, "base_model_path", None)),
        "base_stats_path": _abs_or_none(getattr(args, "base_stats_path", None)),
        "frame_offsets": getattr(args, "frame_offsets", None),
    }


def validate_eval_contract(
    contract: dict[str, Any] | None,
    eval_context: dict[str, Any],
    *,
    profile: str | None = None,
) -> list[str]:
    if not contract:
        return ["missing_contract"]

    mismatches: list[str] = []
    model = contract.get("model", {})
    expected_eval = dict(contract.get("expected_eval", {}))
    if profile is not None:
        profile_spec = EVALUATION_PROFILES.get(profile)
        if profile_spec is None:
            raise ValueError(f"Unknown evaluation profile: {profile}")
        for key in profile_spec["ignore"]:
            expected_eval[key] = None
        expected_eval.update(profile_spec["overrides"])
    artifacts = contract.get("artifacts", {})
    task_extension = contract.get("task_extension", {})

    expected_task = expected_eval.get("task") or model.get("task_family")
    if expected_task and expected_task != eval_context.get("task"):
        mismatches.append(f"task:{expected_task}!={eval_context.get('task')}")

    expected_env_variant = expected_eval.get("env_variant")
    if not expected_env_variant and isinstance(task_extension.get("placewipe"), dict):
        expected_env_variant = task_extension["placewipe"].get("env_variant")
    if expected_env_variant and eval_context.get("env_variant") and expected_env_variant != eval_context.get("env_variant"):
        mismatches.append(f"env_variant:{expected_env_variant}!={eval_context.get('env_variant')}")

    expected_env_class = expected_eval.get("env_class")
    if expected_env_class and eval_context.get("env_class") and expected_env_class != eval_context.get("env_class"):
        mismatches.append(f"env_class:{expected_env_class}!={eval_context.get('env_class')}")

    expected_modes = _sorted_ints(expected_eval.get("modes"))
    actual_modes = _sorted_ints(eval_context.get("modes"))
    if expected_modes and actual_modes and expected_modes != actual_modes:
        mismatches.append(f"modes:{expected_modes}!={actual_modes}")

    expected_frame_offsets = model.get("frame_offsets", [0])
    actual_frame_offsets = eval_context.get("frame_offsets")
    if actual_frame_offsets is not None and list(expected_frame_offsets) != list(actual_frame_offsets):
        mismatches.append(
            f"frame_offsets:{list(expected_frame_offsets)}!={list(actual_frame_offsets)}"
        )

    for key in ("seeds",):
        expected = _sorted_ints(expected_eval.get(key))
        actual = _sorted_ints(eval_context.get(key))
        if expected and actual and expected != actual:
            mismatches.append(f"{key}:{expected}!={actual}")

    for key in ("repeats", "max_steps", "replan_freq", "selector_mode", "sponge_scale", "video_policy"):
        expected = expected_eval.get(key)
        actual = eval_context.get(key)
        if expected is not None and actual is not None and expected != actual:
            mismatches.append(f"{key}:{expected}!={actual}")

    if not _same_path(artifacts.get("checkpoint_path"), eval_context.get("checkpoint_path")):
        mismatches.append("checkpoint_path")
    if not _same_path(artifacts.get("stats_path"), eval_context.get("adapter_stats_path")) and not _same_stats_artifact(
        contract, eval_context.get("adapter_stats_path")
    ):
        mismatches.append("adapter_stats_path")
    if not _same_path(artifacts.get("base_model_path"), eval_context.get("base_model_path")):
        mismatches.append("base_model_path")
    if not _same_path(artifacts.get("base_stats_path"), eval_context.get("base_stats_path")):
        mismatches.append("base_stats_path")

    mismatches.extend(contract_integrity_mismatches(contract, eval_context))
    return mismatches


def success_key(row: dict[str, Any]) -> str | None:
    for key in ("task_success", "placewipe_success", "wipe_success", "success", "generator_acceptance_success"):
        if key in row:
            return key
    return None


def success_by_mode(rows: list[dict[str, Any]]) -> dict[str, Any]:
    totals: Counter[Any] = Counter()
    successes: Counter[Any] = Counter()
    for row in rows:
        mode = row.get("mode", "all")
        totals[mode] += 1
        key = success_key(row)
        if key and bool(row.get(key)):
            successes[mode] += 1
    out = {}
    for mode in sorted(totals, key=lambda x: str(x)):
        total = totals[mode]
        success = successes[mode]
        out[str(mode)] = {
            "success": int(success),
            "total": int(total),
            "rate": float(success) / total if total else 0.0,
        }
    return out


def failure_breakdown_by_mode(rows: list[dict[str, Any]], task: str) -> dict[str, Any]:
    if task == "wipe":
        gates = [
            "tray_visited_waiting_pad", "tray_returned", "sponge_home",
            "coverage_ok", "tray_tilt_ok", "handle_drift_ok",
            "motion_quality_ok", "tray_dropped", "sponge_tray_collision",
        ]
    else:
        gates = [
            "cube_visited_dest", "cube_returned", "sponge_home",
            "coverage_ok", "sponge_cube_collision",
        ]
    out: dict[str, Any] = {}
    for row in rows:
        mode = str(row.get("mode", "all"))
        bucket = out.setdefault(mode, {"failures": 0, "gates": Counter(), "near_threshold_return_miss": 0})
        key = success_key(row)
        if key and bool(row.get(key)):
            continue
        bucket["failures"] += 1
        for gate in gates:
            if gate not in row:
                continue
            value = bool(row.get(gate))
            if gate.endswith("_collision") or gate == "tray_dropped":
                if value:
                    bucket["gates"][gate] += 1
            elif not value:
                bucket["gates"][gate] += 1
        if task == "placewipe":
            dist = row.get("cube_to_return_final")
            if (
                isinstance(dist, (int, float))
                and bool(row.get("cube_visited_dest"))
                and bool(row.get("sponge_home"))
                and bool(row.get("coverage_ok"))
                and not bool(row.get("cube_returned"))
                and 0.08 <= float(dist) <= 0.10
            ):
                bucket["near_threshold_return_miss"] += 1
    return {
        mode: {
            "failures": int(value["failures"]),
            "gates": dict(sorted(value["gates"].items())),
            "near_threshold_return_miss": int(value["near_threshold_return_miss"]),
        }
        for mode, value in sorted(out.items())
    }


def compact_contract_summary(contract: dict[str, Any] | None) -> dict[str, Any] | None:
    if not contract:
        return None
    data = contract.get("data", {})
    model = contract.get("model", {})
    return {
        "contract_id": contract.get("contract", {}).get("contract_id"),
        "task_family": model.get("task_family"),
        "method": model.get("method"),
        "num_arms": model.get("num_arms"),
        "backbone": model.get("backbone"),
        "horizon": model.get("horizon"),
        "data_policy": data.get("data_policy") or data.get("singlearm_data_policy"),
        "loaded_rollout_count": data.get("loaded_rollout_count"),
        "twoarm_train_count": data.get("twoarm_train_count"),
        "singlearm_train_count": data.get("singlearm_train_count"),
        "env_variant": contract.get("expected_eval", {}).get("env_variant")
            or (contract.get("task_extension", {}).get("placewipe") or {}).get("env_variant"),
    }
