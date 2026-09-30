"""Strict loader for low-var tied-return combined training manifests."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


SCHEMA = "lowvar_tied_return_combined_training_manifest.v1"
HIGHVAR_V3_SCHEMA = "highvar_tied_return_v3_combined_training_manifest.v1"
LOWVAR_PROFILE = "lowvar_tied_return_v1"
HIGHVAR_V3_PROFILE = "highvar_tied_return_v3"
TASKS = ("place_return_neg_y", "place_return_pos_y", "wipe")


def _under(path: str, root: str) -> bool:
    return os.path.commonpath([os.path.abspath(path), os.path.abspath(root)]) == os.path.abspath(root)


def _attested_json(attestation: dict[str, Any], label: str) -> dict[str, Any]:
    if not isinstance(attestation, dict) or not attestation.get("path") or not attestation.get("sha256"):
        raise ValueError(f"missing {label} source attestation")
    path = Path(attestation["path"])
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != attestation["sha256"]:
        raise ValueError(f"{label} source hash mismatch")
    return json.loads(data)


def _verify_files(multiarm_files: list[str], singlearm_files: list[str], *, multiarm_root: str, singlearm_roots: list[str]) -> None:
    all_files = multiarm_files + singlearm_files
    if len(all_files) != len(set(all_files)):
        raise ValueError("combined manifest contains duplicate rollout files")
    missing = [path for path in all_files if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(f"combined manifest references missing rollouts: {missing[:5]}")
    if any(not _under(path, multiarm_root) for path in multiarm_files):
        raise ValueError("combined manifest has a multi-arm rollout outside the configured root")
    if any(not any(_under(path, root) for root in singlearm_roots) for path in singlearm_files):
        raise ValueError("combined manifest has a single-arm rollout outside configured roots")


def load_lowvar_combined_manifest(path: str, *, multiarm_root: str, singlearm_roots: list[str]) -> dict[str, Any]:
    raw_bytes = Path(path).read_bytes()
    raw = json.loads(raw_bytes)
    schema = raw.get("schema")
    if schema not in (SCHEMA, HIGHVAR_V3_SCHEMA):
        raise ValueError(f"unsupported combined schema: {schema!r}")
    expected_profile = LOWVAR_PROFILE if schema == SCHEMA else HIGHVAR_V3_PROFILE
    if raw.get("geometry_profile") != expected_profile:
        raise ValueError("combined manifest has the wrong geometry profile")
    budget = int(raw.get("per_mode_demo_budget", -1))
    if budget not in (5, 10, 15):
        raise ValueError(f"unsupported combined per-mode budget: {budget}")
    attestations = raw.get("source_attestations")
    if not isinstance(attestations, dict):
        raise ValueError("combined manifest has no source attestations")
    threearm_label = "lowvar_threearm" if schema == SCHEMA else "highvar_threearm"
    for label in (threearm_label, "legacy_singlearm", "tray_drag_singlearm"):
        _attested_json(attestations.get(label, {}), label)

    by_mode = raw.get("threearm", {}).get("files_by_mode", {})
    if set(by_mode) != {"0", "1", "2", "3"} or any(len(by_mode[mode]) != budget for mode in by_mode):
        raise ValueError("combined manifest must contain exactly budget three-arm files in modes 0..3")
    multiarm_files = [str(path) for mode in ("0", "1", "2", "3") for path in by_mode[mode]]

    legacy = raw.get("legacy_singlearm", {}).get("files_by_task_mode", {})
    if set(legacy) != set(TASKS):
        raise ValueError("combined manifest legacy sources do not match the six forward-only modes")
    legacy_files = []
    for task in TASKS:
        task_modes = legacy[task]
        if set(task_modes) != {"0", "1"} or any(len(task_modes[mode]) != budget for mode in task_modes):
            raise ValueError(f"combined manifest has invalid legacy selection for {task}")
        legacy_files.extend(str(path) for mode in ("0", "1") for path in task_modes[mode])

    tray_section = raw.get("tray_drag_singlearm", {})
    tray_included = bool(tray_section.get("included"))
    tray_files: list[str] = []
    if tray_included:
        tray_by_mode = tray_section.get("files_by_mode", {})
        if set(tray_by_mode) != {"0", "1"} or any(len(tray_by_mode[mode]) != budget for mode in tray_by_mode):
            raise ValueError("combined manifest has invalid tray-drag selection")
        tray_files = [str(path) for mode in ("0", "1") for path in tray_by_mode[mode]]
    elif tray_section.get("files_by_mode") not in ({}, None):
        raise ValueError("baseline combined manifest must not carry tray-drag files")

    singlearm_files = legacy_files + tray_files
    counts = raw.get("counts", {})
    expected = {
        "multiarm_train_files": 4 * budget,
        "legacy_singlearm_train_files": 6 * budget,
        "tray_drag_singlearm_train_files": 2 * budget if tray_included else 0,
        "singlearm_train_files": 8 * budget if tray_included else 6 * budget,
        "total_train_files": 12 * budget if tray_included else 10 * budget,
    }
    if {key: counts.get(key) for key in expected} != expected:
        raise ValueError("combined manifest counts are inconsistent")
    if raw.get("training_status") != "pending_user_approval":
        raise ValueError("combined manifest must remain pending user approval before launch")
    _verify_files(multiarm_files, singlearm_files, multiarm_root=multiarm_root, singlearm_roots=singlearm_roots)
    return {
        "schema": schema,
        "geometry_profile": expected_profile,
        "source_manifest_path": os.path.abspath(path),
        "source_manifest_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "selection_seed": 0,
        "per_mode_demo_budget": budget,
        "twoarm_per_mode_demo_budget": budget,
        "singlearm_per_mode_demo_budget": budget,
        "multiarm_train_files": multiarm_files,
        "singlearm_train_files": singlearm_files,
        "multiarm_validation_files": [],
        "singlearm_validation_files": [],
        "legacy_singlearm_train_files": legacy_files,
        "tray_drag_singlearm_train_files": tray_files,
        "source_manifest": raw,
    }
