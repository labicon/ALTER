"""Validation and normalization for exact low-data manifests."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path


EXPECTED_SINGLEARM_MODES = {
    "place_return_neg_y": {"0", "1", "2", "3"},
    "place_return_pos_y": {"0", "1", "2", "3"},
    "wipe": {"0", "1"},
}
PLACEWIPE_SCHEMA = "placewipe_direction_exact_manifest.v1"
# Physical two-arm port. Validated in envs/arm/train/hardware_placewipe_manifest.py;
# named here only so load_exact_manifest can dispatch to it.
HARDWARE_SCHEMA = "placewipe_hardware_exact_manifest.v1"
PLACEWIPE_ALL_EXPERT_SCHEMA = "placewipe_forwardonly_all_expert_exact_manifest.v1"
FORWARDONLY_DISTILLED_SCHEMA = "threearm_forwardonly_distilled_exact_manifest.v1"
TRAY_DRAG_AUXILIARY_SCHEMA = "threearm_forwardonly_tray_drag_auxiliary_manifest.v1"
FORWARDONLY_SINGLEARM_MODES = {
    "place_return_neg_y": {"0", "1"},
    "place_return_pos_y": {"0", "1"},
    "wipe": {"0", "1"},
}
PLACEWIPE_DIRECTIONS = {
    "forward_only": {"place_return": {"0", "1"}, "wipe": {"0", "1"}},
    "two_frame_all_modes": {"place_return": {"0", "1", "2", "3"}, "wipe": {"0", "1"}},
}


def _under(path: str, root: str) -> bool:
    return os.path.commonpath([os.path.abspath(path), os.path.abspath(root)]) == os.path.abspath(root)


def _validate_files(multiarm_files, singlearm_files, *, multiarm_root, singlearm_roots):
    all_files = multiarm_files + singlearm_files
    if len(all_files) != len(set(all_files)):
        raise ValueError("Exact manifest contains duplicate rollout files")
    missing = [item for item in all_files if not os.path.isfile(item)]
    if missing:
        raise FileNotFoundError(f"Exact manifest files are missing: {missing[:5]}")
    if any(not _under(item, multiarm_root) for item in multiarm_files):
        raise ValueError("Exact manifest contains files outside the configured multi-arm root")
    if any(not any(_under(item, root) for root in singlearm_roots) for item in singlearm_files):
        raise ValueError("Single-arm manifest contains files outside configured single-arm roots")


def _normalized(raw, raw_bytes, path, budget, multiarm_files, singlearm_files):
    return {
        "schema": raw["schema"],
        "direction": raw.get("direction"),
        "source_manifest_path": os.path.abspath(path),
        "source_manifest_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "selection_seed": int(raw["selection_seed"]),
        "per_mode_demo_budget": budget,
        "twoarm_per_mode_demo_budget": int(
            raw.get("twoarm_per_mode_demo_budget", budget)
        ),
        "singlearm_per_mode_demo_budget": int(
            raw.get("singlearm_per_mode_demo_budget", budget)
        ),
        "multiarm_train_files": multiarm_files,
        "singlearm_train_files": singlearm_files,
        "multiarm_validation_files": [],
        "singlearm_validation_files": [],
        "source_manifest": raw,
    }


def _load_exact_threearm(raw, raw_bytes, path, *, multiarm_root, singlearm_roots):
    if raw.get("schema") != "threearm_realign_v2_exact_manifest.v1":
        raise ValueError(f"Unsupported exact manifest schema: {raw.get('schema')!r}")
    budget = int(raw["per_mode_demo_budget"])
    if budget not in (5, 10, 15):
        raise ValueError(f"Unexpected per-mode budget: {budget}")

    ta_by_mode = raw["threearm"]["files_by_mode"]
    if set(ta_by_mode) != {"0", "1", "2", "3"}:
        raise ValueError("Three-arm manifest must contain exactly modes 0..3")
    if any(len(files) != budget for files in ta_by_mode.values()):
        raise ValueError("Three-arm per-mode count does not match manifest budget")
    multiarm_files = [item for mode in ("0", "1", "2", "3") for item in ta_by_mode[mode]]

    if set(raw["singlearm"]) != set(EXPECTED_SINGLEARM_MODES):
        raise ValueError("Single-arm source buckets do not match the realign-v2 protocol")
    singlearm_files = []
    for source_name, expected_modes in EXPECTED_SINGLEARM_MODES.items():
        by_mode = raw["singlearm"][source_name]["files_by_mode"]
        if set(by_mode) != expected_modes:
            raise ValueError(f"Unexpected modes for single-arm source {source_name}")
        if any(len(files) != budget for files in by_mode.values()):
            raise ValueError(f"Single-arm per-mode count mismatch for {source_name}")
        singlearm_files.extend(item for mode in sorted(expected_modes) for item in by_mode[mode])

    _validate_files(
        multiarm_files, singlearm_files,
        multiarm_root=multiarm_root, singlearm_roots=singlearm_roots,
    )
    return _normalized(raw, raw_bytes, path, budget, multiarm_files, singlearm_files)


def _load_exact_placewipe(raw, raw_bytes, path, *, multiarm_root, singlearm_roots):
    direction = raw.get("direction")
    if direction not in PLACEWIPE_DIRECTIONS:
        raise ValueError(f"Unsupported placewipe direction: {direction!r}")
    singlearm_budget = int(raw.get("singlearm_per_mode_demo_budget", raw["per_mode_demo_budget"]))
    twoarm_budget = int(raw.get("twoarm_per_mode_demo_budget", 10))
    all_expert = raw.get("schema") == PLACEWIPE_ALL_EXPERT_SCHEMA
    if all_expert and twoarm_budget != singlearm_budget:
        raise ValueError(
            "PlaceWipe all-expert two-arm per-mode budget must match the single-arm budget"
        )
    if not all_expert and twoarm_budget != 10:
        raise ValueError(
            "Placewipe two-arm density-matched manifests require exactly 10 demonstrations/mode, "
            f"got {twoarm_budget}"
        )
    if singlearm_budget not in (5, 10, 15):
        raise ValueError(
            "Placewipe single-arm budget must be one of 5, 10, or 15 demonstrations/mode, "
            f"got {singlearm_budget}"
        )
    ta_by_mode = raw["twoarm"]["files_by_mode"]
    expected_ta_modes = {"0", "1", "2", "3"}
    if set(ta_by_mode) != expected_ta_modes:
        raise ValueError("Placewipe two-arm manifest must contain exactly modes 0..3")
    if any(len(ta_by_mode[mode]) != twoarm_budget for mode in expected_ta_modes):
        raise ValueError("Placewipe two-arm per-mode count does not match the manifest budget")
    multiarm_files = [item for mode in sorted(expected_ta_modes) for item in ta_by_mode[mode]]

    expected_sources = PLACEWIPE_DIRECTIONS[direction]
    if set(raw["singlearm"]) != set(expected_sources):
        raise ValueError(f"Unexpected single-arm sources for direction {direction}")
    singlearm_files = []
    for source_name, expected_modes in expected_sources.items():
        by_mode = raw["singlearm"][source_name]["files_by_mode"]
        if set(by_mode) != expected_modes:
            raise ValueError(f"Unexpected modes for placewipe source {source_name}")
        if any(len(by_mode[mode]) != singlearm_budget for mode in expected_modes):
            raise ValueError(f"Placewipe per-mode count mismatch for {source_name}")
        singlearm_files.extend(item for mode in sorted(expected_modes) for item in by_mode[mode])

    _validate_files(
        multiarm_files, singlearm_files,
        multiarm_root=multiarm_root, singlearm_roots=singlearm_roots,
    )
    return _normalized(
        raw, raw_bytes, path, singlearm_budget, multiarm_files, singlearm_files
    )



def _load_forwardonly_distilled(raw, raw_bytes, path, *, multiarm_root, singlearm_roots):
    """Load the immutable v3-policy / v2-expert forward-only protocol."""
    budget = int(raw["per_mode_demo_budget"])
    if budget not in (5, 10, 15):
        raise ValueError(f"Unexpected forward-only per-mode budget: {budget}")
    if raw.get("direction") != "forward_only":
        raise ValueError("Forward-only distilled manifest must declare direction=forward_only")
    if not raw.get("expert_manifest_sha256") or not raw.get("distilled_pool_provenance"):
        raise ValueError("Forward-only distilled manifest is missing frozen provenance")
    ta_by_mode = raw["threearm"]["files_by_mode"]
    if set(ta_by_mode) != {"0", "1", "2", "3"} or any(len(files) != budget for files in ta_by_mode.values()):
        raise ValueError("Three-arm files must be exactly the requested count for modes 0..3")
    multiarm_files = [item for mode in ("0", "1", "2", "3") for item in ta_by_mode[mode]]
    if set(raw["singlearm"]) != set(FORWARDONLY_SINGLEARM_MODES):
        raise ValueError("Forward-only source buckets must be negative-place, positive-place, and wipe")
    singlearm_files, content_hashes = [], set()
    for source_name, expected_modes in FORWARDONLY_SINGLEARM_MODES.items():
        by_mode = raw["singlearm"][source_name]["files_by_mode"]
        if set(by_mode) != expected_modes:
            raise ValueError(f"Forward-only source {source_name} must contain exactly modes 0 and 1")
        for mode in sorted(expected_modes):
            files = by_mode[mode]
            if len(files) != budget:
                raise ValueError(f"Forward-only per-mode count mismatch for {source_name}/mode{mode}")
            for item in files:
                try:
                    with open(item, "rb") as stream:
                        rollout = pickle.load(stream)
                except Exception as exc:
                    raise ValueError(f"Cannot read distilled rollout {item}: {exc}") from exc
                if rollout.get("task") != source_name or str(rollout.get("mode")) != mode:
                    raise ValueError(f"Invalid distilled task/mode label in {item}")
                if rollout.get("metrics", {}).get("task_success") is not True:
                    raise ValueError(f"Distilled rollout is not successful: {item}")
                digest = hashlib.sha256(Path(item).read_bytes()).hexdigest()
                if digest in content_hashes:
                    raise ValueError(f"Duplicate distilled rollout content: {item}")
                content_hashes.add(digest)
            singlearm_files.extend(files)
    _validate_files(multiarm_files, singlearm_files, multiarm_root=multiarm_root, singlearm_roots=singlearm_roots)
    return _normalized(raw, raw_bytes, path, budget, multiarm_files, singlearm_files)


def _attested_json(attestation, *, label):
    if not isinstance(attestation, dict) or not attestation.get("path") or not attestation.get("sha256"):
        raise ValueError(f"Tray-drag auxiliary manifest is missing {label} provenance")
    attestation_path = Path(attestation["path"])
    try:
        data = attestation_path.read_bytes()
    except OSError as exc:
        raise ValueError(f"Cannot read {label} provenance: {attestation_path}") from exc
    if hashlib.sha256(data).hexdigest() != attestation["sha256"]:
        raise ValueError(f"{label} provenance hash mismatch")
    try:
        return json.loads(data)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid {label} provenance JSON") from exc


def _load_forwardonly_tray_drag_auxiliary(raw, raw_bytes, path, *, multiarm_root, singlearm_roots):
    """Load the immutable forward-only pilot with zero-residual tray-drag data."""
    if raw.get("direction") != "forward_only":
        raise ValueError("Tray-drag auxiliary manifest must declare direction=forward_only")
    if raw.get("training_status") != "pending_user_approval":
        raise ValueError("Tray-drag auxiliary manifest must remain pending user approval")
    budget = int(raw["per_mode_demo_budget"])
    if budget not in (5, 10, 15):
        raise ValueError(f"Unexpected forward-only per-mode budget: {budget}")

    source = raw.get("source_forwardonly_manifest", {})
    source_raw = _attested_json(source, label="forward-only source")
    if source.get("schema") != FORWARDONLY_DISTILLED_SCHEMA or source_raw.get("schema") != FORWARDONLY_DISTILLED_SCHEMA:
        raise ValueError("Tray-drag auxiliary manifest has the wrong forward-only source schema")
    if int(source_raw.get("per_mode_demo_budget", -1)) != budget:
        raise ValueError("Tray-drag auxiliary source budget does not match the pilot budget")

    ta_by_mode = raw["threearm"]["files_by_mode"]
    if set(ta_by_mode) != {"0", "1", "2", "3"} or any(len(files) != budget for files in ta_by_mode.values()):
        raise ValueError("Three-arm files must be exactly the requested count for modes 0..3")
    multiarm_files = [item for mode in ("0", "1", "2", "3") for item in ta_by_mode[mode]]

    forwardonly_files = list(raw["forwardonly_singlearm"]["train_files"])
    expected_forwardonly_count = 6 * budget
    if len(forwardonly_files) != expected_forwardonly_count:
        raise ValueError("Forward-only single-arm count does not match the source budget")

    aux = raw["tray_drag_auxiliary"]
    if aux.get("objective") != "zero_residual":
        raise ValueError("Tray-drag auxiliary data must be marked zero_residual")
    if int(aux.get("accepted_per_mode", -1)) != 100:
        raise ValueError("Tray-drag auxiliary manifest must contain exactly 100 accepted rollouts/mode")
    by_mode = aux.get("files_by_mode", {})
    if set(by_mode) != {"0", "1"} or any(len(by_mode[mode]) != 100 for mode in ("0", "1")):
        raise ValueError("Tray-drag auxiliary files must contain exactly 100 rollouts for modes 0 and 1")
    tray_files = [item for mode in ("0", "1") for item in by_mode[mode]]
    file_hashes = aux.get("file_sha256", {})
    if set(file_hashes) != set(tray_files) or len(set(file_hashes.values())) != len(tray_files):
        raise ValueError("Tray-drag auxiliary file hashes are incomplete or non-unique")
    records = aux.get("records", [])
    if len(records) != 200:
        raise ValueError("Tray-drag auxiliary manifest must contain exactly 200 records")
    record_keys = set()
    for record in records:
        mode = str(record.get("mode"))
        if mode not in {"0", "1"} or record.get("path") not in by_mode[mode]:
            raise ValueError("Tray-drag auxiliary record has an invalid mode/path")
        key = (mode, int(record.get("seed", -1)))
        if key in record_keys:
            raise ValueError("Tray-drag auxiliary records must have unique mode/seed keys")
        record_keys.add(key)
        if record.get("file_sha256") != file_hashes.get(record["path"]) or not record.get("action_sha256"):
            raise ValueError("Tray-drag auxiliary record hash mismatch")
    if {record["path"] for record in records} != set(tray_files):
        raise ValueError("Tray-drag auxiliary records do not match their files")

    collection = _attested_json(aux.get("collection_contract"), label="collection")
    if collection.get("schema") != "singlearm_tray_drag_collection.v1" or not collection.get("immutable_accepted_rollouts"):
        raise ValueError("Tray-drag collection contract is not immutable")
    review = _attested_json(aux.get("review_gate"), label="review")
    if review.get("schema") != "singlearm_tray_drag_review.v1" or not review.get("user_review_required"):
        raise ValueError("Tray-drag review gate is missing")

    counts = raw.get("counts", {})
    expected_counts = {
        "multiarm_train_files": len(multiarm_files),
        "forwardonly_singlearm_train_files": len(forwardonly_files),
        "tray_drag_auxiliary_train_files": len(tray_files),
        "singlearm_train_files_total": len(forwardonly_files) + len(tray_files),
    }
    if {key: counts.get(key) for key in expected_counts} != expected_counts:
        raise ValueError("Tray-drag auxiliary manifest counts are inconsistent")

    singlearm_files = forwardonly_files + tray_files
    _validate_files(multiarm_files, singlearm_files, multiarm_root=multiarm_root, singlearm_roots=singlearm_roots)
    normalized = _normalized(raw, raw_bytes, path, budget, multiarm_files, singlearm_files)
    normalized.update({
        "forwardonly_singlearm_train_files": forwardonly_files,
        "tray_drag_auxiliary_train_files": tray_files,
        "tray_drag_objective": aux["objective"],
    })
    return normalized


def load_exact_manifest(path: str, *, multiarm_root: str, singlearm_roots: list[str]):
    raw_bytes = Path(path).read_bytes()
    raw = json.loads(raw_bytes)
    if raw.get("schema") == HARDWARE_SCHEMA:
        # The hardware contract lives in its own module and enforces its own
        # counts; dispatching here leaves both simulator contracts untouched.
        from envs.arm.train.hardware_placewipe_manifest import (
            normalize_hardware_manifest,
        )

        return normalize_hardware_manifest(
            raw, raw_bytes, path,
            multiarm_root=multiarm_root, singlearm_roots=singlearm_roots,
        )
    if raw.get("schema") in {PLACEWIPE_SCHEMA, PLACEWIPE_ALL_EXPERT_SCHEMA}:
        return _load_exact_placewipe(
            raw, raw_bytes, path,
            multiarm_root=multiarm_root, singlearm_roots=singlearm_roots,
        )
    if raw.get("schema") == FORWARDONLY_DISTILLED_SCHEMA:
        return _load_forwardonly_distilled(
            raw, raw_bytes, path,
            multiarm_root=multiarm_root, singlearm_roots=singlearm_roots,
        )
    if raw.get("schema") == TRAY_DRAG_AUXILIARY_SCHEMA:
        return _load_forwardonly_tray_drag_auxiliary(
            raw, raw_bytes, path,
            multiarm_root=multiarm_root, singlearm_roots=singlearm_roots,
        )
    return _load_exact_threearm(
        raw, raw_bytes, path,
        multiarm_root=multiarm_root, singlearm_roots=singlearm_roots,
    )


def load_exact_threearm_manifest(path: str, *, multiarm_root: str, singlearm_roots: list[str]):
    """Backward-compatible loader that continues to reject non-three-arm schemas."""
    raw_bytes = Path(path).read_bytes()
    raw = json.loads(raw_bytes)
    return _load_exact_threearm(
        raw, raw_bytes, path,
        multiarm_root=multiarm_root, singlearm_roots=singlearm_roots,
    )
