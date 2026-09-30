"""Loader for the full eight-mode hard-wide frozen single-arm base corpus."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


SCHEMA_V1 = "singlearm_hardwide_traydrag_frozen_base_manifest.v1"
SCHEMA_V2 = "singlearm_frozen_base_manifest.v2"
LEGACY_BUCKETS = {
    "singlearm/place_return_neg_y/mode0", "singlearm/place_return_neg_y/mode1",
    "singlearm/place_return_pos_y/mode0", "singlearm/place_return_pos_y/mode1",
    "singlearm/wipe/mode0", "singlearm/wipe/mode1",
}


def _under(path: str, root: str) -> bool:
    return os.path.commonpath([os.path.abspath(path), os.path.abspath(root)]) == os.path.abspath(root)


def _attest(attestation: dict[str, Any], label: str) -> None:
    if not isinstance(attestation, dict) or not attestation.get("path") or not attestation.get("sha256"):
        raise ValueError(f"frozen-base manifest lacks {label} attestation")
    content = Path(attestation["path"]).read_bytes()
    if hashlib.sha256(content).hexdigest() != attestation["sha256"]:
        raise ValueError(f"frozen-base {label} attestation hash mismatch")


def load_frozen_base_manifest(path: str, *, rollout_dirs: list[str]) -> dict[str, Any]:
    raw_bytes = Path(path).read_bytes()
    raw = json.loads(raw_bytes)
    schema = raw.get("schema")
    if schema == SCHEMA_V1:
        if raw.get("geometry_profile") != "threearm_hard_wide_v1":
            raise ValueError("unsupported v1 frozen-base geometry profile")
    elif schema == SCHEMA_V2:
        if (
            raw.get("legacy_geometry_profile") != "threearm_hard_wide_v1"
            or raw.get("tray_drag_profile") != "singlearm_tray_drag_learnable_v2"
        ):
            raise ValueError("unsupported v2 frozen-base geometry profile")
    else:
        raise ValueError("unsupported frozen-base manifest schema")
    if raw.get("training_status") != "pending_user_approval":
        raise ValueError("frozen-base manifest is not authorized for this launch")
    legacy = raw.get("legacy_singlearm", {})
    _attest(legacy.get("source_manifest", {}), "legacy source")
    buckets = legacy.get("buckets", {})
    if set(buckets) != LEGACY_BUCKETS or any(len(rows) != 100 for rows in buckets.values()):
        raise ValueError("frozen-base legacy corpus must contain six buckets of 100 files")
    legacy_files = []
    for bucket in sorted(LEGACY_BUCKETS):
        for row in buckets[bucket]:
            if not row.get("sha256"):
                raise ValueError(f"frozen-base legacy record lacks a hash: {bucket}")
            legacy_files.append(str(row["path"]))

    tray = raw.get("tray_drag_singlearm", {})
    _attest(tray.get("source_collection_contract", {}), "tray collection")
    _attest(tray.get("source_attempt_ledger", {}), "tray attempt ledger")
    by_mode = tray.get("files_by_mode", {})
    if set(by_mode) != {"0", "1"} or any(len(by_mode[mode]) != 100 for mode in by_mode):
        raise ValueError("frozen-base tray corpus must contain both 100-file modes")
    records = tray.get("records", [])
    if len(records) != 200 or len({(int(row["seed"]), str(row["path"])) for row in records}) != 200:
        raise ValueError("frozen-base tray records must have 200 unique seed/path entries")
    tray_files = [str(item) for mode in ("0", "1") for item in by_mode[mode]]
    if {str(row["path"]) for row in records} != set(tray_files):
        raise ValueError("frozen-base tray records do not match files_by_mode")
    if any(not row.get("sha256") or not row.get("action_sha256") for row in records):
        raise ValueError("frozen-base tray records lack a rollout/action hash")

    train_files = legacy_files + tray_files
    counts = raw.get("counts", {})
    if counts != {"legacy_singlearm": 600, "tray_drag_singlearm": 200, "total_singlearm": 800}:
        raise ValueError("frozen-base manifest counts are inconsistent")
    if len(train_files) != len(set(train_files)):
        raise ValueError("frozen-base manifest has duplicate rollout paths")
    missing = [item for item in train_files if not os.path.isfile(item)]
    if missing:
        raise FileNotFoundError(f"frozen-base manifest references missing rollouts: {missing[:5]}")
    if any(not any(_under(item, root) for root in rollout_dirs) for item in train_files):
        raise ValueError("frozen-base manifest includes a rollout outside --rollout-dirs")
    return {
        "source_manifest_path": os.path.abspath(path),
        "source_manifest_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "train_files": train_files,
        "legacy_singlearm_train_files": legacy_files,
        "tray_drag_singlearm_train_files": tray_files,
        "source_manifest": raw,
    }
