"""Exact manifest contract for the physical two-arm PlaceWipe port.

The simulator PlaceWipe task is a 2x2 factorial (destination pad x sponge pad)
and its manifest contract in ``exact_threearm_manifest`` hard-requires two-arm
modes 0..3 at ten demonstrations each.  The hardware task is narrower: the
two-arm pickup/cardboard setup has a single route, and each single-arm source
contributes a single forward-only mode.  That cannot be expressed under the
simulator schema, and the simulator contract must not be relaxed to fit it.

This module therefore defines a separate, equally strict hardware schema.  One
per-mode budget governs every mode in the experiment:

* exactly one two-arm mode (``mode0``), holding ``per_mode_demo_budget`` files;
* exactly one mode per single-arm source (``pickup``, ``cardboard_forward``),
  each holding ``per_mode_demo_budget`` files;
* the same file-existence, duplicate, and containment-root checks as the
  simulator loader.

So a budget of N means N two-arm demonstrations and 2N single-arm ones.  Both
domains scale together across the 5/10/15 sweep, exactly as the simulator's
per-mode budget scales its four two-arm modes.  Strictness is preserved; only
the number of modes differs.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from envs.arm.train.exact_threearm_manifest import _validate_files


HARDWARE_SCHEMA = "placewipe_hardware_exact_manifest.v1"

# Hardware runs the forward-only protocol: the cardboard task's backward half is
# not part of the frozen base pool, and the two-arm route is single-mode.
HARDWARE_DIRECTIONS = {"forward_only"}

# Single-arm distillation sources. Each contributes exactly one forward mode.
EXPECTED_SINGLEARM_SOURCES = {"pickup", "cardboard_forward"}

# One budget governs every mode: the single two-arm mode and both single-arm
# modes. Budget N therefore means N two-arm and 2N single-arm demonstrations.
# 5/10/15 is the low-data sweep. 39 is the full Aug 22 2026 two-arm pool
# (39 paired bird/cardboard recordings), so a 39-budget manifest trains on
# every two-arm demo plus a matched 39 from each single-arm source.
DEMO_BUDGETS = (5, 10, 15, 20, 25, 39)
# The builder's default sweep. 39 is opt-in (--budgets 39) so the default
# 5/10/15 manifests are unchanged.
SWEEP_BUDGETS = (5, 10, 15)

# The two-arm task has one route, so every two-arm file is tagged mode0.
TWOARM_MODES = {"0"}
SINGLEARM_MODES = {"0"}


def _require_mode_map(container, *, label, expected_modes):
    """Return files_by_mode for a source after checking its mode key set."""
    if "files_by_mode" not in container:
        raise ValueError(f"{label} is missing 'files_by_mode'")
    by_mode = container["files_by_mode"]
    if set(by_mode) != expected_modes:
        raise ValueError(
            f"{label} must declare exactly modes {sorted(expected_modes)}, "
            f"got {sorted(by_mode)}"
        )
    return by_mode


def load_hardware_manifest(path: str, *, multiarm_root: str, singlearm_roots: list[str]):
    """Validate and normalize a hardware PlaceWipe exact manifest.

    The returned dict matches the shape produced by the simulator loader so the
    shared coordination trainer consumes either without special-casing.
    """
    raw_bytes = Path(path).read_bytes()
    raw = json.loads(raw_bytes)
    return normalize_hardware_manifest(
        raw, raw_bytes, path,
        multiarm_root=multiarm_root, singlearm_roots=singlearm_roots,
    )


def normalize_hardware_manifest(
    raw, raw_bytes, path, *, multiarm_root: str, singlearm_roots: list[str]
):
    if raw.get("schema") != HARDWARE_SCHEMA:
        raise ValueError(f"Unsupported hardware manifest schema: {raw.get('schema')!r}")

    direction = raw.get("direction")
    if direction not in HARDWARE_DIRECTIONS:
        raise ValueError(
            f"Hardware manifests support only {sorted(HARDWARE_DIRECTIONS)}, got {direction!r}"
        )

    budget = int(raw["per_mode_demo_budget"])
    if budget not in DEMO_BUDGETS:
        raise ValueError(
            f"Hardware per-mode budget must be one of {', '.join(map(str, DEMO_BUDGETS))} demonstrations, "
            f"got {budget}"
        )

    # The two-arm task has a single mode, so the budget binds it directly: the
    # multi-arm pool scales with the sweep rather than staying pinned.
    ta_by_mode = _require_mode_map(
        raw["twoarm"], label="Hardware two-arm manifest", expected_modes=TWOARM_MODES
    )
    multiarm_files = list(ta_by_mode["0"])
    if len(multiarm_files) != budget:
        raise ValueError(
            "Hardware two-arm per-mode count mismatch: "
            f"expected {budget}, got {len(multiarm_files)}"
        )

    if set(raw["singlearm"]) != EXPECTED_SINGLEARM_SOURCES:
        raise ValueError(
            f"Hardware single-arm sources must be exactly "
            f"{sorted(EXPECTED_SINGLEARM_SOURCES)}, got {sorted(raw['singlearm'])}"
        )
    # Optional per-source budget overrides (e.g. the distilled cardboard slot
    # carries N forward + N backward = 2N files under a per-mode budget of N).
    # Absent overrides preserve the original rule: every source == budget.
    per_source = {str(k): int(v) for k, v in
                  (raw.get("per_source_demo_budget") or {}).items()}
    unknown = set(per_source) - EXPECTED_SINGLEARM_SOURCES
    if unknown:
        raise ValueError(f"per_source_demo_budget names unknown sources: {sorted(unknown)}")
    singlearm_files = []
    for source_name in sorted(EXPECTED_SINGLEARM_SOURCES):
        by_mode = _require_mode_map(
            raw["singlearm"][source_name],
            label=f"Hardware single-arm source {source_name}",
            expected_modes=SINGLEARM_MODES,
        )
        files = list(by_mode["0"])
        expected = per_source.get(source_name, budget)
        if len(files) != expected:
            raise ValueError(
                f"Hardware per-source count mismatch for {source_name}: "
                f"expected {expected}, got {len(files)}"
            )
        singlearm_files.extend(files)

    _validate_files(
        multiarm_files, singlearm_files,
        multiarm_root=multiarm_root, singlearm_roots=singlearm_roots,
    )

    return {
        "schema": raw["schema"],
        "direction": direction,
        "source_manifest_path": os.path.abspath(path),
        "source_manifest_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "selection_seed": int(raw["selection_seed"]),
        # One budget governs both domains; the per-domain fields the shared
        # trainer reads therefore carry the same value.
        "per_mode_demo_budget": budget,
        "twoarm_per_mode_demo_budget": budget,
        "singlearm_per_mode_demo_budget": budget,
        "twoarm_demo_count": len(multiarm_files),
        "singlearm_demo_count": len(singlearm_files),
        "multiarm_train_files": multiarm_files,
        "singlearm_train_files": singlearm_files,
        "multiarm_validation_files": [],
        "singlearm_validation_files": [],
        "source_manifest": raw,
    }
