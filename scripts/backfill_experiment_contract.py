#!/usr/bin/env python3
"""Create audited experiment-contract sidecars for legacy checkpoints."""

from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiment_contract import (
    build_training_contract,
    contract_path_for_checkpoint,
    finalize_contract,
    sha256_file,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--stats-path", required=True)
    parser.add_argument(
        "--manifest-path",
        help="Optional saved JSON manifest. It must exactly match an embedded dataset_manifest.",
    )
    parser.add_argument("--base-model-path")
    parser.add_argument("--base-stats-path")
    parser.add_argument("--task-family", choices=["placewipe", "wipe", "cube", "pot"])
    parser.add_argument("--method", choices=["coord", "fromscratch", "finetune"])
    parser.add_argument("--env-variant", choices=["hard", "glass", "standard"])
    parser.add_argument("--num-arms", type=int)
    parser.add_argument("--training-seed", type=int)
    parser.add_argument("--expected-eval-max-steps", type=int, default=1800)
    parser.add_argument("--expected-eval-replan-freq", type=int, default=15)
    parser.add_argument("--expected-eval-repeats", type=int, default=1)
    parser.add_argument("--expected-eval-sponge-scale", type=float)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing checkpoint-specific sidecar.",
    )
    return parser.parse_args(argv)


def _load_stats(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        stats = pickle.load(handle)
    if not isinstance(stats, dict):
        raise TypeError(f"Stats must contain a dictionary: {path}")
    return copy.deepcopy(stats)


def _load_manifest(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise TypeError(f"Manifest must contain a JSON object: {path}")
    return manifest


def _normalized_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _set_artifact_override(
    stats: dict[str, Any],
    *,
    key: str,
    override: str | None,
    inferred_fields: list[str],
) -> None:
    if override is None:
        return
    override_abs = os.path.abspath(override)
    saved = stats.get(key)
    if saved and os.path.abspath(os.fspath(saved)) != override_abs:
        inferred_fields.append(f"{key}:override_saved_value")
    elif not saved:
        inferred_fields.append(f"{key}:provided_for_backfill")
    stats[key] = override_abs


def _require_hash(path: str | None, label: str) -> str:
    if not path:
        raise ValueError(f"Missing required {label} path")
    digest = sha256_file(path)
    if digest is None:
        raise FileNotFoundError(f"Cannot hash {label}: {path}")
    return digest


def build_backfill_contract(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = Path(args.checkpoint_path).resolve()
    stats_path = Path(args.stats_path).resolve()
    _require_hash(str(checkpoint), "checkpoint")
    _require_hash(str(stats_path), "stats")

    stats = _load_stats(stats_path)
    inferred_fields: list[str] = []
    _set_artifact_override(
        stats,
        key="base_model_path",
        override=args.base_model_path,
        inferred_fields=inferred_fields,
    )
    _set_artifact_override(
        stats,
        key="base_stats_path",
        override=args.base_stats_path,
        inferred_fields=inferred_fields,
    )
    for key in ("base_model_path", "base_stats_path"):
        if stats.get(key):
            _require_hash(os.fspath(stats[key]), key)

    manifest = None
    manifest_path = None
    if args.manifest_path:
        manifest_path = Path(args.manifest_path).resolve()
        manifest = _load_manifest(manifest_path)
        embedded = stats.get("dataset_manifest")
        if embedded is not None and _normalized_json(embedded) != _normalized_json(manifest):
            raise ValueError(
                "--manifest-path differs from the dataset_manifest saved in stats"
            )
        stats["dataset_manifest"] = copy.deepcopy(manifest)
        if embedded is None:
            inferred_fields.append("dataset_manifest:imported_exact_saved_json")

    if args.training_seed is not None:
        saved_seed = stats.get("training_seed")
        if saved_seed is not None and int(saved_seed) != int(args.training_seed):
            inferred_fields.append("training_seed:override_saved_value")
        elif saved_seed is None:
            inferred_fields.append("training_seed:provided_for_backfill")
        stats["training_seed"] = int(args.training_seed)

    contract_args = argparse.Namespace(
        checkpoint_prefix=args.method or checkpoint.stem,
        expected_env_variant=args.env_variant,
        task_family=args.task_family,
        task=args.task_family,
        num_arms=args.num_arms,
        max_train_steps=stats.get("max_train_steps"),
        epochs=stats.get("epochs"),
        checkpoint_every_steps=stats.get("checkpoint_every_steps"),
        lr=stats.get("lr"),
        batch_size=stats.get("batch_size"),
        num_workers=stats.get("num_workers"),
        prefetch_factor=stats.get("prefetch_factor"),
        data_subset_seed=stats.get("data_subset_seed"),
        training_seed=stats.get("training_seed"),
        wandb_run_id=stats.get("wandb_run_id"),
        expected_eval_max_steps=args.expected_eval_max_steps,
        expected_eval_replan_freq=args.expected_eval_replan_freq,
        expected_eval_repeats=args.expected_eval_repeats,
        expected_eval_sponge_scale=args.expected_eval_sponge_scale,
    )
    contract = build_training_contract(
        args=contract_args,
        stats=stats,
        checkpoint_path=str(checkpoint),
        stats_path=str(stats_path),
        producer_script=str(Path(__file__).resolve()),
        repo_root=str(REPO_ROOT),
        task_family=args.task_family,
    )

    if args.method:
        if contract["model"].get("method") != args.method:
            inferred_fields.append("model.method:provided_for_backfill")
        contract["model"]["method"] = args.method
        if args.method == "coord" and contract["expected_eval"].get("selector_mode") is None:
            contract["expected_eval"]["selector_mode"] = "coord"
    if args.num_arms is not None:
        if contract["model"].get("num_arms") != args.num_arms:
            inferred_fields.append("model.num_arms:provided_for_backfill")
        contract["model"]["num_arms"] = int(args.num_arms)

    exact_manifest_source = "stats_fields"
    if manifest is not None:
        exact_manifest_source = "saved_json"
    elif isinstance(stats.get("dataset_manifest"), dict):
        exact_manifest_source = "embedded_dataset_manifest"
        manifest = copy.deepcopy(stats["dataset_manifest"])

    contract["contract"]["provenance"] = "legacy_backfill"
    contract["provenance"] = {
        "source": "legacy_backfill",
        "backfilled_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint_existed_before_contract": True,
        "stats_existed_before_contract": True,
        "exact_manifest_source": exact_manifest_source,
    }
    if manifest is not None:
        contract["data"]["source_manifest"] = manifest
    if manifest_path is not None:
        contract["data"]["source_manifest_path"] = str(manifest_path)
        contract["data"]["source_manifest_sha256"] = _require_hash(
            str(manifest_path), "manifest"
        )
    contract["inferred_fields"] = sorted(set(inferred_fields))
    return finalize_contract(contract)


def write_backfill_contract(
    contract: dict[str, Any],
    *,
    checkpoint_path: str,
    force: bool = False,
) -> Path:
    output = contract_path_for_checkpoint(checkpoint_path)
    if output.exists() and not force:
        raise FileExistsError(
            f"Contract already exists: {output}; pass --force to replace it"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=output.name + ".", suffix=".tmp", dir=output.parent
    )
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(contract, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, output)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return output


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = contract_path_for_checkpoint(args.checkpoint_path)
    if output.exists() and not args.force:
        raise FileExistsError(
            f"Contract already exists: {output}; pass --force to replace it"
        )
    contract = build_backfill_contract(args)
    preview = {
        "dry_run": bool(args.dry_run),
        "would_write": str(output),
        "contract": contract,
    }
    if args.dry_run:
        print(json.dumps(preview, indent=2, sort_keys=True))
        return 0
    written = write_backfill_contract(
        contract,
        checkpoint_path=args.checkpoint_path,
        force=args.force,
    )
    print(json.dumps({"written": str(written), "contract_id": contract["contract"]["contract_id"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
