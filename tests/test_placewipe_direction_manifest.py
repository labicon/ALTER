import json
import pickle

import pytest

from envs.arm.train.exact_threearm_manifest import load_exact_manifest
from scripts.build_placewipe_direction_manifests import build_manifests


def _rollouts(root, modes):
    paths = []
    root.mkdir()
    for mode in modes:
        for seed in range(10):
            path = root / f"demo_seed{seed}_mode{mode}.pkl"
            path.touch()
            paths.append(str(path.resolve()))
    return paths


@pytest.mark.parametrize(
    "direction,expected_singlearm",
    [("forward_only", 40), ("two_frame_all_modes", 60)],
)
def test_build_and_load_direction_manifest(tmp_path, direction, expected_singlearm):
    ta_dir = tmp_path / "twoarm"
    place_dir = tmp_path / "singlearm_place_return"
    wipe_dir = tmp_path / "singlearm_wipe"
    stats = {
        "twoarm_selected_rollout_files": _rollouts(ta_dir, range(4)),
        "singlearm_selected_rollout_files": (
            _rollouts(place_dir, range(4)) + _rollouts(wipe_dir, range(2))
        ),
    }
    stats_path = tmp_path / "stats.pkl"
    stats_path.write_bytes(pickle.dumps(stats))
    raw = build_manifests(str(stats_path), selection_seed=0)[direction]
    manifest_path = tmp_path / f"{direction}.json"
    manifest_path.write_text(json.dumps(raw))
    loaded = load_exact_manifest(
        str(manifest_path), multiarm_root=str(ta_dir),
        singlearm_roots=[str(place_dir), str(wipe_dir)],
    )
    assert len(loaded["multiarm_train_files"]) == 40
    assert len(loaded["singlearm_train_files"]) == expected_singlearm
    assert loaded["direction"] == direction


@pytest.mark.parametrize("budget", [5, 15])
def test_placewipe_loader_allows_fixed_twoarm_with_variable_singlearm_budget(tmp_path, budget):
    ta_dir = tmp_path / "twoarm"
    place_dir = tmp_path / "singlearm_place_return"
    wipe_dir = tmp_path / "singlearm_wipe"
    ta_files = _rollouts(ta_dir, range(4))
    place_files = _rollouts(place_dir, range(2))
    wipe_files = _rollouts(wipe_dir, range(2))
    for root, files in ((place_dir, place_files), (wipe_dir, wipe_files)):
        for mode in (0, 1):
            for seed in range(10, 15):
                path = root / f"demo_seed{seed}_mode{mode}.pkl"
                path.touch()
                files.append(str(path.resolve()))
    raw = {
        "schema": "placewipe_direction_exact_manifest.v1",
        "direction": "forward_only",
        "selection_seed": 20260728,
        "per_mode_demo_budget": budget,
        "twoarm_per_mode_demo_budget": 10,
        "singlearm_per_mode_demo_budget": budget,
        "twoarm": {"files_by_mode": {
            str(mode): [path for path in ta_files if path.endswith(f"mode{mode}.pkl")]
            for mode in range(4)
        }},
        "singlearm": {
            "place_return": {"files_by_mode": {
                str(mode): [path for path in place_files if path.endswith(f"mode{mode}.pkl")][:budget]
                for mode in (0, 1)
            }},
            "wipe": {"files_by_mode": {
                str(mode): [path for path in wipe_files if path.endswith(f"mode{mode}.pkl")][:budget]
                for mode in (0, 1)
            }},
        },
    }
    manifest_path = tmp_path / f"placewipe_{budget}.json"
    manifest_path.write_text(json.dumps(raw))
    loaded = load_exact_manifest(
        str(manifest_path), multiarm_root=str(ta_dir),
        singlearm_roots=[str(place_dir), str(wipe_dir)],
    )
    assert len(loaded["multiarm_train_files"]) == 40
    assert len(loaded["singlearm_train_files"]) == 4 * budget
    assert loaded["twoarm_per_mode_demo_budget"] == 10
    assert loaded["singlearm_per_mode_demo_budget"] == budget
