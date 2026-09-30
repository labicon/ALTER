#!/usr/bin/env python3
"""Select the best two 40-episode checkpoints per model deterministically."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


STEP_RE = re.compile(r"step(\d+)\.pt$")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-frame-offsets", default="0")
    return parser.parse_args(argv)


def _latest_result(directory: Path) -> Path:
    candidates = sorted(directory.rglob("eval_*.json"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No evaluation JSON under {directory}")
    return candidates[-1]


def main(argv=None) -> int:
    args = parse_args(argv)
    root = Path(args.scan_root)
    expected_offsets = [int(value) for value in args.expected_frame_offsets.split(",") if value]
    selected = {}
    for model in ("coord", "fs"):
        candidates = []
        for step in range(100000, 900001, 100000):
            result_path = _latest_result(root / f"{model}_step{step}")
            data = json.loads(result_path.read_text())
            checkpoint = str(data.get("checkpoint_path", ""))
            match = STEP_RE.search(checkpoint)
            if match is None or int(match.group(1)) != step:
                raise ValueError(f"Checkpoint/step mismatch in {result_path}")
            if data.get("num_runs") != 40 or data.get("frame_offsets", [0]) != expected_offsets:
                raise ValueError(f"Invalid scan protocol in {result_path}")
            by_mode = data.get("success_by_mode", {})
            if set(by_mode) != {"0", "1", "2", "3"}:
                raise ValueError(f"Missing mode results in {result_path}")
            total_success = int(data["aggregate"]["task_success"]["count"])
            min_mode_success = min(int(by_mode[str(mode)]["success"]) for mode in range(4))
            candidates.append({
                "model": model, "step": step, "checkpoint_path": checkpoint,
                "stats_path": data["adapter_stats_path"], "result_path": str(result_path.resolve()),
                "total_success": total_success, "min_mode_success": min_mode_success,
                "success_by_mode": by_mode,
            })
        candidates.sort(key=lambda item: (-item["total_success"], -item["min_mode_success"], item["step"]))
        selected[model] = candidates[:2]
    payload = {
        "selection_protocol": "total_success_desc,min_mode_success_desc,step_asc",
        "scan_episodes_per_checkpoint": 40,
        "frame_offsets": expected_offsets,
        "selected": selected,
    }
    output = Path(args.output)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if output.exists() and output.read_text() != encoded:
        raise FileExistsError(f"Refusing to replace different selection: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(encoded)
    print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
