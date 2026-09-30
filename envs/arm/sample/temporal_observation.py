"""Control-step image history for temporal policy evaluation."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping

import numpy as np
import torch

from src.temporal import normalize_frame_offsets


class TemporalObservationBuffer:
    """Keep exact control-step camera history and form model-ready batches."""

    def __init__(self, frame_offsets=(0,)):
        self.frame_offsets = normalize_frame_offsets(frame_offsets)
        self._maxlen = max(self.frame_offsets) + 1
        self._histories: dict[str, deque[np.ndarray]] = {}

    def reset(self, frames: Mapping[str, np.ndarray]) -> None:
        self._histories = {
            str(key): deque([np.asarray(frame, dtype=np.uint8).copy()], maxlen=self._maxlen)
            for key, frame in frames.items()
        }

    def append(self, frames: Mapping[str, np.ndarray]) -> None:
        if set(frames) != set(self._histories):
            raise ValueError(
                "Temporal camera keys changed within an episode: "
                f"expected={sorted(self._histories)}, got={sorted(frames)}"
            )
        for key, frame in frames.items():
            self._histories[key].append(np.asarray(frame, dtype=np.uint8).copy())

    def batch(self, key: str, device) -> torch.Tensor:
        if key not in self._histories:
            raise KeyError(f"Temporal camera history is missing {key!r}")
        history = list(self._histories[key])
        latest = len(history) - 1
        frames = [history[max(0, latest - offset)] for offset in self.frame_offsets]
        tensor = torch.stack(
            [torch.from_numpy(frame.copy()).permute(2, 0, 1).float() / 255.0 for frame in frames],
            dim=0,
        ).to(device)
        if len(self.frame_offsets) == 1:
            return tensor
        return tensor.unsqueeze(0)
