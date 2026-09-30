"""Shared temporal-image conditioning helpers.

Frame offsets are non-negative lookback distances.  For example ``[15, 0]``
means condition on images from ``[t - 15, t]``.  The order is significant and
is persisted in stats and checkpoints.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping


DEFAULT_FRAME_OFFSETS = (0,)


def normalize_frame_offsets(offsets: Iterable[int] | None) -> tuple[int, ...]:
    """Validate and normalize a temporal lookback specification."""
    if offsets is None:
        return DEFAULT_FRAME_OFFSETS
    normalized = tuple(int(offset) for offset in offsets)
    if not normalized:
        raise ValueError("frame_offsets must contain at least one offset")
    if any(offset < 0 for offset in normalized):
        raise ValueError(f"frame_offsets must be non-negative, got {list(normalized)}")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"frame_offsets must be unique, got {list(normalized)}")
    if normalized[-1] != 0:
        raise ValueError(
            "frame_offsets must end with 0 so the current image is last; "
            f"got {list(normalized)}"
        )
    if any(left <= right for left, right in zip(normalized, normalized[1:])):
        raise ValueError(
            "frame_offsets must be strictly descending lookbacks ending in 0; "
            f"got {list(normalized)}"
        )
    return normalized


def frame_offsets_from_stats(stats: Mapping | None) -> tuple[int, ...]:
    """Read frame offsets from stats, treating legacy stats as single-frame."""
    if not stats:
        return DEFAULT_FRAME_OFFSETS
    return normalize_frame_offsets(stats.get("frame_offsets", DEFAULT_FRAME_OFFSETS))


def frame_indices(timestep: int, frame_offsets: Iterable[int]) -> tuple[int, ...]:
    """Return clamped trajectory indices for a timestep.

    Clamping duplicates the earliest available image at episode start.
    """
    timestep = int(timestep)
    if timestep < 0:
        raise ValueError(f"timestep must be non-negative, got {timestep}")
    offsets = normalize_frame_offsets(frame_offsets)
    return tuple(max(0, timestep - offset) for offset in offsets)


def require_checkpoint_frame_offsets(
    checkpoint: Mapping,
    expected_offsets: Iterable[int],
    *,
    artifact: str,
) -> tuple[int, ...]:
    """Reject a checkpoint whose temporal setting differs from the model."""
    expected = normalize_frame_offsets(expected_offsets)
    actual = normalize_frame_offsets(
        checkpoint.get("frame_offsets", DEFAULT_FRAME_OFFSETS)
    )
    if actual != expected:
        raise ValueError(
            f"{artifact} frame offset mismatch: "
            f"checkpoint={list(actual)}, model={list(expected)}"
        )
    return actual
