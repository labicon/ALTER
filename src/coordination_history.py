"""Historical timestamp/preprocessing helpers for frozen dataset and audit parity.

Runtime camera history has been retired. These helpers remain because the
retained current-frame trainer uses the original A/B eligibility filter, and
historical data audits reconstruct observations from raw timestamps.
"""

import numpy as np
import torch
from torchvision.transforms import functional as TF


def validate_lookbacks(seconds):
    values = tuple(float(value) for value in seconds)
    if not values or values[-1] != 0 or any(not np.isfinite(value) or value < 0 for value in values):
        raise ValueError("History lookbacks must be finite, nonnegative, and end at zero")
    if any(previous <= following for previous, following in zip(values, values[1:])):
        raise ValueError("History lookbacks must be strictly descending")
    return values


def history_indices(timestamps_ns, anchor, seconds, max_lag_ns=250_000_000):
    lookbacks = validate_lookbacks(seconds)
    timestamps = np.asarray(timestamps_ns, dtype=np.int64)
    if not 0 <= anchor < len(timestamps):
        raise ValueError("History anchor out of bounds")
    targets = timestamps[anchor] - np.rint(np.asarray(lookbacks) * 1e9).astype(np.int64)
    gaps = np.flatnonzero(np.diff(timestamps[:anchor + 1]) > max_lag_ns)
    segment_start = int(gaps[-1] + 1) if len(gaps) else 0
    indices = np.searchsorted(timestamps[:anchor + 1], targets, side="right") - 1
    indices = np.maximum(indices, segment_start)
    lag = targets - timestamps[indices]
    if np.any(lag > max_lag_ns):
        raise ValueError("Camera history contains a gap larger than the allowed lag")
    return indices


def history_tensor(images):
    array = np.stack(images)
    tensor = torch.from_numpy(array).permute(0, 3, 1, 2).float() / 255.0
    return TF.resize(tensor, [128, 128], antialias=True)
