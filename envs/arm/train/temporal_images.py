"""Temporal image extraction with augmentation shared across all frames."""

from __future__ import annotations

import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import functional as TF

from src.temporal import frame_indices, normalize_frame_offsets


def _to_chw_float(frame: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(frame).copy()).permute(2, 0, 1).float() / 255.0


def _stack_frames(sequence: np.ndarray, timestep: int, frame_offsets) -> torch.Tensor:
    offsets = normalize_frame_offsets(frame_offsets)
    frames = [_to_chw_float(sequence[index]) for index in frame_indices(timestep, offsets)]
    if len(frames) == 1:
        return frames[0]
    return torch.stack(frames, dim=0)


def _map_frames(images: torch.Tensor, fn) -> torch.Tensor:
    if images.ndim == 3:
        return fn(images)
    return torch.stack([fn(frame) for frame in images], dim=0)


def prepare_temporal_views(
    eih_sequence: np.ndarray,
    shoulder_sequence: np.ndarray,
    timestep: int,
    frame_offsets=(0,),
    *,
    augment: bool,
    output_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract camera histories and apply one shared crop to every image.

    The crop parameters are shared across time and camera views.  At episode
    start, lookbacks before frame zero are clamped to frame zero.
    """
    offsets = normalize_frame_offsets(frame_offsets)
    eih = _stack_frames(eih_sequence, timestep, offsets)
    shoulder = _stack_frames(shoulder_sequence, timestep, offsets)
    if augment:
        if len(offsets) == 1:
            # Preserve the legacy single-frame behavior: camera views receive
            # independently sampled crops. Temporal runs share one crop across
            # every history frame (and camera, when both views are enabled).
            def independent_crop(image):
                top, left, height, width = transforms.RandomResizedCrop.get_params(
                    image,
                    scale=(0.8, 1.0),
                    ratio=(3.0 / 4.0, 4.0 / 3.0),
                )
                return TF.resized_crop(
                    image, top, left, height, width,
                    [output_size, output_size], antialias=True,
                )

            return independent_crop(eih), independent_crop(shoulder)

        reference = eih if eih.ndim == 3 else eih[0]
        top, left, height, width = transforms.RandomResizedCrop.get_params(
            reference,
            scale=(0.8, 1.0),
            ratio=(3.0 / 4.0, 4.0 / 3.0),
        )

        def apply_crop(image):
            return TF.resized_crop(
                image,
                top,
                left,
                height,
                width,
                [output_size, output_size],
                antialias=True,
            )

        return _map_frames(eih, apply_crop), _map_frames(shoulder, apply_crop)

    def apply_resize(image):
        return TF.resize(image, [output_size, output_size], antialias=True)

    return _map_frames(eih, apply_resize), _map_frames(shoulder, apply_resize)
