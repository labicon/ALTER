"""Light, temporally consistent image augmentation and grasp sampling windows."""

import math

import numpy as np
import torch
from torch.nn import functional as functional


LIGHT_AUGMENTATION = dict(probability=.8, brightness=.1, contrast=.1, saturation=.1,
                          channel_gain=.03, rotation_degrees=2., translation_pixels=2.,
                          scale=.02, blur_probability=.1, blur_sigma=.5,
                          image_size=128, padding_mode="border")


def light_augment(images, generator):
    if images.ndim not in (4, 5) or images.shape[-3:] != (3, 128, 128):
        raise ValueError("Expected BCHW or BTCHW RGB images at128x128")
    if not images.is_floating_point():
        raise ValueError("Expected floating point images in0..1")
    temporal = images.ndim == 5
    sequence = images if temporal else images[:, None]
    batch, frames = sequence.shape[:2]
    random = torch.rand((batch, 12), generator=generator, device=images.device, dtype=images.dtype)
    active = random[:, 0] < LIGHT_AUGMENTATION["probability"]
    angle = (random[:, 1] * 2 - 1) * (2 * math.pi / 180)
    scale = 1 + (random[:, 2] * 2 - 1) * .02
    theta = torch.zeros((batch, 2, 3), device=images.device, dtype=images.dtype)
    theta[:, 0, 0] = torch.cos(angle) / scale
    theta[:, 0, 1] = -torch.sin(angle) / scale
    theta[:, 1, 0] = torch.sin(angle) / scale
    theta[:, 1, 1] = torch.cos(angle) / scale
    theta[:, :, 2] = (random[:, 3:5] * 2 - 1) * (4 / 128)
    flat = sequence.reshape(batch * frames, 3, 128, 128)
    grid = functional.affine_grid(theta.repeat_interleave(frames, dim=0), flat.shape, align_corners=False)
    transformed = functional.grid_sample(flat, grid, mode="bilinear", padding_mode="border", align_corners=False)
    transformed = transformed.reshape_as(sequence)
    grayscale = (transformed * transformed.new_tensor([.2989, .5870, .1140])[None, None, :, None, None]).sum(dim=2, keepdim=True)
    saturation = (1 + (random[:, 5] * 2 - 1) * .1)[:, None, None, None, None]
    transformed = grayscale + saturation * (transformed - grayscale)
    contrast = (1 + (random[:, 6] * 2 - 1) * .1)[:, None, None, None, None]
    average = grayscale.mean(dim=(-2, -1), keepdim=True)
    transformed = average + contrast * (transformed - average)
    brightness = (1 + (random[:, 7] * 2 - 1) * .1)[:, None, None, None, None]
    gains = (1 + (random[:, 8:11] * 2 - 1) * .03)[:, None, :, None, None]
    transformed = (transformed * brightness * gains).clamp(0, 1)
    coordinates = torch.arange(-1, 2, device=images.device, dtype=images.dtype)
    kernel = torch.exp(-coordinates.square() / (2 * .5 ** 2))
    kernel = kernel / kernel.sum()
    kernel = (kernel[:, None] * kernel[None, :])[None, None].expand(3, 1, 3, 3)
    blurred = functional.conv2d(functional.pad(transformed.reshape_as(flat), (1, 1, 1, 1), mode="reflect"), kernel, groups=3).reshape_as(sequence)
    transformed = torch.where((random[:, 11] < .1)[:, None, None, None, None], blurred, transformed)
    result = torch.where(active[:, None, None, None, None], transformed, sequence)
    return result if temporal else result[:, 0]


def grasp_transition_windows(actions, events, horizon=20):
    grasp, release = int(events["grasp"]), int(events["release"])
    if not 0 < grasp < release < len(actions):
        raise ValueError("Invalid grasp/release events")
    candidates = np.arange(grasp + 10, release)
    motion = (actions[candidates, 2] >= actions[grasp, 2] + 15) | (actions[candidates, 1] >= actions[grasp, 1] + 20)
    lifted = candidates[motion & (actions[candidates, 6] < 300)]
    if not len(lifted):
        raise ValueError("No closed-gripper lift/removal after expert grasp")
    lift_end = min(release - horizon + 1, int(lifted[0]) + horizon, len(actions) - horizon + 1)
    windows = dict(approach=(max(0, grasp - horizon), grasp),
                   close=(grasp, grasp + 10), lift=(grasp + 10, lift_end))
    if any(start >= end for start, end in windows.values()):
        raise ValueError("Empty grasp-transition stage")
    return windows
