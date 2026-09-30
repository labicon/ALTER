"""Frozen DINOv2-Small (ViT-S/14) backbone for image-conditioned diffusion policy.

Provides the same interface as ``ResNet18GroupNorm`` in ``image_dit.py``:
    Input:  (B, 3, 128, 128) images in [0, 1]
    Output: (B, tokens_per_image, feature_dim) spatial feature tokens

DINOv2-Small produces 81 tokens (9x9 patch grid) of dimension 384.
All backbone weights are frozen; only downstream layers are trainable.

Weights are downloaded once via ``timm`` and cached locally at
``checkpoints/pretrained/dinov2_vits14.pth`` (git-ignored). Subsequent
loads skip the network entirely and read from the local cache.
"""

import os

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_CACHE_PATH = os.environ.get("CODIFF_DINO_WEIGHTS", os.path.join(
    os.environ.get("CODIFF_DATA_ROOT", os.path.join(_REPO_ROOT, "release-artifacts")),
    "checkpoints", "pretrained", "dinov2_vits14.pth"))


def _load_dinov2_state_dict():
    """Return DINOv2-Small state dict, downloading only on the first call."""
    if os.path.isfile(_CACHE_PATH):
        print(f"Loading cached DINOv2 weights from {_CACHE_PATH}")
        return torch.load(_CACHE_PATH, map_location="cpu", weights_only=True)

    if os.environ.get("CODIFF_ALLOW_BACKBONE_DOWNLOAD") != "1":
        raise FileNotFoundError(
            f"Missing DINOv2 weights: {_CACHE_PATH}. Set CODIFF_DINO_WEIGHTS to a verified "
            "weight file, or explicitly opt into fetching upstream weights with "
            "CODIFF_ALLOW_BACKBONE_DOWNLOAD=1.")
    print("Downloading DINOv2-Small weights (one-time)...")
    tmp_model = timm.create_model(
        "vit_small_patch14_dinov2", pretrained=True, num_classes=0,
        img_size=126,
    )
    sd = tmp_model.state_dict()

    os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
    torch.save(sd, _CACHE_PATH)
    print(f"Cached DINOv2 weights to {_CACHE_PATH}")
    del tmp_model
    return sd


class DINOv2Backbone(nn.Module):
    """Frozen DINOv2 ViT-S/14 outputting spatial patch tokens.

    Input:  (B, 3, 128, 128) images in [0, 1]
    Output: (B, 81, 384) patch tokens — 9x9 grid, 384-dim per token
    """

    feature_dim: int = 384
    tokens_per_image: int = 81  # 9x9 patches from 126x126 input with patch_size=14

    def __init__(self) -> None:
        super().__init__()
        # Build architecture without downloading weights
        self.model = timm.create_model(
            "vit_small_patch14_dinov2", pretrained=False, num_classes=0,
            img_size=126,  # 126/14 = 9x9 = 81 patch tokens
        )
        # Load weights from local cache (downloads once if missing)
        sd = _load_dinov2_state_dict()
        self.model.load_state_dict(sd, strict=True)

        self.model.requires_grad_(False)
        self.model.eval()

        # ImageNet normalization buffers
        self.register_buffer(
            "img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def train(self, mode: bool = True):
        """No-op: backbone always stays in eval mode."""
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 128, 128) images in [0, 1]
        Returns:
            (B, 81, 384) spatial patch tokens
        """
        # Resize to 126x126 so it is divisible by patch_size=14 → 9x9 = 81 patches
        x = F.interpolate(x, size=(126, 126), mode="bilinear", align_corners=False)
        x = (x - self.img_mean) / self.img_std
        # forward_features returns (B, 1+N_patches, dim) with CLS token at index 0
        tokens = self.model.forward_features(x)
        # Remove CLS token — keep only patch tokens
        return tokens[:, 1:, :]  # (B, 81, 384)
