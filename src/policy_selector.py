"""Lightweight visual selector for routing base vs coordination policies."""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn


class ImagePolicySelector(nn.Module):
    """Binary MLP selector on pooled frozen image-policy encoder tokens.

    The selector predicts whether an observation should use the coordination
    head. A positive logit means "route through base + coordination head";
    a negative logit means "use the underlying base policy".
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.net = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward_from_enc_outputs(self, enc_outputs: list[torch.Tensor]) -> torch.Tensor:
        """Return logits from cached base encoder outputs."""
        if not enc_outputs:
            raise ValueError("enc_outputs must be a non-empty list of tensors")
        pooled = enc_outputs[-1].mean(dim=1)
        return self.net(pooled).squeeze(-1)

    @torch.no_grad()
    def predict_proba_from_images(
        self,
        base_model: Any,
        imgs_eih: torch.Tensor | None,
        imgs_shoulder: torch.Tensor,
    ) -> torch.Tensor:
        """Return P(use coordination head) for image observations."""
        was_training = self.training
        self.eval()
        enc_outputs = base_model.F_ema.forward_encoder(imgs_eih, imgs_shoulder)
        probs = torch.sigmoid(self.forward_from_enc_outputs(enc_outputs))
        if was_training:
            self.train()
        return probs

    def checkpoint(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        state = {
            "selector": self.state_dict(),
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
        }
        if extra:
            state.update(extra)
        return state

    def save(self, path: str, extra: dict[str, Any] | None = None) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.checkpoint(extra=extra), path)


def load_image_policy_selector(path: str, device: torch.device | str = "cpu") -> ImagePolicySelector:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    selector = ImagePolicySelector(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dim=int(checkpoint.get("hidden_dim", 128)),
        dropout=float(checkpoint.get("dropout", 0.1)),
    ).to(device)
    selector.load_state_dict(checkpoint["selector"])
    selector.eval()
    return selector
