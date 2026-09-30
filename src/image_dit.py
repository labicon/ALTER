"""Encoder-decoder DiT for end-to-end image-conditioned diffusion policy.

Architecture:
    - Shared ResNet18 (GroupNorm) encodes both camera views into spatial tokens
    - Self-attention encoder fuses image tokens across views
    - DiT decoder with adaLN modulation denoises action trajectories
    - CFG via learned null token that replaces encoder outputs during training

Target: ~17M params (11M ResNet18 + 6M transformer).
"""

import math
import torch
import torch.nn as nn
import torchvision.models as tvm

from src.temporal import normalize_frame_offsets


# ---------------------------------------------------------------------------
# ResNet18 with GroupNorm (matches dit-policy's diffusion_policy convention)
# ---------------------------------------------------------------------------

def _make_gn(num_channels, num_groups=None) -> nn.GroupNorm:
    """GroupNorm with num_channels // 16 groups (dit-policy convention)."""
    if num_groups is None:
        num_groups = max(1, num_channels // 16)
    return nn.GroupNorm(num_groups, num_channels)


class ResNet18GroupNorm(nn.Module):
    """ResNet18 with GroupNorm, outputting spatial feature tokens.

    Input:  (B, 3, 128, 128)
    Output: (B, 16, 512)   — 4x4 spatial grid flattened, 512-dim per token
    """

    def __init__(self) -> None:
        super().__init__()
        base = tvm.resnet18(weights=None)

        # Replace all BatchNorm with GroupNorm
        base.bn1 = _make_gn(64)
        for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
            layer = getattr(base, layer_name)
            for block in layer:
                block.bn1 = _make_gn(block.conv1.out_channels)
                block.bn2 = _make_gn(block.conv2.out_channels)
                if block.downsample is not None:
                    # downsample[1] is the norm
                    out_ch = block.downsample[0].out_channels
                    block.downsample[1] = _make_gn(out_ch)

        # Remove global avg-pool and fc (we want spatial features)
        base.avgpool = nn.Identity()
        base.fc = nn.Identity()

        self.backbone = base

        # ImageNet normalization buffers
        self.register_buffer(
            "img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 128, 128) images in [0, 1]
        Returns:
            (B, 16, 512) spatial feature tokens
        """
        x = (x - self.img_mean) / self.img_std

        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        # x: (B, 512, 4, 4) for 128x128 input

        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).permute(0, 2, 1)  # (B, 16, 512)
        return x


# ---------------------------------------------------------------------------
# adaLN modulation layers (ported from dit-policy)
# ---------------------------------------------------------------------------

class ShiftScaleMod(nn.Module):
    """Affine modulation: x * scale(c) + shift(c)."""

    def __init__(self, dim: int):
        super().__init__()
        self.act = nn.SiLU()
        self.scale = nn.Linear(dim, dim)
        self.shift = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # x: (B, S, D),  c: (B, D)
        c = self.act(c)
        return x * self.scale(c).unsqueeze(1) + self.shift(c).unsqueeze(1)

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.scale.weight)
        nn.init.xavier_uniform_(self.shift.weight)
        nn.init.zeros_(self.scale.bias)
        nn.init.zeros_(self.shift.bias)


class ZeroScaleMod(nn.Module):
    """Gate modulation (zero-initialized): x * scale(c)."""

    def __init__(self, dim: int):
        super().__init__()
        self.act = nn.SiLU()
        self.scale = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        c = self.act(c)
        return x * self.scale(c).unsqueeze(1)

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.scale.bias)


# ---------------------------------------------------------------------------
# Encoder block (pre-norm self-attention + FFN)
# ---------------------------------------------------------------------------

class SelfAttentionEncoderBlock(nn.Module):
    """Pre-norm self-attention encoder block.

    Pattern from dit-policy's _SelfAttnEncoder, adapted for batch_first.
    """

    def __init__(self, d_model: int = 256, n_heads: int = 4,
                 dim_feedforward: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.GELU(approximate="tanh")

    def forward(self, src: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src: (B, S, D)
            pos: (B, S, D) or (1, S, D) positional encoding
        Returns:
            (B, S, D)
        """
        q = k = src + pos
        src2, _ = self.self_attn(q, k, value=src, need_weights=False)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src


# ---------------------------------------------------------------------------
# DiT decoder block (self-attention + FFN with adaLN modulation)
# ---------------------------------------------------------------------------

class DiTDecoderBlock(nn.Module):
    """DiT decoder block with adaLN modulation.

    Conditioning: cond = mean(encoder_output, dim=seq) + time_embedding.
    Pattern from dit-policy's _DiTDecoder.
    """

    def __init__(self, d_model: int = 256, n_heads: int = 4,
                 dim_feedforward: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.GELU(approximate="tanh")

        # adaLN modulation layers
        self.attn_mod1 = ShiftScaleMod(d_model)
        self.attn_mod2 = ZeroScaleMod(d_model)
        self.mlp_mod1 = ShiftScaleMod(d_model)
        self.mlp_mod2 = ZeroScaleMod(d_model)

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    (B, H, D) — noisy action tokens
            t:    (B, D)    — time embedding
            cond: (B, S, D) — encoder output for this layer
        Returns:
            (B, H, D)
        """
        # Conditioning vector: pool encoder output + add time embedding
        c = cond.mean(dim=1) + t  # (B, D)

        x2 = self.attn_mod1(self.norm1(x), c)
        x2, _ = self.self_attn(x2, x2, x2, need_weights=False)
        x = self.attn_mod2(self.dropout1(x2), c) + x

        x2 = self.mlp_mod1(self.norm2(x), c)
        x2 = self.linear2(self.dropout2(self.activation(self.linear1(x2))))
        x2 = self.mlp_mod2(self.dropout3(x2), c)
        return x + x2


# ---------------------------------------------------------------------------
# Final layer (adaLN + linear projection)
# ---------------------------------------------------------------------------

class FinalLayer(nn.Module):
    """adaLN-conditioned final projection layer."""

    def __init__(self, hidden_size: int, out_dim: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    (B, H, D)
            t:    (B, D)    — time embedding
            cond: (B, S, D) — last encoder output
        Returns:
            (B, H, out_dim)
        """
        c = cond.mean(dim=1) + t  # (B, D)
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = self.norm_final(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return self.linear(x)


# ---------------------------------------------------------------------------
# Time embedding (same pattern as Co-Diff src/model.py)
# ---------------------------------------------------------------------------

class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(1, dim), nn.Mish(), nn.Linear(dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


# ---------------------------------------------------------------------------
# Main model: ImageDiT
# ---------------------------------------------------------------------------

class ImageDiT(nn.Module):
    """End-to-end image-conditioned DiT for action prediction.

    Architecture:
        Encoder: ResNet18 (shared) -> project -> positional encoding -> 3x self-attention
        Decoder: project noisy actions -> positional encoding -> 3x DiT blocks -> final layer

    Parameters (~17.2M total):
        - ResNet18GroupNorm: ~11.17M
        - Encoder blocks (3x): ~2.36M
        - Decoder blocks (3x): ~3.54M
        - Projections + embeddings: ~0.13M
    """

    def __init__(self, x_dim: int = 7, d_model: int = 256, n_heads: int = 4,
                 depth: int = 3, dim_feedforward: int = 1024,
                 dropout: float = 0.1, horizon: int = 20, num_cameras: int = 2,
                 backbone: str = "resnet18", frame_offsets=(0,)):
        super().__init__()
        self.x_dim = x_dim
        self.d_model = d_model
        self.horizon = horizon
        self.num_cameras = num_cameras
        self.backbone_type = backbone
        self.frame_offsets = normalize_frame_offsets(frame_offsets)
        self.num_frames = len(self.frame_offsets)

        # Shared image encoder
        if backbone == "dinov2":
            from src.dino_backbone import DINOv2Backbone
            self.resnet = DINOv2Backbone()
            self.tokens_per_camera = DINOv2Backbone.tokens_per_image  # 81
            feature_dim = DINOv2Backbone.feature_dim  # 384
        else:
            self.resnet = ResNet18GroupNorm()
            self.tokens_per_camera = 16
            feature_dim = 512

        enc_seq_len = self.tokens_per_camera * num_cameras * self.num_frames

        # Project backbone features to d_model
        self.img_proj = nn.Linear(feature_dim, d_model)

        # Camera embedding to distinguish EIH vs shoulder views
        self.cam_embed = nn.Embedding(2, d_model)

        # Single-frame models intentionally omit this parameter so legacy
        # checkpoints retain an identical state dict and parameter count.
        if self.num_frames > 1:
            self.temporal_embed = nn.Embedding(self.num_frames, d_model)

        # Learnable positional encoding for image tokens (16 per camera)
        self.enc_pos = nn.Parameter(torch.empty(1, enc_seq_len, d_model))
        nn.init.xavier_uniform_(self.enc_pos.data.view(enc_seq_len, d_model))

        # Encoder: 3 self-attention blocks
        self.encoder_blocks = nn.ModuleList([
            SelfAttentionEncoderBlock(d_model, n_heads, dim_feedforward, dropout)
            for _ in range(depth)
        ])

        # Time embedding
        self.time_emb = TimeEmbedding(d_model)

        # Action projection
        self.ac_proj = nn.Linear(x_dim, d_model)

        # Learnable positional encoding for action tokens
        self.dec_pos = nn.Parameter(torch.empty(1, horizon, d_model))
        nn.init.xavier_uniform_(self.dec_pos.data.view(horizon, d_model))

        # Decoder: 3 DiT blocks
        self.decoder_blocks = nn.ModuleList([
            DiTDecoderBlock(d_model, n_heads, dim_feedforward, dropout)
            for _ in range(depth)
        ])

        # Final projection
        self.final_layer = FinalLayer(d_model, x_dim)

        # Null token for CFG dropout (replaces encoder outputs)
        self.null_token = nn.Parameter(torch.empty(1, enc_seq_len, d_model))
        nn.init.normal_(self.null_token, std=0.02)

        self.initialize_weights()

    def initialize_weights(self) -> None:
        """Weight initialization following Co-Diff's DiT1d pattern."""
        def _basic_init(module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        # Apply to transformer components (skip ResNet — it has its own init)
        for module_list in [self.encoder_blocks, self.decoder_blocks]:
            module_list.apply(_basic_init)
        self.ac_proj.apply(_basic_init)
        self.img_proj.apply(_basic_init)

        # Time embedding: Normal(0.02)
        nn.init.normal_(self.time_emb.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_emb.mlp[2].weight, std=0.02)

        # Zero-init all adaLN modulation output layers (decoder blocks)
        for block in self.decoder_blocks:
            block.attn_mod2.reset_parameters()  # zero-init gate
            block.mlp_mod2.reset_parameters()   # zero-init gate
            block.attn_mod1.reset_parameters()   # xavier for shift/scale
            block.mlp_mod1.reset_parameters()    # xavier for shift/scale

        # Zero-init final layer
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def _encode_camera(self, images: torch.Tensor, camera_id: int) -> torch.Tensor:
        """Encode one camera's temporal stack with shared backbone weights."""
        if images.ndim == 4:
            if self.num_frames != 1:
                raise ValueError(
                    f"Expected {self.num_frames} temporal frames, got a 4-D image batch"
                )
            images = images[:, None]
        if images.ndim != 5:
            raise ValueError(
                "Images must have shape (B,C,H,W) or (B,F,C,H,W); "
                f"got {tuple(images.shape)}"
            )
        batch_size, num_frames, channels, height, width = images.shape
        if num_frames != self.num_frames:
            raise ValueError(
                f"Expected {self.num_frames} temporal frames, got {num_frames}"
            )

        flat = images.reshape(batch_size * num_frames, channels, height, width)
        tokens = self.img_proj(self.resnet(flat))
        tokens = tokens.reshape(
            batch_size, num_frames, self.tokens_per_camera, self.d_model
        )
        camera_ids = torch.full(
            (batch_size, num_frames, self.tokens_per_camera),
            camera_id,
            dtype=torch.long,
            device=images.device,
        )
        tokens = tokens + self.cam_embed(camera_ids)
        if self.num_frames > 1:
            temporal_ids = torch.arange(num_frames, device=images.device).view(
                1, num_frames, 1
            )
            tokens = tokens + self.temporal_embed(temporal_ids)
        return tokens.flatten(1, 2)

    def forward_encoder(self, imgs_eih: torch.Tensor,
                        imgs_shoulder: torch.Tensor) -> list:
        """Encode images from available cameras.

        Args:
            imgs_eih:      (B, [F,] 3, 128, 128) eye-in-hand images, or None
            imgs_shoulder: (B, [F,] 3, 128, 128) shoulder images

        Returns:
            List of encoder layer outputs, each (B, S, d_model)
            where S = spatial_tokens * frames * num_active_cameras
        """
        token_groups = []

        if imgs_eih is not None:
            token_groups.append(self._encode_camera(imgs_eih, camera_id=0))
        token_groups.append(self._encode_camera(imgs_shoulder, camera_id=1))

        # Concatenate all camera tokens
        tokens = torch.cat(token_groups, dim=1)  # (B, S, d_model)

        # Self-attention encoder with positional encoding
        outputs = []
        x = tokens
        for block in self.encoder_blocks:
            x = block(x, self.enc_pos)
            outputs.append(x)

        return outputs

    def forward_decoder(self, noisy_actions: torch.Tensor,
                        time: torch.Tensor,
                        encoder_outputs: list) -> torch.Tensor:
        """Decode noisy actions conditioned on encoder outputs.

        Args:
            noisy_actions:   (B, H, x_dim) noisy action trajectory
            time:            (B, 1) noise level (c_noise)
            encoder_outputs: list of (B, 32, d_model) from encoder

        Returns:
            (B, H, x_dim) predicted clean actions
        """
        t = self.time_emb(time)  # (B, d_model)

        # Project actions to d_model and add positional encoding
        x = self.ac_proj(noisy_actions) + self.dec_pos  # (B, H, d_model)

        # DiT decoder blocks
        for block, enc_out in zip(self.decoder_blocks, encoder_outputs):
            x = block(x, t, enc_out)

        # Final layer
        x = self.final_layer(x, t, encoder_outputs[-1])
        return x

    def forward(self, noisy_actions: torch.Tensor, time: torch.Tensor,
                imgs_eih: torch.Tensor, imgs_shoulder: torch.Tensor,
                cfg_mask: torch.Tensor = None) -> torch.Tensor:
        """Full forward pass with optional CFG dropout.

        Args:
            noisy_actions:  (B, H, x_dim)
            time:           (B, 1)
            imgs_eih:       (B, 3, 128, 128) or None for shoulder-only
            imgs_shoulder:  (B, 3, 128, 128)
            cfg_mask:       (B,) bool tensor — True means drop conditioning (use null token)

        Returns:
            (B, H, x_dim)
        """
        enc_outputs = self.forward_encoder(imgs_eih, imgs_shoulder)

        if cfg_mask is not None and cfg_mask.any():
            B = imgs_shoulder.shape[0]
            null_expanded = self.null_token.expand(B, -1, -1)
            enc_outputs = [
                torch.where(cfg_mask[:, None, None], null_expanded, out)
                for out in enc_outputs
            ]

        return self.forward_decoder(noisy_actions, time, enc_outputs)
