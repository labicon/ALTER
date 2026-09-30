"""End-to-end image coordination head for multi-arm diffusion policy.

Learns a residual correction on top of a frozen ``ImageConditional_ODE``
base model.  The head does **not** have its own ResNet — it reuses the
frozen base model's encoder outputs (list of 3 × (B, 32, 256)) and wraps
a small decoder-only transformer (~750K params) that produces a residual.

Optionally includes a small **side network** — a lightweight CNN that
processes the same raw images in parallel with the frozen backbone.  The
side network's tokens are added to the projected frozen encoder outputs,
giving the head access to task-relevant visual features without modifying
the frozen backbone.

Architecture mirrors ``LatentCoordinationHead`` in spirit but operates on
image encoder outputs rather than pre-computed latent vectors.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as Fnn
from copy import deepcopy

from src.image_dit import (
    DiTDecoderBlock,
    FinalLayer,
    ShiftScaleMod,
    TimeEmbedding,
    ZeroScaleMod,
)
from src.temporal import normalize_frame_offsets, require_checkpoint_frame_offsets


LEGACY_DECODER_EXECUTION = "legacy_zip"
ALL_BLOCKS_DECODER_EXECUTION = "all_blocks_repeat_last"
SIDE_NET_FUSION_ADD = "add"
SIDE_NET_FUSION_CONCAT = "concat"
DECODER_CONDITIONING_POOLED = "pooled"
DECODER_CONDITIONING_CROSS_ATTN = "cross_attn"
PLAN_MEMORY_ADDITIVE = "additive"
PLAN_MEMORY_REPLACE = "replace"
PLAN_MEMORY_DUAL = "dual"
VALID_DECODER_EXECUTIONS = {
    LEGACY_DECODER_EXECUTION,
    ALL_BLOCKS_DECODER_EXECUTION,
}
VALID_SIDE_NET_FUSIONS = {
    SIDE_NET_FUSION_ADD,
    SIDE_NET_FUSION_CONCAT,
}
VALID_DECODER_CONDITIONING = {
    DECODER_CONDITIONING_POOLED,
    DECODER_CONDITIONING_CROSS_ATTN,
}
VALID_PLAN_MEMORY_MODES = {
    PLAN_MEMORY_ADDITIVE,
    PLAN_MEMORY_REPLACE,
    PLAN_MEMORY_DUAL,
}


def coordination_architecture_name(plan_memory_mode: str) -> str:
    """Stable checkpoint architecture tag for vanilla residual heads."""
    if plan_memory_mode == PLAN_MEMORY_ADDITIVE:
        return "coord_image_residual_v1"
    return "coord_plan_memory_v1"


def plan_memory_kwargs_from_stats(coord_stats: dict) -> dict:
    """Return the plan-fusion setting with an explicit legacy fallback."""
    return {
        "plan_memory_mode": str(
            coord_stats.get("plan_memory_mode", PLAN_MEMORY_ADDITIVE)
        )
    }


def side_net_kwargs_from_stats(coord_stats: dict, base_tokens_per_camera: int) -> dict:
    """Return ImageCoordinationHead side-net kwargs with legacy fallbacks."""
    tokens_per_camera = int(
        coord_stats.get("tokens_per_camera", base_tokens_per_camera)
    )
    return {
        "tokens_per_camera": tokens_per_camera,
        "use_side_net": bool(coord_stats.get("use_side_net", False)),
        "side_net_input_size": int(coord_stats.get("side_net_input_size", 128)),
        "side_net_fusion": str(coord_stats.get("side_net_fusion", SIDE_NET_FUSION_ADD)),
        "side_net_tokens_per_camera": int(
            coord_stats.get("side_net_tokens_per_camera", tokens_per_camera)
        ),
    }


# ---------------------------------------------------------------------------
# Lightweight side CNN for side-tuning
# ---------------------------------------------------------------------------

class _SideCNN(nn.Module):
    """Small CNN that extracts task-relevant features from raw images.

    Input:  (B, 3, input_size, input_size)
    Output: (B, tokens_per_camera, d_model)

    Architecture: 4 conv layers with GroupNorm + GELU, ending at a spatial
    grid that gets flattened into tokens.  ~180K params for d_model=128.
    """

    def __init__(
        self,
        d_model: int = 128,
        tokens_per_camera: int = 16,
        input_size: int = 128,
    ):
        super().__init__()
        self.tokens_per_camera = tokens_per_camera
        self.input_size = int(input_size)
        spatial = int(tokens_per_camera ** 0.5)  # 16->4, 64->8, 81->9
        if spatial * spatial != tokens_per_camera:
            raise ValueError(
                "tokens_per_camera must be a perfect square for _SideCNN; "
                f"got {tokens_per_camera}"
            )

        # 128x128 -> 64 -> 32 -> 16 -> 8 before the adaptive pool.
        # For larger input_size, the same lightweight CNN sees sharper inputs
        # and the adaptive pool controls the final token grid.
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),   # -> 64x64
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # -> 32x32
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # -> 16x16
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv2d(128, d_model, 3, stride=2, padding=1),  # -> 8x8
            nn.GroupNorm(8, d_model),
            nn.GELU(),
        )

        # Adaptive pool to match tokens_per_camera spatial resolution.
        self.pool = nn.AdaptiveAvgPool2d(spatial)

        # ImageNet normalization
        self.register_buffer(
            "img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) -> (B, tokens_per_camera, d_model)"""
        if x.shape[-2:] != (self.input_size, self.input_size):
            x = Fnn.interpolate(
                x, size=(self.input_size, self.input_size),
                mode="bilinear", align_corners=False,
            )
        x = (x - self.img_mean) / self.img_std
        x = self.conv(x)              # (B, d_model, H, W)
        x = self.pool(x)              # (B, d_model, s, s)
        B, C, H, W = x.shape
        return x.reshape(B, C, H * W).permute(0, 2, 1)  # (B, tokens, d_model)


class _CrossAttentionDecoderBlock(nn.Module):
    """Coordination-head decoder block with token-level visual conditioning."""

    def __init__(self, d_model: int = 256, n_heads: int = 4,
                 dim_feedforward: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm_cross = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout_cross = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.GELU(approximate="tanh")

        self.attn_mod1 = ShiftScaleMod(d_model)
        self.attn_mod2 = ZeroScaleMod(d_model)
        self.cross_mod1 = ShiftScaleMod(d_model)
        self.cross_mod2 = ZeroScaleMod(d_model)
        self.mlp_mod1 = ShiftScaleMod(d_model)
        self.mlp_mod2 = ZeroScaleMod(d_model)

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                cond: torch.Tensor) -> torch.Tensor:
        c = cond.mean(dim=1) + t

        x2 = self.attn_mod1(self.norm1(x), c)
        x2, _ = self.self_attn(x2, x2, x2, need_weights=False)
        x = self.attn_mod2(self.dropout1(x2), c) + x

        x2 = self.cross_mod1(self.norm_cross(x), c)
        x2, _ = self.cross_attn(x2, cond, cond, need_weights=False)
        x = self.cross_mod2(self.dropout_cross(x2), c) + x

        x2 = self.mlp_mod1(self.norm2(x), c)
        x2 = self.linear2(self.dropout2(self.activation(self.linear1(x2))))
        x2 = self.mlp_mod2(self.dropout3(x2), c)
        return x + x2


# ---------------------------------------------------------------------------
# Inner network — all trainable params (clean EMA copying)
# ---------------------------------------------------------------------------

class _ImageCoordNet(nn.Module):
    """Small decoder-only transformer that produces a residual correction.

    Takes the last encoder outputs from the frozen base model, projects them
    into the head width, and produces a (B, H, x_dim) residual. Corrected
    checkpoints execute every configured decoder block.
    """

    def __init__(
        self,
        x_dim: int = 7,
        base_d_model: int = 256,
        d_model: int = 128,
        n_heads: int = 4,
        depth: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        horizon: int = 20,
        num_enc_layers_used: int = 2,
        num_cameras: int = 2,
        tokens_per_camera: int = 16,
        use_side_net: bool = False,
        side_net_input_size: int = 128,
        side_net_fusion: str = SIDE_NET_FUSION_ADD,
        side_net_tokens_per_camera: int | None = None,
        decoder_execution: str = LEGACY_DECODER_EXECUTION,
        decoder_conditioning: str = DECODER_CONDITIONING_POOLED,
        plan_memory_mode: str = PLAN_MEMORY_ADDITIVE,
        frame_offsets=(0,),
    ):
        super().__init__()
        if decoder_execution not in VALID_DECODER_EXECUTIONS:
            raise ValueError(
                f"Unknown decoder_execution={decoder_execution!r}; "
                f"expected one of {sorted(VALID_DECODER_EXECUTIONS)}"
            )
        if side_net_fusion not in VALID_SIDE_NET_FUSIONS:
            raise ValueError(
                f"Unknown side_net_fusion={side_net_fusion!r}; "
                f"expected one of {sorted(VALID_SIDE_NET_FUSIONS)}"
            )
        if decoder_conditioning not in VALID_DECODER_CONDITIONING:
            raise ValueError(
                f"Unknown decoder_conditioning={decoder_conditioning!r}; "
                f"expected one of {sorted(VALID_DECODER_CONDITIONING)}"
            )
        if plan_memory_mode not in VALID_PLAN_MEMORY_MODES:
            raise ValueError(
                f"Unknown plan_memory_mode={plan_memory_mode!r}; "
                f"expected one of {sorted(VALID_PLAN_MEMORY_MODES)}"
            )
        if (
            plan_memory_mode != PLAN_MEMORY_ADDITIVE
            and decoder_conditioning != DECODER_CONDITIONING_CROSS_ATTN
        ):
            raise ValueError(
                "plan-memory fusion requires decoder_conditioning='cross_attn' "
                "so action tokens attend to visual and frozen-base plan tokens"
            )
        self.d_model = d_model
        self.horizon = horizon
        self.num_enc_layers_used = num_enc_layers_used
        self.use_side_net = use_side_net
        self.num_cameras = num_cameras
        self.tokens_per_camera = tokens_per_camera
        self.frame_offsets = normalize_frame_offsets(frame_offsets)
        self.num_frames = len(self.frame_offsets)
        self.side_net_input_size = int(side_net_input_size)
        self.side_net_fusion = side_net_fusion
        self.side_net_tokens_per_camera = int(
            tokens_per_camera if side_net_tokens_per_camera is None
            else side_net_tokens_per_camera
        )
        self.decoder_execution = decoder_execution
        self.decoder_conditioning = decoder_conditioning
        self.plan_memory_mode = plan_memory_mode

        enc_seq_len = tokens_per_camera * num_cameras
        side_enc_seq_len = (
            self.side_net_tokens_per_camera * num_cameras * self.num_frames
        )
        if self.use_side_net and self.side_net_fusion == SIDE_NET_FUSION_ADD:
            if side_enc_seq_len != enc_seq_len:
                raise ValueError(
                    "side_net_fusion='add' requires side_net_tokens_per_camera "
                    "to match tokens_per_camera; got "
                    f"{self.side_net_tokens_per_camera} vs {tokens_per_camera}"
                )

        # Project last N base encoder outputs to head's d_model
        self.enc_proj = nn.ModuleList([
            nn.Linear(base_d_model, d_model)
            for _ in range(num_enc_layers_used)
        ])

        # Null token for CFG in projected space (zero-init → residual starts at 0)
        self.null_enc_proj = nn.Parameter(torch.zeros(1, enc_seq_len, d_model))

        # Side network: small trainable CNN that runs in parallel with frozen backbone
        if use_side_net:
            self.side_cnn = _SideCNN(
                d_model=d_model,
                tokens_per_camera=self.side_net_tokens_per_camera,
                input_size=self.side_net_input_size,
            )
            # Zero-init projection so side net starts as no-op
            self.side_proj = nn.Linear(d_model, d_model)
            # Null token for side net CFG dropout
            self.null_side = nn.Parameter(torch.zeros(1, side_enc_seq_len, d_model))
            if self.num_frames > 1:
                self.side_temporal_embed = nn.Embedding(self.num_frames, d_model)

        # Time embedding
        self.time_emb = TimeEmbedding(d_model)

        # Action projection
        self.ac_proj = nn.Linear(x_dim, d_model)

        # The legacy head adds a projected base denoising chunk to action
        # tokens. The replacement variant intentionally has no such module.
        if plan_memory_mode != PLAN_MEMORY_REPLACE:
            self.d_base_proj = nn.Linear(x_dim, d_model)

        # Plan-memory tokens are derived solely from D_base. Their positional
        # index is the action-chunk index; they carry no task, mode, robot,
        # privileged-state, temporal-history, or extra-image signal.
        if plan_memory_mode != PLAN_MEMORY_ADDITIVE:
            self.plan_memory_proj = nn.Linear(x_dim, d_model)
            self.plan_memory_pos = nn.Parameter(torch.empty(1, horizon, d_model))
            nn.init.xavier_uniform_(self.plan_memory_pos.data.view(horizon, d_model))

        # Learnable positional encoding for action tokens
        self.dec_pos = nn.Parameter(torch.empty(1, horizon, d_model))
        nn.init.xavier_uniform_(self.dec_pos.data.view(horizon, d_model))

        # Decoder blocks
        decoder_block_cls = (
            _CrossAttentionDecoderBlock
            if decoder_conditioning == DECODER_CONDITIONING_CROSS_ATTN
            else DiTDecoderBlock
        )
        self.decoder_blocks = nn.ModuleList([
            decoder_block_cls(d_model, n_heads, dim_feedforward, dropout)
            for _ in range(depth)
        ])

        # Final projection
        self.final_layer = FinalLayer(d_model, x_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier on projections, zero-init on final layer and gates."""
        # Xavier on projection layers
        for proj in self.enc_proj:
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)
        nn.init.xavier_uniform_(self.ac_proj.weight)
        nn.init.zeros_(self.ac_proj.bias)
        if hasattr(self, "d_base_proj"):
            nn.init.zeros_(self.d_base_proj.weight)
            nn.init.zeros_(self.d_base_proj.bias)
        if hasattr(self, "plan_memory_proj"):
            nn.init.zeros_(self.plan_memory_proj.weight)
            nn.init.zeros_(self.plan_memory_proj.bias)

        # Zero-init side projection so it starts as no-op
        if self.use_side_net:
            nn.init.zeros_(self.side_proj.weight)
            nn.init.zeros_(self.side_proj.bias)

        # Time embedding: Normal(0.02)
        nn.init.normal_(self.time_emb.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_emb.mlp[2].weight, std=0.02)

        # Zero-init adaLN gates in decoder blocks
        for block in self.decoder_blocks:
            block.attn_mod2.reset_parameters()   # zero-init gate
            block.mlp_mod2.reset_parameters()    # zero-init gate
            block.attn_mod1.reset_parameters()   # xavier for shift/scale
            block.mlp_mod1.reset_parameters()    # xavier for shift/scale
            if hasattr(block, "cross_mod2"):
                block.cross_mod2.reset_parameters()
                block.cross_mod1.reset_parameters()

        # Zero-init final layer so residual starts at exactly zero
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def _encode_side_camera(self, images: torch.Tensor) -> torch.Tensor:
        """Run the shared side CNN over a camera's temporal frames."""
        if images.ndim == 4:
            if self.num_frames != 1:
                raise ValueError(
                    f"Expected {self.num_frames} side-net frames, got a 4-D batch"
                )
            images = images[:, None]
        if images.ndim != 5:
            raise ValueError(
                "Side-net images must have shape (B,C,H,W) or (B,F,C,H,W); "
                f"got {tuple(images.shape)}"
            )
        batch_size, num_frames, channels, height, width = images.shape
        if num_frames != self.num_frames:
            raise ValueError(
                f"Expected {self.num_frames} side-net frames, got {num_frames}"
            )
        flat = images.reshape(batch_size * num_frames, channels, height, width)
        tokens = self.side_cnn(flat).reshape(
            batch_size,
            num_frames,
            self.side_net_tokens_per_camera,
            self.d_model,
        )
        if self.num_frames > 1:
            temporal_ids = torch.arange(num_frames, device=images.device).view(
                1, num_frames, 1
            )
            tokens = tokens + self.side_temporal_embed(temporal_ids)
        return tokens.flatten(1, 2)

    def _plan_memory_tokens(
        self, d_base: torch.Tensor, *, batch_size: int
    ) -> torch.Tensor:
        """Return positional memory tokens from only the frozen-base action plan."""
        if d_base is None:
            raise ValueError("plan-memory fusion requires the frozen-base D_base action chunk")
        expected_dim = self.plan_memory_proj.in_features
        if (
            d_base.ndim != 3
            or d_base.shape[0] != batch_size
            or d_base.shape[1] != self.horizon
            or d_base.shape[2] != expected_dim
        ):
            raise ValueError(
                "D_base shape differs from the frozen-base action chunk: "
                f"expected (B, {self.horizon}, {expected_dim}), got {tuple(d_base.shape)}"
            )
        return self.plan_memory_proj(d_base) + self.plan_memory_pos

    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        enc_outputs_base: list,
        cfg_mask: torch.Tensor = None,
        d_base: torch.Tensor = None,
        imgs_eih: torch.Tensor = None,
        imgs_shoulder: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x:                (B, H, x_dim) — scaled noisy actions (c_in already applied)
            sigma:            (B, 1) — c_noise(sigma) time conditioning
            enc_outputs_base: list of (B, 32, base_d_model) from frozen base encoder
            cfg_mask:         (B,) bool — True means drop conditioning (use null token)
            d_base:           (B, H, x_dim) — base model denoising prediction (raw, not c_in scaled)
            imgs_eih:         (B, 3, 128, 128) eye-in-hand images (for side net)
            imgs_shoulder:    (B, 3, 128, 128) shoulder images (for side net)

        Returns:
            (B, H, x_dim) raw network output (before c_out scaling)
        """
        # Project last N encoder outputs to head dimension
        projected = []
        for i in range(self.num_enc_layers_used):
            idx = -(self.num_enc_layers_used - i)
            projected.append(self.enc_proj[i](enc_outputs_base[idx]))

        # CFG on base projected outputs first. For additive fusion this keeps
        # the old behavior: both base and side streams are separately nulled.
        if cfg_mask is not None and cfg_mask.any():
            null_expanded = self.null_enc_proj.expand(x.shape[0], -1, -1)
            projected = [
                torch.where(cfg_mask[:, None, None], null_expanded, p)
                for p in projected
            ]

        # Side network: extract task-relevant features from raw images
        if self.use_side_net and imgs_shoulder is not None:
            side_tokens = []
            if imgs_eih is not None:
                side_tokens.append(self._encode_side_camera(imgs_eih))
            side_tokens.append(self._encode_side_camera(imgs_shoulder))
            side_combined = torch.cat(side_tokens, dim=1)
            side_out = self.side_proj(side_combined)

            # CFG: replace side tokens with null where mask is True
            if cfg_mask is not None and cfg_mask.any():
                null_side_expanded = self.null_side.expand(x.shape[0], -1, -1)
                side_out = torch.where(cfg_mask[:, None, None], null_side_expanded, side_out)

            if self.side_net_fusion == SIDE_NET_FUSION_ADD:
                # Backwards-compatible path: side features must align 1:1 with
                # the frozen base tokens and are added in projected space.
                projected = [p + side_out for p in projected]
            else:
                # New path: keep the frozen base token grid intact and append a
                # separate trainable side stream. The current decoder pools
                # conditioning tokens, so this decouples side resolution without
                # changing old checkpoints or the frozen base.
                projected = [torch.cat([p, side_out], dim=1) for p in projected]

        if self.plan_memory_mode != PLAN_MEMORY_ADDITIVE:
            plan_tokens = self._plan_memory_tokens(d_base, batch_size=x.shape[0])
            # Cross-attention keys/values contain both frozen visual tokens
            # and the positional representation of the same base action plan.
            projected = [torch.cat([p, plan_tokens], dim=1) for p in projected]

        # Time embedding
        t = self.time_emb(sigma)  # (B, d_model)

        # Project actions + positional encoding + D_base
        h = self.ac_proj(x) + self.dec_pos  # (B, H, d_model)
        if d_base is not None and self.plan_memory_mode != PLAN_MEMORY_REPLACE:
            h = h + self.d_base_proj(d_base)

        if self.decoder_execution == LEGACY_DECODER_EXECUTION:
            for block, enc_out in zip(self.decoder_blocks, projected):
                h = block(h, t, enc_out)
        else:
            # Extra decoder blocks reuse the deepest projected encoder output.
            for i, block in enumerate(self.decoder_blocks):
                enc_out = projected[min(i, len(projected) - 1)]
                h = block(h, t, enc_out)

        # Final layer uses last projected encoder output
        h = self.final_layer(h, t, projected[-1])
        return h


# ---------------------------------------------------------------------------
# Outer class — Karras preconditioning + training logic
# ---------------------------------------------------------------------------

class ImageCoordinationHead(nn.Module):
    """End-to-end image coordination head with Karras preconditioning.

    Mirrors ``LatentCoordinationHead`` but operates on frozen base model
    encoder outputs instead of pre-computed latent vectors.
    """

    def __init__(
        self,
        x_dim: int = 7,
        base_d_model: int = 256,
        d_model: int = 128,
        n_heads: int = 4,
        depth: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        horizon: int = 20,
        sigma_data: float = 1.0,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        num_cameras: int = 2,
        tokens_per_camera: int = 16,
        d_base_drop_prob: float = 0.1,
        use_side_net: bool = False,
        side_net_input_size: int = 128,
        side_net_fusion: str = SIDE_NET_FUSION_ADD,
        side_net_tokens_per_camera: int | None = None,
        decoder_execution: str = LEGACY_DECODER_EXECUTION,
        decoder_conditioning: str = DECODER_CONDITIONING_POOLED,
        plan_memory_mode: str = PLAN_MEMORY_ADDITIVE,
        frame_offsets=(0,),
    ):
        super().__init__()
        self.x_dim = x_dim
        self.sigma_data = sigma_data
        self.d_base_drop_prob = d_base_drop_prob
        self.decoder_execution = decoder_execution
        self.decoder_conditioning = decoder_conditioning
        self.plan_memory_mode = plan_memory_mode
        self.frame_offsets = normalize_frame_offsets(frame_offsets)

        self.F = _ImageCoordNet(
            x_dim=x_dim,
            base_d_model=base_d_model,
            d_model=d_model,
            n_heads=n_heads,
            depth=depth,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            horizon=horizon,
            num_cameras=num_cameras,
            tokens_per_camera=tokens_per_camera,
            use_side_net=use_side_net,
            side_net_input_size=side_net_input_size,
            side_net_fusion=side_net_fusion,
            side_net_tokens_per_camera=side_net_tokens_per_camera,
            decoder_execution=decoder_execution,
            decoder_conditioning=decoder_conditioning,
            plan_memory_mode=plan_memory_mode,
            frame_offsets=self.frame_offsets,
        )
        self.F.train()

        # EMA copy for inference
        self.F_ema = deepcopy(self.F).requires_grad_(False).eval()

        if weight_decay < 0:
            raise ValueError("weight_decay must be nonnegative")
        # Optimise only the trainable head. Keeping the default at 1e-2
        # preserves legacy Coord checkpoints; new campaigns pass their
        # intended value explicitly and seal it in the launch contract.
        self.optimizer = torch.optim.AdamW(
            [parameter for parameter in self.F.parameters() if parameter.requires_grad],
            lr=lr,
            weight_decay=weight_decay,
        )
        self.checkpoint_metadata = {}
        self.last_train_metrics = {}
        self.training_metric_ema = {}

    def ema_update(self, decay: float = 0.999):
        for p, p_ema in zip(self.F.parameters(), self.F_ema.parameters()):
            p_ema.data = decay * p_ema.data + (1 - decay) * p.data

    # ------------------------------------------------------------------
    # Karras preconditioning helpers (identical to base model)
    # ------------------------------------------------------------------

    def c_skip(self, sigma):
        return self.sigma_data ** 2 / (self.sigma_data ** 2 + sigma ** 2)

    def c_out(self, sigma):
        return sigma * self.sigma_data / (self.sigma_data ** 2 + sigma ** 2).sqrt()

    def c_in(self, sigma):
        return 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()

    def c_noise(self, sigma):
        return 0.25 * sigma.log()

    def loss_weighting(self, sigma):
        return (self.sigma_data ** 2 + sigma ** 2) / ((sigma * self.sigma_data) ** 2)

    # ------------------------------------------------------------------
    # Forward — residual prediction
    # ------------------------------------------------------------------

    def forward_residual(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        enc_outputs_base: list,
        use_ema: bool = False,
        cfg_mask: torch.Tensor = None,
        d_base: torch.Tensor = None,
        imgs_eih: torch.Tensor = None,
        imgs_shoulder: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute residual correction: c_out * F(c_in * x, c_noise(sigma), enc, d_base).

        Args:
            x:                (B, H, x_dim) noisy trajectory
            sigma:            (B, 1, 1) noise level
            enc_outputs_base: list of (B, 32, base_d_model) from frozen base encoder
            use_ema:          use EMA weights
            cfg_mask:         (B,) bool — True = drop conditioning
            d_base:           (B, H, x_dim) — base model denoising prediction (raw)
            imgs_eih:         (B, 3, 128, 128) eye-in-hand images (for side net)
            imgs_shoulder:    (B, 3, 128, 128) shoulder images (for side net)

        Returns:
            (B, H, x_dim) residual correction
        """
        c_out = self.c_out(sigma)
        c_in = self.c_in(sigma)
        c_noise = self.c_noise(sigma)

        F = self.F_ema if use_ema else self.F
        return c_out * F(c_in * x, c_noise.squeeze(-1), enc_outputs_base, cfg_mask,
                         d_base=d_base, imgs_eih=imgs_eih, imgs_shoulder=imgs_shoulder)

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def _training_domain_loss(
        self,
        x_GT: torch.Tensor,
        imgs_eih: torch.Tensor,
        imgs_shoulder: torch.Tensor,
        base_model,
        *,
        zero_residual_target: bool,
        cfg_mask: torch.Tensor = None,
        diagnostics: dict = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build one domain objective without stepping the optimizer.

        Multi-arm data uses the usual expert denoising objective. Single-arm
        data uses the same Karras weighting but trains the head residual to
        zero, leaving the frozen base prediction unchanged.
        """
        B = x_GT.shape[0]
        device = x_GT.device
        sigma = base_model.sample_noise_distribution(B)
        x_t = x_GT + torch.randn_like(x_GT) * sigma

        if cfg_mask is None:
            cfg_mask = torch.rand(B, device=device) < 0.2

        with torch.no_grad():
            enc_out = base_model.F_ema.forward_encoder(imgs_eih, imgs_shoulder)
            null_token = base_model.F_ema.null_token.expand(B, -1, -1)
            enc_out_masked = [
                torch.where(cfg_mask[:, None, None], null_token, out)
                for out in enc_out
            ]
            D_base = base_model._D_from_enc(
                x_t, sigma, enc_out_masked, use_ema=True
            )

        if self.d_base_drop_prob > 0:
            d_base_mask = torch.rand(B, device=device) < self.d_base_drop_prob
            D_base_input = torch.where(
                d_base_mask[:, None, None], torch.zeros_like(D_base), D_base
            )
        else:
            D_base_input = D_base

        delta = self.forward_residual(
            x_t, sigma, enc_out, use_ema=False, cfg_mask=cfg_mask,
            d_base=D_base_input,
            imgs_eih=imgs_eih, imgs_shoulder=imgs_shoulder,
        )
        weights = base_model.loss_weighting(sigma)
        error = delta if zero_residual_target else (D_base + delta - x_GT)
        loss = (weights * error.square()).mean()
        if diagnostics is not None:
            diagnostics.update(
                sigma=sigma.detach().flatten(),
                loss=(weights * error.square()).detach().mean(dim=(1, 2)),
                base_error=(D_base - x_GT).detach().square().mean(dim=(1, 2)),
                residual_rms=delta.detach().square().mean(dim=(1, 2)).sqrt(),
            )
        return loss, delta

    def update(
        self,
        x_GT: torch.Tensor,
        imgs_eih: torch.Tensor,
        imgs_shoulder: torch.Tensor,
        base_model,
        cfg_mask: torch.Tensor = None,
    ) -> tuple:
        """Single training step.

        Args:
            x_GT:          (B, H, x_dim) ground-truth normalized actions
            imgs_eih:      (B, 3, 128, 128) eye-in-hand images
            imgs_shoulder: (B, 3, 128, 128) shoulder images
            base_model:    frozen ImageConditional_ODE
            cfg_mask:      (B,) bool — optional CFG dropout mask

        Returns:
            (loss, grad_norm)
        """
        loss, _ = self._training_domain_loss(
            x_GT, imgs_eih, imgs_shoulder, base_model,
            zero_residual_target=False, cfg_mask=cfg_mask,
        )

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
        self.optimizer.step()

        self.ema_update()

        return loss.item(), grad_norm.item()

    def update_mixed(
        self,
        twoarm_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        singlearm_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        base_model,
        singlearm_weight: float = 1.0,
    ) -> tuple[float, float, dict[str, float]]:
        """Optimize one two-arm denoising batch plus one single-arm no-op batch."""
        ta_eih, ta_shoulder, ta_actions = twoarm_batch
        sa_eih, sa_shoulder, sa_actions = singlearm_batch
        twoarm_loss, twoarm_delta = self._training_domain_loss(
            ta_actions, ta_eih, ta_shoulder, base_model,
            zero_residual_target=False,
        )
        singlearm_loss, singlearm_delta = self._training_domain_loss(
            sa_actions, sa_eih, sa_shoulder, base_model,
            zero_residual_target=True,
        )
        total_loss = twoarm_loss + float(singlearm_weight) * singlearm_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
        self.optimizer.step()
        self.ema_update(decay=0.999)

        metrics = {
            "twoarm_loss": float(twoarm_loss.detach().item()),
            "singlearm_loss": float(singlearm_loss.detach().item()),
            "twoarm_residual_rms": float(twoarm_delta.detach().square().mean().sqrt().item()),
            "singlearm_residual_rms": float(singlearm_delta.detach().square().mean().sqrt().item()),
            "singlearm_weight": float(singlearm_weight),
        }
        self.last_train_metrics = metrics
        if not self.training_metric_ema:
            self.training_metric_ema = dict(metrics)
        else:
            self.training_metric_ema = {
                key: 0.99 * self.training_metric_ema[key] + 0.01 * value
                for key, value in metrics.items()
            }
        return float(total_loss.detach().item()), float(grad_norm.item()), metrics

    @torch.no_grad()
    def validation_loss(
        self,
        x_GT: torch.Tensor,
        imgs_eih: torch.Tensor,
        imgs_shoulder: torch.Tensor,
        base_model,
        cfg_drop_prob: float = 0.2,
        generator: torch.Generator | None = None,
        sigma=None,
    ) -> float:
        """Coordination-head denoising loss without optimizer or EMA updates.

        If `generator` is provided, sigma/eps/cfg_mask sampling is reproducible.
        If `sigma` is provided, it overrides sigma sampling (used for stratified
        validation over a fixed log-spaced sigma grid).
        """
        was_training = self.F.training
        self.F.eval()

        B = x_GT.shape[0]
        device = x_GT.device

        if sigma is None:
            sigma = base_model.sample_noise_distribution(B, generator=generator)
        eps = torch.randn(
            x_GT.shape, dtype=x_GT.dtype, device=device, generator=generator
        ) * sigma
        x_t = x_GT + eps
        cfg_mask = (
            torch.rand(B, device=device, generator=generator) < cfg_drop_prob
        )

        enc_out = base_model.F_ema.forward_encoder(imgs_eih, imgs_shoulder)
        null_token = base_model.F_ema.null_token.expand(B, -1, -1)
        enc_out_masked = [
            torch.where(cfg_mask[:, None, None], null_token, out)
            for out in enc_out
        ]
        D_base = base_model._D_from_enc(x_t, sigma, enc_out_masked, use_ema=True)

        delta = self.forward_residual(
            x_t, sigma, enc_out, use_ema=False, cfg_mask=cfg_mask,
            d_base=D_base, imgs_eih=imgs_eih, imgs_shoulder=imgs_shoulder,
        )
        D_pred = D_base + delta

        weights = base_model.loss_weighting(sigma)
        loss = (weights * (D_pred - x_GT) ** 2).mean()

        if was_training:
            self.F.train()
        return loss.item()

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path, metadata=None):
        checkpoint_metadata = dict(self.checkpoint_metadata)
        if metadata:
            checkpoint_metadata.update(metadata)
        if self.last_train_metrics:
            checkpoint_metadata["last_train_metrics"] = dict(self.last_train_metrics)
        if self.training_metric_ema:
            checkpoint_metadata["training_metric_ema"] = dict(self.training_metric_ema)
        state = {
            "model": self.F.state_dict(),
            "model_ema": self.F_ema.state_dict(),
            "coord_architecture": coordination_architecture_name(self.plan_memory_mode),
            "decoder_execution": self.decoder_execution,
            "decoder_conditioning": self.decoder_conditioning,
            "plan_memory_mode": self.plan_memory_mode,
            "frame_offsets": list(self.frame_offsets),
            "metadata": checkpoint_metadata,
        }
        torch.save(state, path)
        print(f"Saved image coordination head to {path}")

    def load(self, path, device=None):
        if device is None:
            device = next(self.parameters()).device

        checkpoint = torch.load(path, map_location=device, weights_only=True)
        require_checkpoint_frame_offsets(
            checkpoint, self.frame_offsets, artifact="Coordination checkpoint",
        )
        saved_side_offsets = normalize_frame_offsets(
            checkpoint.get("side_frame_offsets", checkpoint.get("frame_offsets", (0,)))
        )
        if saved_side_offsets != self.frame_offsets:
            raise ValueError(
                "Independent head camera history is retired: "
                f"checkpoint side_frame_offsets={saved_side_offsets}, "
                f"frame_offsets={self.frame_offsets}. Use the original experiment snapshot "
                "to reproduce history variant B."
            )
        saved_execution = checkpoint.get("decoder_execution")
        saved_conditioning = checkpoint.get(
            "decoder_conditioning", DECODER_CONDITIONING_POOLED
        )
        saved_plan_memory = checkpoint.get("plan_memory_mode", PLAN_MEMORY_ADDITIVE)
        expected_architecture = getattr(
            self, "architecture", coordination_architecture_name(self.plan_memory_mode)
        )
        saved_architecture = checkpoint.get("coord_architecture")
        if saved_architecture is None:
            if expected_architecture != "coord_image_residual_v1":
                raise ValueError(
                    "Coordination checkpoint has no plan-memory architecture tag; "
                    "legacy vanilla checkpoints are incompatible with this model"
                )
        elif saved_architecture != expected_architecture:
            raise ValueError(
                "Coordination checkpoint architecture mismatch: "
                f"checkpoint={saved_architecture!r}, model={expected_architecture!r}"
            )
        if saved_execution is not None and saved_execution != self.decoder_execution:
            raise ValueError(
                "Coordination checkpoint decoder execution mismatch: "
                f"checkpoint={saved_execution!r}, model={self.decoder_execution!r}"
            )
        if saved_conditioning != self.decoder_conditioning:
            raise ValueError(
                "Coordination checkpoint decoder conditioning mismatch: "
                f"checkpoint={saved_conditioning!r}, "
                f"model={self.decoder_conditioning!r}"
            )
        if saved_plan_memory != self.plan_memory_mode:
            raise ValueError(
                "Coordination checkpoint plan-memory architecture mismatch: "
                f"checkpoint={saved_plan_memory!r}, model={self.plan_memory_mode!r}"
            )

        if "model" in checkpoint:
            missing, unexpected = self.F.load_state_dict(checkpoint["model"], strict=False)
            print(f"Loaded image coordination head from {path}")
        else:
            missing, unexpected = self.F.load_state_dict(checkpoint, strict=False)
            print(f"Loaded image coordination head from {path} (legacy format)")
        if missing:
            print(f"  Missing keys (new params, zero-initialized): {missing}")
        if unexpected:
            print(f"  Unexpected keys: {unexpected}")

        if "model_ema" in checkpoint:
            missing_ema, _ = self.F_ema.load_state_dict(checkpoint["model_ema"], strict=False)
            print("  Loaded EMA weights")
            if missing_ema:
                print(f"  EMA missing keys (zero-initialized): {missing_ema}")
        else:
            self.F_ema.load_state_dict(self.F.state_dict())
            print("  No EMA weights found, copied from training model")

        return True
