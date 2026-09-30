"""Karras/EDM diffusion wrapper for end-to-end image-conditioned policy.

Mirrors ``Conditional_ODE`` from ``src/diffusion.py`` but takes raw images
instead of pre-computed latent vectors. Key difference: during sampling the
encoder runs once and its outputs are cached; only the decoder runs per
denoising step.
"""

from __future__ import annotations

import os
import torch
import torch.nn as nn
from copy import deepcopy

from src.image_dit import ImageDiT
from src.temporal import (
    normalize_frame_offsets,
    require_checkpoint_frame_offsets,
)


class ImageConditional_ODE:
    """Karras/EDM flow-matching wrapper for ImageDiT.

    Same preconditioning, noise schedule, and Euler ODE integrator as
    ``Conditional_ODE``, but operating on images end-to-end.
    """

    def __init__(
        self,
        x_dim: int = 7,
        sigma_data: float = 1.0,
        sigma_min: float = 0.001,
        sigma_max: float = 50,
        rho: float = 7,
        p_mean: float = -1.2,
        p_std: float = 1.2,
        d_model: int = 256,
        n_heads: int = 4,
        depth: int = 3,
        dim_feedforward: int = 1024,
        horizon: int = 20,
        device: str = "cpu",
        N: int = 50,
        lr: float = 2e-4,
        cfg_drop_prob: float = 0.2,
        num_cameras: int = 2,
        backbone: str = "resnet18",
        frame_offsets=(0,),
    ):
        self.x_dim = x_dim
        self.sigma_data = sigma_data
        self.sigma_min, self.sigma_max = sigma_min, sigma_max
        self.rho, self.p_mean, self.p_std = rho, p_mean, p_std
        self.device = device
        self.cfg_drop_prob = cfg_drop_prob
        self.frame_offsets = normalize_frame_offsets(frame_offsets)

        self.num_cameras = num_cameras
        self.F = ImageDiT(
            x_dim=x_dim, d_model=d_model, n_heads=n_heads, depth=depth,
            dim_feedforward=dim_feedforward, dropout=0.1, horizon=horizon,
            num_cameras=num_cameras, backbone=backbone,
            frame_offsets=self.frame_offsets,
        ).to(device)
        self.F.train()

        self.F_ema = deepcopy(self.F).requires_grad_(False).eval()

        self.optim = torch.optim.AdamW(
            [p for p in self.F.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4
        )
        self.set_N(N)

    # ------------------------------------------------------------------
    # Noise schedule (identical to Conditional_ODE)
    # ------------------------------------------------------------------

    def ema_update(self, decay: float = 0.999):
        for p, p_ema in zip(self.F.parameters(), self.F_ema.parameters()):
            p_ema.data = decay * p_ema.data + (1 - decay) * p.data

    def set_N(self, N: int):
        self.N = N
        self.sigma_s = (
            self.sigma_max ** (1 / self.rho)
            + torch.arange(N, device=self.device) / (N - 1)
            * (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))
        ) ** self.rho
        self.t_s = self.sigma_s
        self.scale_s = torch.ones_like(self.sigma_s)
        self.dot_sigma_s = torch.ones_like(self.sigma_s)
        self.dot_scale_s = torch.zeros_like(self.sigma_s)
        self.coeff1 = self.dot_sigma_s / self.sigma_s + self.dot_scale_s / self.scale_s
        self.coeff2 = self.dot_sigma_s / self.sigma_s * self.scale_s

    # ------------------------------------------------------------------
    # Karras preconditioning (identical to Conditional_ODE)
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

    def sample_noise_distribution(self, N: int, generator: torch.Generator | None = None):
        log_sigma = torch.randn(
            (N, 1, 1), device=self.device, generator=generator
        ) * self.p_std + self.p_mean
        return log_sigma.exp()

    # ------------------------------------------------------------------
    # Denoiser D — operates on images
    # ------------------------------------------------------------------

    def D(self, x, sigma, imgs_eih, imgs_shoulder, use_ema=False, cfg_mask=None):
        """Karras denoiser: D(x, sigma) = c_skip * x + c_out * F(c_in * x, c_noise)."""
        c_skip = self.c_skip(sigma)
        c_out = self.c_out(sigma)
        c_in = self.c_in(sigma)
        c_noise = self.c_noise(sigma)

        F = self.F_ema if use_ema else self.F
        return c_skip * x + c_out * F(
            c_in * x, c_noise.squeeze(-1), imgs_eih, imgs_shoulder, cfg_mask
        )

    def _D_from_enc(self, x, sigma, enc_outputs, use_ema=True):
        """Denoiser using pre-computed encoder outputs (for fast sampling)."""
        c_skip = self.c_skip(sigma)
        c_out = self.c_out(sigma)
        c_in = self.c_in(sigma)
        c_noise = self.c_noise(sigma)

        F = self.F_ema if use_ema else self.F
        pred = F.forward_decoder(c_in * x, c_noise.squeeze(-1), enc_outputs)
        return c_skip * x + c_out * pred

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def _training_loss(self, x, imgs_eih, imgs_shoulder):
        """Return one batch's mean EDM denoising loss."""
        B = x.shape[0]
        sigma = self.sample_noise_distribution(B)
        eps = torch.randn_like(x) * sigma

        # CFG dropout mask
        cfg_mask = (torch.rand(B, device=self.device) < self.cfg_drop_prob)

        pred = self.D(x + eps, sigma, imgs_eih, imgs_shoulder, use_ema=False, cfg_mask=cfg_mask)
        return (self.loss_weighting(sigma) * (pred - x) ** 2).mean()

    def update(self, x, imgs_eih, imgs_shoulder):
        """Single training step on one batch."""
        loss = self._training_loss(x, imgs_eih, imgs_shoulder)

        self.optim.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.F.parameters(), 10.0)
        self.optim.step()
        self.ema_update()

        return loss.item(), grad_norm.item()

    def update_mixed(
        self,
        multiarm_batch,
        singlearm_batch,
        multiarm_weight: float = 1.0,
        singlearm_weight: float = 1.0,
    ):
        """Update from separately reduced multi- and single-arm batches.

        Each batch is ``(actions, eye_in_hand_images, shoulder_images)``. The
        two batch losses are averaged independently and then summed, matching
        the domain-balanced batching used by the Coord training path.
        """
        self.optim.zero_grad()
        multiarm_loss = self._training_loss(*multiarm_batch)
        (multiarm_weight * multiarm_loss).backward()
        singlearm_loss = self._training_loss(*singlearm_batch)
        (singlearm_weight * singlearm_loss).backward()
        loss = multiarm_weight * multiarm_loss.detach() + singlearm_weight * singlearm_loss.detach()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.F.parameters(), 10.0)
        self.optim.step()
        self.ema_update()

        return loss.item(), grad_norm.item(), {
            "multiarm_loss": multiarm_loss.item(),
            "singlearm_loss": singlearm_loss.item(),
        }

    @torch.no_grad()
    def validation_loss(
        self,
        x,
        imgs_eih,
        imgs_shoulder,
        cfg_drop_prob=None,
        generator: torch.Generator | None = None,
        sigma=None,
    ):
        """Diffusion denoising loss without optimizer or EMA updates.

        If `generator` is provided, all noise sampling (sigma, eps, cfg_mask) is
        drawn from it for reproducibility. If `sigma` is provided (shape (1,1,1)
        or broadcastable), it overrides the sampled sigma — useful for
        stratifying the validation loss over a fixed log-spaced grid.
        """
        was_training = self.F.training
        self.F.eval()

        B = x.shape[0]
        if sigma is None:
            sigma = self.sample_noise_distribution(B, generator=generator)
        eps = torch.randn(
            x.shape, dtype=x.dtype, device=x.device, generator=generator
        ) * sigma

        drop_prob = self.cfg_drop_prob if cfg_drop_prob is None else float(cfg_drop_prob)
        cfg_mask = (
            torch.rand(B, device=self.device, generator=generator) < drop_prob
        )

        pred = self.D(x + eps, sigma, imgs_eih, imgs_shoulder, use_ema=False, cfg_mask=cfg_mask)
        loss = (self.loss_weighting(sigma) * (pred - x) ** 2).mean()

        if was_training:
            self.F.train()
        return loss.item()

    # ------------------------------------------------------------------
    # Sampling (Euler ODE integration with CFG)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(self, imgs_eih, imgs_shoulder, traj_len, n_samples=1,
               w=1.5, N=None, x_init=None, warm_start_skip=0.75):
        """Sample action trajectories conditioned on images.

        The encoder runs once; only the decoder runs per denoising step.

        Args:
            imgs_eih:       (B, 3, 128, 128) eye-in-hand images
            imgs_shoulder:  (B, 3, 128, 128) shoulder images
            traj_len:       action horizon
            n_samples:      number of samples (should match B of images)
            w:              CFG guidance weight
            N:              number of denoising steps (overrides default)
            x_init:         warm-start trajectory
            warm_start_skip: fraction of noisy steps to skip for warm start

        Returns:
            (n_samples, traj_len, x_dim) denoised trajectories
        """
        if N is not None and N != self.N:
            self.set_N(N)

        # Cache encoder outputs for conditioned path
        enc_cond = self.F_ema.forward_encoder(imgs_eih, imgs_shoulder)

        # Null encoder outputs for unconditioned path
        null_token = self.F_ema.null_token.expand(n_samples, -1, -1)
        enc_uncond = [null_token for _ in enc_cond]

        # Initialize trajectory
        if x_init is not None:
            i_start = int(self.N * warm_start_skip)
            i_start = min(i_start, self.N - 1)
            x = x_init
        else:
            i_start = 0
            x = torch.randn(
                (n_samples, traj_len, self.x_dim), device=self.device
            ) * self.sigma_s[0] * self.scale_s[0]

        # Euler ODE integration
        for i in range(i_start, self.N):
            sigma_i = torch.ones((n_samples, 1, 1), device=self.device) * self.sigma_s[i]

            # Conditioned and unconditioned predictions
            D_cond = self._D_from_enc(x / self.scale_s[i], sigma_i, enc_cond)
            D_uncond = self._D_from_enc(x / self.scale_s[i], sigma_i, enc_uncond)

            # CFG combination
            D = w * D_cond + (1 - w) * D_uncond

            # Euler step
            delta = self.coeff1[i] * x - self.coeff2[i] * D
            dt = self.t_s[i] - self.t_s[i + 1] if i != self.N - 1 else self.t_s[i]
            x = x - delta * dt

        return x

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def save(self, path):
        state = {
            "model": self.F.state_dict(),
            "model_ema": self.F_ema.state_dict(),
            "frame_offsets": list(self.frame_offsets),
        }
        torch.save(state, path)

    def load(self, path) -> bool:
        if os.path.isfile(path):
            print(f"Loading {path}")
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
            require_checkpoint_frame_offsets(
                checkpoint,
                self.frame_offsets,
                artifact="ImageConditional_ODE checkpoint",
            )
            self.F.load_state_dict(checkpoint["model"])
            self.F_ema.load_state_dict(checkpoint["model_ema"])
            return True
        else:
            print(f"File {path} not found.")
            return False
