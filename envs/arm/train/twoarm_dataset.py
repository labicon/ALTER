"""Shared dataset for two-arm fine-tuning (coordination head, OFT, LoRA).

Loads two-arm rollout pkls and produces samples for BOTH agents.
Actions are normalized with the BASE MODEL's action stats so fine-tuned
models operate in the same action space as the base.
"""

from __future__ import annotations

import os
import pickle as pkl

import numpy as np
import torch
from torch.utils.data import Dataset

from envs.arm.train import _train_utils
from envs.arm.train.temporal_images import prepare_temporal_views
from src.temporal import normalize_frame_offsets


class TwoArmE2EImageDataset(Dataset):
    """Dataset for two-arm fine-tuning with raw images.

    Each sample returns:
        img_eih:      (3, 128, 128) float32 in [0, 1]
        img_shoulder: (3, 128, 128) float32 in [0, 1]
        action_chunk: (horizon, 7) float32 normalized actions
    """

    def __init__(
        self,
        rollout_dir: str,
        base_action_mean: np.ndarray,
        base_action_std: np.ndarray,
        horizon: int = 20,
        augment: bool = True,
        modes: list[int] | None = [2, 3, 4, 5],
        num_arms: int = 2,
        data_fraction: float = 1.0,
        data_subset_seed: int = 0,
        rollout_files: list[str] | None = None,
        frame_offsets=(0,),
        other_arm_cutout: dict | None = None,
        other_arm_cutout_prob: float = 0.0,
    ):
        self.horizon = horizon
        self.augment = augment
        # {agent_idx: (x0, y0, x1, y1) fractions}: with probability
        # other_arm_cutout_prob, that region of the agent's own camera frame is
        # replaced by the mean colour of the rest of the frame. Used to stop
        # the coordination head keying on the OTHER arm's posture (a spurious
        # cue that the live arm, with a different IK elbow solution, never
        # reproduces) and make it use the object state instead.
        self.other_arm_cutout = dict(other_arm_cutout or {})
        self.other_arm_cutout_prob = float(other_arm_cutout_prob)
        self.trajectory_agent_idx = []
        self.action_mean = base_action_mean
        self.action_std = base_action_std
        self.num_arms = num_arms
        self.data_fraction = float(data_fraction)
        self.data_subset_seed = int(data_subset_seed)
        self.frame_offsets = normalize_frame_offsets(frame_offsets)

        pkl_files = sorted(
            os.path.abspath(os.path.join(root, fname))
            for root, _, files in os.walk(rollout_dir)
            for fname in files
            if fname.endswith(".pkl")
        )
        if modes is not None:
            mode_strs = {f"mode{m}" for m in modes}
            pkl_files = [
                path
                for path in pkl_files
                if any(ms in os.path.basename(path) for ms in mode_strs)
            ]
        if not pkl_files:
            raise FileNotFoundError(f"No pkl files found in {rollout_dir}")
        self.available_pkl_files = list(pkl_files)

        if rollout_files is not None:
            requested = {
                os.path.abspath(
                    path if os.path.isabs(path) else os.path.join(rollout_dir, path)
                )
                for path in rollout_files
            }
            missing = sorted(requested.difference(self.available_pkl_files))
            if missing:
                raise FileNotFoundError(
                    f"Requested rollout files are not available in {rollout_dir}: {missing}"
                )
            pkl_files = sorted([path for path in pkl_files if path in requested])
            if not pkl_files:
                raise FileNotFoundError("No rollout files left after explicit rollout_files filter.")
        else:
            if not (0.0 < self.data_fraction <= 1.0):
                raise ValueError(f"data_fraction must be in (0, 1], got {self.data_fraction}")
            if self.data_fraction < 1.0:
                pkl_files = self._subsample_rollout_files(
                    pkl_files, self.data_fraction, self.data_subset_seed
                )
        self.selected_pkl_files = list(pkl_files)

        print(
            f"Found {len(self.available_pkl_files)} rollout files in {rollout_dir}; "
            f"using {len(self.selected_pkl_files)} "
            f"(fraction={self.data_fraction}, subset_seed={self.data_subset_seed})"
        )

        all_imgs_eih = []
        all_imgs_shoulder = []
        all_actions = []
        self.index_map = []  # (traj_idx, timestep)
        # Per-sample importance weight, aligned with index_map. 1.0 unless the
        # rollout carries an optional ``frame_weights`` array of shape
        # (T, num_arms): a per-arm, per-frame multiplier used by
        # --sampling-policy weighted to oversample e.g. the frames whose action
        # chunk contains a release. Ignored by every other policy.
        self.sample_weights = []
        self.trajectory_group_keys = []
        self.trajectory_rollout_ids = []

        for path in pkl_files:
            group_key = f"multiarm/{_train_utils.mode_key(os.path.basename(path))}"
            rollout_id = os.path.abspath(path)
            with open(path, "rb") as f:
                rollout = pkl.load(f)

            actions_full = np.array(rollout["actions"], dtype=np.float32)

            # Process each agent from each rollout
            for agent_idx in range(self.num_arms):
                eih_key = f"camera_obs{agent_idx}"
                shoulder_key = f"camera_obs_shoulder{agent_idx}"
                action_slice = slice(agent_idx * 7, (agent_idx + 1) * 7)

                if eih_key not in rollout:
                    continue
                if shoulder_key not in rollout:
                    continue

                imgs_eih = rollout[eih_key]
                imgs_shoulder = rollout[shoulder_key]
                actions_7 = actions_full[:, action_slice]

                T = min(len(actions_7), len(imgs_eih), len(imgs_shoulder))
                if T < horizon:
                    continue

                traj_idx = len(all_imgs_eih)
                all_imgs_eih.append(np.asarray(imgs_eih[:T], dtype=np.uint8))
                all_imgs_shoulder.append(np.asarray(imgs_shoulder[:T], dtype=np.uint8))
                all_actions.append(actions_7[:T])
                self.trajectory_group_keys.append(group_key)
                self.trajectory_rollout_ids.append(rollout_id)
                self.trajectory_agent_idx.append(agent_idx)

                fw = rollout.get("frame_weights")
                if fw is not None:
                    fw = np.asarray(fw, dtype=np.float32)
                    w_arm = fw[:T, agent_idx] if fw.ndim == 2 else fw[:T]
                else:
                    w_arm = np.ones(T, dtype=np.float32)
                for t in range(T):
                    self.index_map.append((traj_idx, t))
                    self.sample_weights.append(float(w_arm[t]))

        if len(all_actions) == 0:
            raise RuntimeError("No valid trajectories loaded.")

        self.all_imgs_eih = all_imgs_eih
        self.all_imgs_shoulder = all_imgs_shoulder
        self.all_actions = all_actions

        print(
            f"Loaded {len(all_actions)} agent trajectories, "
            f"{len(self.index_map)} samples, horizon={horizon}"
        )

    @staticmethod
    def _mode_key(filename: str) -> str:
        return _train_utils.mode_key(filename)

    @classmethod
    def _subsample_rollout_files(
        cls, pkl_files: list[str], data_fraction: float, seed: int
    ) -> list[str]:
        return _train_utils.subsample_rollout_files(pkl_files, data_fraction, seed)

    def _cutout(self, seq, t, box):
        """Return a copy of the frames prepare_temporal_views will read (t and
        its lookbacks) with `box` blanked by the mean of the rest of the frame."""
        H, W = seq.shape[1:3]
        x0, y0, x1, y1 = (int(box[0] * W), int(box[1] * H), int(box[2] * W), int(box[3] * H))
        lo = max(0, t - max(self.frame_offsets))
        out = np.array(seq[lo:t + 1])  # copy only the frames that will be read
        for f in out:
            keep = np.ones((H, W), dtype=bool); keep[y0:y1, x0:x1] = False
            f[y0:y1, x0:x1] = f[keep].reshape(-1, 3).mean(0).astype(np.uint8)
        # re-embed so absolute indexing by t still works
        padded = np.concatenate([seq[:lo], out], axis=0) if lo > 0 else out
        return padded

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        traj_idx, t = self.index_map[idx]

        # Action chunk with end-padding
        actions = self.all_actions[traj_idx]
        T = len(actions)
        if t + self.horizon > T:
            chunk = actions[t:]
            pad_len = self.horizon - len(chunk)
            chunk = np.concatenate([chunk, np.tile(actions[-1], (pad_len, 1))], axis=0)
        else:
            chunk = actions[t: t + self.horizon]

        # Normalize with base model stats
        chunk = (chunk - self.action_mean) / self.action_std

        imgs_eih = self.all_imgs_eih[traj_idx]
        imgs_shoulder = self.all_imgs_shoulder[traj_idx]
        box = self.other_arm_cutout.get(self.trajectory_agent_idx[traj_idx])
        if box is not None and self.other_arm_cutout_prob > 0 and np.random.rand() < self.other_arm_cutout_prob:
            imgs_shoulder = self._cutout(imgs_shoulder, t, box)
            imgs_eih = imgs_shoulder if imgs_eih is self.all_imgs_shoulder[traj_idx] else self._cutout(imgs_eih, t, box)
        img_eih, img_shoulder = prepare_temporal_views(
            imgs_eih,
            imgs_shoulder,
            t,
            self.frame_offsets,
            augment=self.augment,
        )

        return img_eih, img_shoulder, torch.FloatTensor(chunk)
