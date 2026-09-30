#!/usr/bin/env python3
"""Co-Diff single-arm inference node.

Subscribes to the top camera. Samples a (horizon, 7) action chunk from the
trained policy, denormalizes it, and publishes it as a Float32MultiArray to
/codiff/actions for the executor to consume. Sampling is driven either by a
fixed-rate timer (default) or one-chunk-per-request (--on-request).

Layout of the published Float32MultiArray:
    dim[0] = {label: "horizon", size: H, stride: H*7}
    dim[1] = {label: "pose7",   size: 7, stride: 7}
    data   = flat list of H*7 floats (row-major: pose 0, pose 1, ...)

Each pose is [x_mm, y_mm, z_mm, roll, pitch, yaw, gripper] (matches RobotMsg.pose
convention used by the data collected via record_node).
"""

import argparse
import os
import pickle as pkl
import sys
import time

import cv2
import numpy as np
import torch
from torchvision.transforms import functional as TF

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Empty, Float32MultiArray, MultiArrayDimension
from xarm_msgs.msg import RobotMsg

CO_DIFF_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), ".."
))
if CO_DIFF_ROOT not in sys.path:
    sys.path.insert(0, CO_DIFF_ROOT)

from src.image_diffusion import ImageConditional_ODE  # noqa: E402




class CoDiffInferenceNode(Node):
    def __init__(self, args):
        super().__init__("xarm_codiff_inference_node")
        self.bridge = CvBridge()
        self.device = torch.device(args.device)
        self.cfg_w = args.cfg_w
        self.anchor = bool(args.anchor)
        self.anchor_gripper = bool(args.anchor_gripper)
        self.publish_from_index = int(args.publish_from_index)
        self.prescale_hw = tuple(args.prescale_hw) if args.prescale_hw else None
        self.current_pose = None  # latest TCP from /xarm/robot_states (6,) numpy
        self.fresh_frame_on_request = bool(args.fresh_frame_on_request)
        self.fresh_frame_timeout = args.fresh_frame_timeout
        self._pending_frame_request = None
        self._image_stamp_ns = 0
        self._sample_request_ns = 0
        self.dump_dir = args.dump_frames
        self._dump_i = 0
        if self.dump_dir:
            os.makedirs(self.dump_dir, exist_ok=True)

        with open(args.stats, "rb") as f:
            stats = pkl.load(f)
        self.action_mean = np.asarray(stats["action_mean"], dtype=np.float32)
        self.action_std = np.asarray(stats["action_std"], dtype=np.float32)
        self.horizon = int(stats["horizon"])
        self.get_logger().info(
            f"Loaded stats: horizon={self.horizon}, mean={self.action_mean.round(3)}, std={self.action_std.round(3)}"
        )

        self.model = ImageConditional_ODE(
            x_dim=7,
            sigma_data=float(stats["sigma_data"]),
            d_model=int(stats["d_model"]),
            n_heads=int(stats["n_heads"]),
            depth=int(stats["depth"]),
            dim_feedforward=int(stats["dim_feedforward"]),
            horizon=self.horizon,
            device=self.device,
            N=args.n_steps,
            lr=1e-4,  # unused at inference
            cfg_drop_prob=float(stats["cfg_drop_prob"]),
            num_cameras=int(stats.get("num_cameras", 1)),
            backbone=str(stats.get("backbone", "resnet18")),
        )
        if not self.model.load(args.checkpoint):
            raise RuntimeError(f"Failed to load checkpoint {args.checkpoint}")
        self.get_logger().info(f"Loaded checkpoint: {args.checkpoint}")

        self.latest_image = None
        self.create_subscription(Image, args.image_topic, self._image_cb, 10)
        if self.anchor:
            self.create_subscription(RobotMsg, "/xarm/robot_states", self._robot_state_cb, 10)
        self.action_pub = self.create_publisher(Float32MultiArray, args.actions_topic, 10)

        # Two drive modes:
        #   free-running (default) -- a timer samples at --rate-hz regardless of
        #     whether anyone consumed the last chunk. Pairs with the servo-mode
        #     executor, which just tracks the newest chunk.
        #   on-request -- sample exactly once per Empty message on --request-topic.
        #     Pairs with xarm_codiff_deploy_executor.py, which blocks on each pose
        #     and would otherwise be fed chunks faster than it can execute them.
        self.on_request = bool(args.on_request)
        if self.on_request:
            self.create_subscription(Empty, args.request_topic, self._request_cb, 10)
            if self.fresh_frame_on_request:
                self.create_timer(0.05, self._expire_frame_request)
                self.get_logger().info(
                    f"Fresh-frame requests enabled: timeout={self.fresh_frame_timeout}s, "
                    f"actions_topic={args.actions_topic}")
            self.get_logger().info(
                f"Inference ON REQUEST via {args.request_topic}, image_topic={args.image_topic}, "
                f"anchor={self.anchor} (gripper={self.anchor_gripper}), "
                f"publish_from_index={self.publish_from_index}"
            )
        else:
            self.inference_period = 1.0 / float(args.rate_hz)
            self.create_timer(self.inference_period, self._tick)
            self.get_logger().info(
                f"Inference @ {args.rate_hz:.1f} Hz, image_topic={args.image_topic}, "
                f"anchor={self.anchor} (gripper={self.anchor_gripper}), "
                f"publish_from_index={self.publish_from_index}"
            )

    def _request_cb(self, _msg: Empty):
        if self.fresh_frame_on_request:
            if self._pending_frame_request is None:
                self._pending_frame_request = (
                    self.get_clock().now().nanoseconds,
                    time.monotonic() + self.fresh_frame_timeout)
        else:
            self._tick()

    def _expire_frame_request(self):
        if (self._pending_frame_request is not None
                and time.monotonic() >= self._pending_frame_request[1]):
            self._pending_frame_request = None
            self.get_logger().error("fresh-frame timeout: no newly timestamped image; no chunk published")

    def _robot_state_cb(self, msg: RobotMsg):
        # msg.pose is [x_mm, y_mm, z_mm, roll, pitch, yaw]
        self.current_pose = np.asarray(msg.pose, dtype=np.float32)

    def _image_cb(self, msg: Image):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().error(f"image conversion failed: {exc}")
            return
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self.latest_image = rgb
        self._image_stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        if self.fresh_frame_on_request:
            self._expire_frame_request()
            if self._pending_frame_request is not None:
                requested_ns, _ = self._pending_frame_request
                now_ns = self.get_clock().now().nanoseconds
                # Reject queued/pre-request, zero-stamp and future-dated frames.
                if requested_ns < self._image_stamp_ns <= now_ns:
                    self._pending_frame_request = None
                    self._sample_request_ns = requested_ns
                    self.get_logger().info(
                        f"fresh frame: stamp_ns={self._image_stamp_ns} request_ns={requested_ns}")
                    self._tick()

    def _preprocess(self, rgb: np.ndarray) -> torch.Tensor:
        # rgb: H x W x 3 uint8 -> 1 x 3 x 128 x 128 float in [0,1]
        # With --prescale-hw H W, mirror the new training pipeline:
        #   cv2.resize(orig -> HxW, INTER_AREA)  [matches the conversion step]
        #   torchvision.F.resize(HxW -> 128x128, antialias=True)  [matches dataset eval]
        # Without --prescale-hw, fall back to a single cv2 resize for backward
        # compatibility with checkpoints trained on full-resolution pkls.
        if self.prescale_hw is not None:
            h, w = self.prescale_hw
            rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
            t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            t = TF.resize(t, [128, 128], antialias=True)
        else:
            resized = cv2.resize(rgb, (128, 128), interpolation=cv2.INTER_AREA)
            t = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        return t.unsqueeze(0).to(self.device)

    def _sample_anchored(self, img_shoulder: torch.Tensor, anchor_norm: np.ndarray,
                          anchor_dims: int) -> torch.Tensor:
        """Diffusion sampling with x[:, 0, :anchor_dims] pinned to the current pose
        at every denoising step.

        Re-implements ImageConditional_ODE.sample() with the inpainting constraint.
        anchor_norm: (7,) normalized; only the first ``anchor_dims`` entries are used.

        Trick: at every Euler step, force x[:, 0, :anchor_dims] = anchor BEFORE
        the model predicts D. This way pred[1..H-1] is denoised consistent with
        "the chunk starts at the current arm pose."
        """
        model = self.model
        n_samples = 1
        traj_len = self.horizon
        w = self.cfg_w
        device = model.device

        anchor_t = torch.from_numpy(anchor_norm.astype(np.float32)).to(device)  # (7,)

        enc_cond = model.F_ema.forward_encoder(None, img_shoulder)
        null_token = model.F_ema.null_token.expand(n_samples, -1, -1)
        enc_uncond = [null_token for _ in enc_cond]

        x = torch.randn(
            (n_samples, traj_len, model.x_dim), device=device
        ) * model.sigma_s[0] * model.scale_s[0]

        for i in range(model.N):
            # Pin the first pose's anchored dims at every step.
            x[:, 0, :anchor_dims] = anchor_t[:anchor_dims]

            sigma_i = torch.ones((n_samples, 1, 1), device=device) * model.sigma_s[i]
            D_cond = model._D_from_enc(x / model.scale_s[i], sigma_i, enc_cond)
            D_uncond = model._D_from_enc(x / model.scale_s[i], sigma_i, enc_uncond)
            D = w * D_cond + (1 - w) * D_uncond

            delta = model.coeff1[i] * x - model.coeff2[i] * D
            dt = model.t_s[i] - model.t_s[i + 1] if i != model.N - 1 else model.t_s[i]
            x = x - delta * dt

        # Final pin so the read-out x[:, 0] is exactly the anchor.
        x[:, 0, :anchor_dims] = anchor_t[:anchor_dims]
        return x

    def _tick(self):
        if self.latest_image is None:
            self.get_logger().warn("no camera frame yet — skipping inference")
            return
        img = self._preprocess(self.latest_image)

        t0 = time.time()
        if self.anchor:
            if self.current_pose is None:
                self.get_logger().warn("anchor mode but no /xarm/robot_states yet — skipping")
                return
            anchor_dims = 7 if self.anchor_gripper else 6
            # Build a 7-dim raw anchor; gripper slot is 0 if not anchored (unused since
            # we only pin the first 6 dims).
            anchor_raw = np.zeros(7, dtype=np.float32)
            anchor_raw[:6] = self.current_pose
            if self.anchor_gripper:
                # Hold last commanded gripper if known, else mean from training data.
                anchor_raw[6] = float(self.action_mean[6])
            anchor_norm = (anchor_raw - self.action_mean) / self.action_std
            with torch.no_grad():
                chunk_norm = self._sample_anchored(img, anchor_norm, anchor_dims=anchor_dims)
        else:
            with torch.no_grad():
                chunk_norm = self.model.sample(
                    imgs_eih=None,
                    imgs_shoulder=img,
                    traj_len=self.horizon,
                    n_samples=1,
                    w=self.cfg_w,
                )  # (1, H, 7)
        infer_ms = (time.time() - t0) * 1000.0

        chunk = chunk_norm.squeeze(0).cpu().numpy()  # (H, 7)
        chunk = chunk * self.action_std + self.action_mean
        chunk = chunk.astype(np.float32)

        if self.dump_dir:
            # What the policy saw and what it answered, for offline replay —
            # same fields as the coordination node's dumps.
            np.savez_compressed(
                os.path.join(self.dump_dir, f"tick_{self._dump_i:05d}.npz"),
                rgb=self.latest_image,
                pose=(self.current_pose if self.current_pose is not None
                      else np.zeros(6, np.float32)),
                chunk=chunk, t_wall=time.time(),
                image_stamp_ns=self._image_stamp_ns, request_stamp_ns=self._sample_request_ns,
            )
            self._dump_i += 1

        # When anchoring, pred[0] is just the current pose echoed back — useless to
        # send to the executor. Strip it (or strip more via --publish-from-index).
        start = self.publish_from_index if self.publish_from_index > 0 else (1 if self.anchor else 0)
        published = chunk[start:]

        msg = Float32MultiArray()
        msg.layout.dim = [
            MultiArrayDimension(label="horizon", size=published.shape[0], stride=published.size),
            MultiArrayDimension(label="pose7", size=published.shape[1], stride=published.shape[1]),
        ]
        msg.data = published.flatten().tolist()
        self.action_pub.publish(msg)
        self.get_logger().info(
            f"published shape={published.shape} infer={infer_ms:.0f} ms "
            f"anchor={'on' if self.anchor else 'off'} "
            f"pose[start]={published[0].round(2)}"
        )


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--stats", required=True)
    p.add_argument("--image-topic", default="/camera_top/camera_top/color/image_raw")
    p.add_argument("--rate-hz", type=float, default=5.0,
                   help="how often to run inference (Hz). Diffusion sampling is not free. "
                        "Ignored under --on-request.")
    p.add_argument("--on-request", action="store_true",
                   help="sample once per std_msgs/Empty on --request-topic instead of on a "
                        "timer. Use with xarm_codiff_deploy_executor.py so every chunk is "
                        "generated from the pose the arm actually reached.")
    p.add_argument("--request-topic", default="/codiff/request",
                   help="topic the executor pings to request one chunk (--on-request only)")
    p.add_argument("--actions-topic", default="/codiff/actions")
    p.add_argument("--fresh-frame-on-request", action="store_true",
                   help="wait for an image captured after each request before sampling")
    p.add_argument("--fresh-frame-timeout", type=float, default=1.0)
    p.add_argument("--n-steps", type=int, default=50, help="denoising steps")
    p.add_argument("--cfg-w", type=float, default=1.5, help="CFG guidance weight")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--anchor", action="store_true",
                   help="diffusion inpainting: pin x[:, 0, :] to the current arm pose at every "
                        "denoising step. Anchors the chunk in the present, mimicking a "
                        "state-conditioned model. Subscribes to /xarm/robot_states.")
    p.add_argument("--anchor-gripper", action="store_true",
                   help="also anchor the gripper dim (otherwise only the 6 TCP dims are pinned).")
    p.add_argument("--publish-from-index", type=int, default=0,
                   help="strip leading poses before publishing. 0 = auto (strip pred[0] when "
                        "anchoring, otherwise nothing).")
    p.add_argument("--dump-frames", default=None, metavar="DIR",
                   help="save each tick's raw camera frame, measured pose, and full "
                        "denormalized chunk as tick_NNNNN.npz under DIR, exactly as the "
                        "coordination node does — the offline-replay debugging workflow")
    p.add_argument("--prescale-hw", type=int, nargs=2, metavar=("H", "W"), default=None,
                   help="Intermediate prescale size H W for two-stage resize that mirrors "
                        "training. Use when the model was trained on data pre-resized at "
                        "conversion time (e.g. --prescale-hw 192 256). Off by default for "
                        "backward compatibility with checkpoints trained on full-res pkls.")
    args = p.parse_args()
    if args.fresh_frame_on_request and not args.on_request:
        p.error("--fresh-frame-on-request requires --on-request")
    if not np.isfinite(args.fresh_frame_timeout) or args.fresh_frame_timeout <= 0:
        p.error("--fresh-frame-timeout must be finite and positive")
    return args


def main():
    args = parse_args()
    rclpy.init()
    node = CoDiffInferenceNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
