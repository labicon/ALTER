#!/usr/bin/env python3
"""Co-Diff coordination-head inference node.

Sibling of xarm_codiff_inference_node.py. Loads:
    1. A frozen single-arm base policy (ImageConditional_ODE)
    2. A residual coordination head (ImageCoordinationHead)
trained on top of that base by envs/arm/train/train_mixed_coordination_e2e_shoulder.py
(scripts/train_hardware_placewipe_parity.sh STAGE=coord), which writes
mixed_coord_head_*_{stepN,final}.pt plus a self-contained stats pkl that also
records the base checkpoint and base stats paths.

The sampling loop below mirrors _sample_coordinated() in
envs/arm/sample/sample_coord_twoarm_placewipe_shoulder.py; keep them in step.

Sampling re-implements ImageConditional_ODE.sample() with the head's residual
added inside the CFG-guided Euler loop:

    D_cond   = D_base_cond   + delta_cond
    D_uncond = D_base_uncond + delta_uncond
    D       = w * D_cond + (1 - w) * D_uncond

Anchoring (--anchor) pins x[:, 0, :] to the current arm pose at every step,
identical to the single-arm node.

Both arms on a two-machine setup run this same node — the coord head was
trained on agent-0 and agent-1 samples interleaved, so it generalises to
either arm based on the camera view it sees. No per-arm flag needed.

Layout of the published Float32MultiArray matches the single-arm node:
    dim[0] = horizon, dim[1] = pose7, row-major H*7 floats.

Two inference modes, as in the single-arm node:
    timer       -- sample every 1/--rate-hz seconds (servo executor).
    --on-request -- sample exactly once per std_msgs/Empty on --request-topic.
                   Required by xarm_codiff_deploy_executor.py, which pings
                   /codiff/request and waits for the chunk.
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
from std_msgs.msg import Empty, Float32, Float32MultiArray, MultiArrayDimension
from xarm_msgs.msg import RobotMsg

CO_DIFF_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), ".."
))
if CO_DIFF_ROOT not in sys.path:
    sys.path.insert(0, CO_DIFF_ROOT)

from src.image_diffusion import ImageConditional_ODE  # noqa: E402
from src.image_coordination import ImageCoordinationHead  # noqa: E402
from hardware_training.current_frame import require_current_frame_hardware




class CoDiffCoordinationInferenceNode(Node):
    def __init__(self, args):
        super().__init__("xarm_codiff_coordination_inference_node")
        self.bridge = CvBridge()
        self.device = torch.device(args.device)
        self.cfg_w = float(args.cfg_w)
        self.anchor_state = bool(args.anchor_state)
        # --anchor-state implies --anchor: it is full-state row-0 anchoring.
        self.anchor = bool(args.anchor) or self.anchor_state
        self.anchor_gripper = bool(args.anchor_gripper)
        self.measured_gripper = None  # latest value from --gripper-state-topic
        self._warned_no_gripper = False
        self.publish_from_index = int(args.publish_from_index)
        self.prescale_hw = tuple(args.prescale_hw) if args.prescale_hw else None
        self.log_residual = bool(args.log_residual)
        self._last_residual = None
        self.mask_x0 = float(args.mask_other_arm) if args.mask_other_arm is not None else None
        self.dump_dir = args.dump_frames
        self._dump_i = 0
        if self.dump_dir:
            os.makedirs(self.dump_dir, exist_ok=True)
        self.current_pose = None

        # ------------------------------------------------------------------
        # Load coord stats (self-contained — has both head and base arch).
        # ------------------------------------------------------------------
        with open(args.coord_stats, "rb") as f:
            stats = pkl.load(f)
        require_current_frame_hardware(stats)
        self.action_mean = np.asarray(stats["action_mean"], dtype=np.float32)
        self.action_std = np.asarray(stats["action_std"], dtype=np.float32)
        self.horizon = int(stats["horizon"])
        self.num_cameras = int(stats.get("num_cameras", 1))
        sigma_data = float(stats["sigma_data"])
        self.get_logger().info(
            f"Loaded coord stats: horizon={self.horizon}, num_cameras={self.num_cameras}, "
            f"mean={self.action_mean.round(3)}, std={self.action_std.round(3)}"
        )

        # ------------------------------------------------------------------
        # Frozen base policy
        # ------------------------------------------------------------------
        base_ckpt = args.base_checkpoint or stats.get("base_model_path", "")
        if not base_ckpt or not os.path.isfile(os.path.expanduser(base_ckpt)):
            raise FileNotFoundError(
                f"Base checkpoint not found: {base_ckpt}. Pass --base-checkpoint."
            )

        # Older coord stats files only saved base_d_model; fall back to the
        # base stats pkl for the rest of the base architecture.
        missing_base_fields = {"base_n_heads", "base_depth", "base_dim_feedforward"} - set(stats)
        if missing_base_fields:
            base_stats_path = args.base_stats or stats.get("base_stats_path", "")
            base_stats_path = os.path.expanduser(base_stats_path) if base_stats_path else ""
            if not base_stats_path or not os.path.isfile(base_stats_path):
                raise FileNotFoundError(
                    f"Coord stats is missing {sorted(missing_base_fields)} and base stats "
                    f"path ({base_stats_path!r}) is not available. Pass --base-stats."
                )
            with open(base_stats_path, "rb") as f:
                base_stats = pkl.load(f)
            self.get_logger().info(
                f"Coord stats missing {sorted(missing_base_fields)}; "
                f"loaded base arch from {base_stats_path}"
            )
        else:
            base_stats = stats

        self.base_model = ImageConditional_ODE(
            x_dim=7,
            sigma_data=sigma_data,
            d_model=int(stats["base_d_model"]),
            n_heads=int(base_stats.get("base_n_heads", base_stats.get("n_heads"))),
            depth=int(base_stats.get("base_depth", base_stats.get("depth"))),
            dim_feedforward=int(base_stats.get("base_dim_feedforward", base_stats.get("dim_feedforward"))),
            horizon=self.horizon,
            device=self.device,
            N=args.n_steps,
            lr=1e-4,
            cfg_drop_prob=float(stats.get("base_cfg_drop_prob",
                                          base_stats.get("cfg_drop_prob", 0.2))),
            num_cameras=self.num_cameras,
            backbone=str(stats.get("backbone", base_stats.get("backbone", "resnet18"))),
        )
        if not self.base_model.load(os.path.expanduser(base_ckpt)):
            raise RuntimeError(f"Failed to load base checkpoint {base_ckpt}")
        for p in self.base_model.F.parameters():
            p.requires_grad = False
        self.base_model.F.eval()
        self.base_model.F_ema.eval()
        self.get_logger().info(f"Loaded base checkpoint: {base_ckpt}")

        # ------------------------------------------------------------------
        # Coordination head
        # ------------------------------------------------------------------
        self.coord_head = ImageCoordinationHead(
            x_dim=7,
            base_d_model=int(stats["base_d_model"]),
            d_model=int(stats["head_d_model"]),
            n_heads=int(stats["head_n_heads"]),
            depth=int(stats["head_depth"]),
            dim_feedforward=int(stats["head_dim_feedforward"]),
            horizon=self.horizon,
            sigma_data=sigma_data,
            lr=1e-4,
            num_cameras=self.num_cameras,
            tokens_per_camera=int(stats["tokens_per_camera"]),
            d_base_drop_prob=float(stats.get("d_base_drop_prob", 0.1)),
            use_side_net=bool(stats.get("use_side_net", False)),
            side_net_input_size=int(stats.get("side_net_input_size", 128)),
            side_net_fusion=str(stats.get("side_net_fusion", "add")),
            side_net_tokens_per_camera=int(stats.get(
                "side_net_tokens_per_camera", stats["tokens_per_camera"]
            )),
            decoder_execution=str(stats.get("decoder_execution", "legacy_zip")),
            decoder_conditioning=str(stats.get("decoder_conditioning", "pooled")),
        ).to(self.device)
        if not self.coord_head.load(args.coord_checkpoint):
            raise RuntimeError(f"Failed to load coord head {args.coord_checkpoint}")
        self.coord_head.F.eval()
        self.coord_head.F_ema.eval()
        self.get_logger().info(f"Loaded coord head: {args.coord_checkpoint}")

        # ------------------------------------------------------------------
        # I/O
        # ------------------------------------------------------------------
        self.latest_image = None
        self.fresh_frame_on_request = bool(args.fresh_frame_on_request)
        self.fresh_frame_timeout = args.fresh_frame_timeout
        self._pending_frame_request = None
        self._image_stamp_ns = 0
        self._sample_request_ns = 0
        self.create_subscription(Image, args.image_topic, self._image_cb, 10)
        if self.anchor:
            self.create_subscription(RobotMsg, "/xarm/robot_states", self._robot_state_cb, 10)
        if self.anchor_state:
            self.create_subscription(Float32, args.gripper_state_topic,
                                     self._gripper_state_cb, 10)
        self.action_pub = self.create_publisher(Float32MultiArray, args.actions_topic, 10)

        self.on_request = bool(args.on_request)
        if self.on_request:
            self.create_subscription(Empty, args.request_topic, self._request_cb, 10)
            if self.fresh_frame_on_request:
                self.create_timer(0.05, self._expire_frame_request)
                self.get_logger().info(
                    f"Fresh-frame requests enabled: timeout={self.fresh_frame_timeout}s, "
                    f"actions_topic={args.actions_topic}")
            self.get_logger().info(
                f"Coord inference ON REQUEST via {args.request_topic}, "
                f"image_topic={args.image_topic}, cfg_w={self.cfg_w}, "
                f"anchor={self.anchor} (gripper={self.anchor_gripper}), "
                f"prescale_hw={self.prescale_hw}"
            )
        else:
            self.create_timer(1.0 / float(args.rate_hz), self._tick)
            self.get_logger().info(
                f"Coord inference @ {args.rate_hz:.1f} Hz, image_topic={args.image_topic}, "
                f"cfg_w={self.cfg_w}, anchor={self.anchor} (gripper={self.anchor_gripper}), "
                f"prescale_hw={self.prescale_hw}"
            )

    # ------------------------------------------------------------------
    # callbacks
    # ------------------------------------------------------------------
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
            self.get_logger().error(
                "fresh-frame timeout: no newly timestamped image; no chunk published")

    def _robot_state_cb(self, msg: RobotMsg):
        self.current_pose = np.asarray(msg.pose, dtype=np.float32)

    def _gripper_state_cb(self, msg: Float32):
        self.measured_gripper = float(msg.data)

    def _image_cb(self, msg: Image):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().error(f"image conversion failed: {exc}")
            return
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self.latest_image = rgb
        self._image_stamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        if self.fresh_frame_on_request:
            self._expire_frame_request()
        if self.fresh_frame_on_request and self._pending_frame_request is not None:
            requested_ns, _ = self._pending_frame_request
            now_ns = self.get_clock().now().nanoseconds
            if requested_ns < self._image_stamp_ns <= now_ns:
                self._pending_frame_request = None
                self._sample_request_ns = requested_ns
                self.get_logger().info(
                    f"fresh frame: stamp_ns={self._image_stamp_ns} request_ns={requested_ns}")
                self._tick()

    # ------------------------------------------------------------------
    # preprocessing (mirrors single-arm inference node)
    # ------------------------------------------------------------------
    def _preprocess(self, rgb: np.ndarray) -> torch.Tensor:
        if self.mask_x0 is not None:
            # Blank the other arm's region with the mean colour of the rest of
            # the frame (in-distribution "table"), so the head decides from the
            # basket/lid instead of the other arm's posture. See
            # TwoArmE2EImageDataset other_arm_cutout for the training-time twin.
            x0 = int(self.mask_x0 * rgb.shape[1])
            rgb = rgb.copy()
            rgb[:, x0:] = rgb[:, :x0].reshape(-1, 3).mean(0).astype(np.uint8)
        if self.prescale_hw is not None:
            h, w = self.prescale_hw
            rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
            t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            t = TF.resize(t, [128, 128], antialias=True)
        else:
            resized = cv2.resize(rgb, (128, 128), interpolation=cv2.INTER_AREA)
            t = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        return t.unsqueeze(0).to(self.device)

    # ------------------------------------------------------------------
    # Coord-head sampling: base D_base + head delta inside the CFG Euler loop.
    # ------------------------------------------------------------------
    def _sample(self, img_shoulder: torch.Tensor,
                anchor_norm: np.ndarray = None,
                anchor_dims: int = 0) -> torch.Tensor:
        base = self.base_model
        head = self.coord_head
        device = base.device
        n_samples = 1
        traj_len = self.horizon
        w = self.cfg_w

        anchor_t = None
        if anchor_norm is not None and anchor_dims > 0:
            anchor_t = torch.from_numpy(anchor_norm.astype(np.float32)).to(device)

        # Real encoder outputs + null tokens for the BASE CFG branch.
        enc_cond = base.F_ema.forward_encoder(None, img_shoulder)
        null_token = base.F_ema.null_token.expand(n_samples, -1, -1)
        enc_uncond = [null_token for _ in enc_cond]

        # CFG mask for the HEAD: head applies its own null masking internally.
        cfg_false = torch.zeros(n_samples, dtype=torch.bool, device=device)
        cfg_true = torch.ones(n_samples, dtype=torch.bool, device=device)

        x = torch.randn(
            (n_samples, traj_len, base.x_dim), device=device
        ) * base.sigma_s[0] * base.scale_s[0]

        res_acc = []
        for i in range(base.N):
            if anchor_t is not None:
                x[:, 0, :anchor_dims] = anchor_t[:anchor_dims]

            sigma_i = torch.ones((n_samples, 1, 1), device=device) * base.sigma_s[i]
            x_in = x / base.scale_s[i]

            # Base denoising for the two CFG branches.
            D_base_cond = base._D_from_enc(x_in, sigma_i, enc_cond)
            D_base_uncond = base._D_from_enc(x_in, sigma_i, enc_uncond)

            # Head residual for each branch (pass real enc_cond; head masks
            # internally via cfg_mask). The head applies its own Karras c_in
            # to x, so it takes the UNSCALED x -- exactly as in the canonical
            # sampler and the trainer. scale_s is identically 1 today, which is
            # why x_in used to work here; do not reintroduce it.
            delta_cond = head.forward_residual(
                x, sigma_i, enc_cond, use_ema=True,
                cfg_mask=cfg_false, d_base=D_base_cond,
                imgs_eih=None, imgs_shoulder=img_shoulder,
            )
            delta_uncond = head.forward_residual(
                x, sigma_i, enc_cond, use_ema=True,
                cfg_mask=cfg_true, d_base=D_base_uncond,
                imgs_eih=None, imgs_shoulder=img_shoulder,
            )

            if self.log_residual:
                # normalized-action units; the trainer reported ~0.008 on
                # single-arm scenes and ~0.33 on two-arm scenes.
                r = float(delta_cond.pow(2).mean().sqrt())
                res_acc.append(r)

            D_cond = D_base_cond + delta_cond
            D_uncond = D_base_uncond + delta_uncond
            D = w * D_cond + (1 - w) * D_uncond

            delta_step = base.coeff1[i] * x - base.coeff2[i] * D
            dt = base.t_s[i] - base.t_s[i + 1] if i != base.N - 1 else base.t_s[i]
            x = x - delta_step * dt

        if anchor_t is not None:
            x[:, 0, :anchor_dims] = anchor_t[:anchor_dims]
        if res_acc:
            self._last_residual = float(np.mean(res_acc))
        return x

    # ------------------------------------------------------------------
    # tick
    # ------------------------------------------------------------------
    def _tick(self):
        if self.latest_image is None:
            self.get_logger().warn("no camera frame yet — skipping inference")
            return
        rgb = self.latest_image
        img = self._preprocess(rgb)

        anchor_norm = None
        anchor_dims = 0
        if self.anchor:
            if self.current_pose is None:
                self.get_logger().warn("anchor mode but no /xarm/robot_states yet — skipping")
                return
            anchor_dims = 7 if self.anchor_gripper else 6
            anchor_raw = np.zeros(7, dtype=np.float32)
            anchor_raw[:6] = self.current_pose
            if self.anchor_gripper:
                anchor_raw[6] = float(self.action_mean[6])
            if self.anchor_state:
                # Full-state anchoring: row 0 carries the MEASURED gripper,
                # matching training semantics (data rows hold the true state).
                if self.measured_gripper is not None:
                    anchor_raw[6] = self.measured_gripper
                    anchor_dims = 7
                elif not self._warned_no_gripper:
                    self.get_logger().warn(
                        "--anchor-state set but no gripper state received yet "
                        "on the gripper-state topic; anchoring pose only")
                    self._warned_no_gripper = True
            anchor_norm = (anchor_raw - self.action_mean) / self.action_std

        t0 = time.time()
        with torch.no_grad():
            chunk_norm = self._sample(img, anchor_norm, anchor_dims)
        infer_ms = (time.time() - t0) * 1000.0

        chunk = chunk_norm.squeeze(0).cpu().numpy()
        chunk = chunk * self.action_std + self.action_mean
        chunk = chunk.astype(np.float32)

        start = (self.publish_from_index if self.publish_from_index > 0
                 else (1 if self.anchor else 0))
        published = chunk[start:]

        if self.dump_dir:
            # What the policy saw and what it answered, for offline replay.
            # rgb is the raw camera frame (pre-resize) so any preprocessing
            # can be re-run on it exactly as the node does.
            np.savez_compressed(
                os.path.join(self.dump_dir, f"tick_{self._dump_i:05d}.npz"),
                rgb=rgb, pose=(self.current_pose if self.current_pose is not None
                                             else np.zeros(6, np.float32)),
                chunk=chunk, residual_rms=(self._last_residual or -1.0),
                t_wall=time.time(),
                image_stamp_ns=self._image_stamp_ns,
                request_stamp_ns=self._sample_request_ns,
                measured_gripper=(self.measured_gripper
                                  if self.measured_gripper is not None else -1.0),
            )
            self._dump_i += 1

        msg = Float32MultiArray()
        msg.layout.dim = [
            MultiArrayDimension(label="horizon", size=published.shape[0], stride=published.size),
            MultiArrayDimension(label="pose7", size=published.shape[1], stride=published.shape[1]),
        ]
        msg.data = published.flatten().tolist()
        self.action_pub.publish(msg)
        extra = ""
        if self.log_residual and self._last_residual is not None:
            extra = f" head_residual_rms={self._last_residual:.3f}"
        self.get_logger().info(
            f"published shape={published.shape} infer={infer_ms:.0f} ms "
            f"anchor={'on' if self.anchor else 'off'} "
            f"pose[start]={published[0].round(2)} "
            f"grip[1,5,10,-1]={published[[0, 4, 9, -1], 6].round(0)}{extra}"
        )


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--coord-checkpoint", required=True,
                   help="Path to mixed_coord_head_*_{stepN,final}.pt from the parity coord stage.")
    p.add_argument("--coord-stats", required=True,
                   help="Path to coord stats pkl (self-contained: holds base arch too).")
    p.add_argument("--base-checkpoint", default=None,
                   help="Path to the frozen base singlearm_hardware_shoulder_*.pt. "
                        "If omitted, uses base_model_path stored inside the coord stats.")
    p.add_argument("--base-stats", default=None,
                   help="Path to the base policy stats pkl. Only needed for older coord stats "
                        "files that don't carry the full base architecture inline. "
                        "If omitted, uses base_stats_path stored inside the coord stats.")
    p.add_argument("--image-topic", default="/camera_top/camera_top/color/image_raw")
    p.add_argument("--rate-hz", type=float, default=1.0,
                   help="Inference rate in timer mode. Coord head adds a second model "
                        "forward per step, so each tick is ~1.5-2x the single-arm node.")
    p.add_argument("--on-request", action="store_true",
                   help="sample once per std_msgs/Empty on --request-topic instead of on a "
                        "timer. Required when driving the arm with "
                        "xarm_codiff_deploy_executor.py.")
    p.add_argument("--request-topic", default="/codiff/request",
                   help="topic the deploy executor pings to request one chunk (--on-request only)")
    p.add_argument("--actions-topic", default="/codiff/actions",
                   help="Float32MultiArray topic for predicted chunks")
    p.add_argument("--fresh-frame-on-request", action="store_true",
                   help="wait for an image timestamped after each request before sampling")
    p.add_argument("--fresh-frame-timeout", type=float, default=2.0,
                   help="seconds to wait for a fresh image after a request")
    p.add_argument("--n-steps", type=int, default=50, help="denoising steps")
    p.add_argument("--cfg-w", type=float, default=1.2,
                   help="CFG guidance weight; 1.2 per configs/hardware_placewipe_execution.json")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--anchor", action="store_true",
                   help="pin x[:, 0, :] to the current arm pose at every diffusion step.")
    p.add_argument("--anchor-gripper", action="store_true")
    p.add_argument("--anchor-state", action="store_true",
                   help="anchor the FULL 7-D state in row 0: pose from /xarm/robot_states "
                        "plus the measured gripper from --gripper-state-topic (published "
                        "by the executor). Matches training-row semantics.")
    p.add_argument("--gripper-state-topic", default="/codiff/gripper_state")
    p.add_argument("--publish-from-index", type=int, default=0,
                   help="strip leading poses before publishing. 0 = auto.")
    p.add_argument("--mask-other-arm", type=float, default=None, metavar="X0FRAC",
                   help="blank the image right of this width fraction (mean fill) before "
                        "inference, e.g. 0.515 hides the cardboard arm in the bird camera. "
                        "Stopgap for heads that key on the other arm's posture.")
    p.add_argument("--dump-frames", default=None, metavar="DIR",
                   help="save each tick's raw camera frame, arm pose, predicted chunk and head "
                        "residual as DIR/tick_NNNNN.npz for offline replay of a live run.")
    p.add_argument("--log-residual", action="store_true",
                   help="log the head's mean residual RMS per chunk (normalized units): "
                        "~0.01 means the head is inactive (single-arm scene), ~0.3 means "
                        "it is intervening (two-arm scene).")
    p.add_argument("--prescale-hw", type=int, nargs=2, metavar=("H", "W"), default=None,
                   help="Two-stage resize H W. Use --prescale-hw 192 256 to match the "
                        "hardware coordination training pipeline.")
    args = p.parse_args()
    if args.fresh_frame_on_request and not args.on_request:
        p.error("--fresh-frame-on-request requires --on-request")
    if not np.isfinite(args.fresh_frame_timeout) or args.fresh_frame_timeout <= 0:
        p.error("--fresh-frame-timeout must be finite and positive")
    return args


def main():
    args = parse_args()
    rclpy.init()
    node = CoDiffCoordinationInferenceNode(args)
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
