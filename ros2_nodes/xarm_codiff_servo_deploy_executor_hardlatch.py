#!/usr/bin/env python3
"""Co-Diff deploy executor with servo-mode motion.

Same closed loop and same safety envelope as xarm_codiff_deploy_executor.py --
request/reply replanning, step / rotation / workspace gates, confirmation prompt,
per-pose logging -- but every gated pose is reached by STREAMING toward it in
mode 1 with /xarm/set_servo_cartesian, the way the SpaceNav teleop that recorded
the demonstrations moved the arm. Not with a planned, blocking mode-0
/xarm/set_position move.

Why this exists. The xArm7 is redundant: a TCP pose fixes six of seven joints.
In servo mode the controller integrates small deltas from the current joint
state, so the elbow stays wherever the continuous path leaves it -- which is why
40 teleoperated demos land within 1-4 deg of each other at the same TCP. In
position mode every set_position is a fresh planned move, and with stop-and-go,
replan jumps and slew hops the elbow wanders (observed live: cardboard arm at
the correct TCP with its elbow on the table). The policies see their own arm
through the shoulder camera and have only ever seen mode-1 postures. Executing
through the demos' controller restores the collection/execution contract in
documentation/hardware/experiments.md; nothing about the models changes.

Motion primitive: from the last commanded pose, walk toward the target in
--step-mm increments at --rate-hz (default 3 mm @ 20 Hz = 60 mm/s, the demos'
mean speed), rotation interpolated on the wrapped difference. A legitimately
large step (e.g. the post-drop retreat) is therefore many small ticks, never a
jump; --max-step-mm stays as the abort gate for genuinely bad targets. While
waiting for a chunk the last pose is re-sent each tick so mode 1 does not
freeze (same as xarm_codiff_executor.py).

Rows of /codiff/actions are [x_mm, y_mm, z_mm, roll, pitch, yaw, gripper].
Run the inference node with --on-request.

    # Terminal A
    python ros2_nodes/xarm_codiff_coordination_inference_node.py --on-request --anchor \\
        --prescale-hw 192 256 --cfg-w 1.2 --n-steps 50 --coord-checkpoint ... --coord-stats ...
    # Terminal B
    python ros2_nodes/xarm_codiff_servo_deploy_executor.py --replan-steps 20 --max-steps 900 \\
        --max-step-mm 100

--dry-run exercises the whole loop (request, gates, streaming schedule) without
calling any arm service; the arm driver need not even be running.
"""

import argparse
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node

from std_msgs.msg import Empty, Float32, Float32MultiArray
from xarm_msgs.msg import RobotMsg
from xarm_msgs.srv import (
    GetInt16, GripperMove, MoveCartesian, SetFloat32, SetInt16, SetInt16ById,
)

GRIPPER_MIN = 0.0
GRIPPER_MAX = 850.0


class GripperGate:
    """Optional gates between the model's raw gripper channel and actuation.

    hysteresis  Binary state machine. Once CLOSED, only a decisive open
                (>= open_thr) or a tighter grip actuates -- any loosening
                short of that (mid-ramp diffusion samples like 100..500,
                which physically flutter the fingers and can drop the
                payload early) holds the last position. Once OPEN, only a
                decisive close (<= close_thr) or further opening actuates.
    latch       One-way: after the release, close commands are ignored.
                Neither arm ever legitimately re-grasps in this task, and
                the post-release state is thin in the training data, where
                the base sometimes drifts back to "closed" (live: the bird
                arm re-closed over the basket 7 s after a clean drop and
                froze). The release is a decisive open after >=
                min_grasp_steps consecutive close-side commands -- a real
                grasp is held closed for hundreds of steps (the carry),
                whereas the anticipatory dip at the tail of an approach
                chunk lasts only a few and must NOT arm the latch (live:
                a 285 tail-dip one chunk before the grasp armed it, and
                the true close was then refused -- no grasp at all).
    grasp_lock  Position-aware hold. Once >= lock_engage consecutive
                close-side commands land (a sustained close -- above the
                ~5-step approach dips, within a real grasp chunk's close
                run), the TCP position is recorded as the grasp site, and
                every open command is suppressed until the TCP is more
                than lock_radius mm away. At a replan boundary the model
                often re-samples the grasp program from the top (chunk
                head ~800 open) before it has visually accepted the grasp;
                without the lock that pries the fingers off the payload
                once per replan (live: grab/release cycling at the bird).
                The release happens ~620 mm away at the basket and is
                unaffected -- and only such a far open can arm the latch,
                which kills the latch's own mis-arm mode (live: a grasp
                struggle armed it at the pickup and stranded the run).
                Trade-off: a sustained close on air also engages the lock,
                so a genuinely missed grasp cannot re-open on its own;
                restart the episode instead (2 of 403 demos retry, vs the
                re-open cycle which killed most runs it appeared in).
    hard_latch  Commitment device for policy mode-oscillation (live 0911:
                in the two-arm scene the head alternated full-open hover
                chunks with grasp chunks at successive replans, so the
                fingers cycled and the lid never came off). The first
                command <= close_thr commits the grasp; after that, open
                commands actuate only once >= min_open_steps CONSECUTIVE
                commands >= open_thr arrive. Mid-ramp values (between the
                two thresholds) neither count nor reset the streak; any
                close resets it. Suppressed opens do NOT arm the
                one-way latch -- only the open that actually actuates does.
                Position-agnostic, so unlike grasp_lock it is safe for the
                cardboard arm's ~10 mm grasp/release geometry.
    """

    def __init__(self, args):
        self.hysteresis = args.gripper_hysteresis
        self.latch = args.gripper_latch
        self.grasp_lock = args.gripper_grasp_lock
        self.lock_radius = float(args.gripper_grasp_lock_radius)
        self.lock_engage = int(args.gripper_lock_engage_steps)
        self.open_thr = float(args.gripper_open_thr)
        self.close_thr = float(args.gripper_close_thr)
        self.slack = float(args.gripper_threshold)
        self.min_grasp = int(args.gripper_min_grasp_steps)
        self.hard_latch = args.gripper_hard_latch
        self.min_open = int(args.gripper_min_open_steps)
        self.closed_value = float(args.gripper_closed_value)
        self.state = None       # "open" / "closed" once a decisive command lands
        self.last = None        # last actuated value
        self.close_streak = 0   # consecutive close-side commands seen
        self.open_streak = 0    # consecutive open-side commands while held
        self.held = False       # hard latch: a close has committed the grasp
        self.released = False   # a real grasp ended in a decisive open
        self.grasp_site = None  # TCP xyz where the sustained close engaged
        self.suppressed = None  # why the last command was held back

    def _near_grasp(self, tcp):
        if self.grasp_site is None:
            return False
        if tcp is None:
            return True  # no pose feedback -> assume still at the grasp site
        return float(np.linalg.norm(np.asarray(tcp[:3], dtype=float)
                                    - self.grasp_site)) < self.lock_radius

    def filter(self, position, tcp=None):
        """Return the position to actuate, or None to hold the current one.

        tcp is the measured TCP pose (>= 3 dims used) for the grasp lock;
        pass None to skip position awareness for this call.
        """
        closing = position <= self.close_thr
        opening = position >= self.open_thr
        if self.latch and self.released and closing:
            self.suppressed = "latch"
            return None
        if closing:
            self.close_streak += 1
            if (self.grasp_lock and not self.released and self.grasp_site is None
                    and self.close_streak >= self.lock_engage and tcp is not None):
                self.grasp_site = np.asarray(tcp[:3], dtype=float).copy()
        elif (self.grasp_lock and not self.released and self._near_grasp(tcp)):
            # Near the grasp site nothing legitimately loosens; this also
            # runs before the release bookkeeping, so an open here can
            # neither move the fingers nor arm the latch.
            self.suppressed = "grasp-lock"
            return None
        if self.hard_latch:
            if closing:
                self.open_streak = 0
                self.held = True  # the first close commits the grasp
                # Drive to a full close rather than holding the command that
                # happened to cross the threshold: that first close-side value
                # is often a marginal 280-300 from the head of a ramp, and when
                # the policy oscillates no tighter command ever follows, so the
                # fingers would latch half open on the payload.
                position = self.closed_value
            elif opening and self.held and not self.released:
                self.open_streak += 1
                if self.open_streak < self.min_open:
                    # Runs before the release bookkeeping, so a suppressed
                    # open neither moves the fingers nor arms the latch.
                    self.suppressed = "hard-latch"
                    return None
        if opening:
            if self.close_streak >= self.min_grasp:
                self.released = True
                self.grasp_site = None
            self.close_streak = 0
        if self.hysteresis:
            if self.state == "closed" and not opening and (
                    self.last is None or position > self.last + self.slack):
                self.suppressed = "mid-ramp"   # loosening without a decisive open
                return None
            if self.state == "open" and not closing and (
                    self.last is None or position < self.last - self.slack):
                self.suppressed = "mid-ramp"   # drooping without a decisive close
                return None
            if self.state is None and not (closing or opening):
                self.suppressed = "mid-ramp"
                return None
        if closing:
            self.state = "closed"
        elif opening:
            self.state = "open"
        self.last = position
        self.suppressed = None
        return position


def wrap_pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


class CoDiffServoDeployExecutorNode(Node):
    def __init__(self, args):
        super().__init__("xarm_codiff_servo_deploy_executor")
        self.args = args
        self.dry = bool(args.dry_run)

        self._lock = threading.Lock()
        self._chunk = None            # newest (H, 7) chunk, or None while awaiting one
        self._current_pose = None     # latest measured TCP from /xarm/robot_states
        self._current_joints = None   # latest measured joints, if the message carries them
        self._cmd_pose = None         # last pose actually streamed (mode 1 integrates from here)
        self._last_gripper = None
        self.gripper_gate = GripperGate(args)
        self._tick_s = 1.0 / float(args.rate_hz)

        if not self.dry:
            self.motion_enable = self.create_client(SetInt16ById, "/xarm/motion_enable")
            self.set_mode = self.create_client(SetInt16, "/xarm/set_mode")
            self.set_state = self.create_client(SetInt16, "/xarm/set_state")
            self.get_state = self.create_client(GetInt16, "/xarm/get_state")
            self.servo_cart = self.create_client(MoveCartesian, "/xarm/set_servo_cartesian")
            self.gripper_move = self.create_client(GripperMove, "/xarm/set_gripper_position")
            self.gripper_enable = self.create_client(SetInt16, "/xarm/set_gripper_enable")
            self.gripper_mode = self.create_client(SetInt16, "/xarm/set_gripper_mode")
            self.gripper_speed = self.create_client(SetFloat32, "/xarm/set_gripper_speed")
            for client, name in [
                (self.motion_enable, "motion_enable"), (self.set_mode, "set_mode"),
                (self.set_state, "set_state"), (self.get_state, "get_state"),
                (self.servo_cart, "set_servo_cartesian"),
                (self.gripper_move, "set_gripper_position"),
                (self.gripper_enable, "set_gripper_enable"),
                (self.gripper_mode, "set_gripper_mode"),
                (self.gripper_speed, "set_gripper_speed"),
            ]:
                if not client.wait_for_service(timeout_sec=5.0):
                    raise RuntimeError(f"service /xarm/{name} not available — is xarm.launch.py up?")

        self.create_subscription(Float32MultiArray, "/codiff/actions", self._actions_cb, 10)
        self.create_subscription(RobotMsg, "/xarm/robot_states", self._robot_state_cb, 10)
        self.request_pub = self.create_publisher(Empty, args.request_topic, 10)
        # Executor's belief of the physical gripper, for the inference node's
        # --anchor-state (full-state row-0 anchoring). Episodes start open by
        # protocol; updated on every gated actuation.
        self.gripper_state_pub = self.create_publisher(
            Float32, args.gripper_state_topic, 10)
        self._grip_belief = 800.0

        gates = []
        if args.max_step_mm > 0.0:
            gates.append(f"step<={args.max_step_mm} mm")
        if args.max_rot_step_rad > 0.0:
            gates.append(f"rot<={args.max_rot_step_rad} rad")
        if args.workspace_min is not None and args.workspace_max is not None:
            gates.append(f"box {args.workspace_min}..{args.workspace_max}")
        if args.gripper_hysteresis:
            gates.append(f"grip hysteresis {args.gripper_close_thr:.0f}/{args.gripper_open_thr:.0f}")
        if args.gripper_latch:
            gates.append("grip latch")
        if args.gripper_grasp_lock:
            gates.append(f"grasp lock r{args.gripper_grasp_lock_radius:.0f}")
        if args.gripper_hard_latch:
            gates.append(f"grip hard latch open>={args.gripper_min_open_steps}")
        self.get_logger().info(
            f"servo deploy executor up{' [DRY RUN]' if self.dry else ''}: "
            f"replan_steps={args.replan_steps}, max_steps={args.max_steps}, "
            f"stream {args.step_mm} mm @ {args.rate_hz:.0f} Hz "
            f"(= {args.step_mm * args.rate_hz:.0f} mm/s), gates={', '.join(gates) if gates else 'NONE'}"
        )
        if not gates:
            self.get_logger().warn(
                "all safety gates disabled — every pose the policy emits is commanded "
                "unchecked. Keep the e-stop in hand."
            )

    # ------------------------------------------------------------------
    # callbacks
    # ------------------------------------------------------------------
    def _robot_state_cb(self, msg: RobotMsg):
        self._current_pose = np.asarray(msg.pose, dtype=np.float32)
        ang = getattr(msg, "angle", None)
        if ang is not None and len(ang) >= 7:
            self._current_joints = np.degrees(np.asarray(ang[:7], dtype=np.float32))

    def _actions_cb(self, msg: Float32MultiArray):
        data = np.asarray(msg.data, dtype=np.float32)
        if data.size == 0 or data.size % 7 != 0:
            self.get_logger().warn(f"bad chunk size {data.size}, expected a multiple of 7")
            return
        with self._lock:
            self._chunk = data.reshape(-1, 7)

    # ------------------------------------------------------------------
    # service helpers
    # ------------------------------------------------------------------
    def _call(self, client, request):
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

    def enable(self):
        if self.dry:
            self.get_logger().info("[dry] would: motion_enable, set_mode(1), set_state(0), gripper setup")
            return
        self._call(self.motion_enable, SetInt16ById.Request(id=8, data=1))
        self._call(self.set_mode, SetInt16.Request(data=1))   # 1 = servo mode, as the teleop node
        self._call(self.set_state, SetInt16.Request(data=0))
        self._call(self.gripper_enable, SetInt16.Request(data=1))
        self._call(self.gripper_mode, SetInt16.Request(data=0))
        self._call(self.gripper_speed, SetFloat32.Request(data=float(self.args.gripper_speed)))
        time.sleep(0.1)
        self.get_logger().info("arm in mode 1 (servo) + state 0; gripper enabled")

    def _send_servo(self, pose6):
        """Fire-and-forget one servo target (no wait), exactly as the servo executor."""
        self._cmd_pose = np.asarray(pose6, dtype=np.float64)
        if self.dry:
            return
        req = MoveCartesian.Request()
        req.pose = [float(v) for v in pose6]
        req.speed = float(self.args.speed)
        req.acc = float(self.args.acc)
        req.mvtime = float(self.args.mvtime)
        self.servo_cart.call_async(req)

    def hold_tick(self):
        """Re-send the last commanded pose so mode 1 does not freeze while we wait."""
        if self._cmd_pose is not None:
            self._send_servo(self._cmd_pose)

    def stream_to(self, target6):
        """Walk from the last commanded pose to target6 in --step-mm ticks at --rate-hz."""
        if self._cmd_pose is None:
            self._cmd_pose = np.asarray(self._current_pose[:6], dtype=np.float64)
        start = self._cmd_pose.copy()
        end = np.asarray(target6, dtype=np.float64)
        gap = float(np.linalg.norm(end[:3] - start[:3]))
        drot = wrap_pi(end[3:6] - start[3:6])
        n = max(1, int(np.ceil(gap / self.args.step_mm)))
        t_next = time.monotonic()
        for i in range(1, n + 1):
            f = i / n
            w = np.empty(6)
            w[:3] = start[:3] + (end[:3] - start[:3]) * f
            w[3:6] = start[3:6] + drot * f
            self._send_servo(w)
            t_next += self._tick_s
            dt = t_next - time.monotonic()
            if dt > 0:
                time.sleep(dt)
            rclpy.spin_once(self, timeout_sec=0.0)
        return n

    def settle(self):
        """Optionally wait until the measured pose is near the last commanded one,
        so the next replan is sampled from where the arm actually is."""
        if self.args.settle_mm <= 0 or self.dry or self._cmd_pose is None:
            return
        deadline = time.monotonic() + self.args.settle_timeout
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.0)
            if self._current_pose is not None:
                if np.linalg.norm(self._current_pose[:3] - self._cmd_pose[:3]) <= self.args.settle_mm:
                    return
            self.hold_tick()
            time.sleep(self._tick_s)
        self.get_logger().warn(f"settle: arm still >{self.args.settle_mm} mm from command after "
                               f"{self.args.settle_timeout:.1f}s")

    def move_gripper(self, position):
        position = float(np.clip(position, GRIPPER_MIN, GRIPPER_MAX))
        gated = self.gripper_gate.filter(position, self._current_pose)
        if gated is None:
            if self.gripper_gate.suppressed == "latch":
                self.get_logger().info(
                    f"gripper latch: ignoring re-close command {position:.0f}",
                    throttle_duration_sec=5.0)
            elif self.gripper_gate.suppressed == "grasp-lock":
                self.get_logger().info(
                    f"grasp lock: holding grip against re-open command {position:.0f}",
                    throttle_duration_sec=5.0)
            return
        if (self._last_gripper is not None
                and abs(gated - self._last_gripper) <= self.args.gripper_threshold):
            return
        if not self.dry:
            self._call(self.gripper_move, GripperMove.Request(pos=gated))
        self._last_gripper = gated
        self._grip_belief = gated

    # ------------------------------------------------------------------
    # replan cycle (identical contract to the position-mode deploy executor)
    # ------------------------------------------------------------------
    def wait_for_pose(self, timeout_s=10.0):
        deadline = time.monotonic() + timeout_s
        while rclpy.ok() and self._current_pose is None and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        if self._current_pose is None and self.dry:
            self._current_pose = np.zeros(6, np.float32)
            self.get_logger().warn("[dry] no /xarm/robot_states; using a zero pose")
        return self._current_pose is not None

    def request_chunk(self):
        """Ping the inference node and block until a fresh chunk lands, holding
        the arm (re-sending the last pose) while we wait."""
        with self._lock:
            self._chunk = None
        self.gripper_state_pub.publish(Float32(data=float(self._grip_belief)))
        self.request_pub.publish(Empty())
        deadline = time.monotonic() + self.args.request_timeout
        last_retry = time.monotonic()
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.0)
            with self._lock:
                if self._chunk is not None:
                    return self._chunk
            self.hold_tick()
            time.sleep(self._tick_s)
            if time.monotonic() - last_retry > self.args.request_retry:
                self.request_pub.publish(Empty())
                last_retry = time.monotonic()
                self.get_logger().warn("no chunk yet — re-requesting")
        return None

    def resample_discontinuous(self, chunk):
        """Re-request while the chunk's first row is unreachably far from the arm.

        With --anchor, row 0 of the SAMPLED chunk is pinned to the measured pose,
        but that row is dropped before publishing -- so the first row we execute
        is unconstrained. When the head switches program mid-task (live 0912: the
        cardboard arm holding the lid at y=230 drew a chunk whose every real row
        sat at the home pose with the gripper open, 193.6 mm away) the whole chunk
        is a teleport. check_target rightly refuses it, but that ends the run on a
        single unlucky draw. The sample is cheap; ask for another one instead.
        Falls through to check_target -- and the SAFETY STOP -- once the retries
        are spent, so the envelope is unchanged.
        """
        if chunk is None or self.args.max_chunk_jump_mm <= 0.0:
            return chunk
        for attempt in range(self.args.max_chunk_resamples):
            if self._current_pose is None:
                return chunk
            jump = float(np.linalg.norm(chunk[0, :3] - self._current_pose[:3]))
            if jump <= self.args.max_chunk_jump_mm:
                return chunk
            self.get_logger().warn(
                f"discontinuous chunk: row 0 is {jump:.1f} mm from the arm "
                f"(> --max-chunk-jump-mm {self.args.max_chunk_jump_mm:.0f}); "
                f"resampling {attempt + 1}/{self.args.max_chunk_resamples}"
            )
            fresh = self.request_chunk()
            if fresh is None:
                return chunk
            chunk = fresh
        return chunk

    def check_target(self, target):
        """Reject a target before commanding it. Returns an error string or None."""
        cur = self._current_pose
        if self.args.max_step_mm > 0.0:
            if cur is None:
                return "no /xarm/robot_states — cannot bound the step"
            step = float(np.linalg.norm(target[0:3] - cur[0:3]))
            if step > self.args.max_step_mm:
                return f"step {step:.1f} mm from current pose exceeds --max-step-mm {self.args.max_step_mm}"
        if self.args.max_rot_step_rad > 0.0:
            if cur is None:
                return "no /xarm/robot_states — cannot bound the rotation step"
            rot_delta = np.abs(wrap_pi(target[3:6] - cur[3:6]))
            if (rot_delta > self.args.max_rot_step_rad).any():
                return (f"rotation step {np.round(rot_delta, 3)} rad exceeds "
                        f"--max-rot-step-rad {self.args.max_rot_step_rad}")
        if self.args.workspace_min is not None and self.args.workspace_max is not None:
            lo = np.asarray(self.args.workspace_min, dtype=np.float32)
            hi = np.asarray(self.args.workspace_max, dtype=np.float32)
            if (target[0:3] < lo).any() or (target[0:3] > hi).any():
                return f"target {np.round(target[0:3], 1)} outside --workspace-min/--workspace-max"
        return None

    def _joints_str(self):
        if self._current_joints is None:
            return ""
        return f" joints {np.round(self._current_joints, 0).astype(int).tolist()}"

    def run(self):
        if not self.wait_for_pose():
            self.get_logger().error("no /xarm/robot_states — is the driver up?")
            return
        self.get_logger().info(f"current pose: {np.round(self._current_pose, 2)}{self._joints_str()}")

        self.enable()
        if not (self.args.yes or self.dry or confirm("Start closed-loop Co-Diff execution on the arm?")):
            self.get_logger().info("aborted before any motion")
            return

        executed = 0
        while rclpy.ok() and executed < self.args.max_steps:
            self.settle()
            chunk = self.resample_discontinuous(self.request_chunk())
            if chunk is None:
                self.get_logger().error(
                    f"no chunk within {self.args.request_timeout:.1f}s — is the inference node "
                    f"running with --on-request --request-topic {self.args.request_topic}?"
                )
                return
            if self.args.align_chunk_start and self._current_pose is not None:
                # With --anchor only row 0 of the sampled chunk is pinned to the
                # measured pose; near the drop the model's manifold sits up the
                # approach corridor and the published rows begin 40-70 mm behind
                # the arm. Executing them re-runs the approach once per replan
                # (the hover limit cycle). Start from the row nearest the arm
                # instead: on a healthy chunk that is row 0 (no change), on a
                # pull-back chunk it skips only the physical retreat.
                dists = np.linalg.norm(chunk[:, :3] - self._current_pose[:3], axis=1)
                skip = int(np.argmin(dists))
                # Only act on a real pull-back: row 0 clearly far AND the nearest
                # row clearly closer. In a near-stationary chunk (grasp ramp,
                # release, hold) every row is mm-close and argmin is noise --
                # skipping there would scramble the gripper ramp's timing.
                if (skip > 0 and dists[0] > self.args.align_min_jump_mm
                        and dists[skip] < 0.5 * dists[0]):
                    self.get_logger().info(
                        f"chunk align: skipping {skip} leading rows "
                        f"(row 0 was {dists[0]:.1f} mm away, row {skip} is {dists[skip]:.1f} mm)"
                    )
                    chunk = chunk[skip:]
            n_exec = min(len(chunk), self.args.replan_steps)
            if len(chunk) < self.args.replan_steps:
                self.get_logger().warn(
                    f"chunk {len(chunk)} < --replan-steps {self.args.replan_steps}; executing all"
                )
            g = chunk[:n_exec, 6]
            self.get_logger().info(
                f"chunk grip: min={g.min():.0f} max={g.max():.0f} | executing 0..{n_exec - 1}{self._joints_str()}"
            )

            for action in chunk[:n_exec]:
                rclpy.spin_once(self, timeout_sec=0.0)
                err = self.check_target(action)
                if err is not None:
                    self.get_logger().error(f"SAFETY STOP: {err}")
                    self.hold_tick()
                    return
                ticks = self.stream_to(action[0:6])
                self.move_gripper(action[6])
                executed += 1
                if executed % 5 == 0:
                    self.get_logger().info(
                        f"step {executed}: xyz {np.round(action[0:3], 1)} "
                        f"rpy {np.round(action[3:6], 3)} gripper {action[6]:.0f} ({ticks} ticks)"
                    )
                if executed >= self.args.max_steps:
                    break

        self.get_logger().info(f"done after {executed} steps{self._joints_str()}")

    def shutdown(self):
        if self.dry:
            return
        try:
            self._call(self.set_mode, SetInt16.Request(data=0))
            self._call(self.set_state, SetInt16.Request(data=0))
        except Exception as exc:
            self.get_logger().error(f"shutdown sequence failed: {exc}")


def confirm(msg):
    try:
        return input(f"{msg} [type 'yes']: ").strip().lower() == "yes"
    except (KeyboardInterrupt, EOFError):
        print()
        return False


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--replan-steps", type=int, default=15,
                   help="poses to execute from each chunk before requesting a new one")
    p.add_argument("--align-chunk-start", action="store_true",
                   help="start executing each chunk from the row nearest the measured TCP "
                        "instead of row 0 -- skips the 40-70 mm up-corridor pull-back the "
                        "anchor-to-manifold gap produces near the drop, which otherwise "
                        "executes as a back-and-forth hover once per replan")
    p.add_argument("--align-min-jump-mm", type=float, default=20.0,
                   help="alignment engages only when row 0 is at least this far from the "
                        "TCP (and the nearest row at most half that) -- near-stationary "
                        "chunks are left untouched so gripper ramps keep their timing")
    p.add_argument("--max-steps", type=int, default=400, help="stop after this many executed poses")
    # servo streaming
    p.add_argument("--step-mm", type=float, default=3.0,
                   help="servo increment; 3 mm @ 20 Hz is the demos' 60 mm/s mean")
    p.add_argument("--rate-hz", type=float, default=20.0, help="servo tick rate; demos were 20 Hz")
    p.add_argument("--speed", type=float, default=200.0, help="passed to set_servo_cartesian")
    p.add_argument("--acc", type=float, default=2000.0, help="passed to set_servo_cartesian")
    p.add_argument("--mvtime", type=float, default=0.1, help="passed to set_servo_cartesian")
    p.add_argument("--settle-mm", type=float, default=3.0,
                   help="before each replan, wait until the measured pose is within this of the "
                        "last command (0 = don't wait), so chunks are sampled from where the arm is")
    p.add_argument("--settle-timeout", type=float, default=1.0)
    # gates (same semantics as the position-mode executor)
    p.add_argument("--max-step-mm", type=float, default=0.0,
                   help="abort if a commanded pose is further than this from the measured pose. 0 = off")
    p.add_argument("--max-rot-step-rad", type=float, default=0.0,
                   help="abort if any of roll/pitch/yaw jumps more than this in one step. 0 = off")
    p.add_argument("--workspace-min", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    p.add_argument("--workspace-max", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    p.add_argument("--gripper-hysteresis", action="store_true",
                   help="actuate only decisive gripper commands (<= close-thr or >= open-thr); "
                        "mid-ramp diffusion samples hold position instead of fluttering")
    p.add_argument("--gripper-latch", action="store_true",
                   help="after the release (a close, then a decisive open), ignore close "
                        "commands -- the arm never re-grasps in this task")
    p.add_argument("--gripper-min-grasp-steps", type=int, default=25,
                   help="close-side steps a grasp must persist before a decisive open "
                        "counts as the release that arms --gripper-latch")
    p.add_argument("--max-chunk-jump-mm", type=float, default=100.0,
                   help="re-request a chunk whose first row is farther than this from the "
                        "arm, instead of letting it hit the SAFETY STOP. A whole-chunk "
                        "teleport is the head switching program mid-task; another draw is "
                        "usually fine. 0 disables. Keep <= --max-step-mm")
    p.add_argument("--max-chunk-resamples", type=int, default=3,
                   help="how many times to re-request before executing the chunk anyway "
                        "(and letting --max-step-mm abort the run)")
    p.add_argument("--gripper-hard-latch", action="store_true",
                   help="the first close-side command commits the grasp; afterwards opens "
                        "actuate only after --gripper-min-open-steps CONSECUTIVE open-side "
                        "commands (a real release publishes open chunks indefinitely). "
                        "Position-agnostic alternative to --gripper-grasp-lock for arms "
                        "whose grasp and release sites are close together")
    p.add_argument("--gripper-closed-value", type=float, default=0.0,
                   help="gripper position commanded once --gripper-hard-latch commits the "
                        "grasp, and held until the release (0 = fully closed; successful "
                        "demo grasps settle around 8-17)")
    p.add_argument("--gripper-min-open-steps", type=int, default=40,
                   help="consecutive commands >= --gripper-open-thr required for a release "
                        "once --gripper-hard-latch has committed a grasp. 40 = two full "
                        "chunks: the oscillation publishes whole open chunks, so a threshold "
                        "shorter than one chunk (19) does not survive it")
    p.add_argument("--gripper-grasp-lock", action="store_true",
                   help="after a sustained close, suppress open commands until the TCP is "
                        "> --gripper-grasp-lock-radius mm from where the grasp happened -- "
                        "replan-boundary re-samples of the approach must not pry the fingers "
                        "off the payload. The release at the basket is unaffected, and only "
                        "such a far open can arm --gripper-latch")
    p.add_argument("--gripper-grasp-lock-radius", type=float, default=250.0,
                   help="mm from the grasp site within which opens are suppressed "
                        "(pickup->basket is ~620 mm)")
    p.add_argument("--gripper-lock-engage-steps", type=int, default=10,
                   help="consecutive close-side commands that engage the grasp lock; above "
                        "the ~5-step anticipatory approach dips, within a real grasp "
                        "chunk's close run")
    p.add_argument("--gripper-open-thr", type=float, default=600.0)
    p.add_argument("--gripper-close-thr", type=float, default=300.0)
    p.add_argument("--gripper-threshold", type=float, default=20.0)
    p.add_argument("--gripper-speed", type=float, default=2000.0)
    p.add_argument("--request-topic", default="/codiff/request")
    p.add_argument("--gripper-state-topic", default="/codiff/gripper_state",
                   help="publish the executor's gripper belief here before every replan "
                        "request, for the inference node's --anchor-state")
    p.add_argument("--request-timeout", type=float, default=30.0)
    p.add_argument("--request-retry", type=float, default=5.0)
    p.add_argument("--yes", action="store_true", help="skip the interactive confirmation")
    p.add_argument("--dry-run", action="store_true",
                   help="run the loop against the inference node without touching the arm")
    return p.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    node = CoDiffServoDeployExecutorNode(args)
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.shutdown()
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
