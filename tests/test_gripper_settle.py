"""Offline executor/observation sequencing tests; no ROS node or hardware."""
import contextlib
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ros2_nodes import xarm_codiff_servo_deploy_executor as ex
from ros2_nodes import xarm_codiff_inference_node as inf
from ros2_nodes import xarm_codiff_coordination_inference_node as coord


class Clock:
    def __init__(self):
        self.value = 0.0

    def now(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds


class GripperSettleTests(unittest.TestCase):
    def setUp(self):
        with patch.object(sys, 'argv', ['executor', '--gripper-settle', '--gripper-hysteresis',
                                       '--gripper-latch', '--gripper-open-thr', '500']):
            self.args = ex.parse_args()
        self.clock = Clock()
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(ex.time, 'monotonic', self.clock.now))
        self.stack.enter_context(patch.object(ex.time, 'sleep', self.clock.sleep))
        self.stack.enter_context(patch.object(ex.rclpy, 'ok', return_value=True))
        self.stack.enter_context(patch.object(ex.rclpy, 'spin_once'))
        self.node = SimpleNamespace(args=self.args, dry=False, _last_gripper=200.0,
                                    _grip_belief=200.0, gripper_state_pub=Mock(),
                                    _tick_s=0.05, gripper_position=Mock(), gripper_move=Mock(),
                                    hold_tick=Mock(), get_logger=Mock(return_value=Mock()),
                                    _current_pose=np.zeros(6), gripper_gate=ex.GripperGate(self.args))
        self.node._gripper_call = MethodType(ex.CoDiffServoDeployExecutorNode._gripper_call, self.node)

    def test_disabled_and_dry_modes_do_not_read_hardware(self):
        for dry, enabled in [(False, False), (True, True)]:
            self.node.dry = dry
            self.args.gripper_settle = enabled
            ex.CoDiffServoDeployExecutorNode.settle_gripper(self.node)
        self.node.gripper_position.call_async.assert_not_called()

    def test_settles_to_original_target_with_two_consecutive_reads(self):
        self.node._gripper_call = Mock(side_effect=[SimpleNamespace(ret=0, data=v)
                                                   for v in (800, 210, 260, 205, 200)])
        ex.CoDiffServoDeployExecutorNode.settle_gripper(self.node)
        self.assertEqual(self.node._gripper_call.call_count, 5)
        self.assertEqual(self.node._last_gripper, 200)
        self.assertEqual(self.node._grip_belief, 200)
        self.assertGreater(self.node.hold_tick.call_count, 0)
        self.assertLess(self.clock.value, self.args.gripper_settle_timeout)

    def test_unreached_target_times_out_without_replan_or_forced_close(self):
        self.node._gripper_call = Mock(return_value=SimpleNamespace(ret=0, data=800))
        with self.assertRaisesRegex(RuntimeError, 'gripper settle timeout'):
            ex.CoDiffServoDeployExecutorNode.settle_gripper(self.node)
        self.assertLessEqual(self.clock.value, self.args.gripper_settle_timeout)
        self.node.gripper_move.call_async.assert_not_called()

    def test_bad_feedback_stops(self):
        for ret, data in ((1, 200), (0, float('nan')), (0, -10), (0, 999)):
            with self.subTest(ret=ret, data=data):
                self.node._gripper_call = Mock(return_value=SimpleNamespace(ret=ret, data=data))
                with self.assertRaisesRegex(RuntimeError, 'invalid gripper feedback'):
                    ex.CoDiffServoDeployExecutorNode.settle_gripper(self.node)

    def test_stalled_service_is_cancelled_and_hold_continues(self):
        future = Mock()
        future.done.return_value = False
        self.node.gripper_position.call_async.return_value = future
        with self.assertRaisesRegex(RuntimeError, 'service timed out'):
            self.node._gripper_call(self.node.gripper_position, object(), 0.15)
        self.assertEqual(self.clock.value, 0.15)
        future.cancel.assert_called_once()
        self.assertGreaterEqual(self.node.hold_tick.call_count, 3)

    def test_command_error_does_not_update_accepted_target(self):
        self.node._gripper_call = Mock(return_value=SimpleNamespace(ret=1, message='driver error'))
        with self.assertRaisesRegex(RuntimeError, 'command failed'):
            ex.CoDiffServoDeployExecutorNode.move_gripper(self.node, 100)
        self.assertEqual(self.node._last_gripper, 200)

    def test_command_keeps_policy_target_and_logs_return(self):
        self.node._gripper_call = Mock(return_value=SimpleNamespace(ret=0, message=''))
        ex.CoDiffServoDeployExecutorNode.move_gripper(self.node, 120)
        request = self.node._gripper_call.call_args.args[1]
        self.assertEqual(request.pos, 120)
        self.assertFalse(request.wait)
        self.assertEqual(self.node._last_gripper, 120)
        self.assertEqual(self.node._grip_belief, 120)
        self.node.get_logger().info.assert_called()

    def test_no_new_chunk_requested_when_settling_fails(self):
        self.node.wait_for_pose = Mock(return_value=True)
        self.node._joints_str = Mock(return_value='')
        self.node.enable = Mock()
        self.node.settle = Mock()
        self.node.settle_gripper = Mock(side_effect=RuntimeError('settle failed'))
        self.node.request_chunk = Mock()
        self.args.yes = True
        with self.assertRaisesRegex(RuntimeError, 'settle failed'):
            ex.CoDiffServoDeployExecutorNode.run(self.node)
        self.node.request_chunk.assert_not_called()

    def test_request_retry_zero_sends_only_one_request(self):
        import threading
        self.args.request_retry = 0
        self.args.request_timeout = 0.15
        self.node._lock = threading.Lock()
        self.node._chunk = None
        self.node.request_pub = Mock()
        self.assertIsNone(ex.CoDiffServoDeployExecutorNode.request_chunk(self.node))
        self.node.request_pub.publish.assert_called_once()
        self.node.gripper_state_pub.publish.assert_called_once()
        self.assertEqual(self.node.gripper_state_pub.publish.call_args.args[0].data, 200)


class FreshFrameTests(unittest.TestCase):
    node_class = inf.CoDiffInferenceNode

    def setUp(self):
        self.clock = Clock()
        self.patcher = patch.object(inf.time, 'monotonic', self.clock.now)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.node = SimpleNamespace(fresh_frame_on_request=True, fresh_frame_timeout=1.0,
                                    _pending_frame_request=None, _sample_request_ns=0,
                                    get_logger=Mock(return_value=Mock()), _tick=Mock(),
                                    bridge=SimpleNamespace(imgmsg_to_cv2=lambda *a, **k: np.zeros((2, 2, 3), np.uint8)))
        self.node.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=int((10+self.clock.value)*1e9)))
        self.node._expire_frame_request = MethodType(self.node_class._expire_frame_request, self.node)

    def image(self, seconds):
        ns = int(seconds * 1e9)
        msg = SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(sec=ns//10**9, nanosec=ns%10**9)))
        self.node_class._image_cb(self.node, msg)

    def test_cached_stale_zero_and_future_frames_cannot_trigger_sample(self):
        self.node_class._request_cb(self.node, None)
        self.node._tick.assert_not_called()
        for stamp in (0, 9, 10, 12):
            self.image(stamp)
        self.node._tick.assert_not_called()
        self.clock.value = 0.1
        self.image(10.05)
        self.node._tick.assert_called_once()
        self.assertIsNone(self.node._pending_frame_request)
        self.assertEqual(self.node._sample_request_ns, 10_000_000_000)
        self.image(10.06)
        self.node._tick.assert_called_once()

    def test_timeout_does_not_publish_or_extend_on_duplicate_request(self):
        self.node_class._request_cb(self.node, None)
        deadline = self.node._pending_frame_request
        self.clock.value = 0.5
        self.node_class._request_cb(self.node, None)
        self.assertEqual(self.node._pending_frame_request, deadline)
        self.clock.value = 1.1
        self.image(11.05)
        self.node._tick.assert_not_called()
        self.assertIsNone(self.node._pending_frame_request)
        self.node.get_logger().error.assert_called_once()

    def test_default_mode_samples_immediately(self):
        self.node.fresh_frame_on_request = False
        self.node_class._request_cb(self.node, None)
        self.node._tick.assert_called_once()


class CoordinationFreshFrameTests(FreshFrameTests):
    node_class = coord.CoDiffCoordinationInferenceNode


if __name__ == '__main__':
    unittest.main()
