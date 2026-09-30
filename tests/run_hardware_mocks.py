"""Run executor logic tests with inert ROS types; never instantiate a ROS node."""
import sys
from pathlib import Path
from types import ModuleType
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def forbidden(*args, **kwargs):
    raise RuntimeError('ROS execution is forbidden in offline tests')

class Message:
    def __init__(self, **kwargs):
        self.wait = False
        self.__dict__.update(kwargs)

Message.Request = Message

for name, symbols in {
    'rclpy': ['init', 'shutdown', 'spin', 'spin_once', 'ok'],
    'rclpy.node': ['Node'], 'cv_bridge': ['CvBridge'],
    'sensor_msgs.msg': ['Image'],
    'std_msgs.msg': ['Empty', 'Float32', 'Float32MultiArray', 'MultiArrayDimension'],
    'xarm_msgs.msg': ['RobotMsg'],
    'xarm_msgs.srv': ['GetFloat32', 'GetInt16', 'GripperMove', 'MoveCartesian', 'SetFloat32', 'SetInt16', 'SetInt16ById'],
}.items():
    module = ModuleType(name)
    for symbol in symbols:
        setattr(module, symbol, forbidden if name == 'rclpy' else Message)
    sys.modules[name] = module
    if '.' in name:
        parent, child = name.rsplit('.', 1)
        sys.modules.setdefault(parent, ModuleType(parent))
        setattr(sys.modules[parent], child, module)
import pytest
raise SystemExit(pytest.main(['-q', str(Path(__file__).with_name('test_gripper_settle.py')), *sys.argv[1:]]))
