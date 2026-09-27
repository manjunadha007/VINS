#!/usr/bin/env python3
"""
Run the ROS-dependent test suite on a machine without ROS 2.

`test_virtual_camera.py` imports dashboard.py, which pulls in rclpy, cv_bridge
and the ROS message types. This harness injects stand-ins for those so the wiring
between the engine, the callbacks and the UDP output can be checked from a plain
shell:

    python3 run_offline_tests.py

Honest about what this is: the message stubs are hand-written, so a passing run
proves the *wiring* is sound, not that it behaves identically against real ROS.
The authoritative run is still inside the Isaac container:

    python3 -m unittest src.test_virtual_camera -v

The other two suites need nothing at all and should be run directly:

    python3 -m unittest test_pose_transform test_freed_protocol
"""
import os
import sys
import types
import unittest
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PROJECT = os.path.dirname(os.path.abspath(__file__))


# ── rclpy ────────────────────────────────────────────────────────────────────
def _install_rclpy():
    rclpy = types.ModuleType("rclpy")
    rclpy.ok = lambda: True
    rclpy.init = lambda *a, **k: None
    rclpy.shutdown = lambda *a, **k: None
    rclpy.spin = lambda *a, **k: None
    rclpy.spin_once = lambda *a, **k: None

    node_mod = types.ModuleType("rclpy.node")

    class Node:
        def __init__(self, name):
            self._name = name

        def create_subscription(self, *a, **k):
            return None

        def destroy_node(self):
            return None

    node_mod.Node = Node
    rclpy.node = node_mod

    qos_mod = types.ModuleType("rclpy.qos")

    class ReliabilityPolicy:
        BEST_EFFORT = 1
        RELIABLE = 2

    class QoSProfile:
        def __init__(self, depth=10, reliability=None):
            self.depth = depth
            self.reliability = reliability

    qos_mod.ReliabilityPolicy = ReliabilityPolicy
    qos_mod.QoSProfile = QoSProfile
    rclpy.qos = qos_mod

    sys.modules["rclpy"] = rclpy
    sys.modules["rclpy.node"] = node_mod
    sys.modules["rclpy.qos"] = qos_mod


# ── ROS message types ────────────────────────────────────────────────────────
class _Vec3:
    def __init__(self):
        self.x = self.y = self.z = 0.0


class _Quat:
    def __init__(self):
        self.x = self.y = self.z = 0.0
        self.w = 1.0


class _Stamp:
    def __init__(self):
        self.sec = 0
        self.nanosec = 0


class _Header:
    def __init__(self):
        self.stamp = _Stamp()
        self.frame_id = ""


class _Pose:
    def __init__(self):
        self.position = _Vec3()
        self.orientation = _Quat()


class _PoseWithCov:
    def __init__(self):
        self.pose = _Pose()
        self.covariance = [0.0] * 36


class _Twist:
    def __init__(self):
        self.linear = _Vec3()
        self.angular = _Vec3()


class _TwistWithCov:
    def __init__(self):
        self.twist = _Twist()
        self.covariance = [0.0] * 36


class Odometry:
    def __init__(self):
        self.header = _Header()
        self.pose = _PoseWithCov()
        self.twist = _TwistWithCov()


class PoseWithCovarianceStamped:
    def __init__(self):
        self.header = _Header()
        self.pose = _PoseWithCov()


class Imu:
    def __init__(self):
        self.header = _Header()
        self.orientation = _Quat()
        self.orientation_covariance = [0.0] * 9
        self.angular_velocity = _Vec3()
        self.angular_velocity_covariance = [0.0] * 9
        self.linear_acceleration = _Vec3()
        self.linear_acceleration_covariance = [0.0] * 9


class Image:
    def __init__(self):
        self.header = _Header()
        self.height = 0
        self.width = 0
        self.encoding = "mono8"
        self.data = b""


class CameraInfo:
    def __init__(self):
        self.header = _Header()
        self.height = 0
        self.width = 0
        self.k = [0.0] * 9
        self.d = []


def _install_messages():
    table = [
        ("nav_msgs", {}),
        ("nav_msgs.msg", {"Odometry": Odometry}),
        ("sensor_msgs", {}),
        ("sensor_msgs.msg", {"Image": Image, "Imu": Imu,
                             "CameraInfo": CameraInfo}),
        ("geometry_msgs", {}),
        ("geometry_msgs.msg",
         {"PoseWithCovarianceStamped": PoseWithCovarianceStamped}),
    ]
    for name, attrs in table:
        mod = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(mod, key, value)
        sys.modules[name] = mod
    for parent in ("nav_msgs", "sensor_msgs", "geometry_msgs"):
        sys.modules[parent].msg = sys.modules[f"{parent}.msg"]


def _install_cv_bridge():
    mod = types.ModuleType("cv_bridge")

    class CvBridge:
        def imgmsg_to_cv2(self, msg, encoding=None):
            raise NotImplementedError("not exercised by these tests")

    mod.CvBridge = CvBridge
    sys.modules["cv_bridge"] = mod


def main():
    _install_rclpy()
    _install_messages()
    _install_cv_bridge()

    sys.path.insert(0, PROJECT)
    os.chdir(PROJECT)

    # test_lidar_fusion's scale-reconciliation cases import dashboard.py too,
    # so they need the same stand-ins.
    suite = unittest.TestLoader().loadTestsFromNames(
        ["test_virtual_camera", "test_lidar_fusion"])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
