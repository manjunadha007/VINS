#!/usr/bin/env python3
"""
Virtual-camera sync tests for the HyperFusion dashboard.

Verifies the chain that matters for virtual production: a pose arriving from
Isaac ROS Visual SLAM ends up as the right numbers on the trajectory canvas AND
in the FreeD packet sent to Unreal Engine.

Run inside the Isaac ROS container, where rclpy/cv_bridge exist:

    cd /workspaces/isaac_ros-dev
    python3 -m unittest src.test_virtual_camera -v

On a host without ROS 2 these tests skip themselves. The pure-logic suites,
test_pose_transform.py and test_freed_protocol.py, have no ROS dependency and
cover the maths in far more depth; this file covers the wiring between them.
"""
import os
import sys
import threading
import time
import unittest

# Must be set before PyQt loads so the suite runs on a headless machine / CI.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

# The dashboard pulls in the whole ROS 2 + OpenCV stack at import time.
try:
    import dashboard
    from nav_msgs.msg import Odometry
    from geometry_msgs.msg import PoseWithCovarianceStamped
    from PyQt5.QtWidgets import QApplication
    IMPORT_ERROR = None
except Exception as exc:                      # pragma: no cover - env dependent
    IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

import freed
from pose_transform import (PoseGate, PosePredictor, PoseSmoother, PoseTransformer,
                            matrix_to_quaternion, rpy_to_matrix)
import math


def q_from_rpy(roll_deg=0.0, pitch_deg=0.0, yaw_deg=0.0):
    return matrix_to_quaternion(rpy_to_matrix(math.radians(roll_deg),
                                              math.radians(pitch_deg),
                                              math.radians(yaw_deg)))


class _CapturingSocket:
    """Stands in for the UDP socket so no packets leave the machine."""

    def __init__(self):
        self.sent = []

    def sendto(self, data, addr):
        self.sent.append((data, addr))

    @property
    def last_packet(self):
        return self.sent[-1][0] if self.sent else None


@unittest.skipUnless(IMPORT_ERROR is None,
                     f"ROS 2 / dashboard deps unavailable ({IMPORT_ERROR})")
class VirtualCameraTestCase(unittest.TestCase):
    """Builds a HyperFusionApp without running __init__.

    __init__ opens a serial port, a microphone stream, a ROS node and QThreads.
    None of that is needed to test the pose path, and all of it fails without
    hardware, so the instance is created with __new__ and given exactly the state
    the callbacks touch.
    """

    @classmethod
    def setUpClass(cls):
        cls.qapp = QApplication.instance() or QApplication(sys.argv)

    def _make_gui(self, transform_cfg=None, **tracking):
        gui = dashboard.HyperFusionApp.__new__(dashboard.HyperFusionApp)

        gui.cfg = dashboard.CONFIG
        gui.canvas = dashboard.TrajectoryCanvas()
        gui.sock = _CapturingSocket()
        gui.streaming_enabled = True

        gui.transformer = PoseTransformer(transform_cfg)
        gui.predictor = PosePredictor(
            lead_time_s=tracking.get("lead_time_s", 0.0),
            max_extrapolation_s=tracking.get("max_extrapolation_s", 0.05))
        gui.smoother = PoseSmoother(alpha=tracking.get("smoothing_alpha", 0.0))
        gui.stale_timeout = tracking.get("stale_pose_timeout_s", 0.25)
        # Gate off by default here so these tests exercise the pose path itself;
        # the gate has its own dedicated suite in test_pose_transform.py.
        gui.gate = PoseGate(enabled=tracking.get("gate_enabled", False),
                            max_speed_m_s=tracking.get("gate_max_speed_m_s", 3.0),
                            max_jump_m=tracking.get("gate_max_jump_m", 0.05))

        gui.latest_pose = None
        gui.latest_unreal = None
        gui.slam_status = "WAITING FOR SLAM"
        gui.last_vio_time = 0.0
        gui._packets_sent = 0
        gui._pose_updates = 0
        gui._last_rate_check = time.time()
        gui._pose_rate = 0.0
        gui._send_rate = 0.0
        gui._pose_lock = threading.Lock()

        # Send on every pose: exercises the low-latency path and means the
        # callbacks alone are enough to drive output in tests.
        gui.send_on_pose = True
        return gui

    @staticmethod
    def _odom(x, y, z, q=(0.0, 0.0, 0.0, 1.0), lin=(0.0, 0.0, 0.0),
              ang=(0.0, 0.0, 0.0)):
        msg = Odometry()
        msg.pose.pose.position.x = float(x)
        msg.pose.pose.position.y = float(y)
        msg.pose.pose.position.z = float(z)
        msg.pose.pose.orientation.x = float(q[0])
        msg.pose.pose.orientation.y = float(q[1])
        msg.pose.pose.orientation.z = float(q[2])
        msg.pose.pose.orientation.w = float(q[3])
        msg.twist.twist.linear.x = float(lin[0])
        msg.twist.twist.linear.y = float(lin[1])
        msg.twist.twist.linear.z = float(lin[2])
        msg.twist.twist.angular.x = float(ang[0])
        msg.twist.twist.angular.y = float(ang[1])
        msg.twist.twist.angular.z = float(ang[2])
        return msg

    @staticmethod
    def _vo_pose(x, y, z, q=(0.0, 0.0, 0.0, 1.0)):
        msg = PoseWithCovarianceStamped()
        msg.pose.pose.position.x = float(x)
        msg.pose.pose.position.y = float(y)
        msg.pose.pose.position.z = float(z)
        msg.pose.pose.orientation.x = float(q[0])
        msg.pose.pose.orientation.y = float(q[1])
        msg.pose.pose.orientation.z = float(q[2])
        msg.pose.pose.orientation.w = float(q[3])
        return msg


class TestSlamPoseSync(VirtualCameraTestCase):
    def test_virtual_camera_receives_6dof(self):
        """The core promise: 6DOF in -> virtual camera moves to match."""
        gui = self._make_gui()
        gui.odometry_callback(self._odom(15.0, -5.0, 20.0))

        self.assertIsNotNone(gui.canvas._latest,
                             "canvas never received a trajectory point")
        cx, cy, cz = gui.canvas._latest
        # Y mirrors into Unreal's left-handed frame; X and Z pass through.
        self.assertAlmostEqual(cx, 15.0, places=6)
        self.assertAlmostEqual(cy, 5.0, places=6)
        self.assertAlmostEqual(cz, 20.0, places=6)

    def test_level_camera_reports_level(self):
        """Regression: the old transform reported pan/tilt/roll = 0/90/90."""
        gui = self._make_gui()
        gui.odometry_callback(self._odom(0.0, 0.0, 0.0))
        pose = gui.latest_unreal
        self.assertAlmostEqual(pose.pan, 0.0, places=6)
        self.assertAlmostEqual(pose.tilt, 0.0, places=6)
        self.assertAlmostEqual(pose.roll, 0.0, places=6)

    def test_panning_the_camera_pans_the_virtual_camera(self):
        """Regression: the old transform pinned pan at 0.0 for every input."""
        gui = self._make_gui()
        gui.odometry_callback(self._odom(0, 0, 0, q=q_from_rpy(yaw_deg=40.0)))
        pose = gui.latest_unreal
        self.assertAlmostEqual(pose.pan, -40.0, places=4)
        self.assertAlmostEqual(pose.tilt, 0.0, places=4)
        self.assertAlmostEqual(pose.roll, 0.0, places=4,
                               msg="pan must not leak into roll any more")

    def test_latest_pose_keeps_raw_ros_coordinates(self):
        """Conversion happens on output, so internal state stays in ROS."""
        gui = self._make_gui()
        gui.odometry_callback(self._odom(15.0, -5.0, 20.0))
        position, _quat = gui.latest_pose
        self.assertAlmostEqual(position[0], 15.0, places=6)
        self.assertAlmostEqual(position[1], -5.0, places=6)
        self.assertAlmostEqual(position[2], 20.0, places=6)

    def test_velocity_comes_from_isaac_twist(self):
        """Prediction relies on Isaac's own velocity, not finite differencing."""
        gui = self._make_gui(max_extrapolation_s=1.0)
        gui.odometry_callback(self._odom(0, 0, 0, lin=(2.0, 0.0, 0.0)))
        np.testing.assert_allclose(gui.predictor._lin, [2.0, 0.0, 0.0])

    def test_trajectory_accumulates(self):
        gui = self._make_gui()
        for i in range(5):
            gui.odometry_callback(self._odom(float(i), 0.0, 0.0))
        self.assertEqual(len(gui.canvas._pts), 5)
        self.assertAlmostEqual(gui.canvas._latest[0], 4.0, places=6)

    def test_status_flips_to_tracking(self):
        gui = self._make_gui()
        self.assertEqual(gui.slam_status, "WAITING FOR SLAM")
        gui.odometry_callback(self._odom(0, 0, 0))
        self.assertEqual(gui.slam_status, "TRACKING")


class TestSinglePoseWriter(VirtualCameraTestCase):
    """Only one topic may drive the output stream at a time."""

    def test_vo_pose_works_before_odometry_arrives(self):
        gui = self._make_gui()
        gui.vo_pose_callback(self._vo_pose(1.0, 0.0, 0.0))
        self.assertIsNotNone(gui.latest_unreal)
        self.assertAlmostEqual(gui.latest_unreal.x, 1.0, places=6)

    def test_vo_pose_yields_once_odometry_is_live(self):
        """Both topics carry the same result; two writers would fight."""
        gui = self._make_gui()
        gui.odometry_callback(self._odom(5.0, 0.0, 0.0))
        gui.vo_pose_callback(self._vo_pose(99.0, 0.0, 0.0))
        self.assertAlmostEqual(gui.latest_unreal.x, 5.0, places=6,
                               msg="vo_pose overrode the richer odometry pose")

    def test_imu_callback_does_not_touch_the_pose(self):
        """IMU is telemetry only — Isaac ROS already fused it."""
        gui = self._make_gui()
        gui.odometry_callback(self._odom(3.0, 0.0, 0.0))
        before = gui.latest_unreal.as_tuple()

        class _Imu:
            class _V:
                x = y = z = 0.5
            angular_velocity = _V()
            linear_acceleration = _V()
        gui.imu_callback(_Imu())

        self.assertEqual(gui.latest_unreal.as_tuple(), before)
        self.assertIn("monitoring only", gui.latest_imu)


class TestFreeDOutput(VirtualCameraTestCase):
    def test_pose_reaches_unreal_as_a_valid_freed_packet(self):
        gui = self._make_gui()
        gui.odometry_callback(self._odom(15.0, -5.0, 20.0,
                                         q=q_from_rpy(yaw_deg=30.0)))

        pkt = gui.sock.last_packet
        self.assertIsNotNone(pkt, "no FreeD packet was sent")
        decoded = freed.decode_freed_packet(pkt)    # raises on bad checksum/type

        self.assertEqual(decoded["cam_id"], dashboard.CAMERA_ID)
        self.assertAlmostEqual(decoded["x_m"], 15.0, delta=1e-3)
        self.assertAlmostEqual(decoded["y_m"], 5.0, delta=1e-3)
        self.assertAlmostEqual(decoded["z_m"], 20.0, delta=1e-3)
        self.assertAlmostEqual(decoded["yaw_deg"], -30.0, delta=1e-2)
        self.assertAlmostEqual(decoded["pitch_deg"], 0.0, delta=1e-2)
        self.assertAlmostEqual(decoded["roll_deg"], 0.0, delta=1e-2)

    def test_freed_channels_are_not_crossed(self):
        """Tilt must land on tilt, not on roll."""
        gui = self._make_gui()
        gui.odometry_callback(self._odom(0, 0, 0, q=q_from_rpy(pitch_deg=25.0)))
        decoded = freed.decode_freed_packet(gui.sock.last_packet)
        self.assertAlmostEqual(decoded["pitch_deg"], -25.0, delta=1e-2)
        self.assertAlmostEqual(decoded["roll_deg"], 0.0, delta=1e-2)
        self.assertAlmostEqual(decoded["yaw_deg"], 0.0, delta=1e-2)

    def test_packet_goes_to_the_configured_endpoint(self):
        gui = self._make_gui()
        gui.odometry_callback(self._odom(0.0, 0.0, 0.0))
        self.assertEqual(gui.sock.sent[-1][1],
                         (dashboard.UDP_IP, dashboard.UDP_PORT))

    def test_disabled_stream_sends_nothing(self):
        gui = self._make_gui()
        gui.odometry_callback(self._odom(1.0, 2.0, 3.0))
        gui.streaming_enabled = False
        gui.sock.sent.clear()
        gui._stream_freed()
        self.assertEqual(gui.sock.sent, [], "toggle off must stop the UDP stream")

    def test_nothing_is_sent_before_the_first_pose(self):
        """Never assert a fake origin — Unreal would snap to it."""
        gui = self._make_gui()
        gui._stream_freed()
        self.assertEqual(gui.sock.sent, [])

    def test_tracking_loss_holds_the_last_good_pose(self):
        """Freezing is recoverable on set; teleporting to the origin is not."""
        gui = self._make_gui()
        gui.odometry_callback(self._odom(15.0, -5.0, 20.0))
        gui.sock.sent.clear()

        # Age the pose past the freshness gate.
        gui.predictor._stamp -= 5.0
        gui.last_vio_time -= 5.0
        gui._stream_freed()

        decoded = freed.decode_freed_packet(gui.sock.last_packet)
        self.assertAlmostEqual(decoded["x_m"], 15.0, delta=1e-3)
        self.assertAlmostEqual(decoded["y_m"], 5.0, delta=1e-3)
        self.assertAlmostEqual(decoded["z_m"], 20.0, delta=1e-3)
        self.assertEqual(gui.slam_status, "TRACKING LOST — HOLDING")

    def test_stale_pose_is_not_extrapolated(self):
        """A held pose must not keep drifting at the last known velocity."""
        gui = self._make_gui(max_extrapolation_s=10.0)
        gui.odometry_callback(self._odom(0.0, 0.0, 0.0, lin=(5.0, 0.0, 0.0)))
        gui.sock.sent.clear()
        gui.predictor._stamp -= 5.0
        gui.last_vio_time -= 5.0
        gui._stream_freed()
        decoded = freed.decode_freed_packet(gui.sock.last_packet)
        self.assertAlmostEqual(decoded["x_m"], 0.0, delta=1e-3)

    def test_ground_zero_rebases_the_stream(self):
        gui = self._make_gui()
        gui.odometry_callback(self._odom(10.0, 0.0, 0.0))
        gui.transformer.set_ground_zero((10.0, 0.0, 0.0), (0, 0, 0, 1))
        gui.sock.sent.clear()
        gui.odometry_callback(self._odom(11.0, 0.0, 0.0))
        decoded = freed.decode_freed_packet(gui.sock.last_packet)
        self.assertAlmostEqual(decoded["x_m"], 1.0, delta=1e-3)

    def test_rig_offset_reaches_the_wire(self):
        gui = self._make_gui(
            transform_cfg={"rig_offset": {"translation_m": (0.25, 0.0, 0.0)}})
        gui.odometry_callback(self._odom(0.0, 0.0, 0.0))
        decoded = freed.decode_freed_packet(gui.sock.last_packet)
        self.assertAlmostEqual(decoded["x_m"], 0.25, delta=1e-3)


class TestGateProtectsUnreal(VirtualCameraTestCase):
    """End-to-end: a tracker snap must not reach the wire."""

    def test_metre_jump_never_reaches_unreal(self):
        gui = self._make_gui(gate_enabled=True, gate_max_jump_m=0.05)
        gui.odometry_callback(self._odom(0.0, 0.0, 0.0))
        gui.sock.sent.clear()
        # cuVSLAM relocalises 1.25 m away between two frames.
        gui.odometry_callback(self._odom(1.25, 0.0, 0.0))
        decoded = freed.decode_freed_packet(gui.sock.last_packet)
        self.assertLess(abs(decoded["x_m"]), 0.06,
                        f"a {1.25:.2f} m snap reached Unreal as {decoded['x_m']:.3f} m")

    def test_real_motion_still_passes(self):
        gui = self._make_gui(gate_enabled=True, gate_max_jump_m=0.05)
        gui.odometry_callback(self._odom(0.0, 0.0, 0.0))
        for i in range(1, 11):        # 1 cm steps: ordinary handheld motion
            gui.odometry_callback(self._odom(i * 0.01, 0.0, 0.0))
        decoded = freed.decode_freed_packet(gui.sock.last_packet)
        self.assertAlmostEqual(decoded["x_m"], 0.10, delta=0.005,
                               msg="gate ate real camera movement")

    def test_gate_disabled_lets_the_jump_through(self):
        """Confirms the protection comes from the gate, not something else."""
        gui = self._make_gui(gate_enabled=False)
        gui.odometry_callback(self._odom(0.0, 0.0, 0.0))
        gui.odometry_callback(self._odom(1.25, 0.0, 0.0))
        decoded = freed.decode_freed_packet(gui.sock.last_packet)
        self.assertAlmostEqual(decoded["x_m"], 1.25, delta=1e-3)


class TestCameraInfoIntrinsics(VirtualCameraTestCase):
    class _Info:
        def __init__(self, k, d=()):
            self.k = list(k)
            self.d = list(d)

    def _gui(self):
        gui = self._make_gui()
        gui.aruco_K = np.eye(3)
        gui.aruco_D = np.zeros((5, 1))
        gui._intrinsics_from_camera_info = False
        return gui

    def test_intrinsics_are_taken_from_the_camera(self):
        gui = self._gui()
        gui.camera_info_callback(self._Info(
            [600.0, 0.0, 320.0, 0.0, 601.0, 240.0, 0.0, 0.0, 1.0]))
        self.assertTrue(gui._intrinsics_from_camera_info)
        self.assertAlmostEqual(gui.aruco_K[0, 0], 600.0)
        self.assertAlmostEqual(gui.aruco_K[1, 1], 601.0)
        self.assertAlmostEqual(gui.aruco_K[0, 2], 320.0)

    def test_garbage_camera_info_is_ignored(self):
        gui = self._gui()
        gui.camera_info_callback(self._Info([0.0] * 9))
        self.assertFalse(gui._intrinsics_from_camera_info)

    def test_first_valid_message_wins(self):
        gui = self._gui()
        gui.camera_info_callback(self._Info(
            [600.0, 0.0, 320.0, 0.0, 600.0, 240.0, 0.0, 0.0, 1.0]))
        gui.camera_info_callback(self._Info(
            [999.0, 0.0, 111.0, 0.0, 999.0, 111.0, 0.0, 0.0, 1.0]))
        self.assertAlmostEqual(gui.aruco_K[0, 0], 600.0)


class TestTrajectoryCanvas(VirtualCameraTestCase):
    def test_ring_buffer_is_bounded(self):
        canvas = dashboard.TrajectoryCanvas()
        for i in range(canvas.MAX_POINTS + 500):
            canvas.update_trajectory(float(i), 0.0, 0.0)
        self.assertEqual(len(canvas._pts), canvas.MAX_POINTS,
                         "unbounded trajectory would leak memory over a shoot")

    def test_heading_is_recorded(self):
        canvas = dashboard.TrajectoryCanvas()
        canvas.update_trajectory(0.0, 0.0, 0.0, yaw_deg=42.0)
        self.assertAlmostEqual(canvas._heading_deg, 42.0, places=6)

    def test_paint_with_no_data_does_not_crash(self):
        canvas = dashboard.TrajectoryCanvas()
        canvas.resize(400, 300)
        canvas.repaint()

    def test_paint_with_data_does_not_crash(self):
        canvas = dashboard.TrajectoryCanvas()
        canvas.resize(400, 300)
        for i in range(50):
            canvas.update_trajectory(i * 0.1, i * 0.05, i * 0.02, yaw_deg=i)
        canvas.repaint()


if __name__ == '__main__':
    unittest.main(verbosity=2)
