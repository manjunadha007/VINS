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

import copy

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
from tracking_engine import TrackingEngine
from pose_transform import matrix_to_quaternion, rpy_to_matrix
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
        """Build the app without __init__, over a real TrackingEngine.

        The engine is the same object the production code uses, so these tests
        exercise the actual pose path rather than a parallel copy of it.
        """
        gui = dashboard.HyperFusionApp.__new__(dashboard.HyperFusionApp)

        cfg = copy.deepcopy(dashboard.CONFIG)
        if transform_cfg:
            cfg["transform"].update(transform_cfg)
        # Gate and hold are off unless a test asks for them: they have dedicated
        # suites in test_pose_transform.py, and these tests feed poses with no
        # accompanying IMU stream.
        cfg["tracking"].update({
            "gate_enabled": tracking.get("gate_enabled", False),
            "hold_enabled": tracking.get("hold_enabled", False),
            "lead_time_s": tracking.get("lead_time_s", 0.0),
            "max_extrapolation_s": tracking.get("max_extrapolation_s", 0.05),
            "stale_pose_timeout_s": tracking.get("stale_pose_timeout_s", 0.25),
            "smoothing_alpha": tracking.get("smoothing_alpha", 0.0),
            "gate_max_speed_m_s": tracking.get("gate_max_speed_m_s", 3.0),
            "gate_max_jump_m": tracking.get("gate_max_jump_m", 0.05),
        })

        cfg["fusion"]["enabled"] = False
        gui.cfg = cfg
        gui.engine = TrackingEngine(cfg, log=lambda *a, **k: None)
        gui.canvas = dashboard.TrajectoryCanvas()
        gui.sock = _CapturingSocket()
        gui.streaming_enabled = True
        gui.depth_K = None
        gui._pose_lock = threading.Lock()
        gui._last_rate_check = time.time()
        gui._pose_rate = 0.0
        gui._send_rate = 0.0
        gui._last_pose_count = 0
        gui._last_packet_count = 0

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
        from unittest.mock import patch
        gui = self._make_gui()
        with patch("time.monotonic", return_value=100.) as mono, \
                patch("time.time", return_value=1000.) as wall:
            for i in range(90):
                mono.return_value = 100. + i / 90.
                wall.return_value = 1000. + i / 90.
                gui.vo_pose_callback(self._vo_pose(i * .01, 0.0, 0.0))
                self.assertIsNotNone(gui.latest_unreal)
                self.assertAlmostEqual(gui.latest_unreal.x, i * .01, places=6)
        self.assertEqual(gui.engine.pose_updates, 90)

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
        gui.engine.last_vio_time -= 5.0
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
        gui.engine.last_vio_time -= 5.0
        gui._stream_freed()
        decoded = freed.decode_freed_packet(gui.sock.last_packet)
        self.assertAlmostEqual(decoded["x_m"], 0.0, delta=1e-3)

    def test_ground_zero_rebases_the_stream(self):
        gui = self._make_gui()
        gui.odometry_callback(self._odom(10.0, 0.0, 0.0))
        gui.engine.transformer.set_ground_zero((10.0, 0.0, 0.0), (0, 0, 0, 1))
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
        # Start from a deliberately wrong K so a real camera_info is visible.
        gui.engine.aruco_K = np.eye(3)
        gui.engine.aruco_D = np.zeros((5, 1))
        gui.engine.intrinsics_from_camera_info = False
        return gui

    def test_intrinsics_are_taken_from_the_camera(self):
        gui = self._gui()
        gui.camera_info_callback(self._Info(
            [600.0, 0.0, 320.0, 0.0, 601.0, 240.0, 0.0, 0.0, 1.0]))
        self.assertTrue(gui.engine.intrinsics_from_camera_info)
        self.assertAlmostEqual(gui.aruco_K[0, 0], 600.0)
        self.assertAlmostEqual(gui.aruco_K[1, 1], 601.0)
        self.assertAlmostEqual(gui.aruco_K[0, 2], 320.0)

    def test_garbage_camera_info_is_ignored(self):
        gui = self._gui()
        gui.camera_info_callback(self._Info([0.0] * 9))
        self.assertFalse(gui.engine.intrinsics_from_camera_info)

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


class TestCameraChannelEditing(VirtualCameraTestCase):
    def test_camera_editor_keeps_roll_pitch_and_yaw_in_map_coordinates(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from pose_transform import invert_transform
        gui = self._make_gui()
        alignment = np.eye(4)
        alignment[:3, :3] = rpy_to_matrix(*np.radians([4, -8, 35]))
        alignment[:3, 3] = [3, -2, 1]
        gui._align_rigid = lambda: alignment
        gui.lidar_matcher = SimpleNamespace(map_cloud=object())
        previous = np.eye(4)
        gui._lineup = SimpleNamespace(T_map_cam=previous.copy(), history=[])
        gui._refresh_lineup = Mock()
        desired = np.eye(4)
        desired[:3, :3] = rpy_to_matrix(*np.radians([22, -31, 67]))
        desired[:3, 3] = [1, 2, 3]
        gui._on_camera_gizmo_moved(desired)
        np.testing.assert_allclose(gui._lineup.T_map_cam, invert_transform(alignment) @ desired)
        np.testing.assert_array_equal(gui._lineup.history[-1], previous)
        gui._refresh_lineup.assert_called_once()


class TestMarkerAssistedLineup(VirtualCameraTestCase):
    def _marker_gui(self, full_ui=False):
        from unittest.mock import Mock
        from PyQt5.QtWidgets import QMainWindow, QPushButton
        from PyQt5.QtCore import QEvent
        from map_viewer import MapViewerWidget
        from lidar_map_matcher import LidarMapMatcher
        from test_marker_lineup import fixture
        gui = self._make_gui()
        QMainWindow.__init__(gui)
        gui._lineup = None
        gui._marker_request = None
        gui._marker_busy = False
        gui._marker_shutdown = False
        gui.aruco_enabled = True
        gui.aruco_detectors = [("DICT_4X4_50", None, None, None)]
        gui.lidar_matcher = LidarMapMatcher({})
        gui.lidar_matcher.map_cloud = object()
        if full_ui:
            gui.gnss_enabled = False
            gui._build_ui()
        else:
            gui.map_viewer = MapViewerWidget()
            gui.btn_ar = QPushButton()
            gui.btn_ar.setCheckable(True)
            panel = gui._build_marker_lineup_panel()
            panel.setParent(gui)
        gui._push_map_alignment = Mock(wraps=gui._push_map_alignment)
        obs, truth, camera = fixture()
        gui.engine.set_intrinsics(obs.K)
        for stamp, T in ((10., obs.T_slam_image), (10.1, obs.T_slam_depth)):
            gui.engine.accept_pose(T[:3, 3], matrix_to_quaternion(T[:3, :3]),
                                   [0, 0, 0], [0, 0, 0], stamp=stamp)
        gui.engine.set_depth(obs.depth, obs.depth_K, stamp=10.1)
        gui.map_viewer.set_locator_transform(obs.T_map_locator)
        gui.btn_marker_lineup.setChecked(True)
        def cleanup():
            gui._marker_shutdown = True
            gui.map_viewer.deleteLater()
            gui.deleteLater()
            self.qapp.sendPostedEvents(None, QEvent.DeferredDelete)
        self.addCleanup(cleanup)
        return gui, obs, truth

    def test_arming_keeps_existing_alignment_and_marks_the_floor_target(self):
        gui, obs, _ = self._marker_gui()
        self.assertIsNotNone(gui._marker_request)
        self.assertFalse(gui.engine.is_anchored)
        self.assertFalse(gui.lidar_matcher.is_aligned)
        self.assertAlmostEqual(gui.map_viewer.persp_view._locator_marker_size, 0.1)
        self.assertFalse(gui.marker_width.isEnabled())
        self.assertEqual(gui.btn_marker_lineup.property("lineupState"), "waiting")

    def test_verified_result_keeps_switch_on_without_repeating_detection(self):
        from lidar_map_matcher import MapMeasurement
        gui, obs, truth = self._marker_gui()
        measurement = MapMeasurement(ok=True, T_slam_to_map=truth,
                                     verify_score=0.94, observable_rank=3)
        request = gui._marker_request
        gui._finish_marker_lineup((request, obs, measurement, ""))
        np.testing.assert_allclose(gui.lidar_matcher.T_slam_to_map, truth)
        self.assertTrue(gui.lidar_matcher.is_aligned)
        gui._push_map_alignment.assert_called_once()
        self.assertIsNone(gui._marker_request)
        self.assertTrue(gui.btn_marker_lineup.isChecked())
        self.assertTrue(gui.btn_ar.isChecked())
        self.assertTrue(gui.engine.map_aligned)
        self.assertEqual(gui.btn_marker_lineup.property("lineupState"), "verifying")
        from unittest.mock import patch
        with patch("dashboard.threading.Thread") as worker:
            gui._poll_marker_lineup()
            worker.assert_not_called()

    def test_rejected_scan_leaves_previous_alignment_untouched(self):
        from lidar_map_matcher import MapMeasurement
        gui, obs, _ = self._marker_gui()
        previous = gui.lidar_matcher.T_slam_to_map.copy()
        gui._finish_marker_lineup((gui._marker_request, obs,
                                  MapMeasurement(reason="wrong part of room"), ""))
        np.testing.assert_array_equal(gui.lidar_matcher.T_slam_to_map, previous)
        gui._push_map_alignment.assert_not_called()
        self.assertIn("wrong part of room", gui.lbl_marker_lineup.text())
        self.assertFalse(gui.engine.is_anchored)

    def test_cancelling_discards_a_late_success(self):
        from lidar_map_matcher import MapMeasurement
        gui, obs, truth = self._marker_gui()
        request = gui._marker_request
        gui.btn_marker_lineup.setChecked(False)
        gui._finish_marker_lineup((request, obs,
                                  MapMeasurement(ok=True, T_slam_to_map=truth, observable_rank=3), ""))
        gui._push_map_alignment.assert_not_called()
        self.assertFalse(gui.lidar_matcher.is_aligned)

    def test_moving_locator_discards_inflight_solution(self):
        from lidar_map_matcher import MapMeasurement
        gui, obs, truth = self._marker_gui()
        T = gui.map_viewer.get_locator_transform()
        T[0, 3] += 0.2
        gui.map_viewer.set_locator_transform(T)
        gui._finish_marker_lineup((gui._marker_request, obs,
                                  MapMeasurement(ok=True, T_slam_to_map=truth, observable_rank=3), ""))
        gui._push_map_alignment.assert_not_called()
        self.assertIn("locator moved", gui.lbl_marker_lineup.text())

    def test_changed_scan_resync_and_degenerate_geometry_reject_results(self):
        from lidar_map_matcher import MapMeasurement
        for kind in ("scan", "resync", "rank"):
            with self.subTest(kind=kind):
                gui, obs, truth = self._marker_gui()
                if kind == "scan":
                    gui.lidar_matcher._generation += 1
                elif kind == "resync":
                    gui.engine.gate.resyncs += 1
                measurement = MapMeasurement(ok=True, T_slam_to_map=truth,
                                             observable_rank=1 if kind == "rank" else 3)
                gui._finish_marker_lineup((gui._marker_request, obs, measurement, ""))
                gui._push_map_alignment.assert_not_called()

    def test_only_a_new_detection_starts_one_background_search(self):
        from unittest.mock import patch
        gui, obs, _ = self._marker_gui()
        request = gui._marker_request
        # Cached display corners from before arming must never start a solve.
        gui._marker_detection_sample = ([(obs.family, obs.marker_id, obs.corners)],
                                       10., request["armed_at"]-0.01, (848, 480))
        with patch("dashboard.threading.Thread") as worker:
            gui._poll_marker_lineup()
            worker.assert_not_called()
            gui._marker_detection_sample = ([(obs.family, obs.marker_id, obs.corners)],
                                           10., time.monotonic(), (848, 480))
            gui._poll_marker_lineup()
            gui._poll_marker_lineup()
            worker.assert_called_once()
            worker.return_value.start.assert_called_once()
        gui._marker_busy = False

    def test_worker_error_and_stale_tracking_are_reported(self):
        from lidar_map_matcher import MapMeasurement
        gui, obs, truth = self._marker_gui()
        gui._finish_marker_lineup((gui._marker_request, obs, None, "solver unavailable"))
        self.assertIn("solver unavailable", gui.lbl_marker_lineup.text())
        gui.btn_marker_lineup.setChecked(True)
        gui.engine._last_pose_mono = time.monotonic()-5
        gui._finish_marker_lineup((gui._marker_request, obs,
                                  MapMeasurement(ok=True, T_slam_to_map=truth, observable_rank=3), ""))
        self.assertIn("stale", gui.lbl_marker_lineup.text())
        gui._push_map_alignment.assert_not_called()

    def test_operate_has_one_lineup_checkbox_and_live_camera_channels(self):
        from PyQt5.QtWidgets import QCheckBox, QPushButton
        from unittest.mock import Mock
        gui, _, _ = self._marker_gui(full_ui=True)
        self.assertIsInstance(gui.btn_marker_lineup, QCheckBox)
        self.assertFalse(gui.marker_settings.isAncestorOf(gui.btn_marker_lineup))
        labels = [button.text().lower() for button in gui.findChildren(QPushButton)]
        self.assertFalse(any("lineup" in text or "quick match" in text for text in labels))
        view = gui.map_viewer.persp_view
        view._select_gizmo_row("camera")
        self.assertFalse(view.gizmo_active)
        self.assertTrue(all(not field.isEnabled() for field in view.channel_box.fields.values()))
        gui._start_lineup = Mock()
        gui._on_camera_gizmo_moved(np.eye(4))
        gui._start_lineup.assert_not_called()
        view.select_locator()
        self.assertTrue(view.gizmo_active)
        self.assertTrue(all(field.isEnabled() for field in view.channel_box.fields.values()))

    def _install_verified_marker(self, gui, observation, truth):
        from lidar_map_matcher import MapMeasurement
        gui.cfg["fusion"]["enabled"] = True
        gui.engine.fusion.enabled = True
        gui._finish_marker_lineup((gui._marker_request, observation,
            MapMeasurement(ok=True, T_slam_to_map=truth, verify_score=0.96,
                           observable_rank=3), ""))

    def _fresh_scan(self, gui, timestamp):
        from lidar_map_matcher import MapMeasurement
        measurement = MapMeasurement(ok=True, observable_rank=3, verify_score=0.96)
        measurement.monotonic_timestamp = timestamp
        gui.engine.observe_scan(measurement)

    def test_green_needs_consecutive_fresh_checks_and_drops_on_loss(self):
        from unittest.mock import patch
        with patch("time.monotonic", return_value=100.) as mono:
            gui, obs, truth = self._marker_gui()
            self._install_verified_marker(gui, obs, truth)
            self.assertEqual(gui.btn_marker_lineup.property("lineupState"), "tracking")
            # Re-reading the same successful measurement cannot manufacture readiness.
            for _ in range(5):
                gui.engine.observe_scan(gui.lidar_matcher.last_measurement)
            gui._refresh_marker_indicator()
            self.assertNotEqual(gui.btn_marker_lineup.property("lineupState"), "locked")
            for t in (100.03, 100.06):
                mono.return_value = t
                self._fresh_scan(gui, t)
            gui._refresh_tracked_camera()
            gui._refresh_marker_indicator()
            self.assertEqual(gui.btn_marker_lineup.property("lineupState"), "locked")
            self.assertIn("Fusion active", gui.lbl_marker_lineup.text())
            gui.engine._output_held = True
            gui._refresh_marker_indicator()
            self.assertNotEqual(gui.btn_marker_lineup.property("lineupState"), "locked")
            gui.engine._output_held = False
            # Fresh scan but stale camera must never stay green.
            gui.engine._last_pose_mono = 99.
            gui._refresh_marker_indicator()
            self.assertEqual(gui.btn_marker_lineup.property("lineupState"), "held")
            # Fresh camera but stale scan must not stay green either.
            mono.return_value = 102.
            gui.engine._last_pose_mono = 102.
            gui._refresh_marker_indicator()
            self.assertEqual(gui.btn_marker_lineup.property("lineupState"), "tracking")
            self.assertIn("camera tracking active", gui.lbl_marker_lineup.text())
            # RTK alone does not claim a scan lock.
            for t in (101.94, 101.97, 102.):
                gui.engine.fusion.observe("gnss", True, t)
            gui._refresh_marker_indicator()
            self.assertNotEqual(gui.btn_marker_lineup.property("lineupState"), "locked")

    def test_solved_heading_updates_locator_rotation_and_lock_context(self):
        from marker_lineup import MarkerMeasurement
        from lidar_map_matcher import MapMeasurement
        gui, obs, truth = self._marker_gui()
        before = gui.map_viewer.get_locator_transform()
        solved = before.copy()
        solved[:3, :3] = rpy_to_matrix(0, 0, np.radians(143))
        result = MarkerMeasurement(MapMeasurement(ok=True, T_slam_to_map=truth,
            observable_rank=3, verify_score=.96), solved, .02)
        gui._finish_marker_lineup((gui._marker_request, obs, result, ''))
        actual = gui.map_viewer.get_locator_transform()
        np.testing.assert_array_equal(actual[:3, 3], before[:3, 3])
        np.testing.assert_allclose(actual[:3, :3], solved[:3, :3], atol=1e-12)
        gui._poll_marker_lineup()
        self.assertTrue(gui.btn_marker_lineup.isChecked(), 'its own rotation update must not invalidate the lock')

    def test_scene_edit_invalidates_an_existing_marker_lock(self):
        gui, obs, truth = self._marker_gui()
        self._install_verified_marker(gui, obs, truth)
        locator = gui.map_viewer.get_locator_transform()
        locator[1, 3] += 0.25
        gui.map_viewer.set_locator_transform(locator)
        gui._poll_marker_lineup()
        self.assertFalse(gui.btn_marker_lineup.isChecked())
        self.assertIn("locator moved", gui.lbl_marker_lineup.text())
        self.assertEqual(gui.btn_marker_lineup.property("lineupState"), "error")

    def test_dot_and_freed_follow_motion_in_the_aligned_scan_after_marker_leaves(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from pose_transform import make_transform
        with patch("time.monotonic", return_value=100.) as mono, \
                patch("time.time", return_value=1000.) as wall:
            gui, obs, truth = self._marker_gui()
            display = np.eye(4)
            display[:3, :3] = rpy_to_matrix(*np.radians([7, -11, 32]))
            display[:3, 3] = [3., -4., 0.6]
            # Scan metres -> displayed world, separate from arbitrary SLAM origin.
            gui._align_rigid = lambda: display
            rig = np.eye(4)
            rig[:3, :3] = rpy_to_matrix(*np.radians([2, -5, 3]))
            rig[:3, 3] = [0.07, -0.02, 0.12]
            gui.engine.transformer._rig = rig
            self._install_verified_marker(gui, obs, truth)
            gui._marker_detection_sample = None  # no more marker needed
            previous = None
            for i in range(1, 81):
                mono.return_value = 100. + i * .04
                wall.return_value = 1000. + i * .04
                moved = np.eye(4)
                moved[:3, :3] = rpy_to_matrix(*np.radians([2*i, -3*i, 5*i]))
                moved[:3, 3] = [i * .08, -i * .03, i * .02]
                raw = obs.T_slam_depth @ moved
                gui.engine.accept_pose(raw[:3, 3], matrix_to_quaternion(raw[:3, :3]),
                                       [0, 0, 0], [0, 0, 0], stamp=11. + i * .04)
                if i <= 3:
                    self._fresh_scan(gui, mono.return_value)
                # Continue moving for seconds after scan evidence expires.
                # Camera motion starts on the very first post-lineup frame.
                # An obsolete manual preview must not take ownership of the live dot.
                gui._lineup = SimpleNamespace(T_map_cam=np.eye(4))
                gui._refresh_tracked_camera()
                expected = display @ truth @ raw @ rig
                np.testing.assert_allclose(gui.map_viewer.cam_pose, expected, atol=1e-12)
                np.testing.assert_allclose(gui.map_viewer.persp_view._cam_pos, expected[:3, 3])
                np.testing.assert_allclose(rpy_to_matrix(*np.radians(gui.map_viewer.persp_view._cam_rpy)),
                                           expected[:3, :3], atol=1e-12)
                packet = gui.engine.freed_packet()
                self.assertIsNotNone(packet)
                output = gui.engine.latest_unreal
                np.testing.assert_allclose([output.x, output.y, output.z],
                                           expected[:3, 3] * [1, -1, 1], atol=1e-12)
                if previous is not None:
                    self.assertGreater(np.linalg.norm(expected[:3, 3] - previous[:3, 3]), .05)
                previous = expected.copy()
            gui._refresh_marker_indicator()
            self.assertEqual(gui.btn_marker_lineup.property("lineupState"), "tracking")
            self.assertFalse(gui.engine._relineup_required)
            mono.return_value += 1.
            wall.return_value += 1.
            gui._refresh_tracked_camera()
            np.testing.assert_allclose(gui.map_viewer.cam_pose, previous, atol=1e-12)
            gui._refresh_marker_indicator()
            self.assertEqual(gui.btn_marker_lineup.property("lineupState"), "held")


if __name__ == '__main__':
    unittest.main(verbosity=2)
