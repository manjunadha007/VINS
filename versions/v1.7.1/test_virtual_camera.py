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

    def test_body_frame_twist_is_extrapolated_in_the_world(self):
        """Isaac's twist is in the camera body frame. Panned 90 degrees left,
        a dolly forward must advance the prediction along world +Y, not +X."""
        gui = self._make_gui(max_extrapolation_s=1.0)
        gui.odometry_callback(self._odom(0, 0, 0, q=q_from_rpy(yaw_deg=90.0),
                                         lin=(2.0, 0.0, 0.0)))
        np.testing.assert_allclose(gui.predictor._lin, [0.0, 2.0, 0.0], atol=1e-9)
        position, _ = gui.predictor.predict(gui.predictor._stamp + 0.5)
        np.testing.assert_allclose(position, [0.0, 1.0, 0.0], atol=1e-9)

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


class TestMaximizeHeldAsNormalWindow(VirtualCameraTestCase):
    """The rig's Xwayland freezes a window left in the maximized state."""

    def _window(self, hold=True):
        from PyQt5.QtWidgets import QMainWindow

        class Window(dashboard.MaximizeAsNormalWindow, QMainWindow):
            pass
        window = Window()
        window.hold_maximize_as_normal = hold
        window.setGeometry(100, 100, 400, 300)
        window.show()
        self._settle()
        self.addCleanup(window.close)
        return window

    def _settle(self, seconds=0.6):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.qapp.processEvents()
            time.sleep(0.01)

    def test_maximize_toggles_between_filled_and_previous_size(self):
        from PyQt5.QtCore import Qt
        window = self._window()
        small = window.geometry()
        window.setWindowState(Qt.WindowMaximized)
        self._settle()
        self.assertEqual(window.windowState(), Qt.WindowNoState)
        filled = window.geometry()
        self.assertGreater(filled.width(), small.width())
        self.assertGreater(filled.height(), small.height())

        window.setWindowState(Qt.WindowMaximized)
        self._settle()
        self.assertEqual(window.windowState(), Qt.WindowNoState)
        self.assertEqual(window.geometry(), small)

        window.setWindowState(Qt.WindowMaximized)
        self._settle()
        self.assertEqual(window.geometry(), filled)

    def test_fullscreen_is_left_alone(self):
        from PyQt5.QtCore import Qt
        window = self._window()
        window.setWindowState(Qt.WindowFullScreen)
        self._settle()
        self.assertTrue(window.windowState() & Qt.WindowFullScreen)

    def test_switched_off_leaves_maximize_alone(self):
        from PyQt5.QtCore import Qt
        window = self._window(hold=False)
        window.setWindowState(Qt.WindowMaximized)
        self._settle()
        self.assertTrue(window.windowState() & Qt.WindowMaximized)


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


class TestProjectorSwitch(VirtualCameraTestCase):
    def test_switch_waits_for_camera_confirmation_and_shows_rejections(self):
        from unittest.mock import Mock
        from types import SimpleNamespace
        from PyQt5.QtWidgets import QMainWindow, QPushButton, QLabel
        from projector_control import ProjectorControl
        from test_projector_control import Parameters

        gui = self._make_gui()
        QMainWindow.__init__(gui)
        gui.btn_projector = QPushButton(gui)
        gui.btn_projector.setCheckable(True)
        gui.lbl_projector_status = QLabel(gui)
        gui.btn_projector.toggled.connect(gui._toggle_projector)
        gui.ros_node = Mock()
        gui.ros_node.get_publishers_info_by_topic.return_value = [
            SimpleNamespace(node_name="vp_realsense_splitter")]
        params = Parameters()
        gui._projector = ProjectorControl(params)
        try:
            gui._refresh_projector()
            self.assertFalse(gui.btn_projector.isEnabled())
            params.state(1, True)
            gui._refresh_projector()
            self.assertTrue(gui.btn_projector.isChecked())
            self.assertTrue(gui.btn_projector.isEnabled())
            # Synchronizing the button must not issue a parameter write.
            self.assertEqual(len(params.requests), 1)
            gui.btn_projector.click()
            self.assertFalse(gui.btn_projector.isEnabled())
            self.assertIn("APPLYING", gui.btn_projector.text())
            self.assertTrue(gui.btn_projector.isChecked())
            for _ in range(2):
                params.accepted()
                gui._refresh_projector()
            params.state(0, False)
            gui._refresh_projector()
            self.assertFalse(gui.btn_projector.isChecked())
            self.assertTrue(gui.btn_projector.isEnabled())
            self.assertEqual("IR PROJECTOR: OFF", gui.btn_projector.text())
            gui.btn_projector.click()
            params.accepted(False, "device busy")
            gui._refresh_projector()
            self.assertIn("device busy", gui.lbl_projector_status.text())
            self.assertFalse(gui.btn_projector.isEnabled())
        finally:
            gui._projector.close()
            gui.deleteLater()


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
        # The fixture's printed square, whatever this rig's tracking.yaml says.
        gui.marker_width.setValue(obs.marker_size_m)
        gui.btn_marker_lineup.setChecked(True)
        def cleanup():
            gui._marker_shutdown = True
            gui.map_viewer.deleteLater()
            gui.deleteLater()
            self.qapp.sendPostedEvents(None, QEvent.DeferredDelete)
        self.addCleanup(cleanup)
        return gui, obs, truth

    def _mapped_gui(self, scene=None):
        from marker_map import locator_transform
        from test_marker_map import fixture
        gui, _, _ = self._marker_gui()
        gui.btn_marker_lineup.setChecked(False)
        observation, truth, records = scene or fixture()
        view = gui.map_viewer.persp_view
        for i, record in enumerate(records):
            if i:
                view.duplicate_locator()
            view.set_locator_transform(locator_transform(record))
            view.assign_locator_marker(record['dictionary'], record['id'], record['size_m'])
        marker = observation.markers[0]
        # The engine keeps the first camera_info; this scene's camera is its own.
        gui.engine.intrinsics_from_camera_info = False
        gui.engine.set_intrinsics(marker.K)
        for stamp, T in ((11., marker.T_slam_image), (11.1, marker.T_slam_depth)):
            gui.engine.accept_pose(T[:3,3], matrix_to_quaternion(T[:3,:3]),
                                   [0,0,0], [0,0,0], stamp=stamp)
        gui.engine.set_depth(marker.depth, marker.depth_K, stamp=11.1)
        for m in observation.markers:
            m.captured_at = time.monotonic()
        # These fixtures exercise the rigid solve after scale calibration.
        gui._toggle_marker_lineup(True, calibrate_scale=False)
        return gui, observation, truth, records

    @staticmethod
    def _mapped_measurement(observation, truth):
        from marker_map import MappedMeasurement
        from lidar_map_matcher import MapMeasurement
        scan = MapMeasurement(ok=True, T_slam_to_map=truth, fitness=.9,
                              verify_score=.9, observable_rank=1)
        scan.monotonic_timestamp = observation.captured_at
        return MappedMeasurement(scan, [(m.family, m.marker_id) for m in observation.markers], [], .1)

    def test_mapped_layout_selection_preserves_request_but_pose_edits_cancel(self):
        from marker_map import layout_signature
        gui, observation, truth, records = self._mapped_gui()
        view = gui.map_viewer.persp_view
        request = gui._marker_request
        self.assertEqual(layout_signature(request['records']), layout_signature(records))
        self.assertAlmostEqual(view._locator_marker_size, records[-1]['size_m'])
        view.outliner_list.setCurrentRow(4)
        self.assertEqual(gui._marker_request_changed(request), '')
        view._edit_channel(0, view._locator_pos[0] + .2)
        gui._finish_marker_lineup((request, observation,
            self._mapped_measurement(observation, truth), ''))
        self.assertFalse(gui.lidar_matcher.is_aligned)
        self.assertIn('locator poses changed', gui.lbl_marker_lineup.text())

    def test_rejected_survey_explains_positions_are_kept_and_orientations_measured(self):
        from lidar_map_matcher import MapMeasurement
        gui, observation, truth, records = self._mapped_gui()
        gui._finish_marker_lineup((gui._marker_request, observation,
            MapMeasurement(reason="Survey pose has 10% depth agreement"), ''))
        self.assertIn('10% depth agreement', gui.lbl_marker_lineup.text())
        self.assertIn('positions are kept fixed', gui.lbl_marker_lineup.text())
        self.assertIn('orientations are measured', gui.lbl_marker_lineup.text())
        self.assertNotIn('Clear ID', gui.lbl_marker_lineup.text())

    def _recovery_gui(self):
        gui, observation, truth, records = self._mapped_gui()
        gui.engine.fusion.enabled = gui.cfg['fusion']['enabled'] = True
        gui._finish_marker_lineup((gui._marker_request, observation,
            self._mapped_measurement(observation, truth), ''))
        packet = gui.engine.freed_packet()
        context = gui._marker_lock_context
        gui._marker_detection_sample = None
        gui.engine.gate.resyncs += 1
        gui._poll_marker_lineup()
        self.assertTrue(gui.btn_marker_lineup.isChecked())
        self.assertTrue(gui._marker_request['recovering'])
        self.assertIsNone(gui._marker_lock_context)
        self.assertFalse(gui.lidar_matcher.is_aligned)
        self.assertEqual(gui.engine.freed_packet(), packet)
        return gui, observation, truth, packet, context

    def test_reset_reacquires_with_two_markers_and_preserves_output_continuity(self):
        from dataclasses import replace
        from unittest.mock import patch
        from marker_map import MappedObservation
        with patch('time.monotonic', return_value=100.) as mono, \
                patch('time.time', return_value=1000.) as wall:
            gui, observation, truth, packet, context = self._recovery_gui()
            request = gui._marker_request
            # During the outage the camera actually travelled 40 cm. Recovery
            # must require fresh evidence, then slew the old output toward it.
            moved = observation.T_slam_depth.copy()
            moved[:3, 3] += [.4, 0, 0]
            for i in range(1, 4):
                mono.return_value = 100. + i*.5
                wall.return_value = 1000. + i*.5
                gui.engine.accept_pose(moved[:3, 3], matrix_to_quaternion(moved[:3, :3]),
                    [0,0,0], [0,0,0], stamp=12.+i)
                seen = MappedObservation(tuple(replace(m, captured_at=mono.return_value,
                    T_slam_depth=moved, T_slam_image=moved) for m in observation.markers[:2]))
                result = self._mapped_measurement(seen, truth)
                gui._finish_marker_lineup((request, seen, result, ''))
                self.assertEqual(gui.engine.freed_packet(), packet)
                if i < 3:
                    self.assertTrue(gui.engine._relineup_required)
                    self.assertFalse(gui.lidar_matcher.is_aligned)
            self.assertFalse(gui.engine._relineup_required)
            self.assertIsNone(gui._marker_request)
            self.assertIsNotNone(gui._marker_lock_context)
            self.assertTrue(gui.btn_marker_lineup.isChecked())
            self.assertEqual(gui._output_mode(), 'Recovering')
            mono.return_value += .02
            wall.return_value += .02
            before = gui.engine._output_T.copy()
            self.assertNotEqual(gui.engine.freed_packet(), packet)
            self.assertLessEqual(np.linalg.norm(gui.engine._output_T[:3, 3] - before[:3, 3]), .002001)

    def test_recovery_rejects_duplicate_stale_inconsistent_and_pre_reset_results(self):
        from dataclasses import replace
        from unittest.mock import patch
        from marker_map import MappedObservation
        with patch('time.monotonic', return_value=100.) as mono:
            gui, observation, truth, packet, old_context = self._recovery_gui()
            request = gui._marker_request
            def deliver(at, transform=truth, age=0):
                mono.return_value = at
                gui.engine._last_pose_mono = at
                seen = MappedObservation(tuple(replace(m, captured_at=at-age)
                                              for m in observation.markers[:2]))
                result = self._mapped_measurement(seen, transform)
                gui._finish_marker_lineup((request, seen, result, ''))
                self.assertEqual(gui.engine.freed_packet(), packet)
            deliver(100.5)
            deliver(100.5)  # repeated processing is not another observation
            self.assertEqual(len(request['recovery_results']), 1)
            wrong = truth.copy(); wrong[0, 3] += .2
            deliver(101., wrong)
            self.assertEqual(len(request['recovery_results']), 1)
            deliver(102., age=1.)
            # A stale result is not counted, and is not contrary evidence.
            self.assertEqual(len(request['recovery_results']), 1)
            self.assertEqual(request['recovery_results'][0].monotonic_timestamp, 101.)
            gui._finish_marker_lineup((old_context, observation,
                self._mapped_measurement(observation, truth), ''))
            self.assertIs(gui._marker_request, request)
            self.assertFalse(gui.lidar_matcher.is_aligned)
            gui.engine.gate.resyncs += 1
            gui._poll_marker_lineup()
            self.assertIsNot(gui._marker_request, request)
            self.assertEqual(gui._marker_request['recovery_results'], [])
            self.assertTrue(gui.btn_marker_lineup.isChecked())

    def test_a_rejected_view_between_agreeing_checks_does_not_restart_recovery(self):
        """Recorded 2026-11-14: checks reached 1/3 and 2/3, and every poor view
        in between (markers oblique, depth out of the doorway) reset the count,
        so recovery never completed. A rejected view adds no evidence; the
        agreeing checks still count, and too long a gap still restarts."""
        from dataclasses import replace
        from unittest.mock import patch
        from lidar_map_matcher import MapMeasurement
        from marker_map import MappedObservation
        with patch('time.monotonic', return_value=100.) as mono:
            gui, observation, truth, packet, _ = self._recovery_gui()
            request = gui._marker_request
            def deliver(at, ok=True):
                mono.return_value = at
                gui.engine._last_pose_mono = at
                seen = MappedObservation(tuple(replace(m, captured_at=at)
                                              for m in observation.markers[:2]))
                result = (self._mapped_measurement(seen, truth) if ok else
                          MapMeasurement(reason="Survey pose has 60% depth agreement, median miss 70 mm"))
                gui._finish_marker_lineup((request, seen, result, ''))
            deliver(100.5)
            deliver(101.0, ok=False)
            self.assertEqual(len(request['recovery_results']), 1)
            self.assertIn('1/3', gui.lbl_marker_lineup.text())
            self.assertIn('median miss 70 mm', gui.lbl_marker_lineup.text())
            self.assertNotIn('Turned Locator', gui.lbl_marker_lineup.text())
            deliver(101.5)
            deliver(102.0, ok=False)
            self.assertTrue(gui.engine._relineup_required)
            self.assertEqual(gui.engine.freed_packet(), packet)
            deliver(102.5)
            self.assertFalse(gui.engine._relineup_required)
            self.assertIsNone(gui._marker_request)
            self.assertIsNotNone(gui._marker_lock_context)

    def test_recovery_checks_too_far_apart_in_time_start_the_count_again(self):
        from dataclasses import replace
        from unittest.mock import patch
        from marker_map import MappedObservation
        with patch('time.monotonic', return_value=100.) as mono:
            gui, observation, truth, packet, _ = self._recovery_gui()
            request = gui._marker_request
            for at in (100.5, 101.0, 105.0):
                mono.return_value = at
                gui.engine._last_pose_mono = at
                seen = MappedObservation(tuple(replace(m, captured_at=at)
                                              for m in observation.markers[:2]))
                gui._finish_marker_lineup((request, seen, self._mapped_measurement(seen, truth), ''))
            self.assertEqual(len(request['recovery_results']), 1)
            self.assertTrue(gui.engine._relineup_required)

    def test_recovery_waits_through_occlusion_but_scene_edits_cancel_it(self):
        from unittest.mock import patch
        with patch('time.monotonic', return_value=100.) as mono:
            gui, observation, truth, packet, _ = self._recovery_gui()
            request = gui._marker_request
            for _ in range(10):
                mono.return_value += .1
                gui._poll_marker_lineup()  # no markers visible
                self.assertTrue(gui.btn_marker_lineup.isChecked())
                self.assertEqual(gui.engine.freed_packet(), packet)
            gui.map_viewer.persp_view._edit_channel(0, 2.)
            gui._poll_marker_lineup()
            self.assertFalse(gui.btn_marker_lineup.isChecked())
            self.assertIsNone(gui._marker_request)

    def test_output_mode_distinguishes_held_packets_from_live_tracking(self):
        from unittest.mock import patch
        with patch('time.monotonic', return_value=100.):
            gui, observation, truth, packet, _ = self._recovery_gui()
            self.assertEqual(gui._output_mode(), 'Held')
            gui.streaming_enabled = False
            self.assertEqual(gui._output_mode(), 'Paused')

    def test_locked_lineup_that_must_be_redone_reacquires_by_itself(self):
        """'Camera moved too far during hold' must not leave the lineup dead."""
        from unittest.mock import patch
        with patch('time.monotonic', return_value=100.):
            gui, observation, truth, _ = self._mapped_gui()
            gui.engine.fusion.enabled = gui.cfg['fusion']['enabled'] = True
            gui._finish_marker_lineup((gui._marker_request, observation,
                self._mapped_measurement(observation, truth), ''))
            gui._marker_detection_sample = None
            gui.engine._relineup_required = True
            gui._poll_marker_lineup()
            self.assertTrue(gui.btn_marker_lineup.isChecked())
            self.assertTrue(gui._marker_request['recovering'])

    def test_explicit_lineup_is_not_cancelled_by_a_pending_relineup(self):
        """Arming Auto lineup is how the operator answers RELINEUP; the
        pending RELINEUP must not cancel that same lineup."""
        gui, _, _ = self._marker_gui()
        request = gui._marker_request
        self.assertIsNotNone(request)
        gui._marker_detection_sample = None
        gui.engine._relineup_required = True
        gui._poll_marker_lineup()
        self.assertTrue(gui.btn_marker_lineup.isChecked())
        self.assertIs(gui._marker_request, request)
        self.assertFalse(request.get('recovering', False))

    @staticmethod
    def _steady_frames(gui, mono, detections, times):
        """Show the markers for a few frames: Locator rotations need steady views."""
        for at in times:
            mono.return_value = at
            gui._marker_detection_sample = (detections, 11., at, (848, 480))
            gui._poll_marker_lineup()

    def test_mapped_acquisition_and_live_support_use_background_joint_solver(self):
        from unittest.mock import patch, Mock
        from test_marker_map import room_fixture
        with patch('time.monotonic', return_value=100.) as mono:
            gui, observation, truth, _ = self._mapped_gui(room_fixture())
            measurement = self._mapped_measurement(observation, truth)
            gui.marker_lineup_finished.connect(gui._finish_marker_lineup)
            def run_inline(**kwargs):
                return Mock(start=kwargs['target'])
            detections = [(m.family, m.marker_id, m.corners) for m in observation.markers]
            with patch('dashboard.threading.Thread', side_effect=run_inline), \
                    patch('dashboard.solve_mapped_observation', return_value=measurement) as solve:
                self._steady_frames(gui, mono, detections, (100.01, 100.03))
                solve.assert_not_called()
                mono.return_value = 100.05
                gui._marker_detection_sample = (detections, 11., 100.05, (848,480))
                gui._poll_marker_lineup()
                solve.assert_called_once()
                self.assertEqual(len(solve.call_args.args[0].markers), 3)
                self.assertTrue(gui.lidar_matcher.is_aligned)
                generation = gui.lidar_matcher._generation
                # Same frame is never recycled as new evidence.
                gui._poll_marker_lineup()
                self.assertEqual(solve.call_count, 1)
                mono.return_value = 100.60
                gui.engine._last_pose_mono = mono.return_value
                gui.engine.set_depth(observation.markers[0].depth, observation.markers[0].depth_K, stamp=11.1)
                gui._marker_detection_sample = (detections, 11., 100.60, (848,480))
                measurement.monotonic_timestamp = 100.60
                gui._poll_marker_lineup()
                self.assertEqual(solve.call_count, 2)
                self.assertEqual(gui.lidar_matcher._generation, generation)
                gui._push_map_alignment.assert_called_once()  # no new initial lineup
                self.assertIs(gui.lidar_matcher.last_marker_measurement, measurement)

    def test_lineup_stands_wrongly_rotated_wall_locators_up_before_solving(self):
        from unittest.mock import patch, Mock
        from marker_map import layout_signature, locator_transform
        from lidar_map_matcher import MapMeasurement
        from test_marker_map import ROOM_MARKERS, room_fixture, rotation_error_deg
        # As on the rig: ID 1 left flat after Duplicate, ID 2 stood up sideways.
        with patch('time.monotonic', return_value=100.) as mono:
            gui, observation, truth, records = self._mapped_gui(room_fixture({1: (0, 0, 0), 2: (90, 0, 0)}))
            request = gui._marker_request
            detections = [(m.family, m.marker_id, m.corners) for m in observation.markers]
            with patch('dashboard.threading.Thread', side_effect=lambda **kw: Mock(start=kw['target'])), \
                    patch('dashboard.solve_mapped_observation',
                          return_value=MapMeasurement(reason="Survey pose has 40% depth agreement")) as solve:
                gui.marker_lineup_finished.connect(gui._finish_marker_lineup)
                self._steady_frames(gui, mono, detections, (100.01, 100.03))
                solve.assert_not_called()
                self.assertIn('waiting to measure their orientations', gui.lbl_marker_lineup.text())
                mono.return_value = 100.05
                gui._marker_detection_sample = (detections, 11., 100.05, (848, 480))
                gui._poll_marker_lineup()
        placed = {r['id']: locator_transform(r) for r in gui._mapped_marker_records()}
        for fid, (position, _) in ROOM_MARKERS.items():
            truth_locator = locator_transform(dict(position_m=position, rotation_deg=ROOM_MARKERS[fid][1]))
            self.assertLess(rotation_error_deg(placed[fid], truth_locator), 2, fid)
            np.testing.assert_allclose(placed[fid][:3, 3], position, atol=1e-9)
        solved = {m.marker_id: m.T_map_locator for m in solve.call_args.args[0].markers}
        for fid in (1, 2):
            np.testing.assert_allclose(solved[fid], placed[fid], atol=1e-9)
        self.assertEqual(request['layout_signature'], layout_signature(gui._mapped_marker_records()))
        text = gui.lbl_marker_lineup.text()
        self.assertIn('40% depth agreement', text)
        self.assertIn('Turned Locator ID 1 upright on its wall (90°), ID 2 upright on its wall (90°)', text)
        self.assertNotIn('changed', text)

    def test_unclear_orientations_wait_instead_of_using_the_hand_entered_angles(self):
        from unittest.mock import patch
        from test_marker_map import room_fixture
        with patch('time.monotonic', return_value=100.) as mono:
            gui, observation, _, _ = self._mapped_gui(room_fixture(depth=False))
            request = gui._marker_request
            before = gui.map_viewer.persp_view.locator_entries()
            mono.return_value = 100.05
            detections = [(m.family, m.marker_id, m.corners) for m in observation.markers]
            gui._marker_detection_sample = (detections, 11., 100.05, (848, 480))
            with patch('dashboard.solve_mapped_observation') as solve:
                gui._poll_marker_lineup()
                solve.assert_not_called()
            self.assertIs(gui._marker_request, request)
            self.assertTrue(gui.btn_marker_lineup.isChecked())
            self.assertIn('waiting to measure their orientations', gui.lbl_marker_lineup.text())
            for key, entry in gui.map_viewer.persp_view.locator_entries().items():
                np.testing.assert_array_equal(entry['transform'][:3, 3], before[key]['transform'][:3, 3])

    def _locator_rotations(self, gui):
        return {e['marker']['id']: e['transform'][:3, :3].copy()
                for e in gui.map_viewer.persp_view.locator_entries().values() if e['marker']}

    def test_jittering_rotation_measurements_never_turn_a_locator(self):
        """Recorded 2026-11-14: from a poor view each frame's measured rotation
        was 5-9 degrees out, and turning Locators from it made the layout
        disagree. Frames that do not agree must leave the Locators alone."""
        from unittest.mock import patch
        import dashboard as dash
        from test_marker_map import room_fixture
        real = dash.level_locators_from_view
        flip = [1.]
        def jitter(*args, **kwargs):
            corrections, note = real(*args, **kwargs)
            flip[0] = -flip[0]
            turn = rpy_to_matrix(0, 0, np.radians(4.) * flip[0])
            out = []
            for key, T, degrees, on_wall in corrections:
                T = T.copy()
                T[:3, :3] = T[:3, :3] @ turn
                out.append((key, T, degrees, on_wall))
            return out, note
        with patch('time.monotonic', return_value=100.) as mono:
            gui, observation, _, _ = self._mapped_gui(room_fixture({1: (0, 0, 0), 2: (90, 0, 0)}))
            before = self._locator_rotations(gui)
            detections = [(m.family, m.marker_id, m.corners) for m in observation.markers]
            with patch('dashboard.level_locators_from_view', side_effect=jitter), \
                    patch('dashboard.solve_mapped_observation') as solve:
                self._steady_frames(gui, mono, detections, [100.01 + .02 * i for i in range(8)])
                solve.assert_not_called()
            for fid, rotation in self._locator_rotations(gui).items():
                np.testing.assert_array_equal(rotation, before[fid])
            self.assertIn('Measuring the rotation of marker(s)', gui.lbl_marker_lineup.text())
            self.assertTrue(gui.btn_marker_lineup.isChecked())

    def test_edge_on_marker_is_not_turned_and_the_solve_uses_the_measured_ones(self):
        from unittest.mock import patch, Mock
        from lidar_map_matcher import MapMeasurement
        from test_marker_map import room_fixture
        with patch('time.monotonic', return_value=100.) as mono:
            gui, observation, _, _ = self._mapped_gui(room_fixture({1: (0, 0, 0), 2: (90, 0, 0)}))
            # Marker 3 is seen 50 degrees off face-on in this fixture.
            gui.cfg['aruco']['orientation_max_view_deg'] = 40.
            before = self._locator_rotations(gui)
            request = gui._marker_request
            detections = [(m.family, m.marker_id, m.corners) for m in observation.markers]
            with patch('dashboard.threading.Thread', side_effect=lambda **kw: Mock(start=kw['target'])), \
                    patch('dashboard.solve_mapped_observation',
                          return_value=MapMeasurement(reason="not finished here")) as solve:
                self._steady_frames(gui, mono, detections, (100.01, 100.03, 100.05))
                solve.assert_called_once()
            self.assertEqual(sorted(m.marker_id for m in solve.call_args.args[0].markers), [1, 2])
            np.testing.assert_array_equal(self._locator_rotations(gui)[3], before[3])
            self.assertGreater(np.degrees(np.arccos(np.clip(
                (np.trace(before[1].T @ self._locator_rotations(gui)[1]) - 1) / 2, -1, 1))), 45)
            self.assertIn('Marker(s) 3 are too small or too edge-on', request['level_note'])
            self.assertNotIn(3, [key[1] for key in request['oriented_marker_keys']])

    def test_orientation_measurement_preserves_exact_displayed_centres(self):
        from test_marker_map import room_fixture
        from marker_map import MappedObservation
        from dataclasses import replace
        gui, observation, _, _ = self._mapped_gui(room_fixture({1: (0, -85, 4)}))
        gui.map_viewer.set_alignment(1.1234, -.5678, .9012, 12.345, 8.901, -67.891, 1.)
        rigid = gui._align_rigid()
        view = gui.map_viewer.persp_view
        entries = view.locator_entries()
        for key, entry in entries.items():
            view.set_locator_transform(rigid @ entry['transform'], key)
        before = view.locator_entries()
        request = dict(gui._marker_request, rigid=rigid)
        observation = MappedObservation(tuple(replace(m, up_map=np.array([0., 0., 1.]))
                                               for m in observation.markers))
        for _ in range(3):      # rotations are applied from steady frames only
            gui._level_assigned_locators(request, observation)
        self.assertTrue(request['orientation_ready'])
        for key, entry in view.locator_entries().items():
            np.testing.assert_array_equal(entry['transform'][:3, 3], before[key]['transform'][:3, 3])

    def test_marker_not_seen_during_orientation_measurement_cannot_support_tracking(self):
        from unittest.mock import patch
        from test_marker_map import room_fixture
        with patch('time.monotonic', return_value=100.) as mono:
            gui, obs, _, _ = self._mapped_gui(room_fixture())
            request = gui._marker_request
            request['oriented_marker_keys'] = tuple((m.family, m.marker_id) for m in obs.markers[:2])
            gui._marker_request = None
            gui._marker_lock_context = request
            mono.return_value = 100.1
            m = obs.markers[2]
            gui._marker_detection_sample = ([(m.family, m.marker_id, m.corners)], 11., 100.1, (848, 480))
            with patch('dashboard.solve_mapped_observation') as solve:
                gui._poll_marker_lineup()
                solve.assert_not_called()
            self.assertFalse(gui._marker_busy)

    def _scale_gui(self, measured):
        """A lined-up window with Setup → Scan scale, and measure_scale answering `measured`."""
        from unittest.mock import patch, Mock
        gui, obs, truth = self._marker_gui()
        gui.btn_marker_lineup.setChecked(False)
        gui.lidar_cfg = dict(gui.cfg.get("lidar", {}), map_scale=.01)
        gui._build_scan_scale_panel().setParent(gui)
        gui.scan_scale_finished.connect(gui._finish_scale_measurement)
        matcher = gui.lidar_matcher
        matcher._map_native = np.random.default_rng(0).random((400, 3)) * 300
        gui.map_viewer.set_alignment_scale(.01)
        gui._align_rigid()
        matcher.is_aligned, matcher.T_slam_to_map = True, truth.copy()
        threads = patch('dashboard.threading.Thread', side_effect=lambda **kw: Mock(start=kw['target']))
        measure = patch('dashboard.measure_scale', return_value=dict(
            dict(ok=True, scale=1., sigma=.01, inliers=.8, rmse_m=.015, lever_m=1.2), **measured))
        threads.start()
        self.addCleanup(threads.stop)
        return gui, obs, truth, measure.start(), measure

    def test_scale_sweep_measures_then_apply_rescales_scan_and_locators(self):
        from marker_map import locator_transform
        gui, obs, truth, measure, patcher = self._scale_gui(dict(scale=.95))
        self.addCleanup(patcher.stop)
        view = gui.map_viewer.persp_view
        view.duplicate_locator()
        view.set_locator_transform(locator_transform(dict(position_m=(1., -2., .5), rotation_deg=(0, -90, 90))))
        before = {key: entry['transform'].copy() for key, entry in view.locator_entries().items()}
        gui._toggle_scale_sweep()
        self.assertEqual(gui.btn_measure_scale.text(), "Stop and measure")
        self.assertEqual(gui._scale_sweep['sweep'].frames, 1)
        gui._collect_scale_sweep()            # the same depth frame is not added twice
        self.assertEqual(gui._scale_sweep['sweep'].frames, 1)
        gui._toggle_scale_sweep()             # stop early and measure
        measure.assert_called_once()
        points, scan, T = measure.call_args.args
        self.assertGreater(len(points), 100)
        np.testing.assert_allclose(scan, gui.lidar_matcher._map_native * .01)
        np.testing.assert_allclose(T, truth)
        text = gui.lbl_scan_scale.text()
        self.assertIn("5.0% smaller than the room", text)
        self.assertIn("from 0.01 to 0.010526 m per unit", text)
        self.assertTrue(gui.btn_apply_scale.isEnabled())
        self.assertEqual(gui.btn_measure_scale.text(), "Measure scan scale")
        gui._apply_scan_scale()
        self.assertAlmostEqual(gui.map_viewer.alignment_scale, .01 / .95)
        self.assertAlmostEqual(gui.lidar_matcher.map_scale, .01 / .95)
        self.assertAlmostEqual(gui.lidar_cfg["map_scale"], .01 / .95)
        for key, entry in view.locator_entries().items():
            np.testing.assert_allclose(entry['transform'][:3, 3], before[key][:3, 3] / .95, atol=1e-9)
            np.testing.assert_allclose(entry['transform'][:3, :3], before[key][:3, :3], atol=1e-9)
        self.assertIn("Run Auto lineup again", gui.lbl_scan_scale.text())
        self.assertFalse(gui.btn_apply_scale.isEnabled())

    def test_scale_within_the_cameras_accuracy_needs_no_change(self):
        gui, *_, patcher = self._scale_gui(dict(scale=.985))
        self.addCleanup(patcher.stop)
        gui._toggle_scale_sweep()
        gui._toggle_scale_sweep()
        self.assertIn("No change needed", gui.lbl_scan_scale.text())
        self.assertFalse(gui.btn_apply_scale.isEnabled())

    def test_scale_sweep_refusals_are_reported_and_nothing_changes(self):
        gui, *_, patcher = self._scale_gui(dict(ok=False, reason="This sweep can't measure scale: a corner."))
        self.addCleanup(patcher.stop)
        gui._toggle_scale_sweep()
        gui._toggle_scale_sweep()
        self.assertEqual(gui.lbl_scan_scale.text(), "This sweep can't measure scale: a corner.")
        self.assertFalse(gui.btn_apply_scale.isEnabled())
        self.assertAlmostEqual(gui.map_viewer.alignment_scale, .01)

    def test_scale_needs_a_lineup_and_a_steady_scan(self):
        gui, *_, patcher = self._scale_gui(dict(scale=.95))
        self.addCleanup(patcher.stop)
        gui.lidar_matcher.is_aligned = False
        gui._toggle_scale_sweep()
        self.assertIn("Line up first", gui.lbl_scan_scale.text())
        self.assertIsNone(gui._scale_sweep)
        gui.lidar_matcher.is_aligned = True
        gui._toggle_scale_sweep()
        gui.engine.gate.resyncs += 1
        gui._collect_scale_sweep()
        self.assertIn("Tracking jumped", gui.lbl_scan_scale.text())
        self.assertIsNone(gui._scale_sweep)
        gui._toggle_scale_sweep()
        gui._toggle_scale_sweep()
        self.assertTrue(gui.btn_apply_scale.isEnabled())
        gui.map_viewer.set_alignment_scale(.02)
        gui._align_rigid()
        gui._apply_scan_scale()
        self.assertIn("changed since the measurement", gui.lbl_scan_scale.text())
        self.assertAlmostEqual(gui.map_viewer.alignment_scale, .02)

    def _marker_scale_gui(self, nudges, grow=1.05):
        """Assigned scan Locators; depth verification answers with the known scale."""
        from unittest.mock import patch, Mock
        from test_marker_map import ROOM_MARKERS, room_fixture
        positions = {fid: grow * np.asarray(position) + nudges.get(fid, 0.)
                     for fid, (position, _) in ROOM_MARKERS.items()}
        gui, observation, T_map_slam, _ = self._mapped_gui(room_fixture(positions=positions))
        gui.lidar_cfg = dict(gui.cfg.get("lidar", {}), map_scale=.01)
        gui._build_scan_scale_panel().setParent(gui)
        gui.scan_scale_finished.connect(gui._finish_scale_measurement)
        gui.lidar_matcher._map_native = np.random.default_rng(0).random((400, 3)) * 300
        gui.map_viewer.set_alignment_scale(.01)
        gui._align_rigid()
        self.assertFalse(gui.lidar_matcher.is_aligned)
        truth = T_map_slam.copy()
        truth[:3] *= grow
        threads = patch('dashboard.threading.Thread', side_effect=lambda **kw: Mock(start=kw['target']))
        measure = patch('dashboard.verify_locator_scale', return_value=dict(
            ok=True, scale=grow, sigma=.01, inliers=.8, rmse_m=.015, lever_m=1.2, T=truth))
        threads.start()
        self.addCleanup(threads.stop)
        self.addCleanup(measure.stop)
        detections = [(m.family, m.marker_id, m.corners) for m in observation.markers]

        def show_markers():
            gui._marker_detection_sample = (detections, 11., time.monotonic(), (848, 480))
            gui._collect_scale_sweep()
        return gui, truth, measure.start(), show_markers

    def test_markers_scale_scan_with_all_locator_placements_preserved(self):
        gui, truth, measure, show_markers = self._marker_scale_gui(
            {1: (.006, -.004, .003), 2: (-.005, .003, -.004), 3: (.004, .007, 0.)})
        view = gui.map_viewer.persp_view
        # Include a Locator with no ID and a translated/rotated display frame.
        view.duplicate_locator()
        view.clear_locator_marker()
        gui.map_viewer.set_alignment(2., -1., .5, 10., -8., 25., .01)
        rigid = gui._align_rigid()
        before = view.locator_entries()
        self.assertIsNotNone(gui._marker_request)
        gui._toggle_marker_lineup(True)             # normal Auto lineup starts calibration
        self.assertIsNone(gui._marker_request)      # Auto lineup waits for the result
        self.assertEqual(gui.btn_marker_lineup.text(), "Auto lineup · Measuring scale")
        self.assertEqual(gui.btn_marker_scale.text(), "Stop and measure")
        self.assertFalse(gui.btn_measure_scale.isEnabled())
        for _ in range(3):
            show_markers()
        self.assertIn("Seen: ID 1 ×3, ID 2 ×3, ID 3 ×3", gui.lbl_scan_scale.text())
        gui._toggle_scale_sweep(markers=True)        # stop and measure
        measure.assert_called_once()
        _, scan, fit = measure.call_args.args
        np.testing.assert_allclose(scan, gui.lidar_matcher._map_native * .01)
        self.assertAlmostEqual(fit["scale"], 1.05, delta=.01)

        self.assertAlmostEqual(gui.map_viewer.alignment_scale, .01 / 1.05)
        self.assertAlmostEqual(gui.lidar_matcher.map_scale, .01 / 1.05)
        self.assertAlmostEqual(gui.lidar_cfg["map_scale"], .01 / 1.05)
        for key, entry in view.locator_entries().items():
            old = np.linalg.inv(rigid) @ before[key]["transform"]
            new = np.linalg.inv(rigid) @ entry["transform"]
            np.testing.assert_allclose(new[:3, 3], old[:3, 3] / 1.05, atol=1e-9)
            np.testing.assert_allclose(new[:3, :3], old[:3, :3], atol=1e-9)
            self.assertEqual(entry["marker"], before[key]["marker"])
        placed = {e["marker"]["id"]: np.linalg.inv(rigid) @ e["transform"]
                  for e in view.locator_entries().values() if e["marker"]}
        text = gui.lbl_scan_scale.text()
        self.assertIn("The scan was 5.0% larger than the room", text)
        self.assertIn("from 0.01 to 0.0095238 m per unit", text)
        self.assertIn("All Locators kept their positions on the scan", text)
        self.assertIn("Auto lineup is running", text)
        # Auto lineup starts again from the moved Locators.
        self.assertTrue(gui.btn_marker_lineup.isChecked())
        request = gui._marker_request
        self.assertIsNotNone(request)
        for record in request["records"]:
            np.testing.assert_allclose(record["position_m"], placed[record["id"]][:3, 3], atol=1e-9)
        self.assertEqual(gui.btn_marker_scale.text(), "Scale and line up from markers")
        self.assertTrue(gui.btn_measure_scale.isEnabled())
        self.assertFalse(gui.btn_apply_scale.isEnabled())

    def test_marker_scale_refuses_unseen_or_misplaced_markers_and_changes_nothing(self):
        for nudges, sightings, expected in (
                ({}, 1, "Too few assigned markers were seen steadily (seen: none)"),
                ({3: (1.4, -.6, 0.)}, 3, "do not agree under one uniform scale")):
            with self.subTest(expected=expected):
                gui, _, measure, show_markers = self._marker_scale_gui(nudges)
                before = {key: e["transform"].copy()
                          for key, e in gui.map_viewer.persp_view.locator_entries().items()}
                gui._toggle_scale_sweep(markers=True)
                for _ in range(sightings):
                    show_markers()
                gui._toggle_scale_sweep(markers=True)
                self.assertIn(expected, gui.lbl_scan_scale.text())
                measure.assert_not_called()
                self.assertIsNone(gui._scale_sweep)
                self.assertAlmostEqual(gui.map_viewer.alignment_scale, .01)
                for key, e in gui.map_viewer.persp_view.locator_entries().items():
                    np.testing.assert_allclose(e["transform"], before[key])
                self.assertEqual(gui.btn_marker_scale.text(), "Scale and line up from markers")
                self.assertTrue(gui.btn_measure_scale.isEnabled())

    def test_marker_scale_needs_three_assigned_locators(self):
        gui, *_ = self._marker_scale_gui({})
        view = gui.map_viewer.persp_view
        for key in list(view.locator_entries())[2:]:
            view._locators[key]["marker"] = None
        gui._toggle_scale_sweep(markers=True)
        self.assertIn("Assign marker IDs to at least three Locators", gui.lbl_scan_scale.text())
        self.assertIsNone(gui._scale_sweep)

    def test_cancelled_or_changed_scale_request_cannot_apply_late_result(self):
        for action in ("cancel", "locator", "scan", "tracking"):
            with self.subTest(action=action):
                gui, _, measure, _ = self._marker_scale_gui({})
                gui._toggle_marker_lineup(True)
                request = gui._scale_sweep
                if action == "cancel":
                    gui.btn_marker_lineup.setChecked(False)
                elif action == "locator":
                    T = gui.map_viewer.get_locator_transform().copy()
                    T[0, 3] += .1
                    gui.map_viewer.set_locator_transform(T)
                elif action == "scan":
                    gui.map_viewer.set_alignment(1, 0, 0, 0, 0, 0, .01)
                else:
                    gui.engine.gate.resyncs += 1
                before = gui.map_viewer.persp_view.locator_entries()
                gui._finish_scale_measurement((request, measure.return_value, ""))
                self.assertIsNone(gui._scale_sweep)
                self.assertAlmostEqual(gui.map_viewer.alignment_scale, .01)
                for key, entry in gui.map_viewer.persp_view.locator_entries().items():
                    np.testing.assert_array_equal(entry["transform"], before[key]["transform"])

    def test_cancelled_marker_support_cannot_install_late_result(self):
        gui, observation, truth, _ = self._mapped_gui()
        measurement = self._mapped_measurement(observation, truth)
        gui._finish_marker_lineup((gui._marker_request, observation, measurement, ''))
        job = dict(gui._marker_lock_context, continuous=True)
        gui._marker_tracking_job = job
        gui.btn_marker_lineup.setChecked(False)
        gui._finish_marker_lineup((job, observation, measurement, ''))
        self.assertIsNone(gui.lidar_matcher.last_marker_measurement)
        self.assertIsNone(gui._marker_lock_context)
        np.testing.assert_allclose(gui.lidar_matcher.T_slam_to_map, truth)

    def test_marker_layout_roundtrip_uses_scan_coordinates(self):
        from unittest.mock import patch
        from tempfile import TemporaryDirectory
        from pathlib import Path
        from marker_map import layout_signature, load_layout
        gui, _, _, records = self._mapped_gui()
        gui.btn_marker_lineup.setChecked(False)
        rigid = np.eye(4)
        rigid[:3,:3] = rpy_to_matrix(.1,.2,.3)
        rigid[:3,3] = [3,4,5]
        gui.map_viewer.T_align = rigid.copy()
        view = gui.map_viewer.persp_view
        for key, entry in view.locator_entries().items():
            view.set_locator_transform(rigid @ entry['transform'], key)
        with TemporaryDirectory() as folder:
            path = str(Path(folder)/'markers.json')
            with patch('dashboard.QFileDialog.getSaveFileName', return_value=(path, '')):
                gui._save_marker_layout()
            self.assertEqual(layout_signature(load_layout(path)), layout_signature(records))
            view.clear_locator_marker()
            with patch('dashboard.QFileDialog.getOpenFileName', return_value=(path, '')):
                gui._load_marker_layout()
            self.assertEqual(layout_signature(gui._mapped_marker_records()), layout_signature(records))
            self.assertEqual(len(view.locator_entries()), 3)

    def _save_scene(self, gui, path):
        from unittest.mock import patch
        with patch('dashboard.QFileDialog.getSaveFileName', return_value=(path, '')):
            gui._menu_save_scene()

    def _open_scene(self, gui, path):
        from unittest.mock import patch
        with patch('dashboard.QFileDialog.getOpenFileName', return_value=(path, '')), \
                patch('dashboard.QMessageBox.warning') as warning:
            gui._menu_open_scene()
        return warning

    @staticmethod
    def _one_locator():
        return {"locator": dict(name="Locator", transform=np.eye(4), marker=None)}

    def test_open_scene_restores_what_save_scene_wrote(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path
        gui, *_ = self._mapped_gui()
        gui.btn_marker_lineup.setChecked(False)
        viewer, view = gui.map_viewer, gui.map_viewer.persp_view
        viewer.set_alignment(1.5, -2., .25, 0., 0., 30., .01)
        trail = np.eye(4)
        trail[:3, 3] = [.1, .2, .3]
        view.set_display_transform("trail", trail)
        # The tracking origin is whichever Locator is selected; make it not the first.
        view.outliner_list.setCurrentRow(5)
        active = view._active_locator_id
        self.assertNotEqual(active, "locator")
        before, width = view.locator_entries(), gui.marker_width.value()
        with TemporaryDirectory() as folder:
            path = str(Path(folder) / "scene.json")
            self._save_scene(gui, path)
            viewer.set_alignment(0, 0, 0, 0, 0, 0, 1.)
            view.set_display_transform("trail", np.eye(4))
            view.set_locator_entries(self._one_locator())
            gui.marker_width.setValue(.05)
            self._open_scene(gui, path).assert_not_called()
        self.assertEqual((viewer.tx, viewer.ty, viewer.tz, viewer.yaw), (1.5, -2., .25, 30.))
        self.assertAlmostEqual(viewer.alignment_scale, .01)
        np.testing.assert_allclose(view.get_display_transform("trail"), trail, atol=1e-12)
        after = view.locator_entries()
        self.assertEqual(list(after), list(before))
        for key, entry in before.items():
            self.assertEqual((after[key]["name"], after[key]["marker"]), (entry["name"], entry["marker"]))
            np.testing.assert_allclose(after[key]["transform"], entry["transform"], atol=1e-9)
        self.assertEqual(view._active_locator_id, active)
        np.testing.assert_allclose(gui.engine.anchor_target, before[active]["transform"], atol=1e-9)
        self.assertAlmostEqual(gui.marker_width.value(), width)
        self.assertIn("Scene opened", gui.lbl_marker_lineup.text())
        self.assertNotIn(view.duplicate_locator(), before)

    def test_open_scene_imports_its_scan_first_and_finds_it_beside_the_scene(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path
        from unittest.mock import patch
        gui, *_ = self._mapped_gui()
        gui.btn_marker_lineup.setChecked(False)
        gui.lidar_matcher = None      # the import's own fusion handoff is not under test
        view = gui.map_viewer.persp_view
        before = view.locator_entries()
        with TemporaryDirectory() as folder:
            scan = str(Path(folder) / "room.ply")
            Path(scan).write_text("")
            gui._map_path = "/moved/away/room.ply"
            path = str(Path(folder) / "scene.json")
            self._save_scene(gui, path)
            view.set_locator_entries(self._one_locator())
            with patch.object(gui, "_on_map_loaded") as load:
                self._open_scene(gui, path).assert_not_called()
            load.assert_called_once_with(scan)
            self.assertEqual(len(view.locator_entries()), 1)        # waits for the scan
            with patch('dashboard.QMessageBox.warning') as warning:
                gui._on_import_finished(None, None, None, scan)
            warning.assert_called_once()
            self.assertEqual(len(view.locator_entries()), 1)
            with patch.object(gui, "_on_map_loaded"):
                self._open_scene(gui, path)
            gui._on_import_finished(object(), np.random.default_rng(0).random((50, 3)), None, scan)
        self.assertEqual(list(view.locator_entries()), list(before))
        self.assertEqual(gui._map_path, scan)
        self.assertIsNone(gui._pending_scene)

    def test_reset_scene_keeps_the_scene_and_restarts_auto_lineup(self):
        from unittest.mock import patch
        from dashboard import QMessageBox
        gui, obs, truth = self._marker_gui()
        self._install_verified_marker(gui, obs, truth)
        viewer, view = gui.map_viewer, gui.map_viewer.persp_view
        gui._map_path = "/scans/room.obj"
        map_cloud = gui.lidar_matcher.map_cloud
        viewer.set_alignment(1.5, -2., .25, 0., 0., 30., .02)
        gui._align_rigid()
        gui._push_map_alignment()
        moved = np.eye(4)
        moved[:3, 3] = [.1, .2, .3]
        view.set_display_transform("trail", moved)
        view.duplicate_locator()
        view.set_locator_transform(moved)
        gui._on_locator_moved(view.get_locator_transform())    # as dragging it does
        viewer.update_camera_pose((.4, .5, .6), (0, 0, 0, 1))
        gui._toggle_marker_lineup(False)           # lineup done, box unticked
        gui.engine._relineup_required = True       # stuck: camera moved during a hold
        self.assertTrue(gui.engine.is_anchored)
        self.assertTrue(gui.lidar_matcher.is_aligned)
        scene = (view.locator_entries(), view._active_locator_id, viewer.T_align.copy(),
                 viewer.alignment_scale, view.get_display_transform("trail"),
                 gui.engine.anchor_target.copy())

        # Answering No leaves everything as it was.
        with patch('dashboard.QMessageBox.question', return_value=QMessageBox.No):
            gui._menu_reset_scene()
        self.assertTrue(gui.engine.is_anchored)
        self.assertTrue(gui.engine._relineup_required)
        self.assertFalse(gui.btn_marker_lineup.isChecked())

        with patch('dashboard.QMessageBox.question', return_value=QMessageBox.Yes):
            gui._menu_reset_scene()
        # The scene stays: scan, placement, scale, Locators, display offsets.
        entries, active, T_align, scale, trail, anchor = scene
        self.assertIs(gui.lidar_matcher.map_cloud, map_cloud)
        self.assertEqual(gui._map_path, "/scans/room.obj")
        self.assertEqual(list(view.locator_entries()), list(entries))
        for key, entry in entries.items():
            np.testing.assert_allclose(view.locator_entries()[key]["transform"], entry["transform"])
        self.assertEqual(view._active_locator_id, active)
        np.testing.assert_allclose(viewer.T_align, T_align)
        self.assertAlmostEqual(viewer.alignment_scale, scale)
        np.testing.assert_allclose(view.get_display_transform("trail"), trail)
        np.testing.assert_allclose(gui.engine.anchor_target, anchor)
        # The lineup goes, RELINEUP with it.
        self.assertFalse(gui.lidar_matcher.is_aligned)
        self.assertFalse(gui.engine.is_anchored)
        self.assertFalse(gui.engine.map_aligned)
        self.assertFalse(gui.engine._relineup_required)
        self.assertEqual(len(viewer._trail), 0)
        # Auto lineup is armed again, as ticking its box does.
        self.assertTrue(gui.btn_marker_lineup.isChecked())
        self.assertIsNotNone(gui._marker_request)
        self.assertEqual(gui._marker_request["generation"], gui.lidar_matcher._generation)
        self.assertIn("Waiting for a fresh detection", gui.lbl_marker_lineup.text())
        # Tracking carries on, but nothing goes to Unreal until the new lineup.
        T = obs.T_slam_depth
        gui.engine.accept_pose(T[:3, 3] + [.01, 0, 0], matrix_to_quaternion(T[:3, :3]),
                               [0, 0, 0], [0, 0, 0], stamp=20.)
        self.assertEqual(gui.engine.fusion_status()["state"], "ALIGNING")
        self.assertIsNone(gui.engine.freed_packet())
        # And the marker lineup can complete from here.
        self._install_verified_marker(gui, obs, truth)
        self.assertTrue(gui.engine.is_anchored)
        self.assertTrue(gui.lidar_matcher.is_aligned)

    def test_reset_scene_with_three_markers_starts_with_the_scale_check(self):
        gui, truth, measure, show_markers = self._marker_scale_gui(
            {1: (.006, -.004, .003), 2: (-.005, .003, -.004), 3: (.004, .007, 0.)})
        view = gui.map_viewer.persp_view
        before = view.locator_entries()
        gui._reset_scene()
        self.assertIsNone(gui._marker_request)      # waits for the scale result
        self.assertIsNotNone(gui._scale_sweep)
        self.assertEqual(gui.btn_marker_lineup.text(), "Auto lineup · Measuring scale")
        self.assertEqual(gui.btn_marker_scale.text(), "Stop and measure")
        self.assertEqual(list(view.locator_entries()), list(before))
        # Reset again mid-measurement: the sweep starts over rather than finishing.
        first = gui._scale_sweep
        gui._reset_scene()
        self.assertIsNot(gui._scale_sweep, first)
        self.assertEqual(gui.btn_marker_lineup.text(), "Auto lineup · Measuring scale")
        measure.assert_not_called()

    def test_bad_scene_files_change_nothing(self):
        import json
        from tempfile import TemporaryDirectory
        from pathlib import Path
        gui, *_ = self._mapped_gui()
        gui.btn_marker_lineup.setChecked(False)
        view = gui.map_viewer.persp_view
        before = view.locator_entries()
        with TemporaryDirectory() as folder:
            path = str(Path(folder) / "scene.json")
            self._save_scene(gui, path)
            with open(path) as fh:
                scene = json.load(fh)
            marker = dict(dictionary="DICT_4X4_50", id=3, size_m=.1)
            bad = [dict(format="something.else"),
                   dict(scene, map_file=str(Path(folder) / "gone.ply")),
                   dict(scene, locators=[dict(key=k, name=k, transform=np.eye(4).tolist(), marker=marker)
                                         for k in ("locator", "locator_2")]),
                   dict(scene, alignment=dict(scene["alignment"], scale=0))]
            for content in bad:
                Path(path).write_text(json.dumps(content))
                self._open_scene(gui, path).assert_called_once()
                self.assertEqual(list(view.locator_entries()), list(before))
            # A scene from before Locators were saved: its tracking origin is the one Locator.
            origin = np.eye(4)
            origin[:3, 3] = [2., 1., 0.]
            old = {k: v for k, v in scene.items() if k != "locators"}
            Path(path).write_text(json.dumps(dict(old, locator=origin.tolist())))
            self._open_scene(gui, path).assert_not_called()
        self.assertEqual(list(view.locator_entries()), ["locator"])
        np.testing.assert_allclose(view.get_locator_transform(), origin, atol=1e-9)

    def test_scan_search_keeps_its_assigned_locator_when_selection_changes(self):
        from unittest.mock import patch, Mock
        from lidar_map_matcher import MapMeasurement
        gui, observation, _, _ = self._mapped_gui()
        view = gui.map_viewer.persp_view
        target_id = view._active_locator_id
        original = view.get_locator_transform('locator')
        gui.lidar_matcher._map_native = np.ones((10,3))
        gui._marker_detection_sample = ([(m.family, m.marker_id, m.corners) for m in observation.markers],
                                        11., time.monotonic(), (848,480))
        with patch.dict('sys.modules', {'scan_localizer': Mock()}), \
                patch('dashboard.ScanLocalizeThread') as worker:
            gui._find_marker_in_scan()
            worker.return_value.start.assert_called_once()
        self.assertEqual(gui._find_observation.marker_id, 9)
        self.assertEqual(gui._find_observation.marker_size_m, .2)
        self.assertIsNone(gui._marker_lock_context)
        view.outliner_list.setCurrentRow(4)
        placed = view.get_locator_transform(target_id)
        placed[0,3] += .1
        with patch('dashboard.locator_from_scan', return_value=placed), \
                patch('dashboard.solve_marker_observation', return_value=MapMeasurement(reason='test rejected scan')) as solve:
            gui._found_in_scan(dict(ok=True, T_map_optical=np.eye(4), fitness=.9))
            self.assertFalse(solve.call_args.kwargs['estimate_heading'])
        np.testing.assert_allclose(view.get_locator_transform('locator'), original)
        np.testing.assert_allclose(view.get_locator_transform(target_id), placed)
        self.assertIn('test rejected scan', gui.lbl_marker_lineup.text())

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

    def test_found_marker_moves_its_locator_and_takes_the_lineup_id(self):
        from marker_lineup import MarkerMeasurement
        from marker_map import MarkerSurvey, layout_signature
        from lidar_map_matcher import MapMeasurement
        for auto_place in (True, False):
            with self.subTest(auto_place=auto_place):
                gui, obs, truth = self._marker_gui()
                gui._marker_survey = MarkerSurvey() if auto_place else None
                before = gui.map_viewer.get_locator_transform()
                found = before.copy()
                found[:3, :3] = rpy_to_matrix(0, 0, np.radians(143))
                found[:3, 3] += [.6, -.2, 0]
                result = MarkerMeasurement(MapMeasurement(ok=True, T_slam_to_map=truth,
                    observable_rank=3, verify_score=.96), before, .63, found)
                gui._finish_marker_lineup((gui._marker_request, obs, result, ''))
                np.testing.assert_allclose(gui.map_viewer.get_locator_transform(), found, atol=1e-12)
                records = gui._mapped_marker_records()
                expected = [(obs.family, obs.marker_id, obs.marker_size_m)] if auto_place else []
                self.assertEqual([(r['dictionary'], r['id'], r['size_m']) for r in records], expected)
                self.assertEqual(gui._marker_lock_context.get('layout_signature', ()),
                                 layout_signature(records) if auto_place else ())
                gui._poll_marker_lineup()
                self.assertTrue(gui.btn_marker_lineup.isChecked(),
                                'moving its own Locator must not invalidate the lock')
                self.assertIsNotNone(gui._marker_lock_context)

    def test_locked_camera_places_markers_nobody_assigned(self):
        from unittest.mock import patch
        from marker_lineup import MarkerMeasurement
        from marker_map import MarkerSurvey, layout_signature
        from lidar_map_matcher import MapMeasurement
        from pose_transform import T_BODY_OPTICAL, invert_transform
        from test_tracking_engine import project_marker
        gui, obs, truth = self._marker_gui()
        gui._marker_survey = MarkerSurvey(min_sightings=4)
        locator = gui.map_viewer.get_locator_transform()
        gui._finish_marker_lineup((gui._marker_request, obs, MarkerMeasurement(MapMeasurement(
            ok=True, T_slam_to_map=truth, observable_rank=3, verify_score=.96), locator, 0., locator), ''))
        view = gui.map_viewer.persp_view
        # A second, unassigned marker on the same floor, 27 cm from the first.
        other = locator.copy()
        other[:3, :3] = rpy_to_matrix(0, 0, 1.1)
        other[:3, 3] += [.25, .1, 0]
        to_marker = np.eye(4)
        to_marker[:3, :3] = rpy_to_matrix(0, 0, -np.pi / 2)
        camera = truth @ obs.T_slam_image
        stamp = 10.1

        def show(i):
            # Orbit the marker: a parked camera leaves both planar PnP branches
            # steady, and the survey rightly waits for the view to change.
            nonlocal stamp
            orbit = np.eye(4)
            orbit[:3, :3] = rpy_to_matrix(0, 0, np.radians(7 * i))
            orbit[:3, 3] = other[:3, 3] - orbit[:3, :3] @ other[:3, 3]
            moved = orbit @ camera
            stamp += .25
            slam = invert_transform(truth) @ moved
            gui.engine.accept_pose(slam[:3, 3], matrix_to_quaternion(slam[:3, :3]),
                                   [0, 0, 0], [0, 0, 0], stamp=stamp)
            optical = invert_transform(moved @ T_BODY_OPTICAL) @ other @ to_marker
            corners = project_marker(obs.K, gui.marker_width.value(), optical)
            self.assertTrue(np.all(corners > 0) and np.all(corners < (848, 480)))
            gui._marker_detection_sample = ([(obs.family, obs.marker_id, obs.corners),
                                             (obs.family, 12, corners)],
                                            stamp, time.monotonic(), (848, 480))
            gui._survey_markers()

        def placed():
            return [e for e in view.locator_entries().values()
                    if e['marker'] and e['marker']['id'] == 12]

        with patch.object(gui, '_fusion_locked', return_value=True):
            gui._marker_busy = True      # a support solve in flight holds placement back
            for i in range(6):
                show(i)
            self.assertEqual(placed(), [])
            gui._marker_busy = False
            gui._survey_markers()
        entry, = placed()
        self.assertLess(np.linalg.norm(entry['transform'][:3, 3] - other[:3, 3]), .01)
        delta = entry['transform'][:3, :3].T @ other[:3, :3]
        self.assertLess(np.degrees(np.arccos(np.clip((np.trace(delta) - 1) / 2, -1, 1))), 2.)
        self.assertEqual(entry['marker']['size_m'], gui.marker_width.value())
        self.assertEqual(view._active_locator_id, 'locator', 'placing must not take the selection')
        records = gui._mapped_marker_records()
        self.assertEqual(sorted(r['id'] for r in records), [obs.marker_id, 12])
        self.assertEqual(gui._marker_lock_context['layout_signature'], layout_signature(records))
        self.assertEqual(set(gui._marker_lock_context['oriented_marker_keys']),
                         {(obs.family, obs.marker_id), (obs.family, 12)})
        gui._poll_marker_lineup()
        self.assertTrue(gui.btn_marker_lineup.isChecked())
        self.assertIsNotNone(gui._marker_lock_context)
        # Unchecking ends the survey; placed Locators stay for Save marker layout.
        gui.btn_marker_lineup.setChecked(False)
        self.assertEqual(gui._marker_survey._tracks, {})
        self.assertEqual(len(placed()), 1)

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

    def test_dot_keeps_following_camera_while_freed_waits_for_relineup(self):
        from unittest.mock import patch
        with patch("time.monotonic", return_value=100.) as mono, \
                patch("time.time", return_value=1000.) as wall:
            gui, obs, truth = self._marker_gui()
            self._install_verified_marker(gui, obs, truth)
            gui._marker_detection_sample = None

            def move(i):
                mono.return_value = 100. + i * .04
                wall.return_value = 1000. + i * .04
                moved = np.eye(4)
                moved[:3, 3] = [i * .08, 0., 0.]
                raw = obs.T_slam_depth @ moved
                gui.engine.accept_pose(raw[:3, 3], matrix_to_quaternion(raw[:3, :3]),
                                       [0, 0, 0], [0, 0, 0], stamp=11. + i * .04)
                return raw

            move(1)
            packet = gui.engine.freed_packet()
            gui._refresh_tracked_camera()
            self.assertFalse(gui.map_viewer.persp_view.cam_output_held)
            # A stall long enough to hold, then the camera turns up 40 cm on.
            mono.return_value += 1.
            wall.return_value += 1.
            gui.engine.freed_packet()
            for i in range(30, 40):
                raw = move(i)
                self.assertEqual(gui.engine.freed_packet(), packet)
                self.assertEqual(gui.engine.fusion_status()["state"], "RELINEUP")
                gui.map_viewer._last_view_push = 0.0
                gui._refresh_tracked_camera()
                expected = gui.engine.transformer._zero @ raw @ gui.engine.transformer._rig
                np.testing.assert_allclose(gui.map_viewer.persp_view._cam_pos,
                                           expected[:3, 3], atol=1e-9)
                self.assertTrue(gui.map_viewer.persp_view.cam_output_held)



class TestDepthPreview(VirtualCameraTestCase):
    _marker_gui = TestMarkerAssistedLineup._marker_gui

    def test_depth_preview_is_a_separate_default_visible_panel(self):
        from PyQt5.QtCore import Qt
        gui, _, _ = self._marker_gui(full_ui=True)
        self.assertEqual(gui.camera_views.orientation(), Qt.Horizontal)
        self.assertEqual(gui.camera_views.count(), 2)
        self.assertIsNot(gui.lbl_track, gui.lbl_depth_preview)
        self.assertTrue(gui.btn_depth_view.isChecked())
        self.assertFalse(gui.depth_preview_panel.isHidden())
        gui.btn_depth_view.setChecked(False)
        self.assertTrue(gui.depth_preview_panel.isHidden())

    def test_overlay_uses_same_exposure_then_clears_when_stale(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from test_depth_view import K, render, FLOOR, BACK_WALL, SIDE_WALL
        gui, _, _ = self._marker_gui(full_ui=True)
        gui.depth_K = K
        gui.cv_bridge = Mock()
        gui.cv_bridge.imgmsg_to_cv2.side_effect = lambda msg, encoding: msg.array
        depth = render([FLOOR, BACK_WALL, SIDE_WALL])
        gray = np.full(depth.shape, 80, np.uint8)

        def msg(array, ns, frame):
            return SimpleNamespace(array=array, header=SimpleNamespace(
                stamp=SimpleNamespace(sec=1, nanosec=ns), frame_id=frame))

        gui.depth_preview_callback(msg(gray, 1, "camera_infra1_optical_frame"))
        gui._queue_depth_preview("depth", msg(depth, 2, "camera_depth_optical_frame"))
        gui._refresh_depth_view()
        self.assertIsNone(gui._depth_cov)
        self.assertIn("Waiting", gui.lbl_depth_preview.text())
        gui._queue_depth_preview("depth", msg(depth, 1, "camera_depth_optical_frame"))
        gui._refresh_depth_view()
        self.assertEqual(gui._depth_cov.rank, 3)
        self.assertFalse(gui.lbl_depth_preview.pixmap().isNull())
        self.assertIn("scan agreement", gui.lbl_depth_readiness.text())
        self.assertTrue(gui.lbl_track.pixmap() is None or gui.lbl_track.pixmap().isNull())
        sample = gui._depth_preview_sample
        gui._depth_preview_sample = (sample[0], sample[1], time.monotonic() - 1)
        gui._refresh_depth_view()
        self.assertIsNone(gui._depth_cov)
        self.assertIn("unavailable", gui.lbl_depth_readiness.text())

if __name__ == '__main__':
    unittest.main(verbosity=2)
