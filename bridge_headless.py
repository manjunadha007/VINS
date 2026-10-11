#!/usr/bin/env python3
"""
HyperFusion Headless Bridge — Virtual Production Camera Tracking (No GUI)

Bridges NVIDIA Isaac ROS Visual SLAM to Unreal Engine over FreeD protocol,
and additionally streams camera video and accepts alignment commands over UDP
so the Unreal HyperFusionVP plugin can replace the Python dashboard entirely.

Channels:
    Port 11111 (out) → FreeD D1 camera position packets (unchanged from dashboard.py)
    Port 11112 (out) → JPEG camera frames (4-byte big-endian length + JPEG payload)
    Port 11113 (out) → JSON status + ArUco detections (one JSON object per datagram)
    Port 11114 (in)  ← JSON commands from Unreal (store_pose, reset_pose, set_offset)

Usage:
    python3 bridge_headless.py                    # uses default tracking.yaml
    VP_TRACKING_CONFIG=/path/to/tracking.yaml python3 bridge_headless.py
"""
import rclpy
from rclpy.node import Node
import threading
import socket
import struct
import json
import sys
import signal
import time
import numpy as np

from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, Imu, CameraInfo
from geometry_msgs.msg import PoseWithCovarianceStamped
from cv_bridge import CvBridge
import cv2

try:
    import cv2.aruco as aruco
    _HAS_ARUCO = True
except ImportError:
    _HAS_ARUCO = False
    print("[bridge] cv2.aruco not available — ArUco detection disabled")

from freed import get_freed_packet
from tracking_engine import TrackingEngine
from gnss_receiver import GNSSReceiver
from pose_transform import (PoseGate, PosePredictor, PoseSmoother, PoseTransformer,
                            StationaryHold, invert_transform, make_transform,
                            matrix_to_rpy, matrix_to_quaternion)
import tracking_config

try:
    from lidar_map_matcher import LidarMapMatcher
    _HAS_LIDAR = True
except ImportError:
    _HAS_LIDAR = False

# ──────────────────────────────────────────────────────
# UDP Ports
# ──────────────────────────────────────────────────────
VIDEO_PORT   = 11112   # JPEG frames → Unreal
STATUS_PORT  = 11113   # JSON status/ArUco → Unreal
COMMAND_PORT = 11114   # JSON commands ← Unreal
JPEG_QUALITY = 70      # balance between quality and bandwidth
MAX_UDP_PAYLOAD = 65000  # practical UDP datagram limit


class HeadlessBridge:
    """No-GUI bridge: ROS 2 → FreeD + Video + Commands."""

    def __init__(self):
        self.cfg = tracking_config.load_config()
        net = self.cfg["network"]
        trk = self.cfg["tracking"]
        topics = self.cfg["topics"]
        aru = self.cfg["aruco"]

        self.udp_ip = net["unreal_ip"]
        self.udp_port = net["unreal_port"]
        self.camera_id = net["camera_id"]

        print("=" * 50)
        print("HyperFusion Headless Bridge")
        print("=" * 50)
        print(tracking_config.describe(self.cfg))
        print(f"  Video stream  : {self.udp_ip}:{VIDEO_PORT}")
        print(f"  Status stream : {self.udp_ip}:{STATUS_PORT}")
        print(f"  Command listen: 0.0.0.0:{COMMAND_PORT}")
        print("=" * 50)

        # ── Sockets ──────────────────────────────────────
        self.freed_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.video_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.status_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.cv_bridge = CvBridge()
        self.streaming_enabled = True

        # Depth intrinsics. Left as None unless a depth camera_info arrives: on
        # the D455 the depth image is registered to the left infrared sensor, so
        # the infra1 intrinsics the engine already has are the right fallback.
        self.depth_K = None

        # ── Pose pipeline ────────────────────────────────
        # All of it lives in TrackingEngine, shared with dashboard.py. This class
        # is I/O only: ROS subscriptions in, UDP out.
        self.engine = TrackingEngine(self.cfg, log=print)
        self.gnss_receiver = None
        if self.cfg["gnss"]["enabled"]:
            self.gnss_receiver = GNSSReceiver(self.cfg["gnss"], self.engine.update_gnss)
            self.gnss_receiver.start()

        self._last_rate_check = time.time()
        self._pose_rate = 0.0
        self._send_rate = 0.0
        self._last_pose_count = 0
        self._last_packet_count = 0
        self._cam_frame_n = 0
        self._pose_lock = threading.Lock()

        # ── ArUco (optional) ─────────────────────────────
        self.aruco_enabled = _HAS_ARUCO and bool(aru["enabled"])
        self._latest_marker = None  # last detected marker info
        self.aruco_detectors = []
        if self.aruco_enabled:
            for dict_name in tracking_config.aruco_dictionary_names(aru):
                if not hasattr(aruco, dict_name):
                    print(f"[ArUco] ignoring unknown dictionary {dict_name!r}")
                    continue
                dict_id = getattr(aruco, dict_name)
                dictionary = (aruco.getPredefinedDictionary(dict_id)
                              if hasattr(aruco, "getPredefinedDictionary")
                              else aruco.Dictionary_get(dict_id))
                params = (aruco.DetectorParameters_create()
                          if hasattr(aruco, "DetectorParameters_create")
                          else aruco.DetectorParameters())
                detector = (aruco.ArucoDetector(dictionary, params)
                            if hasattr(aruco, "ArucoDetector") else None)
                self.aruco_detectors.append((dict_name, dictionary, params, detector))
            self.aruco_enabled = bool(self.aruco_detectors)
            print("[ArUco] scanning: " + ", ".join(name for name, *_ in self.aruco_detectors))

        # ── Alignment state (controlled by Unreal commands) ──
        self._alignment_offset = {
            "tx": 0.0, "ty": 0.0, "tz": 0.0,
            "rx": 0.0, "ry": 0.0, "rz": 0.0,
            "scale": 1.0
        }
        self._pose_stored = False

        # ── LiDAR map matching ────────────────────────────
        lidar_cfg = self.cfg.get("lidar", {})
        self.lidar_matcher = None
        self._request_lidar_align = False
        self._lidar_marker_map_pose = None
        self._lidar_measurement = None
        self._last_fix_counter = 0
        if _HAS_LIDAR and lidar_cfg.get("enabled", False):
            self.lidar_matcher = LidarMapMatcher(lidar_cfg)
            if self.lidar_matcher.map_cloud is not None:
                self.lidar_matcher.start_worker()
            print("[bridge] ✅ LiDAR map matching enabled")
        self._depth_frame_n = 0
        self._lidar_validation = None
        self._snap_event = None

        # ── ROS 2 ────────────────────────────────────────
        if not rclpy.ok():
            rclpy.init(args=None)

        self.node = Node('vp_camera_tracking_bridge')
        self.node.create_subscription(
            Odometry, topics["odometry"], self.odometry_callback, 10)
        self.node.create_subscription(
            PoseWithCovarianceStamped, topics["vo_pose"],
            self.vo_pose_callback, 10)
        self.node.create_subscription(
            Image, topics["image"], self.image_callback, 1)
        self.node.create_subscription(
            CameraInfo, topics["camera_info"],
            self.camera_info_callback, 1)
        self.node.create_subscription(
            Imu, topics["imu"], self.imu_callback, 10)
        
        self.node.create_subscription(
            Image, topics.get("depth", "/camera/depth/image_rect_raw"),
            self.depth_callback, 1)
        # Depth's own intrinsics. `depth_K` was previously initialised to None
        # and never assigned, so the point cloud was deprojected with the
        # infra1 focal length -- correct only by coincidence, and only while
        # align_depth stays off at a matching resolution.
        self.node.create_subscription(
            CameraInfo,
            topics.get("depth_camera_info", "/camera/depth/camera_info"),
            self.depth_camera_info_callback, 1)

        # ── Start command listener thread ─────────────────
        self._cmd_thread = threading.Thread(target=self._command_listener, daemon=True)
        self._cmd_thread.start()

        # ── Start FreeD send timer thread ─────────────────
        self.send_on_pose = not net["send_rate_hz"]
        if not self.send_on_pose:
            self._send_interval = 1.0 / max(1, net["send_rate_hz"])
            self._freed_thread = threading.Thread(target=self._freed_timer_loop, daemon=True)
            self._freed_thread.start()

        # ── Start status broadcast thread ─────────────────
        self._status_thread = threading.Thread(target=self._status_broadcast_loop, daemon=True)
        self._status_thread.start()

        print("[bridge] ✅ ROS 2 subscriptions active — waiting for SLAM data...")

    # ──────────────────────────────────────────────────
    # ROS Callbacks
    # ──────────────────────────────────────────────────

    def odometry_callback(self, msg):
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        lin = msg.twist.twist.linear
        ang = msg.twist.twist.angular
        self._accept_pose(
            (p.x, p.y, p.z), (o.x, o.y, o.z, o.w),
            (lin.x, lin.y, lin.z), (ang.x, ang.y, ang.z),
            stamp=self._stamp_seconds(msg), source="odometry")

    def vo_pose_callback(self, msg):
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        self._accept_pose((p.x, p.y, p.z), (o.x, o.y, o.z, o.w),
                          (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
                          stamp=self._stamp_seconds(msg), source="vo")

    # ── Delegating views onto the engine ─────────────────────────────
    @property
    def transformer(self):
        return self.engine.transformer

    @property
    def gate(self):
        return self.engine.gate

    @property
    def hold(self):
        return self.engine.hold

    @property
    def predictor(self):
        return self.engine.predictor

    @property
    def smoother(self):
        return self.engine.smoother

    @property
    def stale_timeout(self):
        return self.engine.stale_timeout

    @property
    def aruco_K(self):
        return self.engine.aruco_K

    @property
    def aruco_D(self):
        return self.engine.aruco_D

    @property
    def marker_size(self):
        return self.engine.marker_size

    @property
    def latest_pose(self):
        return self.engine.latest_pose

    @property
    def slam_status(self):
        return self.engine.slam_status

    @property
    def _packets_sent(self):
        return self.engine.packets_sent

    @property
    def _pose_updates(self):
        return self.engine.pose_updates

    @property
    def request_ground_zero(self):
        return self.engine.anchor_requested

    @request_ground_zero.setter
    def request_ground_zero(self, value):
        self.engine.anchor_requested = bool(value)

    def depth_callback(self, msg):
        """Depth feeds two consumers: the marker cross-check and the LiDAR matcher."""
        try:
            depth = self.cv_bridge.imgmsg_to_cv2(msg, "16UC1")
        except Exception as e:
            print(f"[depth] decode error: {e}")
            return

        # Independent distance measurement for the ArUco solve. Four coplanar
        # corners are pose-ambiguous, so this is what stops a flipped solution
        # becoming the anchor for the whole scene.
        self.engine.set_depth(depth, self.depth_K, self._stamp_seconds(msg))
        self._depth_frame_n += 1

        if self.lidar_matcher is None:
            return
        K = self.depth_K if self.depth_K is not None else self.engine.aruco_K
        pose = self.engine.depth_pose()
        if not self.engine.fusion.enabled:
            pose = self.engine.latest_pose
        if pose is None or K is None:
            return

        # Continuous map localisation. `pose` is the RAW SLAM pose: the fusion
        # estimates the SLAM->map transform, so handing it an already-corrected
        # pose would close a loop around its own output.
        if self.lidar_matcher.is_aligned:
            # Hand off; ICP costs 100-200 ms and must not block the ROS
            # executor that also delivers odometry.
            try:
                self.lidar_matcher.submit(
                    depth, K, pose, time.monotonic(),
                    is_stationary=self.engine.hold.is_stationary)
            except Exception as e:
                print(f"[LiDAR] fusion submit error: {e}")
            if self.lidar_matcher.fix_counter != self._last_fix_counter:
                self._last_fix_counter = self.lidar_matcher.fix_counter
                self._lidar_measurement = self.lidar_matcher.last_measurement
                T_slam_to_map, _ = self.lidar_matcher.map_transform()
                # No mesh-alignment gizmo without a GUI, so the map file's
                # native frame IS the output frame here.
                self.engine.apply_map_alignment(T_slam_to_map)

        # Read-only agreement score, ~3 Hz at 30 fps.
        if self.engine.is_anchored and self._depth_frame_n % 10 == 0:
            try:
                self._lidar_validation = self.lidar_matcher.validate_depth(
                    depth, K, make_transform(pose[0], pose[1]))
            except Exception as e:
                print(f"[LiDAR] validation error: {e}")

    def depth_camera_info_callback(self, msg):
        """Depth stream intrinsics. First valid message wins."""
        if self.depth_K is not None:
            return
        k = list(msg.k) if hasattr(msg, "k") else list(msg.K)
        if len(k) < 9 or k[0] == 0.0:
            return
        self.depth_K = np.asarray(k, dtype=float).reshape(3, 3)
        self.engine.set_depth_intrinsics(self.depth_K)
        print(f"[depth] intrinsics from camera_info: fx={self.depth_K[0,0]:.2f} "
              f"fy={self.depth_K[1,1]:.2f}")

    @staticmethod
    def _stamp_seconds(msg):
        s = msg.header.stamp
        if s.sec == 0 and s.nanosec == 0:
            return None
        return s.sec + s.nanosec * 1e-9

    def _accept_pose(self, position, quaternion, linear, angular, stamp=None, *, source=None):
        """Hand a raw SLAM pose to the engine.

        Deliberately raw. The map correction used to be blended into `position`
        here, which could only ever correct translation, fed the estimator its
        own output, and made the gate and the stationary hold argue with it.
        The fusion now corrects the *frame* instead (see depth_callback), so
        this stays a clean VIO stream and the gate keeps seeing only genuine
        tracker behaviour.
        """
        if not self.engine.accept_pose(position, quaternion, linear, angular, stamp, source=source):
            return

        if self.send_on_pose:
            self._stream_freed()

    def image_callback(self, msg):
        try:
            self._cam_frame_n += 1
            if msg.encoding in ("mono8", "8UC1"):
                gray = self.cv_bridge.imgmsg_to_cv2(msg, "mono8")
            elif msg.encoding in ("mono16", "16UC1"):
                gray = (self.cv_bridge.imgmsg_to_cv2(msg, "16UC1") >> 8).astype(np.uint8)
            else:
                gray = self.cv_bridge.imgmsg_to_cv2(msg, "mono8")

            # Stream video to Unreal (every 3rd frame to save bandwidth: ~30fps from 90fps)
            if self._cam_frame_n % 3 == 0:
                self._send_video_frame(gray)

            # ArUco detection (every 15 frames, or when anchoring is requested)
            if self.aruco_enabled and (self.request_ground_zero or self._cam_frame_n % 15 == 0):
                self._detect_aruco(gray, msg.width, msg.height)

        except Exception:
            pass

    def camera_info_callback(self, msg):
        k = list(msg.k) if hasattr(msg, "k") else list(msg.K)
        d = list(msg.d) if hasattr(msg, "d") else list(msg.D)
        if len(k) >= 9:
            self.engine.set_intrinsics(k, d)

    def imu_callback(self, msg):
        g = msg.angular_velocity
        a = msg.linear_acceleration
        self.engine.update_imu((g.x, g.y, g.z), (a.x, a.y, a.z))

    # ──────────────────────────────────────────────────
    # Video Streaming (UDP JPEG)
    # ──────────────────────────────────────────────────

    def _send_video_frame(self, gray_img):
        """Encode a grayscale frame as JPEG and send over UDP with length header."""
        try:
            ok, jpeg_buf = cv2.imencode('.jpg', gray_img,
                                        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if not ok:
                return
            jpeg_bytes = jpeg_buf.tobytes()
            if len(jpeg_bytes) > MAX_UDP_PAYLOAD:
                # Downscale if frame is too large for a single UDP datagram
                h, w = gray_img.shape[:2]
                small = cv2.resize(gray_img, (w // 2, h // 2))
                ok, jpeg_buf = cv2.imencode('.jpg', small,
                                            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                if not ok:
                    return
                jpeg_bytes = jpeg_buf.tobytes()

            # 4-byte big-endian length header + JPEG payload
            header = struct.pack('>I', len(jpeg_bytes))
            self.video_sock.sendto(header + jpeg_bytes,
                                   (self.udp_ip, VIDEO_PORT))
        except Exception:
            pass

    # ──────────────────────────────────────────────────
    # ArUco Detection
    # ──────────────────────────────────────────────────

    def _find_aruco_markers(self, gray):
        """Return one detection per visible square across configured families."""
        found = []
        for dict_name, dictionary, params, detector in self.aruco_detectors:
            if detector:
                corners, ids, _ = detector.detectMarkers(gray)
            else:
                corners, ids, _ = aruco.detectMarkers(gray, dictionary, parameters=params)
            if ids is None:
                continue
            for corner, marker_id in zip(corners, ids):
                quad = np.asarray(corner, dtype=np.float32).reshape(4, 2)
                # Some OpenCV 4x4 families share codes. The same physical
                # square may therefore decode in more than one family; retain
                # the first configured family instead of anchoring twice.
                if any(np.max(np.abs(quad - old_quad)) < 1.0
                       for _, _, old_quad in found):
                    continue
                found.append((dict_name, int(marker_id[0]), quad))
        return found

    def _detect_aruco(self, gray, img_w, img_h):
        """Detect markers; anchor through the engine when armed."""
        try:
            markers = self._find_aruco_markers(gray)
            if not markers:
                self._latest_marker = None
                return

            dictionary, marker_id, quad = markers[0]
            solution = None

            if self.engine.anchor_requested:
                ok, message = self.engine.try_anchor(quad)
                solution = self.engine.last_marker_solution
                if ok:
                    self._on_anchored(solution)
                else:
                    print(f"[Anchor] ✗ refused: {message}")
            elif self.latest_pose is not None:
                # Keep reporting range to the marker for the status feed.
                solution = self.engine.solve_marker_pose(quad)

            self._latest_marker = {
                "id": marker_id,
                "dictionary": dictionary,
                "tvec": (None if solution is None or not solution.ok
                         else [float(v) for v in
                               solution.T_optical_marker[:3, 3]]),
                "distance_m": (None if solution is None
                               else solution.distance_pnp_m),
                "distance_depth_m": (None if solution is None
                                     else solution.distance_depth_m),
            }
        except Exception as exc:
            print(f"[ArUco] detect error: {exc}")
            self._latest_marker = None

    def _on_anchored(self, solution):
        """Post-anchor: send snap position to Unreal and seed LiDAR alignment."""
        # Compute snap position for Unreal teleport
        snap = self.engine.compute_snap_position()
        if snap is not None:
            self._snap_event = {
                "event": "snap_camera",
                "position": {"x": snap["x"], "y": snap["y"], "z": snap["z"]},
                "rotation": {"pan": snap["pan"], "tilt": snap["tilt"], "roll": snap["roll"]},
                "depth_verified": solution.used_depth if solution else False,
            }
            print(f"[Anchor] snap_camera -> Unreal: "
                  f"X={snap['x']:+.3f} Y={snap['y']:+.3f} Z={snap['z']:+.3f}")

        # Seed the map fusion from the same anchor the engine just used.
        #
        # The marker's map pose is taken from `engine.anchor_target` rather
        # than re-derived from `_lidar_marker_map_pose`. Those raw command
        # values are Unreal world units and (pan, tilt, roll) degrees; this
        # code fed them into a matrix as if they were metres and ROS
        # (roll, pitch, yaw), so the seed came out 100x too far away with two
        # axes mixed up. The engine already converts them correctly, through
        # the same signs and trim as the live pose -- so there is exactly one
        # copy of that conversion, and a flipped axis in tracking.yaml moves
        # the anchor and the fusion together.
        if not (self.lidar_matcher and solution is not None and solution.ok):
            return
        T_map_marker = self.engine.anchor_target
        if T_map_marker is None:
            return

        # T_slam_to_map maps SLAM points into the map, so it must satisfy
        #     T_slam_to_map @ T_slam_marker == T_map_marker
        # giving T_map_marker @ inv(T_slam_marker). This was composed the other
        # way round, which produces the inverse transform: the correction then
        # pushed the pose away from the map instead of onto it.
        # T_slam_to_map maps SLAM points into the map, so it must satisfy
        #     T_slam_to_map @ T_slam_marker == T_map_marker
        # giving T_map_marker @ inv(T_slam_marker).
        T_hint = T_map_marker @ invert_transform(solution.T_world_marker)

        # The marker is a *hint*, not the answer: it is printed, taped up by
        # hand and solved from four coplanar corners, while the scan is
        # surveyed geometry. Installing it verbatim (what this used to do)
        # pins the camera exactly on the marker mapping and leaves continuous
        # fusion to creep to the truth at the slew limit. Solve instead.
        depth = self.engine._depth_mm
        pose = self.engine.depth_pose() if self.engine.fusion.enabled else self.engine.latest_pose
        if depth is None or pose is None:
            self.lidar_matcher.set_initial_alignment(T_hint, source="aruco")
            self.engine.apply_map_alignment(T_hint)
            self._request_lidar_align = False
            print("[LiDAR] no depth yet — holding the marker anchor; "
                  "fusion will refine once depth arrives")
            return

        K = self.depth_K if self.depth_K is not None else self.engine.aruco_K
        meas = self.lidar_matcher.solve_from_hint(
            depth, K, make_transform(pose[0], pose[1]), T_hint, source="aruco")
        self._lidar_measurement = meas
        if meas.ok:
            self.engine.observe_scan(meas)
            T_slam_to_map, _ = self.lidar_matcher.map_transform()
            self.engine.apply_map_alignment(T_slam_to_map)
        else:
            self.lidar_matcher.set_initial_alignment(T_hint, source="aruco")
            self.engine.apply_map_alignment(T_hint)
            print("[LiDAR] ⚠️  could not place the camera on the scan — "
                  "holding the marker anchor alone (pose NOT scan-verified)")
        self._request_lidar_align = False

    # ──────────────────────────────────────────────────
    # FreeD Streaming
    # ──────────────────────────────────────────────────

    def camera_pose_now(self):
        """Predicted, converted, smoothed pose. Returns (UnrealPose, is_valid)."""
        return self.engine.camera_pose_now()

    def _stream_freed(self):
        if not self.streaming_enabled:
            return
        if self.lidar_matcher is not None:
            self.engine.observe_scan(self.lidar_matcher.last_measurement)
        pkt = self.engine.freed_packet()
        if pkt is None:
            return
        try:
            self.freed_sock.sendto(pkt, (self.udp_ip, self.udp_port))
        except Exception:
            pass

    def _freed_timer_loop(self):
        """Background thread for fixed-rate FreeD streaming."""
        while True:
            self._stream_freed()
            time.sleep(self._send_interval)

    # ──────────────────────────────────────────────────
    # Status Broadcast (JSON → Unreal)
    # ──────────────────────────────────────────────────

    def _status_broadcast_loop(self):
        """Send status JSON to Unreal every 500ms."""
        while True:
            try:
                now = time.time()
                elapsed = now - self._last_rate_check
                if elapsed >= 1.0:
                    # The engine owns the counters and they only ever climb, so
                    # rates come from deltas rather than by zeroing them here.
                    self._pose_rate = (self.engine.pose_updates
                                       - self._last_pose_count) / elapsed
                    self._send_rate = (self.engine.packets_sent
                                       - self._last_packet_count) / elapsed
                    self._last_pose_count = self.engine.pose_updates
                    self._last_packet_count = self.engine.packets_sent
                    self._last_rate_check = now

                status = {
                    "status": self.slam_status,
                    "fps_in": round(self._pose_rate, 1),
                    "fps_out": round(self._send_rate, 1),
                    "gate_rejects": self.gate.total_rejects,
                    "gate_resyncs": self.gate.resyncs,
                    "anchored": self.transformer.is_zeroed,
                    "map_aligned": self.engine.map_aligned,
                    "pose_stored": self._pose_stored,
                    "fusion": self.engine.fusion_status(),
                    "gnss": {"reason": self.engine.gnss_reason},
                }
                if self._latest_marker:
                    status["marker"] = self._latest_marker
                if self.lidar_matcher:
                    status["lidar"] = self.lidar_matcher.get_diagnostics()

                # One-shot snap_camera event (consumed after sending)
                snap = self._snap_event
                if snap is not None:
                    status["snap_camera"] = snap
                    self._snap_event = None

                # LiDAR depth validation (updated at ~3 Hz)
                lv = self._lidar_validation
                if lv is not None:
                    status["lidar_validation"] = lv

                # Current virtual camera position
                pose = self.engine.latest_unreal
                if pose is not None:
                    status["virtual_camera"] = {
                        "x": round(pose.x, 4), "y": round(pose.y, 4),
                        "z": round(pose.z, 4),
                        "pan": round(pose.pan, 2), "tilt": round(pose.tilt, 2),
                        "roll": round(pose.roll, 2),
                    }

                payload = json.dumps(status).encode('utf-8')
                self.status_sock.sendto(payload, (self.udp_ip, STATUS_PORT))
            except Exception:
                pass
            time.sleep(0.5)

    # ──────────────────────────────────────────────────
    # Command Listener (JSON ← Unreal)
    # ──────────────────────────────────────────────────

    def _command_listener(self):
        """Listen for JSON commands from Unreal on COMMAND_PORT."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('0.0.0.0', COMMAND_PORT))
        sock.settimeout(1.0)
        print(f"[bridge] Command listener on port {COMMAND_PORT}")

        while True:
            try:
                data, addr = sock.recvfrom(4096)
                cmd = json.loads(data.decode('utf-8'))
                self._handle_command(cmd)
            except socket.timeout:
                continue
            except Exception as e:
                print(f"[bridge] Command error: {e}")

    def _handle_command(self, cmd):
        """Dispatch one JSON command from Unreal.

        The engine owns every command's effect on the pose. This wrapper only
        keeps the LiDAR matcher's extra bookkeeping, which is bridge-local.
        """
        action = cmd.get("cmd", "")

        if action in ("anchor_lidar", "set_anchor"):
            self._lidar_marker_map_pose = (
                cmd.get("marker_map_position", cmd.get("position", [0, 0, 0])),
                cmd.get("marker_map_rotation", cmd.get("rotation", [0, 0, 0])))
            self._request_lidar_align = True
        elif action == "reset_pose":
            self._lidar_marker_map_pose = None
            self._lidar_measurement = None
            self._lidar_validation = None
            # Otherwise the previous map fix keeps driving ground zero and the
            # reset silently does nothing to the output.
            if self.lidar_matcher is not None:
                self.lidar_matcher._generation += 1
                self.lidar_matcher.last_measurement = None
                self.lidar_matcher.is_aligned = False
                self.lidar_matcher.has_correction = False
                self.lidar_matcher.confidence = 0.0
            self._pose_stored = False
        elif action == "store_pose":
            self._pose_stored = True

        reply = self.engine.handle_command(cmd)
        if not reply.get("ok"):
            print(f"[bridge] command '{action}': {reply.get('error')}")
        return reply

    # ──────────────────────────────────────────────────
    # Main Loop
    # ──────────────────────────────────────────────────

    def spin(self):
        """Block on ROS 2 spin. Ctrl+C to exit."""
        try:
            rclpy.spin(self.node)
        except KeyboardInterrupt:
            pass
        finally:
            print("\n[bridge] Shutting down...")
            if self.gnss_receiver is not None:
                self.gnss_receiver.stop()
            if self.lidar_matcher is not None:
                self.lidar_matcher.stop_worker()
            self.node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
            self.freed_sock.close()
            self.video_sock.close()
            self.status_sock.close()
            print("[bridge] ✅ Clean shutdown.")


def main():
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    bridge = HeadlessBridge()
    bridge.spin()


if __name__ == '__main__':
    main()
