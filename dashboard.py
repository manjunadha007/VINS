#!/usr/bin/env python3
"""
HyperFusion Dashboard v5.0 — Virtual Production Camera Tracking

Bridges NVIDIA Isaac ROS Visual SLAM to Unreal Engine over the FreeD protocol,
so a virtual camera in Unreal mirrors a real camera carrying an Intel RealSense
D455.

Pipeline:
    RealSense D455 (stereo IR + IMU)
      -> Isaac ROS Visual SLAM  (visual-inertial fusion, on GPU)
      -> this bridge            (frame conversion, latency prediction, FreeD)
      -> Unreal Engine          (LiveLinkFreeD subject drives a CineCamera)

Design notes for anyone modifying this:

  * Isaac ROS already performs visual-inertial fusion (enable_imu_fusion:=True).
    This bridge does NOT re-fuse. Earlier versions integrated /camera/imu gyro
    here and raced Isaac's own estimate for `latest_pose`; that is gone. The IMU
    subscription is now telemetry only.

  * Rate conversion from Isaac's image-rate odometry to a steady FreeD stream is
    done by constant-velocity prediction using the velocity Isaac publishes in
    Odometry.twist -- see pose_transform.PosePredictor.

  * All frame conversion lives in pose_transform.py, all wire encoding in
    freed.py, all operator settings in tracking.yaml. Keep it that way: those
    modules are unit tested, this file is not.
"""
import rclpy
from rclpy.node import Node
import threading
import socket
import math
import sys
import signal
import time
import os
import json
import datetime
import numpy as np
import collections
# NOTE: pyserial (GNSS) and sounddevice (LTC timecode) are imported lazily inside
# their worker threads. Both features are optional and off-by-default, and the
# Isaac ROS container ships neither -- a module-level import here took the whole
# dashboard down with ModuleNotFoundError even when GNSS was disabled.

from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, Imu, CameraInfo
from geometry_msgs.msg import PoseWithCovarianceStamped
from cv_bridge import CvBridge
import cv2
import cv2.aruco as aruco

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QProgressDialog,
    QVBoxLayout, QHBoxLayout, QWidget, QFrame, QSizePolicy, QScrollArea,
    QSplitter, QTabWidget, QLineEdit, QAction, QFileDialog, QMessageBox,
    QComboBox, QSpinBox, QDoubleSpinBox, QFormLayout, QCheckBox
)
from PyQt5.QtGui import (QImage, QPixmap, QFont, QPainter, QPen, QColor,
                         QBrush, QKeySequence)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QRectF, QPointF

# ──────────────────────────────────────────────────────
# Shared modules. These must be deployed alongside this file;
# launch_isaac.sh copies them into the container workspace.
# ──────────────────────────────────────────────────────
from freed import get_freed_packet
from tracking_engine import TrackingEngine
from gnss_receiver import GNSSReceiver
from pose_transform import (PoseGate, PosePredictor, PoseSmoother, PoseTransformer,
                            StationaryHold,
                            decompose_similarity, invert_transform,
                            make_transform, matrix_to_rpy,
                            matrix_to_quaternion, rpy_to_matrix)
import tracking_config
from map_viewer import MapViewerWidget
from marker_lineup import (capture_marker_observation, locator_from_scan,
                           matcher_snapshot, solve_marker_observation)

try:
    from lidar_map_matcher import LidarMapMatcher
    _HAS_LIDAR = True
except ImportError:
    _HAS_LIDAR = False

CONFIG = tracking_config.load_config()

UDP_IP    = CONFIG["network"]["unreal_ip"]
UDP_PORT  = CONFIG["network"]["unreal_port"]
CAMERA_ID = CONFIG["network"]["camera_id"]

# ──────────────────────────────────────────────────────
# Fast 2D Trajectory Canvas (native QPainter — 60fps, zero lag)
# ──────────────────────────────────────────────────────
class TrajectoryCanvas(QWidget):
    """GPU-light, QPainter-based top-down + side trajectory view.
    Renders at 60fps with negligible CPU cost — no Matplotlib overhead."""
    MAX_POINTS = 3000

    def __init__(self):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(400, 300)
        self.setStyleSheet("background:#0a0a14; border:1px solid #1a1a2e; border-radius:6px;")

        # Thread-safe deque for points
        self._pts  = collections.deque(maxlen=self.MAX_POINTS)  # (x, y, z)
        self._lock = __import__('threading').Lock()
        self._latest = None   # most recent (x, y, z)

        # Camera FoV cone state
        self._heading_deg = 0.0  # yaw in degrees

    # Called from ROS callback thread — must be thread-safe
    def update_trajectory(self, x, y, z, yaw_deg=0.0):
        with self._lock:
            self._pts.append((x, y, z))
            self._latest = (x, y, z)
            self._heading_deg = yaw_deg
        self.update()   # schedule repaint (non-blocking)

    def paintEvent(self, event):
        with self._lock:
            pts     = list(self._pts)
            latest  = self._latest
            heading = self._heading_deg

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        W, H = self.width(), self.height()
        # ── background
        painter.fillRect(0, 0, W, H, QColor('#0a0a14'))

        if not pts:
            painter.setPen(QColor('#333355'))
            painter.setFont(QFont('Arial', 11))
            painter.drawText(0, 0, W, H, Qt.AlignCenter, 'Waiting for tracking data...')
            painter.end()
            return

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        zs = [p[2] for p in pts]

        # ── determine world-space span
        margin = 0.5
        x_min, x_max = min(xs) - margin, max(xs) + margin
        y_min, y_max = min(ys) - margin, max(ys) + margin
        z_min, z_max = min(zs) - margin, max(zs) + margin

        # ── two panel layout: top-down (XY) left, side (XZ) right
        panel_w = (W - 12) // 2
        panel_h = H - 50   # leave room for labels at bottom

        def world_to_screen(wx, wy, wmin_x, wmax_x, wmin_y, wmax_y, ox, oy, pw, ph):
            sx = ox + int((wx - wmin_x) / max(wmax_x - wmin_x, 1e-6) * pw)
            sy = oy + ph - int((wy - wmin_y) / max(wmax_y - wmin_y, 1e-6) * ph)
            return sx, sy

        def draw_panel(pts_a, pts_b, a_range, b_range, ox, oy, pw, ph, label, col_a, col_b):
            # Grid
            painter.setPen(QPen(QColor('#1a1a2e'), 1))
            for i in range(5):
                gx = ox + i * pw // 4
                gy = oy + i * ph // 4
                painter.drawLine(gx, oy, gx, oy + ph)
                painter.drawLine(ox, gy, ox + pw, gy)

            # Trail — gradient from dim to bright
            n = len(pts_a)
            if n >= 2:
                for i in range(1, n):
                    alpha = int(80 + 175 * i / n)
                    pen = QPen(QColor(0, 212, 255, alpha), 1.5)
                    painter.setPen(pen)
                    x0, y0 = world_to_screen(pts_a[i-1], pts_b[i-1], *a_range, *b_range, ox, oy, pw, ph)
                    x1, y1 = world_to_screen(pts_a[i],   pts_b[i],   *a_range, *b_range, ox, oy, pw, ph)
                    painter.drawLine(x0, y0, x1, y1)

            # Current dot
            cx, cy = world_to_screen(pts_a[-1], pts_b[-1], *a_range, *b_range, ox, oy, pw, ph)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor('#ff6600')))
            painter.drawEllipse(QPointF(cx, cy), 6, 6)
            # Glow
            painter.setBrush(QBrush(QColor(255, 102, 0, 60)))
            painter.drawEllipse(QPointF(cx, cy), 12, 12)

            # Label
            painter.setPen(QColor('#8dccf1'))
            painter.setFont(QFont('Arial', 8, QFont.Bold))
            painter.drawText(ox + 4, oy + 14, label)

        # Top-down view: X horizontal, Y vertical
        draw_panel(xs, ys, (x_min, x_max), (y_min, y_max),
                   4, 24, panel_w, panel_h, 'TOP  X→  Y↑', '#8dccf1', '#8dccf1')

        # Side view: X horizontal, Z vertical
        draw_panel(xs, zs, (x_min, x_max), (z_min, z_max),
                   panel_w + 8, 24, panel_w, panel_h, 'SIDE  X→  Z↑', '#00ffaa', '#00ffaa')

        # Divider
        painter.setPen(QPen(QColor('#2a2a3a'), 1))
        painter.drawLine(panel_w + 6, 24, panel_w + 6, 24 + panel_h)

        # Bottom stats
        if latest:
            lx, ly, lz = latest
            painter.setPen(QColor('#888'))
            painter.setFont(QFont('Monospace', 8))
            painter.drawText(4, H - 16, W - 8, 16, Qt.AlignLeft,
                             f'X:{lx:+.3f}  Y:{ly:+.3f}  Z:{lz:+.3f}  Hdg:{heading:+.1f}°   [{len(pts)} pts]')
        painter.end()


# ──────────────────────────────────────────────────────
# GNSS & Timecode Math and Threads
# ──────────────────────────────────────────────────────
def wgs84_to_ecef(lat, lon, alt):
    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = 2*f - f*f
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    N = a / math.sqrt(1 - e2 * math.sin(lat_rad)**2)
    x = (N + alt) * math.cos(lat_rad) * math.cos(lon_rad)
    y = (N + alt) * math.cos(lat_rad) * math.sin(lon_rad)
    z = (N * (1 - e2) + alt) * math.sin(lat_rad)
    return x, y, z

def ecef_to_enu(x, y, z, lat0, lon0, x0, y0, z0):
    dx = x - x0
    dy = y - y0
    dz = z - z0
    slat = math.sin(math.radians(lat0))
    clat = math.cos(math.radians(lat0))
    slon = math.sin(math.radians(lon0))
    clon = math.cos(math.radians(lon0))
    e = -slon * dx + clon * dy
    n = -slat * clon * dx - slat * slon * dy + clat * dz
    u = clat * clon * dx + clat * slon * dy + slat * dz
    return e, n, u

class LTCAudioListenerThread(QThread):
    tc_updated = pyqtSignal(str)

    def __init__(self, cfg=None):
        super().__init__()
        cfg = cfg or {}
        self.samplerate = int(cfg.get("samplerate", 48000))

    def run(self):
        try:
            import sounddevice as sd
            import numpy as np
            import queue
        except ImportError:
            self.tc_updated.emit("TC: NO SIGNAL (MISSING PKG)")
            return
            
        self.bit_buffer = []
        audio_queue = queue.Queue(maxsize=100)
        
        def callback(indata, frames, time_info, status):
            if status: return
            try:
                audio_queue.put_nowait(indata[:, 0].copy())
            except queue.Full:
                pass

        try:
            stream = sd.InputStream(channels=1, samplerate=self.samplerate,
                                    callback=callback)
            with stream:
                while not self.isInterruptionRequested():
                    try:
                        audio = audio_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue
                        
                    if np.max(np.abs(audio)) < 0.10:
                        continue
                        
                    crossings = np.where(np.diff(np.signbit(audio)))[0]
                    if len(crossings) < 2: continue
                    
                    dists = np.diff(crossings)
                    if len(dists) == 0: continue
                    
                    threshold = np.median(dists) * 1.5 
                    
                    bits = []
                    skip = False
                    for d in dists:
                        if skip:
                            skip = False
                            continue
                        if d > threshold:
                            bits.append(0)
                        else:
                            bits.append(1)
                            skip = True
                            
                    self.bit_buffer.extend(bits)
                    
                    if len(self.bit_buffer) > 5000:
                        self.bit_buffer = self.bit_buffer[-5000:]
                        
                    sync_seq = [0,0,1,1, 1,1,1,1, 1,1,1,1, 1,1,0,1]
                    
                    while len(self.bit_buffer) >= 80:
                        found = False
                        for i in range(len(self.bit_buffer) - 16):
                            if self.bit_buffer[i:i+16] == sync_seq:
                                if i >= 64:
                                    fb = self.bit_buffer[i-64:i]
                                    fr = fb[0] + fb[1]*2 + fb[2]*4 + fb[3]*8 + fb[8]*10 + fb[9]*20
                                    sc = fb[16] + fb[17]*2 + fb[18]*4 + fb[19]*8 + fb[24]*10 + fb[25]*20 + fb[26]*40
                                    mn = fb[32] + fb[33]*2 + fb[34]*4 + fb[35]*8 + fb[40]*10 + fb[41]*20 + fb[42]*40
                                    hr = fb[48] + fb[49]*2 + fb[50]*4 + fb[51]*8 + fb[56]*10 + fb[57]*20
                                    
                                    if hr < 24 and mn < 60 and sc < 60 and fr < 30:
                                        self.tc_updated.emit(f"TC: {hr:02d}:{mn:02d}:{sc:02d}:{fr:02d}")
                                
                                self.bit_buffer = self.bit_buffer[i+16:]
                                found = True
                                break
                        if not found:
                            self.bit_buffer = self.bit_buffer[-80:]
                            break
        except Exception as e:
            self.tc_updated.emit("TC: NO SIGNAL (MIC ERROR)")
            print(f"[Error] Audio LTC Thread: {e}")

# ──────────────────────────────────────────────────────
# Background Threads
# ──────────────────────────────────────────────────────
class LidarLoadThread(QThread):
    finished_loading = pyqtSignal(object)

    def __init__(self, lidar_cfg):
        super().__init__()
        self.lidar_cfg = lidar_cfg

    def run(self):
        try:
            matcher = LidarMapMatcher(self.lidar_cfg)
            self.finished_loading.emit(matcher)
        except Exception as e:
            print(f"[LiDAR] Loading thread error: {e}")
            self.finished_loading.emit(None)


class ScanLocalizeThread(QThread):
    """Global registration off the GUI thread; it takes seconds, not milliseconds."""
    located = pyqtSignal(object)          # the scan_localizer result dict

    def __init__(self, points_optical, scan_points):
        super().__init__()
        self.points_optical = points_optical
        self.scan_points = scan_points

    def run(self):
        try:
            import scan_localizer
            self.located.emit(scan_localizer.locate_in_scan(
                self.points_optical, self.scan_points))
        except Exception as exc:
            self.located.emit(dict(ok=False, fitness=0.0,
                                   reason=f"{type(exc).__name__}: {exc}"))


class MapImportThread(QThread):
    """Background thread for importing a 3D file with Maya-style progress."""
    import_done = pyqtSignal(object, object, object, str)   # (cloud, vertices_np, faces_np_or_None, path)

    def __init__(self, path):
        super().__init__()
        self.path = path

    def run(self):
        try:
            import open3d as o3d
            import os
            ext = os.path.splitext(self.path)[1].lower()

            faces = None
            if ext in ('.obj', '.stl', '.fbx', '.gltf', '.glb'):
                mesh = o3d.io.read_triangle_mesh(self.path)
                if not mesh.has_triangles():
                    print(f"[Import] Warning: Mesh has no triangles: {self.path}")
                    self.import_done.emit(None, None, None, self.path)
                    return
                # Import the mesh as-is (vertices + faces) -- no more
                # sample_points_uniformly() collapsing it into a point cloud.
                # A plain point cloud is still derived below for the LiDAR
                # ICP matcher, which operates on points/normals, not faces.
                vertices = np.asarray(mesh.vertices)
                faces = np.asarray(mesh.triangles)
                cloud = o3d.geometry.PointCloud()
                cloud.points = o3d.utility.Vector3dVector(vertices)
                print(f"[Import] Loaded mesh as-is: {len(vertices):,} vertices, {len(faces):,} faces")
            else:
                cloud = o3d.io.read_point_cloud(self.path)
                vertices = np.asarray(cloud.points)

            if len(vertices) == 0:
                print("[Import] Error: No points found.")
                self.import_done.emit(None, None, None, self.path)
                return

            self.import_done.emit(cloud, vertices, faces, self.path)
        except Exception as e:
            print(f"[Import] Failed: {e}")
            self.import_done.emit(None, None, None, self.path)

# ──────────────────────────────────────────────────────
# Main Dashboard
# ──────────────────────────────────────────────────────
class HyperFusionApp(QMainWindow):
    marker_lineup_finished = pyqtSignal(object)
    _marker_request = None
    _marker_detection_sample = None
    _find_thread = None
    _find_observation = None
    _find_generation = None
    _find_rigid = None
    _marker_busy = False
    _marker_shutdown = False
    _marker_lock_context = None
    _marker_detail = "Place the locator at the marker centre, then enable Auto lineup."
    _marker_failed = False
    _automatic_lineup_only = False

    def __init__(self):
        super().__init__()
        self.setWindowTitle("HyperFusion | Camera Tracking")
        self.setGeometry(40, 40, 1440, 940)  # Start smaller but allow resizing
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #10151d; color: #dce4ef; font-family: 'DejaVu Sans'; font-size: 12px; }
            QLabel { background: transparent; }
            QPushButton { background: #202b39; color: #dce4ef; border: 1px solid #344356; border-radius: 6px; padding: 8px 12px; }
            QPushButton:hover { background: #2a3a4d; border-color: #6ba9d9; }
            QPushButton:disabled { color: #64748b; background: #171e28; border-color: #253041; }
            QPushButton:focus { border: 1px solid #7dbce9; }
            QLineEdit { background: #151e29; border: 1px solid #344356; padding: 7px; border-radius: 4px; }
            QTabWidget::pane { border: 1px solid #273447; border-radius: 5px; }
            QTabBar::tab { background: #151d28; color: #95a6ba; padding: 10px 20px; }
            QTabBar::tab:selected { color: #8dccf1; background: #223348; }
            QScrollArea { border: none; }
            QScrollBar:vertical { background: #151d28; width: 8px; }
            QScrollBar::handle:vertical { background: #344356; border-radius: 4px; min-height: 24px; }
            QSplitter::handle { background: #273447; }
        """)

        self.cfg = CONFIG
        net = self.cfg["network"]
        trk = self.cfg["tracking"]
        topics = self.cfg["topics"]
        aru = self.cfg["aruco"]

        print("[HyperFusion] Virtual Production Camera Tracking")
        print(tracking_config.describe(self.cfg))

        self.streaming_enabled = True
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.cv_bridge = CvBridge()
        self.latest_imu = "Waiting for IMU..."

        # Read before _build_ui(), which decides whether to draw the GNSS panel.
        self.gnss_enabled = bool(self.cfg["gnss"]["enabled"])
        self.latest_gnss = None

        # ── Pose pipeline: convert -> predict -> smooth ───────────────────────
        # All pose logic lives in TrackingEngine, shared with bridge_headless.py.
        # This class owns the GUI and the ROS subscriptions, nothing more.
        self.engine = TrackingEngine(self.cfg, log=print)

        self.depth_K = None
        self._last_rate_check = time.time()
        self._pose_rate = 0.0
        self._send_rate = 0.0
        self._last_pose_count = 0
        self._last_packet_count = 0
        self._pose_lock = threading.Lock()

        # ── LiDAR map fusion (continuous absolute localisation) ──────────────
        # The matcher estimates T_slam_to_map -- the VIO drift, as a rigid
        # transform -- and that is pushed into the engine's ground zero, so one
        # fused result drives the FreeD stream, the camera dot and the status
        # readout together. See lidar_map_matcher.py for why a *transform* and
        # not a position offset.
        self.lidar_cfg = self.cfg.get("lidar", {})
        self.lidar_matcher = None
        self._depth_frame_n = 0
        self._lidar_validation = None   # latest validate_depth() result
        self._lidar_measurement = None  # latest MapMeasurement
        self._last_fix_counter = 0
        self._image_lineup = None       # latest read-only lineup score
        self._lineup_frame_n = 0
        self._latest_gray = None        # full-res IR frame, for a manual lineup
        self._lineup = None             # LineupSession while one is open
        
        # Schedule the Maya-style loading animation to happen after UI is drawn
        if _HAS_LIDAR and self.lidar_cfg.get("enabled", False):
            QTimer.singleShot(100, self._start_lidar_load)
        # The viewport's "S" slider and lidar.map_scale are the same quantity.
        # The slider defaults to 1.0 and the dashboard pushes it to the matcher,
        # so without this the configured value is silently overwritten and the
        # fusion is handed a centimetre-scale map to match metre-scale depth
        # against -- which reads as "NOT ON SCAN" and can never resolve.
        QTimer.singleShot(0, self._apply_configured_map_scale)

        # ── ArUco ground zero ────────────────────────────────────────────────
        self.aruco_enabled = bool(aru["enabled"])
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

        # Intrinsics start as a config fallback and are replaced by the real
        # values from camera_info. PnP with wrong intrinsics puts ground zero in
        # the wrong place, so this matters.

        self._build_ui()
        self.marker_lineup_finished.connect(self._finish_marker_lineup, Qt.QueuedConnection)

        # UI refresh timer
        self._timer = QTimer()
        self._timer.timeout.connect(self._refresh)
        self._timer.start(max(1, int(1000 / max(1, self.cfg["ui"]["refresh_hz"]))))

        # FreeD send timer. send_rate_hz = 0 means "send on every pose instead",
        # which the pose callbacks handle directly.
        self.send_on_pose = not net["send_rate_hz"]
        if not self.send_on_pose:
            self._freed_timer = QTimer()
            self._freed_timer.timeout.connect(self._stream_freed)
            self._freed_timer.start(max(1, int(1000 / net["send_rate_hz"])))

        # ── Optional hardware threads ────────────────────────────────────────
        self.gnss_thread = None
        if self.gnss_enabled:
            self._start_gnss()

        self.ltc_thread = None
        if self.cfg["timecode"]["enabled"]:
            self.ltc_thread = LTCAudioListenerThread(self.cfg["timecode"])
            self.ltc_thread.tc_updated.connect(self._cb_tc)
            self.ltc_thread.start()
        self.latest_timecode = None

        self._pose_lock = threading.Lock()

        # ── ROS 2 ────────────────────────────────────────────────────────────
        if not rclpy.ok():
            rclpy.init(args=None)

        class DashboardNode(Node):
            def __init__(self, gui):
                super().__init__('vp_camera_tracking_bridge')
                self.gui = gui
                self.gui._cam_frame_n = 0

                self.create_subscription(
                    Odometry, topics["odometry"], self.gui.odometry_callback, 10)
                self.create_subscription(
                    PoseWithCovarianceStamped, topics["vo_pose"],
                    self.gui.vo_pose_callback, 10)
                self.create_subscription(
                    Image, topics["image"], self.gui.image_callback, 1)
                self.create_subscription(
                    CameraInfo, topics["camera_info"],
                    self.gui.camera_info_callback, 1)
                self.create_subscription(
                    Imu, topics["imu"], self.gui.imu_callback, 10)
                # Depth gives the ArUco solve an independent distance, which is
                # what catches a flipped planar-marker pose before it anchors,
                # and it is the sensor the LiDAR map fusion runs on.
                self.create_subscription(
                    Image, topics.get("depth", "/camera/depth/image_rect_raw"),
                    self.gui.depth_callback, 1)
                # Depth's OWN intrinsics. Previously `depth_K` was initialised
                # to None and never assigned, so every consumer silently fell
                # back to the infra1 matrix. That happens to be correct only
                # while align_depth stays off and both streams run at the same
                # resolution -- the moment either changes, the point cloud is
                # deprojected with the wrong focal length and ICP fits a
                # skewed room to the real one.
                self.create_subscription(
                    CameraInfo,
                    topics.get("depth_camera_info",
                               "/camera/depth/camera_info"),
                    self.gui.depth_camera_info_callback, 1)

        self.ros_node = DashboardNode(self)
        self.ros_thread = threading.Thread(
            target=rclpy.spin, args=(self.ros_node,), daemon=True)
        self.ros_thread.start()

    def _apply_configured_map_scale(self):
        """Push lidar.map_scale into the viewport's alignment slider."""
        viewer = getattr(self, "map_viewer", None)
        if viewer is None:
            return
        configured = float(self.lidar_cfg.get("map_scale", 1.0))
        if viewer.set_alignment_scale(configured):
            print(f"[LiDAR] mesh-alignment scale set from tracking.yaml: "
                  f"{configured} m per map unit")

    def _autodetect_map_scale(self, points):
        """Guess the map file's units from its extent, if still at the default.

        A scan exported from Unreal is in centimetres, and left at 1.0 the
        fusion has no chance -- so rather than fail with a scale slider the
        operator has no reason to suspect, pick the factor that puts the scan
        in a plausible room-to-stage range and say so. An operator who has
        already set a scale is left alone.
        """
        viewer = getattr(self, "map_viewer", None)
        if viewer is None or abs(viewer.alignment_scale - 1.0) > 1e-9:
            return                      # operator (or config) has spoken
        extent = float(np.ptp(np.asarray(points), axis=0).max())
        if 0.5 <= extent <= 50.0:
            return                      # already plausible in metres

        # More than one unit can be "plausible" -- 3095 units is a 31 m stage in
        # centimetres or a 3.1 m room in millimetres -- so rather than take the
        # first match, take whichever lands closest to a typical set size. Ties
        # like that are why the log says which reading it chose.
        TYPICAL_SET_M = 8.0
        candidates = [(1.0, "metres"), (0.01, "centimetres"),
                      (0.001, "millimetres")]
        plausible = [(abs(math.log((extent * f) / TYPICAL_SET_M)), f, unit)
                     for f, unit in candidates if 0.5 <= extent * f <= 50.0]
        if not plausible:
            print(f"[LiDAR] ⚠️  scan is {extent:.0f} units across and no standard "
                  f"unit makes that a plausible room — set lidar.map_scale in "
                  f"tracking.yaml by hand.")
            return
        _, factor, unit = min(plausible)
        viewer.set_alignment_scale(factor)
        print(f"[LiDAR] scan is {extent:.0f} units across — reading that as "
              f"{unit} ({extent * factor:.1f} m); alignment scale set to "
              f"{factor}. Override with lidar.map_scale in tracking.yaml.")

    def _start_lidar_load(self):
        """Shows a Maya-style indeterminate loading bar while loading .obj"""
        self.progress_dialog = QProgressDialog("Loading scan geometry…", None, 0, 0, self)
        self.progress_dialog.setWindowTitle("Loading LiDAR")
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setMinimumDuration(0)
        self.progress_dialog.setCancelButton(None)
        self.progress_dialog.show()

        self.lidar_thread = LidarLoadThread(self.lidar_cfg)
        self.lidar_thread.finished_loading.connect(self._on_lidar_loaded)
        self.lidar_thread.start()

    def _on_lidar_loaded(self, matcher):
        if self.progress_dialog:
            self.progress_dialog.close()
        self.lidar_matcher = matcher
        if matcher is None:
            print("[Dashboard] ❌ LiDAR matcher could not be created.")
            return
        self._map_path = getattr(matcher, "map_file", None) or None

        # The worker starts whether or not a scan is loaded. It idles on an
        # empty map -- update_depth() gates on _ready() -- and starting it here
        # is what lets a scan imported later be fused without a restart. Only
        # starting it on a preloaded map is why an import into an empty scene
        # used to load the mesh and then never match anything against it.
        matcher.start_worker()

        if matcher.map_cloud is not None:
            if matcher._mesh_vertices is not None:
                self.map_viewer.set_map_mesh(matcher._mesh_vertices, matcher._mesh_triangles)
            elif matcher._map_native is not None:
                self.map_viewer.set_map_points(matcher._map_native)
            print("[Dashboard] ✅ LiDAR map matching enabled (with ICP correction)")
        elif not getattr(matcher, "map_file", ""):
            # An empty map_file is the configured default, not a failure.
            print("[Dashboard] ✅ LiDAR fusion ready — import a scan "
                  "(File > Import) to start matching.")
        else:
            print("[Dashboard] ❌ LiDAR map loading failed.")

    def _start_gnss(self):
        self.cfg["gnss"]["port"] = self.gnss_port.text().strip()
        self.gnss_thread = GNSSReceiver(self.cfg["gnss"], self.engine.update_gnss)
        self.gnss_thread.start()

    _MENUBAR_QSS = (
        "QMenuBar{background:#151d28;color:#c3d2e3;border-bottom:1px solid #273447;"
        "padding:2px 6px;}"
        "QMenuBar::item{background:transparent;padding:6px 12px;border-radius:4px;}"
        "QMenuBar::item:selected{background:#223348;color:#8dccf1;}"
        "QMenu{background:#18212d;color:#dce4ef;border:1px solid #2b394d;padding:4px;}"
        "QMenu::item{padding:6px 26px 6px 20px;border-radius:4px;}"
        "QMenu::item:selected{background:#223348;color:#8dccf1;}"
        "QMenu::separator{height:1px;background:#273447;margin:4px 8px;}")

    def _build_menu(self):
        """File menu in the window's top-left corner.

        Import is the same dialog and the same import path the toolbar's
        browse button used to open; the toolbar itself is gone.
        """
        bar = self.menuBar()
        bar.setStyleSheet(self._MENUBAR_QSS)
        bar.setNativeMenuBar(False)     # keep it in the window, not the desktop panel
        file_menu = bar.addMenu("&File")

        act_import = QAction("&Import Scan…", self)
        act_import.setShortcut(QKeySequence("Ctrl+I"))
        act_import.setStatusTip("Import a .obj / .ply / .pcd scan into the scene")
        act_import.triggered.connect(self._menu_import_scan)
        file_menu.addAction(act_import)

        act_save = QAction("&Save Scene…", self)
        act_save.setShortcut(QKeySequence.Save)
        act_save.setStatusTip("Save the scan alignment and tracking origin to a scene file")
        act_save.triggered.connect(self._menu_save_scene)
        file_menu.addAction(act_save)

        file_menu.addSeparator()
        self.act_delete = QAction("&Delete Scan", self)
        self.act_delete.setShortcut(QKeySequence.Delete)
        self.act_delete.setStatusTip("Remove the loaded scan from the scene")
        self.act_delete.triggered.connect(self._menu_delete_scan)
        file_menu.addAction(self.act_delete)
        # Greyed out with nothing loaded, so the menu says whether there is a
        # scan to delete rather than silently doing nothing.
        file_menu.aboutToShow.connect(self._sync_delete_action)

    def _menu_import_scan(self):
        """File > Import — hands off to the map viewer's import dialog."""
        viewer = getattr(self, "map_viewer", None)
        if viewer is not None:
            viewer.import_map()

    def _sync_delete_action(self):
        viewer = getattr(self, "map_viewer", None)
        self.act_delete.setEnabled(viewer is not None and viewer.has_map())

    def _menu_delete_scan(self):
        """File > Delete Scan — remove the loaded scan from the whole session.

        The scan is not just what the viewport draws: the matcher holds a
        downsampled pyramid, a KD-tree and a raycast surface built from it,
        and the engine holds a map alignment solved against it. Clearing only
        the viewport is what "delete does nothing" looked like -- the mesh
        vanished and the fusion carried on matching depth to it.
        """
        viewer = getattr(self, "map_viewer", None)
        if viewer is None or not viewer.has_map():
            return
        name = os.path.basename(self._map_path) if self._map_path else "the scan"
        if QMessageBox.question(
                self, "Delete Scan",
                f"Remove {name} from the scene?\n\n"
                "The alignment and any map lock are dropped with it. "
                "The file on disk is not touched.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No) != QMessageBox.Yes:
            return
        self._delete_scan()

    def _delete_scan(self):
        """Unconditional delete, shared by the menu and the viewport's Del.

        The matcher goes first. Clearing the viewer emits `alignment_changed`,
        and that handler rebases a still-aligned matcher onto the new
        transform -- so clearing in the other order would have the fusion
        re-solve itself against the scan on its way out.
        """
        matcher = getattr(self, "lidar_matcher", None)
        if matcher is not None:
            matcher.clear_map()
        self.map_viewer.clear_map()
        self._map_path = None
        # The pose pipeline was corrected against a scan that is now gone;
        # leave it correcting and the camera keeps a drift estimate solved
        # against nothing.
        self.engine.confirm_lineup()
        self.engine.transformer.clear_ground_zero()
        self.engine.map_aligned = False
        self._lidar_validation = None
        self._lidar_measurement = None
        print("[Dashboard] 🗑  Scan deleted — scene is empty.")

    def _menu_save_scene(self):
        """File > Save — write the scene the operator has set up.

        What is worth keeping across a restart is the work that cannot be
        recomputed: which scan is loaded, where the alignment put it, and
        where the tracking origin sits. Poses stream in again on their own.
        """
        viewer = getattr(self, "map_viewer", None)
        if viewer is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Scene", "scene.json", "HyperFusion scene (*.json);;All (*)")
        if not path:
            return
        if not os.path.splitext(path)[1]:
            path += ".json"
        scene = {
            "format": "hyperfusion.scene",
            "version": 1,
            "saved": datetime.datetime.now().isoformat(timespec="seconds"),
            "map_file": getattr(self, "_map_path", None),
            "alignment": {
                "tx": viewer.tx, "ty": viewer.ty, "tz": viewer.tz,
                "roll": viewer.roll, "pitch": viewer.pitch, "yaw": viewer.yaw,
                "scale": viewer.alignment_scale,
                "stored": viewer.is_pose_stored(),
            },
            "locator": np.asarray(viewer.get_locator_transform(),
                                  dtype=float).tolist(),
            "marker_lineup": {
                "mount": "floor_table",
                "dictionary": self.marker_family.currentText(),
                "marker_id": self.marker_id.value(),
                "marker_size_m": self.marker_width.value(),
            },
            "display_transforms": {
                name: viewer.persp_view.get_display_transform(name).tolist()
                for name in ("trail", "world")
            },
        }
        try:
            with open(path, "w") as fh:
                json.dump(scene, fh, indent=2)
        except OSError as e:
            # A save that silently failed is worse than no save button at all.
            QMessageBox.warning(self, "Save Failed", f"Could not write:\n{path}\n\n{e}")
            print(f"[Dashboard] ❌ Scene save failed: {e}")
            return
        print(f"[Dashboard] ✅ Scene saved: {path}")

    def _build_ui(self):
        self._automatic_lineup_only = True
        self._build_menu()
        shell = QWidget()
        layout = QVBoxLayout(shell)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(14)
        header = QHBoxLayout()
        brand = QVBoxLayout()
        title = QLabel("HyperFusion")
        title.setFont(QFont("DejaVu Sans", 22, QFont.Bold))
        title.setStyleSheet("font-size:26px;font-weight:600;color:#e4edf7;")
        brand.addWidget(title)
        subtitle = QLabel("VIRTUAL PRODUCTION  /  CAMERA TRACKING")
        subtitle.setStyleSheet("color:#8295ad; font-size:10px; letter-spacing:2px;")
        brand.addWidget(subtitle)
        header.addLayout(brand)
        header.addStretch()
        self.lbl_session = QLabel("AWAITING LINEUP")
        self.lbl_session.setStyleSheet("background:#253448; color:#a6cae8; border-radius:6px; padding:10px 16px;")
        header.addWidget(self.lbl_session)
        layout.addLayout(header)

        telemetry = QHBoxLayout()
        self.health_cards = {}
        for key, title in (("motion", "01  MOTION"), ("scan", "02  SCAN VALIDATION"), ("gnss", "03  RTK POSITION")):
            card = QFrame()
            card.setStyleSheet("QFrame{background:#18212d;border:1px solid #2b394d;border-radius:8px;}")
            column = QVBoxLayout(card)
            label = QLabel(title)
            label.setStyleSheet("border:none;color:#91a4bb;font-size:10px;")
            value = QLabel("Waiting for input")
            value.setStyleSheet("border:none;color:#dce4ef;font-size:13px;font-weight:600;")
            column.addWidget(label)
            column.addWidget(value)
            self.health_cards[key] = value
            telemetry.addWidget(card)
        layout.addLayout(telemetry)

        main_split = QSplitter(Qt.Horizontal)
        main_split.setHandleWidth(6)
        tabs = QTabWidget()
        tabs.setMinimumWidth(340)
        tabs.setMaximumWidth(440)
        operate = QWidget()
        left = QVBoxLayout(operate)
        left.setSpacing(10)
        left.setContentsMargins(14, 14, 14, 14)

        def panel(text, minimum=0):
            label = QLabel(text)
            label.setWordWrap(True)
            label.setMinimumHeight(minimum)
            label.setStyleSheet("background:#151e29;color:#aebfd2;padding:10px;border:1px solid #2b394d;border-radius:6px;")
            return label

        left.addWidget(self._hdr("TRACKING & OUTPUT", size=10))
        self.lbl_status = panel("Waiting for camera pose", 86)
        left.addWidget(self.lbl_status)
        left.addWidget(self._hdr("CAMERA LINEUP", size=10))
        left.addWidget(self._build_marker_lineup_panel())
        left.addWidget(self._hdr("LIVE EVIDENCE", size=10))
        self.lbl_pose = panel("Position —\nRotation —", 64)
        self.lbl_pose.setFont(QFont("DejaVu Sans Mono", 9))
        left.addWidget(self.lbl_pose)
        self.btn_udp = QPushButton("FreeD output · Enabled")
        self.btn_udp.setMinimumHeight(44)
        self._style_btn(True)
        self.btn_udp.clicked.connect(self._toggle_stream)
        left.addWidget(self.btn_udp)
        left.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(operate)
        tabs.addTab(scroll, "Operate")

        setup = QWidget()
        settings = QVBoxLayout(setup)
        settings.setContentsMargins(14, 14, 14, 14)
        settings.setSpacing(10)
        settings.addWidget(self._hdr("FLOOR / TABLE MARKER", size=10))
        settings.addWidget(self.marker_settings)
        settings.addWidget(panel("Place the locator at the flat marker centre. Its rotation is calculated automatically. "
                                 "Measure the black square, excluding the white paper margin."))
        self.btn_ar = QCheckBox("Show scan overlay")
        self.btn_ar.toggled.connect(self._toggle_ar_overlay)
        settings.addWidget(self.btn_ar)
        settings.addWidget(self._hdr("SCAN DETAILS", size=10))
        self.lbl_fusion = panel("Load the room scan to begin", 100)
        settings.addWidget(self.lbl_fusion)
        settings.addWidget(self._hdr("INPUT READINESS", size=10))
        self.lbl_readiness = panel("Waiting for sensors")
        settings.addWidget(self.lbl_readiness)
        settings.addWidget(self._hdr("ZED-F9P · OUTDOOR RTK", size=10))
        settings.addWidget(panel("The antenna receives satellites. RTK also needs corrections from an NTRIP service or a fixed base station. Start with the indoor scan workflow."))
        self.gnss_port = QLineEdit(self.cfg["gnss"]["port"])
        self.gnss_port.setPlaceholderText("/dev/serial/by-id/…")
        self.gnss_port.setToolTip("Receiver serial port. Prefer a stable /dev/serial/by-id path. Changes apply on the next connection.")
        settings.addWidget(self.gnss_port)
        self.btn_gnss = QPushButton("Disconnect GNSS" if self.gnss_enabled else "Connect GNSS")
        self.btn_gnss.clicked.connect(self._toggle_gnss)
        settings.addWidget(self.btn_gnss)
        self.lbl_gnss = panel("Reference mode · Correction source and alignment not configured", 90)
        settings.addWidget(self.lbl_gnss)
        settings.addWidget(self._hdr("IMU & TIMECODE", size=10))
        self.lbl_imu = panel("Waiting for IMU")
        self.lbl_tc = panel("LTC unavailable · Display only")
        settings.addWidget(self.lbl_imu)
        settings.addWidget(self.lbl_tc)
        settings.addWidget(self._hdr("PLANNED INPUTS", size=10))
        settings.addWidget(panel("Additional depth cameras ×3 · Not connected\nUWB anchors · Not configured\nIR markers / illumination · Not configured"))
        settings.addWidget(panel("Calibration and outdoor setup: RTK_SETUP.md\nProject assessment: PROJECT_ASSESSMENT.md"))
        settings.addStretch()
        setup_scroll = QScrollArea()
        setup_scroll.setWidgetResizable(True)
        setup_scroll.setWidget(setup)
        tabs.addTab(setup_scroll, "Setup")
        main_split.addWidget(tabs)

        workspace = QSplitter(Qt.Vertical)
        preview = QWidget()
        preview_layout = QVBoxLayout(preview)
        preview_layout.setContentsMargins(10, 4, 0, 4)
        self.lbl_preview_title = self._hdr("LIVE CAMERA  /  SCAN OVERLAY", size=10)
        preview_layout.addWidget(self.lbl_preview_title)
        self.lbl_track = self._cambox(480, 180, "Waiting for D455 infrared image")
        self.lbl_track.setMinimumHeight(180)
        preview_layout.addWidget(self.lbl_track, 1)
        workspace.addWidget(preview)
        scene = QWidget()
        scene_layout = QVBoxLayout(scene)
        scene_layout.setContentsMargins(10, 4, 0, 0)
        scene_layout.addWidget(self._hdr("SCENE ALIGNMENT", size=10))
        self.map_viewer = MapViewerWidget()
        self.map_viewer.persp_view.set_camera_edit_locked(True)
        self.map_viewer.file_loaded.connect(self._on_map_loaded)
        self.map_viewer.locator_changed.connect(self._on_locator_moved)
        self.map_viewer.alignment_changed.connect(self._on_map_alignment_changed)
        self.map_viewer.camera_moved.connect(self._on_camera_gizmo_moved)
        self.map_viewer.delete_requested.connect(self._menu_delete_scan)
        scene_layout.addWidget(self.map_viewer, 1)
        workspace.addWidget(scene)
        workspace.setSizes([260, 500])
        main_split.addWidget(workspace)
        main_split.setSizes([365, 1040])
        main_split.setStretchFactor(1, 1)
        layout.addWidget(main_split, 1)
        self.lbl_route = QLabel(f"FreeD UDP  →  {UDP_IP}:{UDP_PORT}   /   Camera {CAMERA_ID}     •     All positions in metres")
        self.lbl_route.setStyleSheet("color:#8295ad;font-size:11px;")
        layout.addWidget(self.lbl_route)
        self.setCentralWidget(shell)

    #: Class-level defaults so the status text works on a partially built
    #: instance too -- the test harness constructs this with __new__, and Qt
    #: raises RuntimeError rather than AttributeError for a missing attribute
    #: on a QMainWindow whose __init__ never ran, which getattr() cannot catch.
    lidar_matcher = None
    _map_path = None
    _image_lineup = None
    _lineup_frame_n = 0
    _lineup = None
    _lineup_overlay = None
    _latest_gray = None
    _image_mono = None
    _image_stamp = None
    _image_sample = None
    _ar_overlay_on = False
    _ar_overlay = None
    _ar_frame_n = 0

    _AR_BTN_QSS = (
        "QPushButton{background:#263238;color:#b0bec5;border-radius:4px;"
        "border:1px solid #455a64;}"
        "QPushButton:checked{background:#006064;color:#fff;"
        "border:1px solid #00acc1;font-weight:bold;}")

    _LINEUP_BTN_IDLE_QSS = (
        "QPushButton{background:#37474f;color:#fff;border-radius:4px;"
        "border:1px solid #546e7a;} QPushButton:hover{background:#546e7a;}")
    _LINEUP_BTN_LIVE_QSS = (
        "QPushButton{background:#ff6f00;color:#000;border-radius:4px;"
        "border:1px solid #ff8f00;font-weight:bold;}")

    _LOCK_BTN_IDLE_QSS = (
        "QPushButton{background:#214969;color:#fff;border-radius:4px;"
        "border:1px solid #4886b4;} QPushButton:hover{background:#2e5e83;}")
    _LOCK_BTN_OK_QSS = (
        "QPushButton{background:#00695c;color:#fff;border-radius:4px;"
        "border:1px solid #00897b;} QPushButton:hover{background:#00897b;}")
    _LOCK_BTN_BAD_QSS = (
        "QPushButton{background:#b71c1c;color:#fff;border-radius:4px;"
        "border:1px solid #d32f2f;} QPushButton:hover{background:#d32f2f;}")

    def _hdr(self, text, color='#8dccf1', size=10):
        l = QLabel(text)
        l.setFont(QFont("Arial", size, QFont.Bold))
        l.setStyleSheet("color:#91a8bf; padding:4px 0;")
        return l

    def _on_map_loaded(self, path):
        """Triggered by the Import button — starts background import with animation."""
        import os
        basename = os.path.basename(path)
        print(f"[Dashboard] Importing LiDAR map: {basename}")

        # Maya-style indeterminate progress dialog
        self._import_progress = QProgressDialog(
            f"Importing {basename}...\n\nReading mesh vertices and faces.",
            None, 0, 0, self)
        self._import_progress.setWindowTitle("📥 Importing LiDAR Map")
        self._import_progress.setWindowModality(Qt.WindowModal)
        self._import_progress.setMinimumDuration(0)
        self._import_progress.setCancelButton(None)
        self._import_progress.setMinimumWidth(380)
        self._import_progress.show()

        self._import_thread = MapImportThread(path)
        self._import_thread.import_done.connect(self._on_import_finished)
        self._import_thread.start()

    def _on_import_finished(self, cloud, points, faces, path):
        """Called when background import completes."""
        import os
        if hasattr(self, '_import_progress') and self._import_progress:
            self._import_progress.close()

        if cloud is None or points is None:
            print(f"[Dashboard] ❌ Import failed: {os.path.basename(path)}")
            return

        # Update the 3D Viewer
        self._map_path = path
        if faces is not None:
            self.map_viewer.set_map_mesh(points, faces)
            print(f"[Dashboard] ✅ Imported mesh: {len(points):,} vertices, {len(faces):,} faces → 3D Viewer")
        else:
            self.map_viewer.set_map_points(points)
            print(f"[Dashboard] ✅ Imported {len(points):,} points → 3D Viewer")

        self._autodetect_map_scale(points)

        # Hand the new scan to the fusion. Routed through set_map_cloud() so
        # the resolution pyramid and the KD-tree are rebuilt: assigning
        # `map_cloud` directly (what this used to do) left both pointing at
        # the previously loaded map, so ICP kept matching against the old set.
        if getattr(self, "lidar_matcher", None) is not None:
            try:
                import open3d as o3d
                if faces is not None:
                    # Mesh vertices cluster wherever the scan happened to be
                    # dense; uniform surface samples give ICP an even target.
                    # Sampled through the matcher's own reproducible sampler --
                    # Open3D's takes no seed, so the same scan would otherwise
                    # give a different fusion problem on every import.
                    from lidar_map_matcher import sample_mesh_surface
                    cloud = o3d.geometry.PointCloud()
                    cloud.points = o3d.utility.Vector3dVector(
                        sample_mesh_surface(
                            points, faces,
                            max(self.lidar_matcher.mesh_sample_points,
                                len(faces) * 10)))
                self.lidar_matcher._generation += 1
                self.lidar_matcher.last_measurement = None
                self.lidar_matcher.is_aligned = False
                self.engine.confirm_lineup()
                self.engine.transformer.clear_ground_zero()
                self.engine.map_aligned = False
                self.lidar_matcher.map_file = path
                self.lidar_matcher.set_map_scale(self.map_viewer.alignment_scale)
                self.lidar_matcher.set_map_cloud(cloud)
                if faces is not None:
                    # Silhouettes need the surface, not the sampled cloud.
                    self.lidar_matcher.set_map_mesh(points, faces)
                # Idempotent: a no-op when the worker is already running, and
                # the thing that gets fusion going when the dashboard started
                # with an empty scene.
                self.lidar_matcher.start_worker()
                print("[Dashboard] ✅ Scan handed to the map fusion.")
            except Exception as e:
                print(f"[Dashboard] LiDAR matcher update error: {e}")

    def _cambox(self, min_w, min_h, placeholder):
        l = QLabel(placeholder)
        l.setAlignment(Qt.AlignCenter)
        l.setMinimumSize(min_w, min_h)
        l.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        l.setStyleSheet("background:#000; border:1px solid #2a2a3a; border-radius:4px;")
        return l

    def _style_btn(self, on):
        if on:
            s = ("QPushButton{background:#205943;color:#e0f2e9;border-radius:6px;"
                 "padding:10px;border:1px solid #43866a;}"
                 "QPushButton:hover{background:#2b7055;}")
        else:
            s = ("QPushButton{background:#b71c1c;color:#fff;border-radius:6px;"
                 "padding:10px;border:2px solid #d50000;}"
                 "QPushButton:hover{background:#d50000;}")
        self.btn_udp.setStyleSheet(s)

    def _toggle_stream(self):
        self.streaming_enabled = not self.streaming_enabled
        state = "Enabled" if self.streaming_enabled else "Paused"
        self.btn_udp.setText(f"FreeD output · {state}")
        self._style_btn(self.streaming_enabled)

    def _toggle_gnss(self):
        if self.gnss_thread is not None:
            self.gnss_thread.stop()
            self.gnss_thread = None
            self.engine.gnss_fix = None
            self.engine.gnss_reason = "GNSS disconnected"
            self.engine.fusion.observe("gnss", False, time.monotonic(), "Receiver disconnected")
            self.btn_gnss.setText("Connect GNSS")
        else:
            self._start_gnss()
            self.btn_gnss.setText("Disconnect GNSS")

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
                # Some OpenCV 4x4 families share codes. Keep the first family
                # configured for a physical square rather than reporting it four times.
                if any(np.max(np.abs(quad - old_quad)) < 1.0
                       for _, _, old_quad in found):
                    continue
                found.append((dict_name, int(marker_id[0]), quad))
        return found

    # --- ROS Callbacks ---
    def image_callback(self, msg):
        """Throttled ~5fps camera view for ArUco anchoring + visual lag check."""
        try:
            self._cam_frame_n += 1
            # Skip 2 of every 3 frames (~5 fps): this feed is a viewfinder.
            if self._cam_frame_n % 3 != 0:
                return

            # Decode grayscale infra1
            if msg.encoding in ("mono8", "8UC1"):
                gray = self.cv_bridge.imgmsg_to_cv2(msg, "mono8")
            elif msg.encoding in ("mono16", "16UC1"):
                gray = (self.cv_bridge.imgmsg_to_cv2(msg, "16UC1") >> 8).astype(np.uint8)
            else:
                gray = self.cv_bridge.imgmsg_to_cv2(msg, "mono8")

            # The lineup check and AR overlay run off the full-resolution IR
            # frame; the display copy below is downscaled and no use for it.
            self._latest_gray = gray
            self._image_mono = time.monotonic()
            self._image_stamp = self._stamp_seconds(msg)
            self._image_sample = (gray, self._image_stamp, self._image_mono)
            if self.lidar_matcher is not None:
                self.lidar_matcher.set_image(gray, self.engine.aruco_K)

            # Downscale display frame (320x200)
            disp = gray.copy()
            cv_img = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
            if hasattr(self, 'map_viewer'):
                # Pass full res gray for AR overlay (or color if we had it), but disp is fine for UI
                self.map_viewer.update_camera_frame(cv_img)

            # ArUco detection at ~5 Hz (every 6 frames at 30 fps).
            if self._cam_frame_n % 6 == 0:
                markers = self._find_aruco_markers(gray) if self.aruco_enabled else []
                self.last_aruco_markers = markers
                self._marker_detection_sample = (markers, self._image_stamp, self._image_mono,
                                                 (gray.shape[1], gray.shape[0]))
            else:
                markers = getattr(self, 'last_aruco_markers', [])

            if markers:
                corners = [quad.reshape(1, 4, 2) for _, _, quad in markers]
                ids = np.asarray([[marker_id] for _, marker_id, _ in markers], dtype=np.int32)
                scale_x, scale_y = disp.shape[1] / msg.width, disp.shape[0] / msg.height
                disp_corners = [(c * np.array([[scale_x, scale_y]])).astype(np.float32) for c in corners]
                aruco.drawDetectedMarkers(cv_img, disp_corners, ids)


            n_ids = len(markers)
            if self.transformer.is_zeroed:
                st = "ALIGNMENT SET"
            elif n_ids:
                seen = ", ".join(f"{family}/{marker_id}"
                                 for family, marker_id, _ in markers[:3])
                st = f"MARKER {seen} DETECTED"
            else:
                st = "SCANNING..."

            col = (0,255,255) if n_ids or self.transformer.is_zeroed else (150,150,150)
            cv2.putText(cv_img, st, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
            self.latest_cv_image = cv_img.copy()
        except Exception as exc:
            print(f"[ArUco] image callback error: {exc}")



    # ── Delegating views onto the engine ─────────────────────────────────
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
    def latest_unreal(self):
        return self.engine.latest_unreal

    @property
    def slam_status(self):
        return self.engine.slam_status

    @property
    def last_vio_time(self):
        return self.engine.last_vio_time

    @property
    def _intrinsics_from_camera_info(self):
        return self.engine.intrinsics_from_camera_info

    def depth_camera_info_callback(self, msg):
        """Depth stream intrinsics. First valid message wins, like colour."""
        if self.depth_K is not None:
            return
        k = list(msg.k) if hasattr(msg, "k") else list(msg.K)
        if len(k) < 9 or k[0] == 0.0:
            return
        self.depth_K = np.asarray(k, dtype=float).reshape(3, 3)
        self.engine.set_depth_intrinsics(self.depth_K)
        print(f"[Depth] intrinsics from camera_info: fx={self.depth_K[0,0]:.2f} "
              f"fy={self.depth_K[1,1]:.2f} cx={self.depth_K[0,2]:.2f} "
              f"cy={self.depth_K[1,2]:.2f}")

    def _depth_intrinsics(self):
        """Best available depth intrinsics, real ones preferred."""
        if self.depth_K is not None:
            return self.depth_K
        return self.engine.aruco_K

    def depth_callback(self, msg):
        """Depth in: ArUco cross-check, LiDAR map fusion, and validation."""
        try:
            depth = self.cv_bridge.imgmsg_to_cv2(msg, "16UC1")
        except Exception:
            return
        self.engine.set_depth(depth, self.depth_K, self._stamp_seconds(msg))
        self._depth_frame_n += 1

        if self.lidar_matcher is None:
            return
        K = self._depth_intrinsics()
        received = time.monotonic()
        if self.engine.fusion.enabled:
            sample = self.engine.capture_depth()
            if sample is None:
                return
            depth, K, pose, received = sample
        else:
            pose = self.engine.latest_pose
        if pose is None or K is None:
            return

        # Continuous map localisation. `pose` is the RAW SLAM pose, which is
        # what the fusion needs: it estimates the SLAM->map transform, so
        # feeding it an already-corrected pose would close a loop around its
        # own output. (The previous code did exactly that, then subtracted the
        # correction it had already applied, and oscillated.) The matcher
        # rate-limits itself to icp_rate_hz, so calling per frame is cheap.
        if self.lidar_matcher.is_aligned:
            # Hand the frame over and return: a 100-200 ms ICP run here blocks
            # the single-threaded ROS executor, which stops odometry being
            # delivered and drops the FreeD output rate.
            try:
                self.lidar_matcher.submit(
                    depth, K, pose, received,
                    is_stationary=self.engine.hold.is_stationary)
            except Exception as exc:
                print(f"[LiDAR] fusion submit error: {exc}")

        # Read-only agreement score for the operator, ~3 Hz at 30 fps.
        if self._depth_frame_n % 10 == 0 and self.engine.is_anchored:
            try:
                self._lidar_validation = self.lidar_matcher.validate_depth(
                    depth, K, make_transform(pose[0], pose[1]))
            except Exception as exc:
                print(f"[LiDAR] validation error: {exc}")
                self._lidar_validation = None

    def _push_map_alignment(self):
        """Hand the fused SLAM->map transform to the engine as ground zero."""
        T_slam_to_map, _ = self.lidar_matcher.map_transform()
        self.engine.apply_map_alignment(T_slam_to_map, self._align_rigid())

    def _align_rigid(self):
        """The mesh alignment's rigid part, with its scale sent to the matcher.

        `T_align` carries the operator's scale slider in its rotation block,
        which is how a centimetre scan gets reconciled with a metre world. A
        similarity cannot be composed with poses as if it were rigid, so the
        scale is split off here and pushed to the matcher (which then holds
        its map in metres) and only the rigid part is used for transforms.
        """
        viewer = getattr(self, "map_viewer", None)
        if viewer is None:
            return np.eye(4)
        scale, rigid = decompose_similarity(viewer.T_align)
        if self.lidar_matcher is not None:
            self.lidar_matcher.set_map_scale(scale)
        return rigid

    def _on_map_alignment_changed(self, T_align):
        """The operator dragged the mesh-alignment gizmo.

        The camera must stay where it is on screen while the mesh moves under
        it, so ground zero is held and the SLAM->map transform is re-derived
        from it instead of being re-measured. Without this the ICP
        correspondence search would keep centring on the pre-drag location.
        """
        rigid = self._align_rigid()
        if self.lidar_matcher is None or not self.lidar_matcher.is_aligned:
            return
        zero = self.transformer._zero
        if zero is None:
            return
        self.lidar_matcher.rebase_alignment(invert_transform(rigid) @ zero)

    def _build_marker_lineup_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(8)
        # Settings live on Setup; acquisition and health stay on Operate.
        self.marker_settings = QWidget(panel)
        self.marker_family = QComboBox()
        self.marker_family.addItems([name for name, *_ in self.aruco_detectors])
        self.marker_id = QSpinBox()
        self.marker_id.setRange(-1, 9999)
        self.marker_id.setSpecialValueText("One visible marker")
        self.marker_id.setValue(int(self.cfg["aruco"].get("lineup_marker_id", -1)))
        self.marker_id.setToolTip("Use the displayed marker ID, or show exactly one marker.")
        self.marker_width = QDoubleSpinBox()
        self.marker_width.setDecimals(4)
        self.marker_width.setRange(0.005, 5.0)
        self.marker_width.setSingleStep(0.005)
        self.marker_width.setSuffix(" m")
        self.marker_width.setValue(self.engine.marker_size)
        self.marker_width.setKeyboardTracking(False)
        form = QFormLayout(self.marker_settings)
        form.setContentsMargins(0, 0, 0, 0)
        form.addRow("Dictionary", self.marker_family)
        form.addRow("Marker ID", self.marker_id)
        form.addRow("Square width", self.marker_width)
        # Placing the locator by hand is the weak step: the marker is not in the
        # scan, so there is nothing to snap to, and a wrong height or a 90 deg
        # heading both look plausible on screen. This asks the scan instead.
        self.btn_find_marker = QPushButton("Find marker in scan · place Locator")
        self.btn_find_marker.setMinimumHeight(36)
        self.btn_find_marker.setToolTip(
            "Locate the camera inside the scan with no prior, derive the marker's "
            "pose from it, place the Locator there and lock. Needs the marker "
            "visible and enough scanned geometry in view.")
        self.btn_find_marker.clicked.connect(self._find_marker_in_scan)
        layout.addWidget(self.btn_find_marker)
        self.btn_marker_lineup = QCheckBox("Auto lineup")
        self.btn_marker_lineup.setMinimumHeight(42)
        self.btn_marker_lineup.setToolTip(
            "Place the locator at the physical marker centre, then enable. Heading is estimated from the scan. "
            "Green means the live camera and scan fusion are validated. "
            "Uncheck to cancel acquisition; an installed alignment remains. "
            "Uncheck and enable again to repeat lineup.")
        self.btn_marker_lineup.toggled.connect(self._toggle_marker_lineup)
        layout.addWidget(self.btn_marker_lineup)
        self.lbl_marker_lineup = QLabel()
        self.lbl_marker_lineup.setWordWrap(True)
        self.lbl_marker_lineup.setTextFormat(Qt.PlainText)
        self.lbl_marker_lineup.setStyleSheet("color:#aebfd2;font-size:11px;")
        layout.addWidget(self.lbl_marker_lineup)
        self._refresh_marker_indicator()
        return panel

    def _find_marker_in_scan(self):
        """Derive the Locator from the scan, then lock on it.

        Inverts the usual workflow: rather than the operator saying where the
        marker is and the fusion solving for the camera, a hint-free global
        registration says where the camera is and this solves for the marker.
        Height and heading both come out of it, which are exactly the two the
        eye cannot judge against a scan the marker is not part of.
        """
        matcher = self.lidar_matcher
        if not self.aruco_enabled or matcher is None or matcher.map_cloud is None:
            self._marker_status("Import a scan and enable ArUco detection first.")
            return
        if self._marker_busy or self._find_thread is not None:
            return
        if self._lineup is not None:
            self._marker_status("Apply or cancel the manual lineup first.")
            return
        sample = self._marker_detection_sample
        if sample is None:
            self._marker_status("No marker detected yet. Show the marker to the camera.")
            return
        detections, stamp, received, image_size = sample
        # Identity locator: only the depth, corners and paired poses are used
        # here -- the locator is the thing being solved for.
        observation, reason = capture_marker_observation(
            self.engine, detections, stamp, received, image_size,
            self.marker_family.currentText(), self.marker_id.value(),
            self.marker_width.value(), np.eye(4), np.eye(4))
        if observation is None:
            self._marker_status(reason)
            return
        native = getattr(matcher, "_map_native", None)
        if native is None or not len(native):
            self._marker_status("The scan has no point data to localise against.")
            return
        try:
            import scan_localizer
            points = scan_localizer.depth_to_cloud(observation.depth,
                                                   observation.depth_K, stride=2)
        except Exception as exc:
            self._marker_status(f"Scan search unavailable: {exc}")
            return
        self._find_observation = observation
        self._find_rigid = self._align_rigid().copy()
        self._find_generation = matcher._generation
        self._set_find_busy(True)
        self._marker_status("Searching the scan for the camera… (a few seconds)")
        self._find_thread = ScanLocalizeThread(points, native * matcher.map_scale)
        self._find_thread.located.connect(self._found_in_scan)
        self._find_thread.start()

    def _set_find_busy(self, busy):
        if hasattr(self, "btn_find_marker"):
            self.btn_find_marker.setEnabled(not busy)
            self.btn_find_marker.setText(
                "Searching the scan…" if busy else "Find marker in scan · place Locator")

    def _found_in_scan(self, result):
        self._find_thread = None
        self._set_find_busy(False)
        observation = self._find_observation
        self._find_observation = None
        matcher = self.lidar_matcher
        if observation is None or matcher is None:
            return
        if matcher._generation != self._find_generation:
            self._marker_status("The scan changed while searching. Try again.")
            return
        if not result.get("ok"):
            self._marker_status("Scan search failed: " + result.get("reason", "unknown"))
            print(f"[FindMarker] ❌ {result.get('reason')}")
            return
        try:
            locator = locator_from_scan(observation, result["T_map_optical"])
        except Exception as exc:
            self._marker_status(f"Scan search failed: {exc}")
            return
        display_locator = self._find_rigid @ locator
        self.map_viewer.set_locator_transform(display_locator)
        self._on_locator_moved(display_locator)
        self.map_viewer.persp_view.set_locator_marker_size(self.marker_width.value())
        rpy = np.degrees(matrix_to_rpy(locator[:3, :3]))
        print(f"[FindMarker] camera found in the scan at "
              f"{result['T_map_optical'][:3, 3].round(3)} "
              f"({result['fitness']:.0%} matched, {result.get('rmse_m', 0)*1000:.0f} mm rmse)")
        print(f"[FindMarker] locator placed at {locator[:3, 3].round(3)} rpy {rpy.round(1)}")
        # Re-solve through the normal path: the global fix seeds the locator,
        # it does not bypass the verification gates.
        observation.T_map_locator = locator
        try:
            measurement = solve_marker_observation(observation, matcher_snapshot(matcher))
        except Exception as exc:
            self._marker_status(f"Solve failed after placing the locator: {exc}")
            return
        if measurement is None or not measurement.ok:
            self._marker_status(
                "Locator placed from the scan, but the check rejected it: "
                + (measurement.reason if measurement else "no result")
                + " — press again, or enable Auto lineup.")
            return
        min_rank = self.cfg.get("fusion", {}).get("min_scan_rank", 3)
        if measurement.observable_rank < min_rank:
            self._marker_status("Locator placed, but the view does not constrain all "
                                "position axes. Show more room geometry.")
            return
        matcher.accept_hint_solution(measurement, source="aruco",
                                     captured_at=observation.captured_at)
        self.engine.confirm_lineup()
        self._lidar_measurement = measurement
        self.engine.observe_scan(measurement)
        self._push_map_alignment()
        self.smoother.reset()
        self._marker_request = None
        self._set_lock_button(True, measurement)
        self._marker_status(
            f"Located from scan · marker {observation.marker_id} · "
            f"{measurement.verify_score:.0%} scan agreement. Fusion continues.",
            armed=False)
        print(f"[FindMarker] ✅ locked: {measurement.verify_score:.0%} agreement, "
              f"rank {measurement.observable_rank}")

    def _marker_status(self, text, armed=None):
        self._marker_detail = text
        if armed is not None and hasattr(self, "btn_marker_lineup"):
            button = self.btn_marker_lineup
            blocked = button.blockSignals(True)
            button.setChecked(armed)
            button.blockSignals(blocked)
            for field in (self.marker_family, self.marker_id, self.marker_width):
                field.setEnabled(not armed)
        self._refresh_marker_indicator()

    def _refresh_marker_indicator(self, status=None):
        """Green requires current scan evidence AND an advancing supervised pose."""
        if not hasattr(self, "btn_marker_lineup"):
            return
        button = self.btn_marker_lineup
        detail = self._marker_detail
        colour = "#8295ad"
        state, title = "off", "Auto lineup"
        if not button.isChecked():
            if self._marker_failed:
                state, title, colour = "error", "Auto lineup · Retry", "#ee9a9a"
        elif self._marker_request is not None:
            state = "searching" if self._marker_busy else "waiting"
            title = "Auto lineup · Matching scan" if self._marker_busy else "Auto lineup · Waiting"
            colour = "#ebc07d"
        else:
            status = self.engine.fusion_status() if status is None else status
            scan = status["sources"].get("scan", {})
            matcher = self.lidar_matcher
            aligned = matcher is not None and matcher.is_aligned and self.engine.map_aligned
            if (self.engine.fusion.enabled and aligned and status["state"] == "VALIDATED"
                    and scan.get("ready", False) and self.engine._output_T is not None
                    and not self.engine._output_held):
                state, title, colour = "locked", "Auto lineup · Locked", "#82d5b3"
                detail = "Fusion active · The 3D camera follows the physical camera. The marker can leave view."
            elif status["state"] in ("HOLD", "RELINEUP"):
                state, title, colour = "held", "Auto lineup · Tracking held", "#ee9a9a"
                detail = status["reason"]
            else:
                state, title, colour = "verifying", "Auto lineup · Validating", "#ebc07d"
                detail = ("Fusion supervision is disabled in tracking.yaml." if not self.engine.fusion.enabled
                          else "Recovering the camera pose." if self.engine._output_held
                          else status["reason"])
        button.setText(title)
        button.setProperty("lineupState", state)
        self.lbl_marker_lineup.setText(detail)
        fill = colour if button.isChecked() else "#151e29"
        self._set_status_style(button,
            f"QCheckBox {{color:{colour};background:#18212d;border:1px solid {colour};"
            "border-radius:6px;padding:10px;spacing:10px;font-weight:600;}"
            f"QCheckBox::indicator {{width:18px;height:18px;border:2px solid {colour};"
            f"border-radius:4px;background:{fill};}}")

    def _cancel_marker_lineup(self, reason, failed=True):
        self._marker_request = None
        self._marker_lock_context = None
        self._marker_failed = failed
        self._marker_status(reason, armed=False)

    def _toggle_marker_lineup(self, enabled):
        if not enabled:
            self._cancel_marker_lineup("Auto lineup off. Any existing alignment is retained; enable again to realign.", failed=False)
            return
        self._marker_failed = False
        self._marker_lock_context = None
        matcher = self.lidar_matcher
        if not self.aruco_enabled or matcher is None or matcher.map_cloud is None:
            self._cancel_marker_lineup("Load the scan and enable ArUco detection first.")
            return
        if self._lineup is not None:
            self._cancel_marker_lineup("Apply or cancel the manual lineup before using a marker.")
            return
        rigid = self._align_rigid()
        locator = self.map_viewer.get_locator_transform().copy()
        self._marker_request = dict(
            matcher=matcher, generation=matcher._generation,
            resyncs=self.engine.gate.resyncs, rigid=rigid.copy(), locator=locator,
            family=self.marker_family.currentText(), marker_id=self.marker_id.value(),
            width=self.marker_width.value(), armed_at=time.monotonic(), last_image=None)
        self.cfg["aruco"]["lineup_marker_id"] = self.marker_id.value()
        self.map_viewer.persp_view.set_locator_marker_size(self.marker_width.value())
        self.map_viewer.persp_view.select_locator()
        self._marker_status("Show the marker and surrounding room geometry. Waiting for a fresh detection.",
                            armed=True)

    def _marker_request_changed(self, request):
        if request["matcher"] is not self.lidar_matcher:
            return "The scan changed. Place the locator and arm again."
        if request["generation"] != self.lidar_matcher._generation:
            return "The scan or alignment changed. Arm marker lineup again."
        if request["resyncs"] != self.engine.gate.resyncs:
            return "Tracking resynchronized. Wait for stable tracking and arm again."
        if not np.allclose(request["locator"], self.map_viewer.get_locator_transform(), atol=1e-9, rtol=0):
            return "The locator moved. Check its marker placement and arm again."
        _, rigid = decompose_similarity(self.map_viewer.T_align)
        if not np.allclose(request["rigid"], rigid, atol=1e-9, rtol=0):
            return "The displayed scan moved. Check the locator and arm again."
        if self._lineup is not None:
            return "Manual lineup is active. Apply or cancel it before using a marker."
        return ""

    def _poll_marker_lineup(self):
        request = self._marker_request
        context = request if request is not None else self._marker_lock_context
        if context is not None:
            reason = self._marker_request_changed(context)
            if reason:
                self._cancel_marker_lineup(reason)
                return
        if request is None or self._marker_busy:
            return
        sample = self._marker_detection_sample
        if sample is None:
            return
        detections, stamp, received, image_size = sample
        if received <= request["armed_at"] or received == request["last_image"]:
            return
        request["last_image"] = received
        from marker_lineup import capture_marker_observation, matcher_snapshot
        observation, reason = capture_marker_observation(
            self.engine, detections, stamp, received, image_size,
            request["family"], request["marker_id"], request["width"],
            request["locator"], request["rigid"])
        if observation is None:
            self._marker_status(reason)
            return
        solver = matcher_snapshot(request["matcher"])
        self._marker_busy = True
        self._marker_status(
            f"Marker {observation.marker_id} detected. Calculating its heading against the scan…")
        # The ROS callbacks and GUI stay available while registration runs.
        def work():
            from marker_lineup import solve_marker_observation
            try:
                measurement = solve_marker_observation(observation, solver)
                error = ""
            except Exception as exc:
                measurement, error = None, f"{type(exc).__name__}: {exc}"
            if not self._marker_shutdown:
                self.marker_lineup_finished.emit((request, observation, measurement, error))
        threading.Thread(target=work, name="marker-lineup", daemon=True).start()

    def _finish_marker_lineup(self, result):
        request, observation, measurement, error = result
        self._marker_busy = False
        if self._marker_shutdown or self._marker_request is not request:
            return
        reason = self._marker_request_changed(request)
        if not reason and time.monotonic() - observation.captured_at > 15.0:
            reason = "Marker solve took too long. Park the rig and try again."
        if not reason and (self.engine._last_pose_mono is None
                or time.monotonic() - self.engine._last_pose_mono > self.stale_timeout
                or self.engine._pose_rejected):
            reason = "Camera tracking is stale. Restore tracking and arm again."
        if not reason and error:
            reason = "Marker lineup failed: " + error
        if not reason and (measurement is None or not measurement.ok):
            reason = "Scan match rejected: " + (measurement.reason if measurement else "no result")
        min_rank = self.cfg.get("fusion", {}).get("min_scan_rank", 3)
        if not reason and measurement.observable_rank < min_rank:
            reason = "Show more room geometry; the scan match does not constrain all position axes."
        if reason:
            self._cancel_marker_lineup(reason)
            return
        matcher = request["matcher"]
        matcher.accept_hint_solution(measurement, source="aruco",
                                     captured_at=observation.captured_at)
        self.engine.confirm_lineup()
        self._lidar_measurement = measurement
        self.engine.observe_scan(measurement)
        self._push_map_alignment()
        self.smoother.reset()
        solved_locator = getattr(measurement, "T_map_locator", None)
        if solved_locator is not None:
            display_locator = request["rigid"] @ solved_locator
            # Preserve exactly the position the operator placed; update rotation only.
            display_locator[:3, 3] = request["locator"][:3, 3]
            self.map_viewer.set_locator_transform(display_locator)
            self._on_locator_moved(display_locator)
            request = dict(request, locator=display_locator.copy())
        self._marker_request = None
        # Monitor this lock without solving every subsequent marker sighting again.
        self._marker_lock_context = dict(request, generation=matcher._generation)
        if hasattr(self, "btn_ar"):
            self.btn_ar.setChecked(True)
        self._marker_status(
            f"Marker {observation.marker_id} matched. Waiting for fresh scan checks.", armed=True)

    def _solve_from_hint(self, T_hint, T_slam_cam, depth, source, sample_time=None):
        """Register the live depth against the scan, starting from `T_hint`.

        Shared by both entry points so the ArUco marker and the locator gizmo
        mean exactly the same thing: a rough indication of where the camera is,
        which the fusion then improves on using what the camera can actually
        see.
        """
        # The vertical, in the map's frame. The locator's heading ring turns
        # about the displayed vertical, so that is the axis the yaw search must
        # swing about -- whatever the mesh alignment has made it.
        _, rigid = decompose_similarity(
            self.map_viewer.T_align if getattr(self, "map_viewer", None)
            else np.eye(4))
        up_map = rigid[:3, :3].T @ np.array([0.0, 0.0, 1.0])

        measurement = None
        try:
            measurement = self.lidar_matcher.solve_from_hint(
                depth, self._depth_intrinsics(), T_slam_cam, T_hint,
                source=source, up_map=up_map)
        except Exception as exc:
            print(f"[LiDAR] lock failed: {exc}")

        if measurement is not None and sample_time is not None:
            measurement.monotonic_timestamp = sample_time
        self._lidar_measurement = measurement
        ok = measurement is not None and measurement.ok
        if ok:
            self.engine.confirm_lineup()
            self.engine.observe_scan(measurement)
            self._push_map_alignment()
            self.smoother.reset()
        elif source == "aruco":
            # Fall back to the marker's own answer, but say so: the pose is
            # then only as good as the marker, and unverified against the scan.
            print("[LiDAR] ⚠️  could not place the camera on the scan — "
                  "holding the marker anchor alone (pose NOT scan-verified)")
        self._set_lock_button(ok, measurement)
        return ok

    def _set_lock_button(self, ok, measurement):
        if not hasattr(self, "btn_lidar_lock"):
            return
        if ok:
            self.btn_lidar_lock.setText(
                f"Scan matched · {measurement.translation_m * 1000:.0f}mm "
                f"off hint")
            self.btn_lidar_lock.setStyleSheet(self._LOCK_BTN_OK_QSS)
        else:
            self.btn_lidar_lock.setText("Scan match rejected · Retry")
            self.btn_lidar_lock.setStyleSheet(self._LOCK_BTN_BAD_QSS)

    def _lock_to_map_from_locator(self):
        """Solve the camera's pose using the locator as a hint.

        The no-marker workflow: park the camera, drag the locator roughly to
        where it physically is, press this. The locator only has to be close --
        the answer comes from matching what the camera sees against the scan,
        and the button reports how far the solution ended up from the hint.
        """
        if self.lidar_matcher is None or self.lidar_matcher.map_cloud is None:
            print("[LiDAR] no map loaded — import a scan first")
            return
        depth = self.engine._depth_mm
        pose = self.engine.depth_pose() if self.engine.fusion.enabled else self.engine.latest_pose
        if depth is None or pose is None:
            print("[LiDAR] need both a depth frame and a SLAM pose first")
            return

        viewer = getattr(self, "map_viewer", None)
        T_align = self._align_rigid()
        T_locator = (viewer.get_locator_transform() if viewer is not None
                     else np.eye(4))
        T_slam_cam = make_transform(pose[0], pose[1])

        # Hint: the camera is somewhere around the locator. The locator is in
        # the displayed frame and the matcher works in the map's own frame, so
        # the alignment is undone here.
        T_hint = (invert_transform(T_align) @ T_locator
                  @ invert_transform(T_slam_cam))

        self.btn_lidar_lock.setText("Matching view to scan…")
        QApplication.processEvents()
        self._solve_from_hint(T_hint, T_slam_cam, depth, source="locator")

    def _open_guided_lineup(self):
        from lineup_dialog import LineupDialog
        matcher = self.lidar_matcher
        if matcher is None or matcher._mesh_vertices is None:
            self.lbl_fusion.setText("Load the room mesh before opening Camera Lineup Lite.")
            return
        image_sample = self._image_sample
        image_pose = None if image_sample is None else self.engine.pose_at(image_sample[1])
        depth_sample = self.engine.capture_depth()
        if (image_sample is None or time.monotonic() - image_sample[2] > 0.5
                or image_pose is None or depth_sample is None
                or not self.engine.intrinsics_from_camera_info):
            self.lbl_fusion.setText("Lineup needs fresh IR and depth images, real camera intrinsics, and timestamp-paired camera poses. Park the rig and wait for input readiness.")
            return
        gray = image_sample[0].copy()
        depth, depth_K, depth_pose, captured = depth_sample
        depth = depth.copy()
        T_image = make_transform(*image_pose)
        T_depth = make_transform(*depth_pose)
        rigid = self._align_rigid()
        generation = matcher._generation
        resyncs = self.engine.gate.resyncs

        def apply_lineup(T_map_camera):
            if matcher is not self.lidar_matcher or matcher._generation != generation:
                return False, "The scan changed during lineup. Cancel and capture a new frame."
            if self.engine.gate.resyncs != resyncs:
                return False, "Tracking resynchronized during lineup. Cancel and capture a new frame."
            if self.engine._last_pose_mono is None or time.monotonic() - self.engine._last_pose_mono > self.stale_timeout:
                return False, "Camera tracking is stale. Restore tracking before applying."
            T_hint = T_map_camera @ invert_transform(T_image)
            measurement = matcher.verify_pose(depth, depth_K,
                                               T_depth, T_hint)
            if not measurement.ok:
                return False, "Depth verification rejected the lineup: " + measurement.reason
            self.engine.confirm_lineup()
            matcher.set_initial_alignment(T_hint, source="survey points")
            measurement.monotonic_timestamp = captured
            matcher.last_measurement = measurement
            matcher._update_diagnostics(measurement)
            matcher.diagnostics.update(lock_ok=True, lock_verify_score=measurement.verify_score,
                                       lock_verify_median_mm=measurement.verify_median_m * 1000)
            self._lidar_measurement = measurement
            self.engine.observe_scan(measurement)
            self._push_map_alignment()
            self.smoother.reset()
            self.btn_ar.setChecked(True)
            return True, "Lineup applied; waiting for live scan checks"

        dialog = LineupDialog(gray, self.engine.aruco_K,
                              matcher._mesh_vertices * matcher.map_scale,
                              matcher._mesh_triangles, rigid, apply_lineup,
                              aligner=matcher.edge_aligner, parent=self)
        view = self.map_viewer.persp_view
        dialog.scene._orb_center = view._orb_center.copy()
        dialog.scene._orb_dist = view._orb_dist
        dialog.scene._orb_yaw = view._orb_yaw
        dialog.scene._orb_pitch = view._orb_pitch
        dialog.exec_()

    # ── manual lineup (3DE-style, on a frozen frame) ─────────────────────────
    def _build_lineup_panel(self):
        """Nudge controls. Hidden until a lineup is open."""
        from PyQt5.QtWidgets import QGridLayout

        self.lineup_panel = QWidget()
        grid = QGridLayout(self.lineup_panel)
        grid.setContentsMargins(4, 2, 4, 4)
        grid.setSpacing(3)

        self.lbl_lineup = QLabel("—")
        self.lbl_lineup.setFont(QFont("Monospace", 8))
        self.lbl_lineup.setStyleSheet("color:#ffb74d;")
        self.lbl_lineup.setWordWrap(True)
        grid.addWidget(self.lbl_lineup, 0, 0, 1, 6)

        self.btn_lineup_step = QPushButton("STEP: COARSE")
        self.btn_lineup_step.setToolTip("Coarse 20 mm / 1.0°, fine 2 mm / 0.1°")
        self.btn_lineup_step.clicked.connect(self._toggle_lineup_step)
        grid.addWidget(self.btn_lineup_step, 1, 0, 1, 6)

        # Camera-relative moves: how the motion reads on screen.
        moves = [("fwd +", dict(forward=1)),   ("left +", dict(left=1)),
                 ("up +", dict(up=1)),          ("pan ←", dict(pan=1)),
                 ("tilt ↑", dict(tilt=1)),      ("roll ↺", dict(roll=1)),
                 ("fwd −", dict(forward=-1)),  ("right +", dict(left=-1)),
                 ("down +", dict(up=-1)),       ("pan →", dict(pan=-1)),
                 ("tilt ↓", dict(tilt=-1)),     ("roll ↻", dict(roll=-1))]
        for i, (label, kw) in enumerate(moves):
            b = QPushButton(label)
            b.setFixedHeight(24)
            b.setStyleSheet("QPushButton{background:#263238;color:#eee;"
                            "border:1px solid #455a64;border-radius:3px;}"
                            "QPushButton:hover{background:#37474f;}")
            b.clicked.connect(lambda _=False, k=kw: self._lineup_nudge(**k))
            grid.addWidget(b, 2 + i // 6, i % 6)

        actions = [("UNDO", self._lineup_undo, "#455a64"),
                   ("RESET", self._lineup_reset, "#455a64"),
                   ("AUTO", self._lineup_auto, "#4a148c"),
                   ("✔ COMMIT", self._lineup_commit, "#00695c"),
                   ("✖ CANCEL", self._lineup_cancel, "#b71c1c")]
        for i, (label, slot, colour) in enumerate(actions):
            b = QPushButton(label)
            b.setFixedHeight(26)
            b.setStyleSheet(f"QPushButton{{background:{colour};color:#fff;"
                            f"border-radius:3px;font-weight:bold;}}")
            b.clicked.connect(slot)
            grid.addWidget(b, 4, i, 1, 2 if label == "✖ CANCEL" else 1)

        self.lineup_panel.hide()
        return self.lineup_panel

    def _toggle_ar_overlay(self, on):
        self._ar_overlay_on = bool(on)
        self._ar_overlay = None
        self.btn_ar.setText("Show scan overlay")
        if on and (self.lidar_matcher is None
                   or not self.lidar_matcher.is_aligned):
            print("[Overlay] complete Auto lineup before viewing the scan overlay.")

    def _refresh_ar_overlay(self):
        """Redraw the scan over the live frame, a few times a second."""
        matcher = self.lidar_matcher
        if (not self._ar_overlay_on or self._lineup is not None
                or matcher is None or not matcher.is_aligned):
            return
        self._ar_frame_n += 1
        if self._ar_frame_n % 10:            # ~6 Hz at a 60 Hz refresh
            return
        pose = self.engine.latest_pose
        if pose is None:
            return
        try:
            self._ar_overlay = matcher.image_overlay(pose)
        except Exception as exc:
            print(f"[Overlay] error: {exc}")
            self._ar_overlay = None

    def _on_camera_gizmo_moved(self, T_camera_display):
        """The operator dragged the camera dot in the 3D viewport.

        Moving the camera only means anything if it is going to be committed,
        so a drag opens a manual lineup if one is not already running -- the
        gizmo and the nudge buttons then drive the same session, and the
        overlay updates as it moves.
        """
        if self._automatic_lineup_only:
            return
        if self.lidar_matcher is None or self.lidar_matcher.map_cloud is None:
            return
        if self._lineup is None:
            self._start_lineup()
            if self._lineup is None:
                return
        # The gizmo works in the displayed frame; the lineup in the map's.
        T_align = self._align_rigid()
        T_map_cam = invert_transform(T_align) @ np.asarray(T_camera_display,
                                                           dtype=float)
        # The gizmo and channel box both supply the complete camera pose.
        self._lineup.history.append(self._lineup.T_map_cam.copy())
        self._lineup.T_map_cam = T_map_cam.copy()
        self._refresh_lineup()

    def _toggle_lineup(self):
        if self._lineup is not None:
            self._lineup_cancel()
        else:
            self._start_lineup()

    def _start_lineup(self):
        """Freeze the current frame and SLAM pose, and open a lineup on them."""
        from image_alignment import LineupSession

        matcher = self.lidar_matcher
        if matcher is None or matcher.map_cloud is None:
            print("[Lineup] no scan loaded — import one first")
            return
        aligner = getattr(matcher, "edge_aligner", None)
        if aligner is None or not aligner.available:
            print("[Lineup] the scan's surface is not loaded, so its edges "
                  "cannot be drawn — re-import the map as a mesh")
            return
        gray = self._latest_gray
        pose = self.engine.latest_pose
        if self.engine.fusion.enabled:
            pose = self.engine.pose_at(self._image_stamp)
            if self._image_mono is None or time.monotonic() - self._image_mono > 0.5:
                pose = None
        if gray is None or pose is None:
            print("[Lineup] need a camera frame and a SLAM pose first")
            return

        T_slam_cam = make_transform(pose[0], pose[1])
        if matcher.is_aligned:
            # Start from wherever the fusion currently thinks the camera is.
            T_map_cam = matcher.T_slam_to_map @ T_slam_cam
        else:
            # Nothing set yet: start from the locator, same as the hint path.
            viewer = getattr(self, "map_viewer", None)
            T_align = self._align_rigid()
            T_locator = (viewer.get_locator_transform() if viewer is not None
                         else np.eye(4))
            T_map_cam = invert_transform(T_align) @ T_locator

        self._lineup = LineupSession(gray, self.engine.aruco_K,
                                     T_slam_cam, T_map_cam, aligner=aligner)
        self.lineup_panel.show()
        self.btn_lineup.setText("🎯  LINEUP OPEN — FRAME FROZEN")
        self.btn_lineup.setStyleSheet(self._LINEUP_BTN_LIVE_QSS)
        print("[Lineup] frame frozen — nudge until the scan's edges sit on "
              "the picture, then COMMIT")
        self._refresh_lineup()

    def _toggle_lineup_step(self):
        if self._lineup is None:
            return
        self._lineup.fine = not self._lineup.fine
        metres, degrees = self._lineup.step
        self.btn_lineup_step.setText(
            f"STEP: {'FINE' if self._lineup.fine else 'COARSE'}  "
            f"({metres * 1000:.0f}mm / {degrees:.1f}°)")

    def _lineup_nudge(self, **kw):
        if self._lineup is None:
            return
        self._lineup.nudge(**kw)
        self._refresh_lineup()

    def _lineup_undo(self):
        if self._lineup is not None:
            self._lineup.undo()
            self._refresh_lineup()

    def _lineup_reset(self):
        if self._lineup is not None:
            self._lineup.reset()
            self._refresh_lineup()

    def _lineup_auto(self):
        """Let the edge solver finish from here. Suggestion, not gospel."""
        if self._lineup is None:
            return
        result = self._lineup.auto_refine()
        if result is None or not result.ok:
            reason = "unavailable" if result is None else result.reason
            print(f"[Lineup] auto-refine declined: {reason}")
        else:
            print(f"[Lineup] auto-refine moved {result.moved_m * 1000:.0f}mm / "
                  f"{result.moved_deg:.2f}° — check it by eye, UNDO if worse")
        self._refresh_lineup()

    def _lineup_commit(self):
        """Tell the fusion this is where the camera is, and carry on from it."""
        if self._lineup is None:
            return
        T_slam_to_map = self._lineup.slam_to_map()
        moved_m, moved_deg = self._lineup.moved_from_start
        score = self._lineup.score()

        self.engine.confirm_lineup()
        self.lidar_matcher.set_initial_alignment(T_slam_to_map,
                                                 source="manual lineup")
        self._push_map_alignment()
        self.smoother.reset()
        self._lidar_measurement = None
        detail = ("" if score is None
                  else f", {score['score']:.0%} of the scan's edges on the "
                       f"picture ({score['median_px']:.1f}px)")
        print(f"[Lineup] ✅ committed — dialled {moved_m * 1000:.0f}mm / "
              f"{moved_deg:.2f}° from the starting pose{detail}. "
              f"Continuous fusion is running from here.")
        self._end_lineup()

    def _lineup_cancel(self):
        if self._lineup is not None:
            print("[Lineup] cancelled — nothing changed")
        self._end_lineup()

    def _end_lineup(self):
        self._lineup = None
        self.lineup_panel.hide()
        self.btn_lineup.setText("Manual lineup")
        self.btn_lineup.setStyleSheet(self._LINEUP_BTN_IDLE_QSS)

    def _refresh_lineup(self):
        """Redraw the frozen frame with the scan over it, and rescore."""
        if self._lineup is None:
            return
        try:
            score = self._lineup.score()
            moved_m, moved_deg = self._lineup.moved_from_start
            if score is None:
                self.lbl_lineup.setText("scan not visible from this pose")
            else:
                self.lbl_lineup.setText(
                    f"{score['score']:.0%} of the scan's edges on the picture  "
                    f"({score['median_px']:.1f}px median)\n"
                    f"dialled {moved_m * 1000:+.0f}mm / {moved_deg:+.2f}° "
                    f"from the start")
            overlay = self._lineup.overlay()
            if overlay is not None:
                self._lineup_overlay = overlay
        except Exception as exc:
            print(f"[Lineup] redraw error: {exc}")

    def _map_fusion_text(self):
        """Operator readout for the absolute localisation.

        `On scan` is the number that answers "does the camera view match the
        LiDAR": the fraction of the depth image landing within
        `verify_threshold_m` of the scanned surface. Deliberately not ICP's own
        fitness, which at a wide search radius reads near 1.0 almost wherever
        the cloud is placed -- measured on a real scan, a solution 530 mm out
        still scored 0.75.

        `Solved off hint` is how far the fusion moved away from where the
        operator put the locator. It exists to make visible that the locator is
        a reference and the pose came from the sensors, not from the gizmo.
        """
        m = self.lidar_matcher
        if m.map_cloud is None:
            return "── Map Fusion ──\nNo scan loaded"
        if not m.is_aligned:
            text = ("── Map Fusion ──\nPlace the locator at the physical marker centre, "
                    "then enable Auto lineup; its heading is calculated automatically.")
            # Show why the last attempt failed; without this the panel gives
            # the same generic line whether nothing has been tried yet or the
            # scan simply does not match the view near the hint.
            meas = self._lidar_measurement
            if meas is not None and not meas.ok:
                text += f"\n\nLast attempt: {meas.reason}"
                if "search region" in meas.reason:
                    d = m.get_diagnostics()
                    text += (f"\nCheck the locator against the physical marker, show "
                             f"more of the room, or widen hint_radius_m "
                             f"(now ±{d.get('search_region_m', 0.0):.2f}m / "
                             f"±{d.get('search_region_deg', 0.0):.0f}°).")
            return text

        d = m.get_diagnostics()
        meas = self._lidar_measurement
        rank = d.get("observable_rank", 0)
        text = (f"── Map Fusion ({d.get('alignment_source', '?')} hint) ──\n"
                f"Lock: {d.get('lock_verify_score', 0.0):.0%} on scan, "
                f"{d.get('lock_verify_median_mm', 0.0):.0f}mm median miss   "
                f"solved {d.get('hint_offset_mm', 0.0):.0f}mm / "
                f"{d.get('hint_offset_deg', 0.1):.1f}° off hint\n"
                f"Search region: ±{d.get('search_region_m', 0.0):.2f}m / "
                f"±{d.get('search_region_deg', 0.0):.0f}°\n"
                f"Live: {d.get('verify_score', 0.0):.0%} on scan   "
                f"drift {d.get('residual_mm', 0.0):.0f}mm / "
                f"{d.get('residual_deg', 0.0):.2f}°   "
                f"applied {d.get('applied_mm', 0.0):.1f}mm\n"
                f"Axes constrained: {rank}/3   "
                f"Inliers: {d.get('inliers', 0)}   "
                f"ICP: {d.get('last_icp_ms', 0.0):.0f}ms")
        lineup = self._image_lineup
        if lineup is not None:
            text += (f"\nImage lineup: {lineup['score']:.0%} of the scan's "
                     f"edges on the picture ({lineup['median_px']:.1f}px)")
        if meas is not None and not meas.ok:
            text += f"\nLast fix rejected: {meas.reason}"
        return text

    def _on_locator_moved(self, T_ros):
        """The dashboard's own 3D-viewport locator gizmo was dragged.

        It is the anchor target now -- every ArUco anchor maps the marker
        onto wherever this sits, replacing the old "anchor to origin"
        default. Starts at identity, so nothing changes until it's moved.
        """
        self.engine.set_anchor_target_ros(T_ros, source="dashboard_locator")

    def camera_info_callback(self, msg):
        """Real intrinsics from the camera; the engine keeps the first valid set."""
        k = list(msg.k) if hasattr(msg, "k") else list(msg.K)
        d = list(msg.d) if hasattr(msg, "d") else list(msg.D)
        if len(k) >= 9:
            self.engine.set_intrinsics(k, d)

    def imu_callback(self, msg):
        """Telemetry, plus the stationary detector that removes creep.

        Isaac ROS consumes this same IMU internally (enable_imu_fusion:=True) and
        publishes an already-fused pose, so this bridge does not re-fuse. The IMU
        earns its keep here by deciding when the rig is genuinely still.
        """
        g = msg.angular_velocity
        a = msg.linear_acceleration
        self.engine.update_imu((g.x, g.y, g.z), (a.x, a.y, a.z))
        self.latest_imu = (
            f"Acc (m/s²):  X {a.x:+7.2f}  Y {a.y:+7.2f}  Z {a.z:+7.2f}\n"
            f"Gyr (rad/s): X {g.x:+7.3f}  Y {g.y:+7.3f}  Z {g.z:+7.3f}\n"
            f"(fused by Isaac ROS — displayed here for monitoring only)"
        )

    def odometry_callback(self, msg):
        """Primary pose input: Isaac ROS fused odometry, with velocity.

        Isaac fills twist with a real cuVSLAM velocity estimate, which is what
        makes constant-velocity prediction between frames trustworthy.
        """
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        lin = msg.twist.twist.linear
        ang = msg.twist.twist.angular
        self._accept_pose(
            (p.x, p.y, p.z), (o.x, o.y, o.z, o.w),
            (lin.x, lin.y, lin.z), (ang.x, ang.y, ang.z),
            stamp=self._stamp_seconds(msg))

    def vo_pose_callback(self, msg):
        """Visual-odometry pose, used only until odometry starts arriving.

        Both topics carry the same tracking result; odometry is preferred because
        it also carries velocity. Letting both drive the output would mean two
        writers fighting over the stream.
        """
        if self.last_vio_time and (time.time() - self.last_vio_time) < 1.0:
            return
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        self._accept_pose((p.x, p.y, p.z), (o.x, o.y, o.z, o.w),
                          (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
                          stamp=self._stamp_seconds(msg))

    @staticmethod
    def _stamp_seconds(msg):
        """Message timestamp in seconds, or None if the header looks unset."""
        try:
            st = msg.header.stamp
            t = st.sec + st.nanosec * 1e-9
            return t if t > 0.0 else None
        except AttributeError:
            return None

    def _accept_pose(self, position, quaternion, linear, angular, stamp=None):
        """Hand a raw SLAM pose to the engine.

        Deliberately raw. The LiDAR map correction used to be blended into
        `position` here, a fraction at a time, which was wrong three ways: it
        could only correct translation, it fed a corrected pose back into the
        estimator that produced the correction, and it made the gate and the
        stationary hold argue with it. The fusion now corrects the *frame*
        instead -- `depth_callback` pushes the fused SLAM->map transform into
        the engine's ground zero -- so this path stays a clean VIO stream and
        the gate keeps seeing only genuine tracker behaviour.
        """
        self.engine.accept_pose(position, quaternion, linear, angular, stamp)
        if self.send_on_pose:
            self._stream_freed()

    def camera_pose_now(self):
        """(UnrealPose, is_valid) — predicted, converted, smoothed."""
        return self.engine.camera_pose_now()

    def _cb_tc(self, tc_str):
        self.latest_timecode = tc_str
        self.lbl_tc.setText(f"{tc_str} · LTC display only")

    def _cb_gnss(self, lat, lon, alt, fix_quality, timestamp):
        """Reference readout only — GNSS never drives the FreeD stream.

        It used to: when VIO went stale the bridge substituted a GNSS-derived
        position, which is a completely different coordinate frame with a
        different origin. Mid-take that teleports the virtual camera. Tracking
        loss now holds the last good SLAM pose instead.
        """
        if self.lbl_gnss is None:
            return
        if fix_quality == 0:
            self.lbl_gnss.setText(
                f"Connected to {self.cfg['gnss']['port']} — NO SATELLITE FIX "
                f"(indoors?)")
            return

        status_text = {1: "SPS FIX", 4: "RTK FIXED", 5: "RTK FLOAT"}.get(
            fix_quality, f"FIX {fix_quality}")

        x, y, z = wgs84_to_ecef(lat, lon, alt)
        if not hasattr(self, 'gnss_origin') or self.gnss_origin is None:
            self.gnss_origin = (lat, lon, x, y, z)
        e, n, u = ecef_to_enu(x, y, z, *self.gnss_origin)
        self.latest_gnss = (e, n, u, fix_quality)

        self.lbl_gnss.setText(
            f"Lat: {lat:.7f}  Lon: {lon:.7f}  Alt: {alt:.2f}m\n"
            f"ENU from first fix: E {e:+.2f}  N {n:+.2f}  U {u:+.2f} m\n"
            f"Quality: {status_text} | reference only, not streamed")

    def _vins_to_ue_pose(self, tx, ty, tz, q):
        """Legacy 6-tuple shim over PoseTransformer.

        Kept so existing call sites and tests keep working. New code should call
        `self.transformer.ros_to_unreal(...)` and use the UnrealPose directly.
        """
        p = self.transformer.ros_to_unreal((tx, ty, tz), q)
        return p.x, p.y, p.z, p.roll, p.tilt, p.pan

    # --- Timers & UI Updates ---
    def _stream_freed(self):
        """Send one FreeD packet to Unreal Engine and trail it on the canvas."""
        if not self.streaming_enabled:
            return
        if self.lidar_matcher is not None:
            self.engine.observe_scan(self.lidar_matcher.last_measurement)
        pkt = self.engine.freed_packet()
        if pkt is None:
            return   # nothing tracked yet: stay silent rather than assert an origin
        try:
            self.sock.sendto(pkt, (UDP_IP, UDP_PORT))
        except Exception as exc:
            if self.engine.packets_sent <= 1:
                print(f"[FreeD] send failed to {UDP_IP}:{UDP_PORT} — {exc}")

        # TrajectoryCanvas was superseded by MapViewerWidget, but the test
        # harness still builds one; feed it when it is there. The live viewport
        # is driven from _refresh() on the GUI thread instead, because touching
        # widgets from the ROS callback thread is not safe.
        canvas = getattr(self, "canvas", None)
        pose = self.engine.latest_unreal
        if canvas is not None and pose is not None:
            canvas.update_trajectory(pose.x, pose.y, pose.z, pose.pan)

    @staticmethod
    def _set_status_style(widget, stylesheet):
        # setStyleSheet repolishes the widget and invalidates its layout,
        # even for identical CSS. Telemetry refreshes must not continually
        # restart layout while the window manager is resizing the surface.
        if widget.styleSheet() != stylesheet:
            widget.setStyleSheet(stylesheet)

    def _refresh_tracked_camera(self):
        """Draw the same supervised world pose sent to FreeD, including full rotation."""
        viewer = getattr(self, "map_viewer", None)
        if viewer is None:
            return
        pose = self.engine.camera_pose_ros_corrected()
        if pose is not None:
            viewer.update_camera_pose(*pose)

    def _refresh(self):
        self._poll_marker_lineup()
        # Drive the 3D viewport from here, on the GUI thread. Ground-zero
        # corrected (still ROS canonical metres, not Unreal-converted) so the
        # camera dot sits in the same frame as the locator gizmo -- unanchored
        # this is identical to the raw VIO pose (see pose_with_zero_ros), so
        # nothing changes here until the operator actually anchors, at which
        # point the dot visibly snaps next to the locator and tracks relative
        # to it from then on.
        # Pick up anything the fusion worker produced since the last tick.
        # Polling a counter is cheaper and safer than being called back onto
        # the GUI thread from the worker.
        matcher = self.lidar_matcher
        if matcher is not None and matcher.is_aligned:
            self.engine.observe_scan(matcher.last_measurement)
            self._lineup_frame_n += 1
            if self._lineup_frame_n % 30 == 0:      # ~2 Hz at 60 Hz refresh
                try:
                    self._image_lineup = matcher.image_lineup(
                        self.engine.latest_pose)
                except Exception as exc:
                    print(f"[LiDAR] lineup error: {exc}")
                    self._image_lineup = None
            if matcher.fix_counter != self._last_fix_counter:
                self._last_fix_counter = matcher.fix_counter
                self._lidar_measurement = matcher.last_measurement
                self._push_map_alignment()

        self._refresh_ar_overlay()

        self._refresh_tracked_camera()

        self.lbl_imu.setText(self.latest_imu)

        # Thread-safe GUI updates for images
        over = None
        if self._lineup is not None:
            over = self._lineup_overlay
        elif self._ar_overlay_on:
            over = self._ar_overlay
        if over is not None:
            over = np.ascontiguousarray(over)
            h, w, ch = over.shape
            q_img = QImage(over.data, w, h, ch * w, QImage.Format_BGR888)
            lbl_size = self.lbl_track.size()
            self.lbl_track.setPixmap(QPixmap.fromImage(q_img).scaled(
                lbl_size.width(), lbl_size.height(), Qt.KeepAspectRatio,
                Qt.FastTransformation))
            self.latest_cv_image = None
        elif hasattr(self, 'latest_cv_image') and self.latest_cv_image is not None:
            h, w, ch = self.latest_cv_image.shape
            bytes_per_line = ch * w
            q_img = QImage(self.latest_cv_image.data, w, h, bytes_per_line, QImage.Format_BGR888)
            lbl_size = self.lbl_track.size()
            pixmap = QPixmap.fromImage(q_img).scaled(lbl_size.width(), lbl_size.height(), Qt.KeepAspectRatio, Qt.FastTransformation)
            self.lbl_track.setPixmap(pixmap)
            self.latest_cv_image = None

        # ── Rates, once a second ─────────────────────────────────────────────
        now = time.time()
        elapsed = now - self._last_rate_check
        if elapsed >= 1.0:
            # Engine counters only climb, so rates come from deltas.
            self._pose_rate = (self.engine.pose_updates
                               - self._last_pose_count) / elapsed
            self._send_rate = (self.engine.packets_sent
                               - self._last_packet_count) / elapsed
            self._last_pose_count = self.engine.pose_updates
            self._last_packet_count = self.engine.packets_sent
            self._last_rate_check = now

        # ── Pose readout, in the same numbers Unreal receives ────────────────
        pose = self.latest_unreal
        if pose is not None:
            self.lbl_pose.setText(
                f"X: {pose.x:+8.3f} m   Y: {pose.y:+8.3f} m   Z: {pose.z:+8.3f} m\n"
                f"Pan: {pose.pan:+7.2f}°  Tilt: {pose.tilt:+7.2f}°  "
                f"Roll: {pose.roll:+7.2f}°"
            )

        # ── Map fusion, in its own panel so nothing is clipped ───────────────
        if self.lidar_matcher is None:
            fusion_text, fusion_colour = "Disabled in tracking.yaml", "#666"
        else:
            fusion_text = self._map_fusion_text()
            if not self.lidar_matcher.is_aligned:
                fusion_colour = "#ffaa00"
            else:
                # Colour on "does the view match the scan", which is what the
                # operator needs to trust the take.
                score = self.lidar_matcher.get_diagnostics().get(
                    "verify_score", 0.0)
                fusion_colour = ("#00ff88" if score > 0.7 else
                                 "#ff9800" if score > 0.5 else "#ff3355")
            # Read-only depth-vs-scan agreement. It used to sit under the
            # marker panel; with that gone it belongs with the fusion it
            # is checking. Worst of the two colours wins, as before.
            lv = self._lidar_validation
            if lv is not None and lv.get("valid"):
                fit = lv["fitness"]
                fusion_text += (
                    f"\n── Depth vs Scan ({lv['elapsed_ms']:.0f}ms) ──\n"
                    f"Agreement: {fit:.1%}  Mean: {lv['mean_error_mm']:.0f}mm  "
                    f"({lv['n_rays']} rays)")
                if fit <= 0.6:
                    fusion_colour = "#ff9800" if fit > 0.3 else "#ff3355"
        self.lbl_fusion.setText(fusion_text)
        fusion_style = (
            f"background:#151e29; color:{fusion_colour}; padding:8px; "
            f"border:1px solid #2b394d; border-radius:4px;")

        status = self.engine.fusion_status()
        self._refresh_marker_indicator(status)
        state = status["state"]
        if not self.engine.fusion.enabled:
            state = "POSE ONLY"
            status["reason"] = "Fusion supervision disabled in configuration"
        colour = ("#82d5b3" if state == "VALIDATED" else
                  "#9cc9ed" if state == "RTK_ASSISTED" else
                  "#ebc07d" if state in ("DEGRADED", "ALIGNING", "VERIFYING", "POSE ONLY") else "#ee9a9a")
        self.lbl_session.setText(state.replace("_", " "))
        self._set_status_style(self.lbl_session, f"background:#202d3c; color:{colour}; border-radius:6px; padding:10px 16px;")
        mode = "Enabled" if self.streaming_enabled else "Paused"
        self.lbl_status.setText(
            f"{status['reason']}\n"
            f"Motion {self._pose_rate:.1f} Hz  /  FreeD {self._send_rate:.1f} Hz\n"
            f"Output {mode}  ·  {'Aligned' if self.engine.is_anchored else 'Lineup required'}")
        self._set_status_style(self.lbl_status,
            f"background:#151e29;color:{colour};padding:10px;border:1px solid #2b394d;border-radius:6px;")
        mono = time.monotonic()
        motion_age = None if self.engine._last_pose_mono is None else mono - self.engine._last_pose_mono
        self.health_cards["motion"].setText(
            "Waiting for D455" if motion_age is None else
            "Pose stale / rejected" if motion_age > self.stale_timeout or self.engine._pose_rejected else
            f"{self._pose_rate:.0f} Hz  ·  {motion_age * 1000:.0f} ms old")
        scan = status["sources"].get("scan")
        self.health_cards["scan"].setText(
            "Awaiting depth match" if scan is None else
            "Check stale" if not scan["fresh"] else
            f"{scan.get('score', 0):.0%} agreement · {scan.get('rank', 0)}/3 axes" if scan["ok"] else
            "Match rejected")
        if scan is not None and (not scan["fresh"] or not scan["ok"]):
            detail = scan["reason"] if scan["fresh"] else f"Last measurement {scan['age_s']:.1f} s ago"
            self.lbl_fusion.setText(f"CURRENT CHECK UNAVAILABLE\n{detail}\n\nPrevious measurements:\n{fusion_text}")
            fusion_style = "background:#151e29;color:#ebc07d;padding:10px;border:1px solid #2b394d;border-radius:6px;"
        self._set_status_style(self.lbl_fusion, fusion_style)

        fix = self.engine.gnss_fix
        receiver = self.gnss_thread
        if fix is None:
            self.health_cards["gnss"].setText("Reference mode · Setup needed")
            self.lbl_gnss.setText(receiver.status if receiver else "Disconnected · See RTK_SETUP.md")
        else:
            age = max(0.0, mono - fix.received)
            quality = {0: "No fix", 1: "Standalone GPS", 2: "Differential GPS",
                       4: "RTK fixed", 5: "RTK float"}.get(fix.quality, f"Fix {fix.quality}")
            self.health_cards["gnss"].setText("Signal stale" if age > 0.8 else quality)
            accuracy = "Unavailable (enable GST)" if fix.hacc_m is None else f"H {fix.hacc_m*1000:.0f} / V {fix.vacc_m*1000:.0f} mm"
            correction = "Unknown" if fix.correction_age_s is None else f"{fix.correction_age_s:.1f} s"
            self.lbl_gnss.setText(
                f"{quality} · {fix.satellites} satellites · {age:.1f} s old\n"
                f"Correction age: {correction}\nAccuracy estimate: {accuracy}\n"
                f"{self.engine.gnss_reason}")
        depth_ready = self.engine.depth_pose() is not None
        self.lbl_readiness.setText(
            f"Camera intrinsics: {'Received' if self.engine.intrinsics_from_camera_info else 'Waiting'}\n"
            f"Depth / pose pairing: {'Ready' if depth_ready else 'Waiting for synchronized data'}\n"
            f"Room scan: {'Loaded' if matcher is not None and matcher.map_cloud is not None else 'Unavailable'}\n"
            f"World lineup: {'Set' if self.engine.is_anchored else 'Required'}\n"
            f"GNSS alignment: {'Configured' if self.cfg['gnss'].get('calibrated') else 'Not calibrated'}\n"
            f"Unreal: {UDP_IP}:{UDP_PORT} · Camera {CAMERA_ID}")
        image_age = None if self._image_mono is None else mono - self._image_mono
        self.lbl_preview_title.setText("LIVE CAMERA  /  SCAN OVERLAY" if image_age is not None and image_age < 0.5
                                     else "CAMERA IMAGE UNAVAILABLE / STALE")
        self.lbl_route.setText(f"FreeD UDP → {UDP_IP}:{UDP_PORT} / Camera {CAMERA_ID}   ·   {mode}   ·   Metres / degrees   ·   LTC display is not genlock")

    def closeEvent(self, event):
        self._marker_shutdown = True
        self._marker_request = None
        # A scan search runs for seconds; closing mid-search would destroy a
        # live QThread ("Destroyed while thread is still running") and can take
        # the process down with it.
        if self._find_thread is not None:
            self._find_thread.wait(5000)
            self._find_thread = None
        if hasattr(self, 'gnss_thread') and self.gnss_thread:
            self.gnss_thread.stop()
        if hasattr(self, 'ltc_thread') and self.ltc_thread:
            self.ltc_thread.requestInterruption()
            self.ltc_thread.wait(2000)
        if self.lidar_matcher is not None:
            self.lidar_matcher.stop_worker()
        if hasattr(self, 'ros_node'):
            self.ros_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        event.accept()

def main():
    app = QApplication(sys.argv)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    win = HyperFusionApp()
    win.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
