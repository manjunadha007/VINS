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
    QComboBox, QSpinBox, QDoubleSpinBox, QFormLayout, QCheckBox, QSlider
)
from PyQt5.QtGui import (QImage, QPixmap, QFont, QPainter, QPen, QColor,
                         QBrush, QKeySequence)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QRectF, QPointF, QRect, QEvent

# ──────────────────────────────────────────────────────
# Shared modules. These must be deployed alongside this file;
# launch_isaac.sh copies them into the container workspace.
# ──────────────────────────────────────────────────────
from freed import get_freed_packet
from tracking_engine import TrackingEngine
from gnss_receiver import GNSSReceiver
from pose_transform import (PoseGate, PosePredictor, PoseSmoother, PoseTransformer,
                            StationaryHold, T_BODY_OPTICAL,
                            decompose_similarity, invert_transform,
                            make_transform, matrix_to_rpy,
                            matrix_to_quaternion, rpy_to_matrix)
import tracking_config
from map_viewer import MapViewerWidget
from depth_view import depth_coverage, draw_depth_points, verdict
from realsense_streams import FrameJoin, stamp_ns
from projector_control import ProjectorControl, RosProjectorParameters
from gopro_stream import GoProStream
from marker_lineup import (capture_marker_observation, locator_from_scan,
                           matcher_snapshot, solve_marker_observation)
from scan_scale import DepthSweep, measure_scale, fit_locator_scale, verify_locator_scale
from marker_map import (MarkerSurvey, capture_mapped_observation, level_locators_from_view,
                        solve_mapped_observation, validate_layout, save_layout, load_layout, locator_transform,
                        layout_entry, layout_signature, MAX_MARKERS,
                        MarkerSightings)

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


class MaximizeAsNormalWindow:
    """Hold a maximize as a normal window covering the same area.

    On this rig (Ubuntu 24.04, GNOME 46 on Wayland, Xwayland 23.2, NVIDIA 595)
    an X11 window in the *maximized* state stops reaching the screen: the app
    keeps drawing the whole window at 60 Hz, but nothing new is shown and clicks
    look dead until it is restored. A 100-line Qt window, inside the container
    or on the host, freezes the same way; normal and fullscreen windows of the
    same size do not. So a maximize is turned into a normal window at the
    geometry the window manager picked for it, and maximizing again restores
    the previous size so the title-bar button still toggles.
    """
    hold_maximize_as_normal = True
    _restore_geometry = None
    _fill_geometry = None
    _filled = False

    def changeEvent(self, event):
        super().changeEvent(event)
        if (self.hold_maximize_as_normal and event.type() == QEvent.WindowStateChange
                and self.windowState() & Qt.WindowMaximized
                and not self.windowState() & Qt.WindowFullScreen):
            # Give the window manager's maximized geometry time to arrive.
            QTimer.singleShot(150, self._hold_maximize_as_normal)

    def _hold_maximize_as_normal(self):
        if not self.windowState() & Qt.WindowMaximized:
            return
        if self._filled and self._restore_geometry is not None:
            target, self._filled = self._restore_geometry, False
        else:
            self._fill_geometry = target = QRect(self.geometry())
            self._filled = True
        self.setWindowState(self.windowState() & ~Qt.WindowMaximized)
        # The window manager first puts back the pre-maximize size; land after it.
        QTimer.singleShot(150, lambda: self.setGeometry(target))

    def moveEvent(self, event):
        super().moveEvent(event)
        self._track_normal_geometry()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._track_normal_geometry()

    def _track_normal_geometry(self):
        if self.windowState() != Qt.WindowNoState:
            return
        if self._fill_geometry is not None and self.geometry() == self._fill_geometry:
            self._filled = True
        else:
            self._filled = False
            self._restore_geometry = QRect(self.geometry())


# ──────────────────────────────────────────────────────
# Main Dashboard
# ──────────────────────────────────────────────────────
class HyperFusionApp(MaximizeAsNormalWindow, QMainWindow):
    marker_lineup_finished = pyqtSignal(object)
    scan_scale_finished = pyqtSignal(object)
    # One definition so the live state updates can't re-inflate the compact header.
    _SESSION_BADGE_CSS = (
        "background:{bg}; color:{fg}; border-radius:6px; "
        "padding:5px 14px; font-size:11px;"
    )
    _marker_request = None
    _marker_detection_sample = None
    _find_thread = None
    _find_observation = None
    _find_generation = None
    _find_rigid = None
    _marker_busy = False
    _marker_tracking_job = None
    _marker_support_detail = ""
    _marker_shutdown = False
    _marker_lock_context = None
    _marker_survey = None
    _survey_last_image = None
    _marker_detail = "Place the locator at the marker centre, then enable Auto lineup."
    _marker_failed = False
    _automatic_lineup_only = False
    _scale_sweep = None
    _scale_result = None
    _pending_scene = None       # File > Open Scene, waiting on its scan's import

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
        self.hold_maximize_as_normal = bool(self.cfg["ui"].get("hold_maximize_as_normal", True))
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
        # After a lineup, every other marker in view is placed in the scan and
        # joins the joint solve (see MarkerSurvey).
        self._marker_survey = (MarkerSurvey(
            min_sightings=aru.get("auto_place_min_sightings", 6),
            tolerance_m=aru.get("auto_place_tolerance_m", .03),
            tolerance_deg=aru.get("auto_place_tolerance_deg", 6.),
            min_marker_px=aru.get("auto_place_min_marker_px", 24.))
            if aru.get("auto_place_markers", True) else None)

        # Intrinsics start as a config fallback and are replaced by the real
        # values from camera_info. PnP with wrong intrinsics puts ground zero in
        # the wrong place, so this matters.

        self._build_ui()
        self.marker_lineup_finished.connect(self._finish_marker_lineup, Qt.QueuedConnection)
        self.scan_scale_finished.connect(self._finish_scale_measurement, Qt.QueuedConnection)

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

        # Started here, not in _build_ui, so building the UI alone (as the
        # offline tests do) never reaches for USB.
        self._gopro = GoProStream(self.cfg["gopro"])
        self.btn_gopro.setChecked(bool(self.cfg["gopro"]["auto_start"]))

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
                    Image, topics["depth_preview"],
                    self.gui.depth_preview_callback, 5)
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
        self._projector = ProjectorControl(RosProjectorParameters(self.ros_node))
        self._projector_timer = QTimer(self)
        self._projector_timer.timeout.connect(self._refresh_projector)
        self._projector_timer.start(100)
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

        act_open = QAction("&Open Scene…", self)
        act_open.setShortcut(QKeySequence.Open)
        act_open.setStatusTip("Open a saved scene: its scan, alignment, tracking origin and Locators")
        act_open.triggered.connect(self._menu_open_scene)
        file_menu.addAction(act_open)

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
                "mapped_markers": self._mapped_marker_records(),
            },
            "locators": [dict(key=key, name=entry["name"],
                              transform=entry["transform"].tolist(), marker=entry["marker"])
                         for key, entry in viewer.persp_view.locator_entries().items()],
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

    def _menu_open_scene(self):
        """File > Open — restore a scene written by Save Scene.

        The scan goes in first, on File > Import's own background path; the
        alignment and Locators are statements about that scan, so they are
        applied once it has loaded (`_on_import_finished`). The scan is
        imported again even when it is already loaded: the import is what
        drops a lineup solved against the old placement.
        """
        viewer = getattr(self, "map_viewer", None)
        if viewer is None:
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Scene", "", "HyperFusion scene (*.json);;All (*)")
        if not path:
            return
        try:
            scene = self._read_scene(path)
        except (OSError, ValueError, TypeError, KeyError) as e:
            QMessageBox.warning(self, "Open Failed", f"Could not open:\n{path}\n\n{e}")
            print(f"[Dashboard] ❌ Scene open failed: {e}")
            return
        self._cancel_marker_lineup("Opening scene…", failed=False)
        if scene["map_file"]:
            self._pending_scene = scene
            self._on_map_loaded(scene["map_file"])
            return
        if viewer.has_map():
            self._delete_scan()
        self._restore_scene(scene)

    def _read_scene(self, path):
        """Parse and check a scene file before anything in the session changes.

        Open changes nothing on a bad file, so every field is checked here.
        Scenes saved before Locators were (v1.6 and earlier) carry only the
        tracking origin, which becomes the one Locator.
        """
        with open(path) as fh:
            scene = json.load(fh)
        if not isinstance(scene, dict) or scene.get("format") != "hyperfusion.scene":
            raise ValueError("This is not a HyperFusion scene file.")
        if scene.get("version", 1) != 1:
            raise ValueError(f"Scene version {scene.get('version')} is newer than this dashboard.")

        def matrix(value, what):
            T = np.asarray(value, dtype=float)
            if T.shape != (4, 4) or not np.isfinite(T).all():
                raise ValueError(f"{what} is not a 4x4 transform.")
            return T

        map_file = scene.get("map_file") or ""
        if map_file and not os.path.isfile(map_file):
            # A scan moved along with its scene still opens.
            beside = os.path.join(os.path.dirname(os.path.abspath(path)), os.path.basename(map_file))
            if not os.path.isfile(beside):
                raise ValueError(f"The scene's scan is missing:\n{map_file}")
            map_file = beside
        saved = scene.get("alignment") or {}
        alignment = {key: float(saved.get(key, 0.)) for key in ("tx", "ty", "tz", "roll", "pitch", "yaw")}
        alignment["scale"] = float(saved.get("scale", 1.))
        if not np.isfinite(list(alignment.values())).all() or alignment["scale"] <= 0:
            raise ValueError("The scan alignment is not a valid transform.")
        alignment["stored"] = bool(saved.get("stored", False))

        origin = matrix(scene.get("locator", np.eye(4).tolist()), "The tracking origin")
        saved = scene.get("locators") or [dict(key="locator", name="Locator", transform=origin.tolist())]
        if not 1 <= len(saved) <= MAX_MARKERS:
            raise ValueError(f"A scene holds 1 to {MAX_MARKERS} Locators.")
        locators, assigned = {}, []
        for entry in saved:
            key = str(entry["key"])
            if key in locators:
                raise ValueError(f"Locator {key} appears twice.")
            locators[key] = dict(name=str(entry.get("name") or "Locator"),
                                 transform=matrix(entry["transform"], f"Locator {key}"), marker=None)
            if entry.get("marker") is not None:
                assigned.append((key, dict(entry["marker"], position_m=[0, 0, 0], rotation_deg=[0, 0, 0])))
        # IDs, widths and duplicates, as a marker layout is checked.
        for (key, _), record in zip(assigned, validate_layout([record for _, record in assigned])):
            locators[key]["marker"] = dict(dictionary=record["dictionary"], id=record["id"],
                                           size_m=record["size_m"])
        families = {name for name, *_ in self.aruco_detectors}
        if any(locators[key]["marker"]["dictionary"] not in families for key, _ in assigned):
            raise ValueError("Enable the scene's marker dictionaries in tracking.yaml before opening it.")
        # The selected Locator is the one saved as the tracking origin.
        active = next((key for key, entry in locators.items()
                       if np.allclose(entry["transform"], origin, atol=1e-9)), None)

        saved = scene.get("marker_lineup") or {}
        lineup = {}
        if saved.get("dictionary") in families:
            lineup["dictionary"] = saved["dictionary"]
        if "marker_id" in saved:
            lineup["marker_id"] = int(saved["marker_id"])
        if "marker_size_m" in saved:
            lineup["marker_size_m"] = float(saved["marker_size_m"])
            if not np.isfinite(lineup["marker_size_m"]):
                raise ValueError("The lineup marker width is not a number.")
        display = {name: matrix(T, f"The {name} display transform")
                   for name, T in (scene.get("display_transforms") or {}).items()
                   if name in ("trail", "world")}
        return dict(path=path, map_file=map_file, alignment=alignment, locators=locators,
                    active=active, lineup=lineup, display=display)

    def _take_pending_scene(self, path):
        """The scene waiting on this import, if any; it is handed out once."""
        scene, self._pending_scene = self._pending_scene, None
        return scene if scene is not None and scene["map_file"] == path else None

    def _restore_scene(self, scene):
        """Apply a scene `_read_scene` checked, once its scan (if any) is loaded."""
        viewer = self.map_viewer
        view = viewer.persp_view
        # Emits alignment_changed, which hands the scale to the matcher.
        viewer.set_alignment(**scene["alignment"])
        view.set_locator_entries(scene["locators"], scene["active"])
        self._set_locator_display(view._active_locator_id, view.get_locator_transform())
        for name, T in scene["display"].items():
            view.set_display_transform(name, T)
        lineup = scene["lineup"]
        if "dictionary" in lineup:
            self.marker_family.setCurrentText(lineup["dictionary"])
        if "marker_id" in lineup:
            self.marker_id.setValue(lineup["marker_id"])
        if "marker_size_m" in lineup:
            self.marker_width.setValue(lineup["marker_size_m"])
        self._marker_status("Scene opened. Check the Locators against the scan, then enable Auto lineup.",
                            armed=False)
        print(f"[Dashboard] ✅ Scene opened: {scene['path']}")

    def _build_ui(self):
        self._automatic_lineup_only = True
        self._build_menu()
        shell = QWidget()
        layout = QVBoxLayout(shell)
        layout.setContentsMargins(20, 8, 20, 12)
        layout.setSpacing(9)
        header = QHBoxLayout()
        header.setSpacing(12)
        brand = QVBoxLayout()
        # Header is pure chrome — keep it thin so the workspace below gets the pixels.
        brand.setContentsMargins(0, 0, 0, 0)
        brand.setSpacing(0)
        title = QLabel("HyperFusion")
        title.setFont(QFont("DejaVu Sans", 15, QFont.Bold))
        title.setStyleSheet("font-size:18px;font-weight:600;color:#e4edf7;")
        brand.addWidget(title)
        subtitle = QLabel("VIRTUAL PRODUCTION  /  CAMERA TRACKING")
        subtitle.setStyleSheet("color:#8295ad; font-size:9px; letter-spacing:2px;")
        brand.addWidget(subtitle)
        header.addLayout(brand)
        header.addStretch()
        self.btn_projector = QPushButton("IR PROJECTOR: CHECKING…")
        self.btn_projector.setCheckable(True)
        self.btn_projector.setEnabled(False)
        self.btn_projector.setMinimumHeight(40)
        self.btn_projector.setMinimumWidth(240)
        self.btn_projector.setCursor(Qt.PointingHandCursor)
        self.btn_projector.setAccessibleName("IR projector")
        self.btn_projector.setStyleSheet(
            "QPushButton{background:#263447;color:#dce4ef;border:2px solid #63758b;"
            "border-radius:6px;padding:8px 16px;font-size:13px;font-weight:bold;}"
            "QPushButton:hover{border-color:#8dccf1;}"
            "QPushButton:checked{background:#165744;color:#e1fff2;border-color:#61d6a3;}"
            "QPushButton:disabled{background:#18212d;color:#91a4bb;border-color:#344356;}")
        self.btn_projector.setToolTip(
            "Click to turn the D455's infrared projector on or off. "
            "Turn it off if the image is washed out; turn it on when extra "
            "surface texture is needed for depth. In alternating mode, "
            "tracking still uses projector-off frames.")
        self.btn_projector.toggled.connect(self._toggle_projector)
        header.addWidget(self.btn_projector, 0, Qt.AlignVCenter)
        self.lbl_session = QLabel("AWAITING LINEUP")
        self.lbl_session.setStyleSheet(self._SESSION_BADGE_CSS.format(bg="#253448", fg="#a6cae8"))
        header.addWidget(self.lbl_session, 0, Qt.AlignVCenter)
        layout.addLayout(header)
        self.lbl_projector_status = QLabel()
        self.lbl_projector_status.setWordWrap(True)
        self.lbl_projector_status.setStyleSheet("color:#ebc07d;font-size:11px;")
        self.lbl_projector_status.hide()
        layout.addWidget(self.lbl_projector_status)

        telemetry = QHBoxLayout()
        telemetry.setSpacing(10)
        self.health_cards = {}
        for key, title in (("motion", "01  MOTION"), ("scan", "02  SCAN VALIDATION"), ("gnss", "03  RTK POSITION")):
            card = QFrame()
            card.setStyleSheet("QFrame{background:#18212d;border:1px solid #2b394d;border-radius:8px;}")
            column = QVBoxLayout(card)
            column.setContentsMargins(11, 5, 11, 6)
            column.setSpacing(1)
            label = QLabel(title)
            label.setStyleSheet("border:none;color:#91a4bb;font-size:9px;")
            value = QLabel("Waiting for input")
            value.setStyleSheet("border:none;color:#dce4ef;font-size:12px;font-weight:600;")
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
        settings.addWidget(panel("Single unassigned Locator: heading is estimated from the scan. "
                                 "Assigned Locators: set each marker's centre and orientation; +X points to its printed top and +Z out of its face. "
                                 "Auto lineup turns a Locator more than 10° off what the camera sees; positions are never moved. "
                                 "Measure the black square, excluding the white paper margin."))
        self.btn_depth_view = QCheckBox("Show depth preview beside camera")
        self.btn_depth_view.setToolTip(
            "Dots mark camera pixels with usable depth (red near, blue far); no dots, no depth. "
            "The bar shows how much of the view has depth and how many position axes its "
            "surfaces pin down. Auto lineup needs 3/3: aim at a corner or furniture, not just floor and one wall.")
        self.btn_depth_view.toggled.connect(self._toggle_depth_view)
        settings.addWidget(self.btn_depth_view)
        self.btn_depth_view.setChecked(True)
        settings.addWidget(self._hdr("SCAN DETAILS", size=10))
        self.lbl_fusion = panel("Load the room scan to begin", 100)
        settings.addWidget(self.lbl_fusion)
        settings.addWidget(self._hdr("SCAN SCALE", size=10))
        settings.addWidget(self._build_scan_scale_panel())
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
        self.camera_views = QSplitter(Qt.Horizontal)
        preview = QWidget()
        preview_layout = QVBoxLayout(preview)
        preview_layout.setContentsMargins(10, 4, 4, 4)
        self.lbl_preview_title = self._hdr("LIVE CAMERA  /  SCAN OVERLAY", size=10)
        preview_layout.addWidget(self.lbl_preview_title)
        self.lbl_track = self._cambox(240, 135, "Waiting for D455 infrared image")
        preview_layout.addWidget(self.lbl_track, 1)
        preview_layout.addWidget(self._build_ar_overlay_panel())
        self.lbl_tracking_hint = QLabel("Keep clear surface detail in view for motion tracking.")
        self.lbl_tracking_hint.setWordWrap(True)
        self.lbl_tracking_hint.setMinimumHeight(48)
        self.lbl_tracking_hint.setStyleSheet("color:#aebfd2;padding:4px;")
        preview_layout.addWidget(self.lbl_tracking_hint)
        self.camera_views.addWidget(preview)

        self.depth_preview_panel = QWidget()
        depth_layout = QVBoxLayout(self.depth_preview_panel)
        depth_layout.setContentsMargins(4, 4, 0, 4)
        depth_layout.addWidget(self._hdr("DEPTH POINTS  /  AUTO LINEUP", size=10))
        self.lbl_depth_preview = self._cambox(240, 135, "Waiting for synchronized image and depth")
        depth_layout.addWidget(self.lbl_depth_preview, 1)
        self.lbl_depth_readiness = QLabel("Waiting for depth coverage")
        self.lbl_depth_readiness.setWordWrap(True)
        self.lbl_depth_readiness.setMinimumHeight(48)
        self.lbl_depth_readiness.setStyleSheet("color:#ebc07d;padding:4px;")
        depth_layout.addWidget(self.lbl_depth_readiness)
        self.camera_views.addWidget(self.depth_preview_panel)
        self.camera_views.setSizes([500, 500])
        workspace.addWidget(self.camera_views)
        self.scene_views = QSplitter(Qt.Horizontal)
        gopro = QWidget()
        gopro_layout = QVBoxLayout(gopro)
        gopro_layout.setContentsMargins(10, 4, 4, 0)
        gopro_header = QHBoxLayout()
        gopro_header.addWidget(self._hdr("GOPRO  /  CAMERA VIEW", size=10))
        gopro_header.addStretch()
        self.btn_gopro = QPushButton("Start GoPro")
        self.btn_gopro.setCheckable(True)
        self.btn_gopro.setStyleSheet(self._AR_BTN_QSS + "QPushButton{padding:3px 10px;}")
        self.btn_gopro.setMinimumWidth(110)
        self.btn_gopro.toggled.connect(self._toggle_gopro)
        gopro_header.addWidget(self.btn_gopro)
        gopro_layout.addLayout(gopro_header)
        self.lbl_gopro = self._cambox(240, 135, "No GoPro video")
        gopro_layout.addWidget(self.lbl_gopro, 1)
        self.lbl_gopro_status = QLabel("GoPro preview off")
        self.lbl_gopro_status.setWordWrap(True)
        self.lbl_gopro_status.setMinimumHeight(28)
        self.lbl_gopro_status.setStyleSheet("color:#aebfd2;padding:4px;")
        gopro_layout.addWidget(self.lbl_gopro_status)
        self.scene_views.addWidget(gopro)
        scene = QWidget()
        scene_layout = QVBoxLayout(scene)
        scene_layout.setContentsMargins(4, 4, 0, 0)
        scene_layout.addWidget(self._hdr("SCENE ALIGNMENT", size=10))
        self.map_viewer = MapViewerWidget()
        self.map_viewer.persp_view.set_camera_edit_locked(True)
        self.map_viewer.file_loaded.connect(self._on_map_loaded)
        self.map_viewer.locator_changed.connect(self._on_locator_moved)
        self.map_viewer.alignment_changed.connect(self._on_map_alignment_changed)
        self.map_viewer.camera_moved.connect(self._on_camera_gizmo_moved)
        self.map_viewer.delete_requested.connect(self._menu_delete_scan)
        scene_layout.addWidget(self.map_viewer, 1)
        self.scene_views.addWidget(scene)
        self.scene_views.setSizes([420, 620])
        workspace.addWidget(self.scene_views)
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
    _ar_opacity = 0.8
    _depth_view_on = True
    depth_preview_panel = None
    _depth_preview_join = None
    _depth_preview_sample = None
    _depth_preview_frame = None
    _depth_preview_analyzed = None
    _depth_view_next = 0.0
    _depth_cov = None
    _depth_drawn = None
    _gopro = None
    _gopro_seq = None
    _gopro_status = None

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
            if self._take_pending_scene(path) is not None:
                self._marker_status("The scene was not opened: its scan could not be imported.", armed=False)
                QMessageBox.warning(self, "Open Failed",
                                    f"The scene's scan could not be imported:\n{path}")
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

        scene = self._take_pending_scene(path)
        if scene is not None:
            self._restore_scene(scene)

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
        """Viewfinder and marker detection, decimated from the tracking stream."""
        try:
            # Passive mode uses this same exposure for depth. In alternating
            # mode it cannot pair with emitter-on depth and is simply expired.
            self._queue_depth_preview("image", msg)
            self._cam_frame_n += 1
            # Skip two of every three tracking images for the viewfinder.
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
                self.lidar_matcher.set_image(gray, self.engine.aruco_K,
                                             self._image_stamp)

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
                ids = sorted(marker_id for _, marker_id, _ in markers)
                seen = ", ".join(map(str, ids[:8])) + (f" +{n_ids - 8}" if n_ids > 8 else "")
                st = f"{n_ids} MARKER{'S' if n_ids > 1 else ''} · ID {seen}"
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
        self._queue_depth_preview("depth", msg)
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

    def _set_locator_display(self, locator_id, display):
        """Move a Locator in code; the selected one is also the anchor target."""
        view = self.map_viewer.persp_view
        if locator_id == view._active_locator_id:
            self.map_viewer.set_locator_transform(display, locator_id)
            self._on_locator_moved(display)
        else:
            view.set_locator_transform(display, locator_id)

    def _build_scan_scale_panel(self):
        """Setup → Scan scale: measure the scan's size against the camera's depth."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.btn_marker_scale = QPushButton("Scale and line up from markers")
        self.btn_marker_scale.setToolTip(
            "Assign IDs to three Locators placed at their markers on the scan. "
            "RealSense depth measures the marker triangle to scale the scan. "
            "Every Locator stays on its original scan position, then Auto lineup runs.")
        self.btn_marker_scale.clicked.connect(lambda: self._toggle_scale_sweep(markers=True))
        self.btn_measure_scale = QPushButton("Measure scan scale")
        self.btn_measure_scale.setToolTip(
            "Gathers depth while you pan, placed by the tracking, and fits it to the scan "
            "with scale free. Needs a lineup first; a rough one is enough.")
        self.btn_measure_scale.clicked.connect(lambda: self._toggle_scale_sweep())
        self.lbl_scan_scale = QLabel(
            "Place three Locators at their physical markers and assign their IDs. Auto lineup "
            "or Scale and line up from markers measures their triangle using RealSense depth, "
            "scales the scan and keeps the Locators attached. Keep the markers and scanned "
            "surfaces visible. Without three markers, line up first and use Measure scan scale "
            "while panning across opposite walls.")
        self.lbl_scan_scale.setWordWrap(True)
        self.lbl_scan_scale.setTextFormat(Qt.PlainText)
        self.lbl_scan_scale.setStyleSheet("color:#aebfd2;font-size:11px;")
        self.btn_apply_scale = QPushButton("Apply scale")
        self.btn_apply_scale.setEnabled(False)
        self.btn_apply_scale.clicked.connect(self._apply_scan_scale)
        for widget in (self.btn_marker_scale, self.btn_measure_scale, self.lbl_scan_scale,
                       self.btn_apply_scale):
            layout.addWidget(widget)
        self._scale_timer = QTimer(panel)
        self._scale_timer.setInterval(250)
        self._scale_timer.timeout.connect(self._collect_scale_sweep)
        return panel

    def _scale_status(self, text):
        self.lbl_scan_scale.setText(text)
        if self._scale_sweep is not None and self._scale_sweep["records"]:
            self._marker_status(text, armed=True)
        print(f"[Scale] {text}")

    def _scale_request_changed(self, request):
        matcher = self.lidar_matcher
        if matcher is not request["matcher"] or matcher._map_native is not request["native"]:
            return "The scan changed. Measure again."
        if abs(matcher.map_scale - request["old_scale"]) > 1e-12:
            return "The scan scale changed. Measure again."
        _, rigid = decompose_similarity(self.map_viewer.T_align)
        if not np.allclose(rigid, request["rigid"], atol=1e-9, rtol=0):
            return "The displayed scan moved. Measure again."
        if not request["records"] and not matcher.is_aligned:
            return "The lineup was lost. Line up and measure again."
        if request["records"] and layout_signature(self._mapped_marker_records()) != request["signature"]:
            return "The Locators changed during the sweep. Measure again."
        if self.engine.gate.resyncs != request["resyncs"]:
            return "Tracking jumped during the sweep. Measure again."
        return ""

    def _scale_button(self, request):
        return self.btn_marker_scale if request["records"] else self.btn_measure_scale

    def _toggle_scale_sweep(self, markers=False):
        """Start a sweep, or stop the running one and measure.

        With `markers` no lineup is needed: three trusted scan positions and
        depth-backed marker centres determine scale, checked by room depth.
        """
        request = self._scale_sweep
        if request is not None:
            if not request["measuring"]:
                self._finish_scale_sweep()
            return
        matcher = self.lidar_matcher
        if matcher is None or getattr(matcher, "_map_native", None) is None:
            self._scale_status("Load the scan first.")
            return
        records = validate_layout(self._mapped_marker_records()) if markers else []
        if markers and len(records) < 3:
            self._scale_status("Assign marker IDs to at least three Locators, each placed "
                               "at its physical marker on the scan, then press again.")
            return
        if not markers and not matcher.is_aligned:
            self._scale_status("Line up first (Auto lineup; a rough lineup is enough), then measure the scale.")
            return
        if markers:
            # Run the rigid marker solve only after the scan is metric.
            self._cancel_marker_lineup("Measuring the scan's scale from the markers…", failed=False)
        self._scale_result = None
        self.btn_apply_scale.setEnabled(False)
        self._scale_sweep = dict(
            sweep=DepthSweep(), started=time.monotonic(), last=None, measuring=False,
            duration=float(self.lidar_cfg.get("scale_sweep_s", 12.)), matcher=matcher,
            native=matcher._map_native, old_scale=matcher.map_scale, resyncs=self.engine.gate.resyncs,
            records=records, signature=layout_signature(records), rigid=self._align_rigid(),
            sightings=MarkerSightings(), last_image=None, marker_note="")
        self._scale_button(self._scale_sweep).setText("Stop and measure")
        (self.btn_measure_scale if markers else self.btn_marker_scale).setEnabled(False)
        self._scale_timer.start()
        self._collect_scale_sweep()

    def _collect_scale_sweep(self):
        request = self._scale_sweep
        if request is None or request["measuring"]:
            self._scale_timer.stop()
            return
        reason = self._scale_request_changed(request)
        if reason:
            self._end_scale_sweep(reason)
            return
        sample = self.engine.capture_depth()
        if sample is not None and sample[3] != request["last"] and not self.engine._pose_rejected:
            frame, K, pose, received = sample
            request["last"] = received
            from scan_localizer import depth_to_cloud
            request["sweep"].add(depth_to_cloud(frame, K, stride=4), make_transform(*pose) @ T_BODY_OPTICAL)
        if request["records"]:
            self._collect_marker_sightings(request)
        left = request["duration"] - (time.monotonic() - request["started"])
        if left <= 0:
            self._finish_scale_sweep()
            return
        text = ("Keep the three markers and surrounding scanned surfaces visible… " if request["records"]
                else "Pan slowly across the room so the camera sees walls on opposite sides… ")
        text += f"{left:.0f} s left · {request['sweep'].frames} frames."
        if request["records"]:
            seen = request["sightings"].counts()
            text += (" Seen: "
                     + (", ".join(f"ID {key[1]} ×{n}" for key, n in sorted(seen.items())) or "none yet") + ".")
        text += " Press Scale and line up from markers again to stop early." if request["records"] else " Press again to stop early."
        self.lbl_scan_scale.setText(text)
        if request["records"]:
            self._marker_status(text, armed=True)

    def _collect_marker_sightings(self, request):
        """Add where the camera sees each assigned marker in the newest detection."""
        sample = self._marker_detection_sample
        if sample is None or sample[2] == request["last_image"]:
            return
        detections, stamp, received, image_size = sample
        request["last_image"] = received
        observation, reason = capture_mapped_observation(
            self.engine, detections, stamp, received, image_size, request["records"], request["rigid"],
            check_marker_width=False)
        # This path uses only centres intersected with measured depth surfaces;
        # PnP range and the entered print width must not set or block scan scale.
        if observation is None:
            # Keep the reason when an assigned marker was visible but unusable.
            known = {(r["dictionary"], r["id"]) for r in request["records"]}
            if any((family, fid) in known for family, fid, _ in detections):
                request["marker_note"] = reason
            return
        for marker in observation.markers:
            request["sightings"].add(marker)

    def _end_scale_sweep(self, text, success=False):
        request = self._scale_sweep
        self._scale_timer.stop()
        self._scale_sweep = None
        self.btn_marker_scale.setText("Scale and line up from markers")
        self.btn_measure_scale.setText("Measure scan scale")
        for button in (self.btn_marker_scale, self.btn_measure_scale):
            button.setEnabled(True)
        self._scale_status(text)
        if request is not None and request["records"]:
            self._marker_failed = not success
            self._marker_status(text, armed=False)

    def _finish_scale_sweep(self):
        """Fit the gathered depth to the scan off the GUI thread; it takes a few seconds."""
        request = self._scale_sweep
        self._scale_timer.stop()
        reason = self._scale_request_changed(request)
        if reason:
            self._end_scale_sweep(reason)
            return
        if request["records"]:
            T_slam_to_map, reason = self._marker_scale_seed(request)
            if T_slam_to_map is None:
                self._end_scale_sweep(reason)
                return
        else:
            T_slam_to_map = request["matcher"].T_slam_to_map.copy()
        request["measuring"] = True
        self._scale_button(request).setText("Measuring…")
        self._scale_button(request).setEnabled(False)
        self._scale_status(f"Fitting {request['sweep'].frames} frames of depth to the scan…")
        points = request["sweep"].points()
        scan = request["native"] * request["old_scale"]

        def work():
            try:
                result = (verify_locator_scale(points, scan, request["locator_fit"])
                          if request["records"] else measure_scale(points, scan, T_slam_to_map))
                error = ""
            except Exception as exc:
                result, error = None, f"{type(exc).__name__}: {exc}"
            if not self._marker_shutdown:
                self.scan_scale_finished.emit((request, result, error))
        threading.Thread(target=work, name="scan-scale", daemon=True).start()

    def _marker_scale_seed(self, request):
        """Similarity from three trusted Locators and stable metric depth centres."""
        centres = request["sightings"].centres(require_depth=True)
        positions = {(r["dictionary"], r["id"]): np.asarray(r["position_m"], dtype=float)
                     for r in request["records"]}
        keys = [key for key in positions if key in centres]
        request["centres"] = {key: centres[key] for key in keys}
        if len(keys) < 3:
            seen = ", ".join(f"ID {key[1]}" for key in keys) or "none"
            return None, (f"Too few assigned markers were seen steadily (seen: {seen}); three or more with "
                          "RealSense depth are needed. Hold each in view with its surrounding surface. "
                          + request["marker_note"]).strip()
        result = fit_locator_scale([centres[k] for k in keys], [positions[k] for k in keys])
        if not result["ok"]:
            return None, result["reason"]
        request["locator_fit"] = result
        return result["T"], ""

    def _finish_scale_measurement(self, payload):
        request, result, error = payload
        if self._marker_shutdown or self._scale_sweep is not request:
            return
        reason = self._scale_request_changed(request)
        if not reason and error:
            reason = "Scale measurement failed: " + error
        if not reason and not result["ok"]:
            reason = result["reason"]
        if reason:
            self._end_scale_sweep(reason)
            return
        s, sigma = result["scale"], result["sigma"]
        summary = (f"{request['sweep'].frames} frames, {result['inliers']:.0%} of the sweep on the scan, "
                   f"{result['rmse_m'] * 1000:.0f} mm RMS")
        if request["records"]:
            self._apply_marker_scale(request, result, summary)
            return
        if abs(s - 1) <= 2 * sigma:
            self._end_scale_sweep(
                f"The scan's scale matches the room within the camera's accuracy: it measured "
                f"{s:.3f}x, inside ±{2 * sigma:.1%}. No change needed. ({summary})")
            return
        new = request["old_scale"] / s
        self._scale_result = dict(request, new_scale=new)
        self._end_scale_sweep(
            f"The scan is {abs(1 - s):.1%} {'smaller' if s < 1 else 'larger'} than the room the camera "
            f"measured (±{sigma:.1%}; {summary}). Apply sets the scan scale from {request['old_scale']:.5g} "
            f"to {new:.5g} m per unit and moves the Locators with the scan.")
        self.btn_apply_scale.setEnabled(True)

    def _apply_marker_scale(self, request, result, summary):
        """Rescale the scan and all attached Locators uniformly, then line up."""
        s, sigma = result["scale"], result["sigma"]
        old = request["old_scale"]
        new = old if abs(s - 1) <= 2 * sigma else old / s
        if not self.map_viewer.SCALE_MIN <= new <= self.map_viewer.SCALE_MAX:
            self._end_scale_sweep("Measured scale is outside the viewer's supported range; check the scan's units.")
            return
        self._rescale_scan(old, new)
        if new == old:
            text = (f"The scan's scale matches the room within the camera's accuracy ({s:.3f}x, inside "
                    f"±{2 * sigma:.1%}); kept at {old:.5g} m per unit.")
        else:
            text = (f"The scan was {abs(1 - s):.1%} {'smaller' if s < 1 else 'larger'} than the room "
                    f"(±{sigma:.1%}); scan scale set from {old:.5g} to {new:.5g} m per unit.")
        text += " All Locators kept their positions on the scan and scaled with it; marker widths stayed in real metres."
        self._end_scale_sweep(text + f" Auto lineup is running. ({summary}) "
                              "File → Save Scene keeps it after a restart.", success=True)
        self._toggle_marker_lineup(True, calibrate_scale=False)

    def _apply_scan_scale(self):
        """Rescale the scan, keeping each Locator on the same spot of it."""
        result, self._scale_result = self._scale_result, None
        self.btn_apply_scale.setEnabled(False)
        matcher = self.lidar_matcher
        if (result is None or matcher is not result["matcher"] or matcher._map_native is not result["native"]
                or abs(matcher.map_scale - result["old_scale"]) > 1e-12):
            self._scale_status("The scan or its scale changed since the measurement. Measure again.")
            return
        old, new = result["old_scale"], result["new_scale"]
        self._rescale_scan(old, new)
        self._scale_status(
            f"Scan scale set to {new:.5g} m per unit (was {old:.5g}); Locators moved with the scan. "
            f"Run Auto lineup again. To keep it after a restart, File → Save, or set "
            f"lidar.map_scale: {new:.5g} in tracking.yaml.")

    def _rescale_scan(self, old, new):
        """Set the scan scale from `old` to `new` m per unit, keeping each Locator on its spot.

        Locators are spots on the scan, in map metres before the display
        transform; rescaling the scan about its origin moves those spots.
        """
        rigid = self._align_rigid()
        for key, entry in self.map_viewer.persp_view.locator_entries().items():
            T = invert_transform(rigid) @ entry["transform"]
            T[:3, 3] *= new / old
            self._set_locator_display(key, rigid @ T)
        if new != old:
            self.map_viewer.set_alignment_scale(new)
            self._align_rigid()
            self.lidar_cfg["map_scale"] = new

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
        self.lbl_marker_map = QLabel(self._marker_map_hint())
        self.lbl_marker_map.setWordWrap(True)
        form.addRow(self.lbl_marker_map)
        marker_files = QHBoxLayout()
        self.btn_save_markers = QPushButton("Save marker layout")
        self.btn_save_markers.clicked.connect(self._save_marker_layout)
        self.btn_load_markers = QPushButton("Load marker layout")
        self.btn_load_markers.clicked.connect(self._load_marker_layout)
        marker_files.addWidget(self.btn_save_markers)
        marker_files.addWidget(self.btn_load_markers)
        form.addRow(marker_files)
        self.lbl_marker_files = QLabel()
        self.lbl_marker_files.setWordWrap(True)
        self.lbl_marker_files.setTextFormat(Qt.PlainText)
        form.addRow(self.lbl_marker_files)
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
            "Place locators at physical markers, then enable. Three assigned IDs first calibrate "
            "the scan's scale using RealSense depth while keeping each Locator attached to the scan. "
            "Assigned positions stay fixed; orientations are measured from marker corners and depth. "
            "With no assigned IDs, the scan is searched around the Locator for the marker and its heading. "
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

    def _marker_map_hint(self):
        if self._marker_survey is not None:
            return ("After Auto lineup locks, every other marker in view is placed in the scan "
                    "and joins joint matching. Assign IDs by hand to survey them yourself.")
        return "Duplicate Locators in the viewport and assign detected marker IDs to use joint matching."

    def _mapped_marker_records(self):
        viewer = getattr(self, "map_viewer", None)
        if viewer is None:
            return []
        _, rigid = decompose_similarity(viewer.T_align)
        records = []
        for entry in viewer.persp_view.locator_entries().values():
            if entry["marker"] is None:
                continue
            T = invert_transform(rigid) @ entry["transform"]
            records.append(dict(entry["marker"], position_m=T[:3,3].tolist(),
                                rotation_deg=np.degrees(matrix_to_rpy(T[:3,:3])).tolist()))
        return records

    def _save_marker_layout(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save marker layout", "markers.json", "Marker layout (*.json)")
        if not path:
            return
        try:
            save_layout(path, self._mapped_marker_records())
            self.lbl_marker_files.setText(f"Saved marker layout: {path}")
        except (OSError, ValueError) as exc:
            self.lbl_marker_files.setText(str(exc))

    def _load_marker_layout(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load marker layout", "", "Marker layout (*.json)")
        if not path:
            return
        try:
            records = load_layout(path)
            families = {name for name, *_ in self.aruco_detectors}
            if any(e["dictionary"] not in families for e in records):
                raise ValueError("Enable the layout's dictionaries in tracking.yaml before loading it.")
            if not records:
                raise ValueError("This layout has no assigned markers.")
        except (OSError, ValueError, TypeError) as exc:
            self.lbl_marker_files.setText(str(exc))
            return
        self._cancel_marker_lineup("Marker layout loaded. Verify locator positions, then enable Auto lineup.", failed=False)
        view = self.map_viewer.persp_view
        view.select_locator()
        while len(view._locators) > 1:
            view.remove_locator()
        view.clear_locator_marker()
        rigid = self._align_rigid()
        for i, entry in enumerate(records):
            if i:
                view.duplicate_locator()
            view.set_locator_transform(rigid @ locator_transform(entry))
            view.assign_locator_marker(entry["dictionary"], entry["id"], entry["size_m"])
        self.lbl_marker_files.setText(f"Loaded {len(records)} assigned Locators. Check their placement against this scan.")

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
        view = self.map_viewer.persp_view
        locator_id = view._active_locator_id
        entry = view.locator_entries()[locator_id]
        marker = entry['marker']
        # Identity locator: only the depth, corners and paired poses are used
        # here -- the locator is the thing being solved for.
        observation, reason = capture_marker_observation(
            self.engine, detections, stamp, received, image_size,
            marker['dictionary'] if marker else self.marker_family.currentText(),
            marker['id'] if marker else self.marker_id.value(),
            marker['size_m'] if marker else self.marker_width.value(), np.eye(4), np.eye(4))
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
        self._cancel_marker_lineup("Searching the scan for the selected Locator.", failed=False)
        self._find_locator_id, self._find_locator_entry = locator_id, entry
        self._find_matcher, self._find_resyncs = matcher, self.engine.gate.resyncs
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
        if matcher is not self._find_matcher or matcher._generation != self._find_generation:
            self._marker_status("The scan changed while searching. Try again.")
            return
        view = self.map_viewer.persp_view
        entry = view.locator_entries().get(self._find_locator_id)
        if (entry is None or entry['marker'] != self._find_locator_entry['marker']
                or not np.allclose(entry['transform'], self._find_locator_entry['transform'], atol=1e-9, rtol=0)
                or not np.allclose(self._align_rigid(), self._find_rigid, atol=1e-9, rtol=0)
                or self.engine.gate.resyncs != self._find_resyncs):
            self._marker_status("The Locator or tracking changed while searching. Try again.")
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
        self.map_viewer.set_locator_transform(display_locator, self._find_locator_id)
        self._on_locator_moved(display_locator)
        if view._active_locator_id == self._find_locator_id:
            view.set_locator_marker_size(observation.marker_size_m)
        rpy = np.degrees(matrix_to_rpy(locator[:3, :3]))
        print(f"[FindMarker] camera found in the scan at "
              f"{result['T_map_optical'][:3, 3].round(3)} "
              f"({result['fitness']:.0%} matched, {result.get('rmse_m', 0)*1000:.0f} mm rmse)")
        print(f"[FindMarker] locator placed at {locator[:3, 3].round(3)} rpy {rpy.round(1)}")
        # Re-solve through the normal path: the global fix seeds the locator,
        # it does not bypass the verification gates.
        observation.T_map_locator = locator
        try:
            measurement = solve_marker_observation(observation, matcher_snapshot(matcher),
                                                   estimate_heading=entry['marker'] is None)
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

    def _fusion_locked(self, status=None):
        """Green: aligned, validated by fresh scan or marker checks, and advancing."""
        status = self.engine.fusion_status() if status is None else status
        matcher = self.lidar_matcher
        sources = status["sources"]
        return bool(self.engine.fusion.enabled and matcher is not None and matcher.is_aligned
                    and self.engine.map_aligned and status["state"] == "VALIDATED"
                    and (sources.get("scan", {}).get("ready", False)
                         or sources.get("markers", {}).get("ready", False))
                    and self.engine._output_T is not None and not self.engine._output_held)

    def _refresh_marker_indicator(self, status=None):
        """Green requires fresh external evidence AND an advancing supervised pose."""
        if not hasattr(self, "btn_marker_lineup"):
            return
        button = self.btn_marker_lineup
        detail = self._marker_detail
        colour = "#8295ad"
        state, title = "off", "Auto lineup"
        if not button.isChecked():
            if self._marker_failed:
                state, title, colour = "error", "Auto lineup · Retry", "#ee9a9a"
        elif self._scale_sweep is not None and self._scale_sweep["records"]:
            state, title, colour = "scaling", "Auto lineup · Measuring scale", "#ebc07d"
        elif self._marker_request is not None:
            state = "searching" if self._marker_busy else "waiting"
            title = "Auto lineup · Matching scan" if self._marker_busy else "Auto lineup · Waiting"
            colour = "#ebc07d"
        else:
            status = self.engine.fusion_status() if status is None else status
            scan = status["sources"].get("scan", {})
            markers = status["sources"].get("markers", {})
            matcher = self.lidar_matcher
            aligned = matcher is not None and matcher.is_aligned and self.engine.map_aligned
            if self._fusion_locked(status):
                state, title, colour = "locked", "Auto lineup · Locked", "#82d5b3"
                detail = "Fusion active · The 3D camera follows the physical camera. The marker can leave view."
                if markers.get("ready"):
                    detail += f" {markers.get('marker_count', 0)} mapped marker(s) support the scan match."
            elif status["state"] in ("HOLD", "RELINEUP"):
                state, title, colour = "held", "Auto lineup · Tracking held", "#ee9a9a"
                detail = status["reason"]
            elif (self.engine.fusion.enabled and aligned
                    and status["state"] == "DEGRADED" and status["output_allowed"]
                    and not self.engine._output_held):
                state, title, colour = "tracking", "Auto lineup · Camera tracking", "#ebc07d"
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
        if self._scale_sweep is not None and self._scale_sweep["records"]:
            self._end_scale_sweep(reason, success=not failed)
        self._marker_request = None
        self._marker_lock_context = None
        self._marker_tracking_job = None
        if self._marker_survey is not None:
            self._marker_survey.clear()
        if self.lidar_matcher is not None:
            self.lidar_matcher.discard_marker_fixes()
        self.engine.fusion.observe("markers", False, time.monotonic(), reason)
        self._marker_failed = failed
        self._marker_status(reason, armed=False)

    def _toggle_marker_lineup(self, enabled, calibrate_scale=True):
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
        if self._find_thread is not None:
            self._cancel_marker_lineup("Wait for the scan search to finish, then enable Auto lineup.", failed=False)
            return
        rigid = self._align_rigid()
        locator = self.map_viewer.get_locator_transform().copy()
        records = validate_layout(self._mapped_marker_records())
        if calibrate_scale and len(records) >= 3:
            self._toggle_scale_sweep(markers=True)
            return
        self._marker_request = dict(
            matcher=matcher, generation=matcher._generation,
            resyncs=self.engine.gate.resyncs, rigid=rigid.copy(), locator=locator,
            family=self.marker_family.currentText(), marker_id=self.marker_id.value(),
            width=self.marker_width.value(), armed_at=time.monotonic(), last_image=None,
            locator_id=self.map_viewer.persp_view._active_locator_id,
            records=records, layout_signature=layout_signature(records), last_support=0.)
        self.cfg["aruco"]["lineup_marker_id"] = self.marker_id.value()
        if not records:
            self.map_viewer.persp_view.set_locator_marker_size(self.marker_width.value())
        self.map_viewer.persp_view.select_locator()
        self._marker_support_detail = ""
        self._marker_status((f"{len(records)} assigned Locators. Show their markers and scanned geometry." if records
                             else "Show the marker and surrounding room geometry. Waiting for a fresh detection."),
                            armed=True)

    def _marker_request_changed(self, request):
        if request["matcher"] is not self.lidar_matcher:
            return "The scan changed. Place the locator and arm again."
        if request["generation"] != self.lidar_matcher._generation:
            return "The scan or alignment changed. Arm marker lineup again."
        if request["resyncs"] != self.engine.gate.resyncs:
            return "Tracking resynchronized. Wait for stable tracking and arm again."
        if request.get("layout_signature", ()) != layout_signature(self._mapped_marker_records()):
            return "Marker assignments or locator poses changed. Check the layout and enable Auto lineup again."
        if (not request.get("records") and
                (request.get("locator_id", self.map_viewer.persp_view._active_locator_id) not in self.map_viewer.persp_view._locators
                 or not np.allclose(request["locator"], self.map_viewer.get_locator_transform(request.get("locator_id")), atol=1e-9, rtol=0))):
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
        if self._marker_busy:
            return
        continuous = request is None
        if continuous:
            if context is None or not context.get("records"):
                return
            rate = max(.1, self.cfg["aruco"].get("marker_support_rate_hz", 2.0))
            if time.monotonic() - context.get("last_support", 0.) < 1.0 / rate:
                return
            request = context
        sample = self._marker_detection_sample
        if sample is None:
            return
        detections, stamp, received, image_size = sample
        if received <= request["armed_at"] or received == request["last_image"]:
            return
        request["last_image"] = received
        from marker_lineup import capture_marker_observation, matcher_snapshot
        if request.get("records"):
            records = request["records"]
            if continuous and "oriented_marker_keys" in request:
                # An assigned marker hidden during acquisition still has a
                # hand-entered rotation. It cannot supply corner constraints.
                known = request["oriented_marker_keys"]
                records = [r for r in records if (r["dictionary"], r["id"]) in known]
            observation, reason = capture_mapped_observation(self.engine, detections, stamp,
                received, image_size, records, request["rigid"])
        else:
            observation, reason = capture_marker_observation(
                self.engine, detections, stamp, received, image_size,
                request["family"], request["marker_id"], request["width"],
                request["locator"], request["rigid"])
        if observation is None:
            if continuous:
                self._marker_support_detail = reason
            else:
                self._marker_status(reason)
            return
        if request.get("records") and not continuous:
            observation = self._level_assigned_locators(request, observation)
            if not request.get("orientation_ready", True):
                self._marker_status(request["level_note"] +
                    " Locator positions are kept; waiting to measure their orientations.")
                return
        request["last_support"] = time.monotonic()
        if continuous:
            request = dict(request, continuous=True)
            self._marker_tracking_job = request
        solver = matcher_snapshot(request["matcher"])
        self._marker_busy = True
        if not continuous:
            turned = request.get("level_turned", "")
            self._marker_status(f"Marker(s) {observation.marker_id} detected. "
                                + (turned + " " if turned else "") + "Matching against the scan…")
        # The ROS callbacks and GUI stay available while registration runs.
        def work():
            from marker_lineup import solve_marker_observation
            try:
                aru = self.cfg["aruco"]
                measurement = (solve_mapped_observation(observation, solver,
                    aru.get("marker_max_reprojection_px", 3.0)) if request.get("records")
                    else solve_marker_observation(observation, solver,
                        search_radius_m=aru.get("lineup_search_radius_m", 1.0),
                        search_step_m=aru.get("lineup_search_step_m", .15)))
                error = ""
            except Exception as exc:
                measurement, error = None, f"{type(exc).__name__}: {exc}"
            if not self._marker_shutdown:
                self.marker_lineup_finished.emit((request, observation, measurement, error))
        threading.Thread(target=work, name="marker-lineup", daemon=True).start()

    def _finish_marker_lineup(self, result):
        request, observation, measurement, error = result
        self._marker_busy = False
        if request.get("continuous"):
            self._finish_marker_support(request, observation, measurement, error)
            return
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
            for note in (request.get("level_turned"), request.get("level_note")):
                if note:
                    reason += " " + note
            if request.get("records"):
                reason += (" Assigned Locator positions are kept fixed on the scan. "
                           "Their orientations are measured from the camera and surrounding surfaces.")
        min_rank = self.cfg.get("fusion", {}).get("min_scan_rank", 3)
        if not reason and measurement.observable_rank < min_rank and len(getattr(measurement, "marker_keys", ())) < 2:
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
        solved_marker = getattr(measurement, "T_map_marker", None)
        moved_note = ""
        if solved_locator is not None:
            if solved_marker is not None:
                # The search covers lineup_search_radius_m around the Locator,
                # so the marker can be well away from it: move the Locator
                # onto the marker the scan found.
                display_locator = request["rigid"] @ solved_marker
                moved = np.linalg.norm(display_locator[:3, 3] - request["locator"][:3, 3])
                if moved >= .01:
                    moved_note = f" Locator moved {moved*1000:.0f} mm onto the marker."
                    print(f"[ArUco] marker {observation.marker_id} found {moved*1000:.0f} mm "
                          "from the Locator; Locator moved onto it")
            else:
                display_locator = request["rigid"] @ solved_locator
                # Preserve exactly the position the operator placed; update rotation only.
                display_locator[:3, 3] = request["locator"][:3, 3]
            self.map_viewer.set_locator_transform(display_locator, request.get("locator_id"))
            self._on_locator_moved(display_locator)
            request = dict(request, locator=display_locator.copy())
            if solved_marker is not None and self._marker_survey is not None:
                request = self._adopt_lineup_marker(request, observation)
        self._marker_request = None
        # Registered layouts supply gradual corrections after the initial lock.
        self._marker_lock_context = dict(request, generation=matcher._generation)
        if hasattr(self, "btn_ar"):
            self.btn_ar.setChecked(True)
        turned = request.get("level_turned", "")
        self._marker_status(
            f"Marker {observation.marker_id} matched.{moved_note}" + (" " + turned if turned else "")
            + " Waiting for fresh scan checks.", armed=True)

    def _level_assigned_locators(self, request, observation):
        """Trust placed centres; measure orientations before using marker corners.

        An unmeasured hand-set rotation must never enter the joint solve.
        Keep the registered single-marker workflow, which needs a known pose;
        multiple Locators derive their rotations from their centre geometry.
        """
        measure_all = len(request["records"]) >= 2
        corrections, note = level_locators_from_view(observation, T_display_map=request["rigid"],
                                                     min_error_deg=0. if measure_all else 10.,
                                                     snap_mounts=not measure_all,
                                                     log=lambda line: print(f"[ArUco] {line}"))
        required = {(m.family, m.marker_id) for m in observation.markers}
        measured = {key for key, *_ in corrections}
        request["orientation_ready"] = not measure_all or required <= measured
        request["oriented_marker_keys"] = tuple(sorted(
            set(request.get("oriented_marker_keys", ())) | (measured if measure_all else required)))
        if not request["orientation_ready"] and not note:
            note = "Show sharper marker corners and depth on their surrounding surfaces."
        view = self.map_viewer.persp_view
        owners = {(e["marker"]["dictionary"], e["marker"]["id"]): key
                  for key, e in view.locator_entries().items() if e["marker"]}
        turned = []
        for key, T_map_locator, degrees, on_wall in corrections:
            display = request["rigid"] @ T_map_locator
            # Preserve the operator's displayed position bit for bit, including
            # under a translated/rotated display frame.
            display[:3, 3] = view.get_locator_transform(owners[key])[:3, 3]
            self._set_locator_display(owners[key], display)
            turned.append(f"ID {key[1]} {'upright on its wall' if on_wall else 'flat'} ({degrees:.0f}°)")
        request["level_note"], request["level_turned"] = note, ""
        if turned:
            records = validate_layout(self._mapped_marker_records())
            request.update(records=records, layout_signature=layout_signature(records))
            observation = observation.with_layout(records)
            request["level_turned"] = "Turned Locator " + ", ".join(turned) + " to match the camera."
        for line in (request["level_turned"], note):
            if line:
                print(f"[ArUco] {line}")
        return observation

    def _adopt_lineup_marker(self, request, observation):
        """Give the lineup marker's ID to its Locator, so later markers join it.

        The Locator now sits on the solved marker, consistent with the alignment
        just installed, so its marker supports tracking like any assigned one.
        """
        try:
            self.map_viewer.persp_view.assign_marker_to_locator(
                request["locator_id"], observation.family, observation.marker_id,
                observation.marker_size_m)
        except (KeyError, ValueError) as exc:
            print(f"[ArUco] lineup marker {observation.marker_id} not assigned: {exc}")
            return request
        records = validate_layout(self._mapped_marker_records())
        self._marker_support_detail = (f"Lineup marker ID {observation.marker_id} assigned to its Locator; "
                                       "other markers in view are placed as the camera moves.")
        measured = set(request.get("oriented_marker_keys", ())) | {(observation.family, observation.marker_id)}
        return dict(request, records=records, layout_signature=layout_signature(records),
                    oriented_marker_keys=tuple(sorted(measured)))

    def _survey_markers(self):
        """Place every detected but unassigned marker while the lineup is locked."""
        context, survey = self._marker_lock_context, self._marker_survey
        sample = self._marker_detection_sample
        if (survey is None or context is None or sample is None
                or self._marker_request is not None or not self._fusion_locked()):
            return
        matcher = context["matcher"]
        if (matcher is not self.lidar_matcher or not matcher.is_aligned
                or context["generation"] != matcher._generation):
            return
        detections, stamp, received, _ = sample
        if received != self._survey_last_image:
            self._survey_last_image = received
            known = {(r["dictionary"], r["id"]) for r in self._mapped_marker_records()}
            unknown = [d for d in detections if (d[0], d[1]) not in known]
            pose = self.engine.pose_at(stamp) if unknown else None
            if pose is not None:
                T_map_optical = matcher.T_slam_to_map @ make_transform(*pose) @ T_BODY_OPTICAL
                for family, fid, quad in unknown:
                    survey.observe((family, fid), quad, self.engine.aruco_K,
                                   self.marker_width.value(), T_map_optical, received)
        # A support solve in flight was captured with the old layout; placing
        # now would make it look like an operator edit and cancel the lock.
        if self._marker_busy:
            return
        placed, measured_keys = [], set(context.get("oriented_marker_keys", ()))
        for (family, fid), size, T_map_locator in survey.ready():
            try:
                entry = layout_entry(family, fid, size, T_map_locator)
                self.map_viewer.persp_view.add_marker_locator(
                    context["rigid"] @ locator_transform(entry), family, fid, size)
            except ValueError as exc:
                print(f"[ArUco] could not place marker {fid}: {exc}")
                continue
            placed.append(str(fid))
            measured_keys.add((family, fid))
        if placed:
            records = validate_layout(self._mapped_marker_records())
            context.update(records=records, layout_signature=layout_signature(records),
                           oriented_marker_keys=tuple(sorted(measured_keys)))
            self._marker_support_detail = f"Placed marker ID {', '.join(placed)} from the locked camera."
            print(f"[ArUco] placed marker(s) {', '.join(placed)} in the scan; {len(records)} now in the joint solve")

    def _finish_marker_support(self, request, observation, measurement, error):
        if (self._marker_shutdown or self._marker_tracking_job is not request
                or self._marker_lock_context is None):
            return
        self._marker_tracking_job = None
        reason = self._marker_request_changed(request)
        if reason:
            self._cancel_marker_lineup(reason)
            return
        if error or measurement is None or not measurement.ok:
            self._marker_support_detail = error or (measurement.reason if measurement else "No marker match")
            self.engine.fusion.observe("markers", False, observation.captured_at, self._marker_support_detail)
            return
        if (self.engine._pose_rejected or self.engine._last_pose_mono is None
                or time.monotonic() - self.engine._last_pose_mono > self.stale_timeout):
            return
        if request["matcher"].submit_marker_fix(measurement, observation.T_slam_depth, request["generation"]):
            ids = ", ".join(str(key[1]) for key in measurement.marker_keys)
            self._marker_support_detail = f"Marker IDs {ids}: {measurement.marker_rms_px:.1f} px error, {measurement.verify_score:.0%} scan agreement."
            if measurement.rejected_marker_keys:
                self._marker_support_detail += " Ignored inconsistent IDs: " + ", ".join(str(k[1]) for k in measurement.rejected_marker_keys)

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

    def _build_ar_overlay_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.btn_ar = QCheckBox("Show scan wireframe overlay")
        self.btn_ar.setToolTip(
            "Draw the scan's visible triangle edges over the camera. Backface culling is on.")
        layout.addWidget(self.btn_ar)
        transparency_row = QHBoxLayout()
        self.lbl_ar_transparency = QLabel()
        transparency_row.addWidget(self.lbl_ar_transparency)
        self.ar_transparency = QSlider(Qt.Horizontal)
        self.ar_transparency.setRange(0, 100)
        self.ar_transparency.setValue(round((1.0 - self._ar_opacity) * 100))
        self.ar_transparency.setAccessibleName("Scan overlay transparency")
        self.ar_transparency.setMinimumWidth(80)
        self.ar_transparency.setFixedHeight(22)
        self.ar_transparency.setStyleSheet(
            "QSlider::groove:horizontal {height:6px;background:#344356;border-radius:3px;}"
            "QSlider::sub-page:horizontal {background:#6ba9d9;border-radius:3px;}"
            "QSlider::handle:horizontal {width:14px;margin:-4px 0;background:#dce4ef;border-radius:7px;}"
            "QSlider::sub-page:horizontal:disabled {background:#52677e;}"
            "QSlider::handle:horizontal:disabled {background:#8295ad;}")
        self.ar_transparency.setToolTip(
            "0%: solid wireframe. 100%: camera only. Backface culling stays on.")
        self.ar_transparency.setEnabled(self._ar_overlay_on)
        self.ar_transparency.valueChanged.connect(self._set_ar_transparency)
        self.lbl_ar_transparency.setBuddy(self.ar_transparency)
        transparency_row.addWidget(self.ar_transparency, 1)
        layout.addLayout(transparency_row)
        self._set_ar_transparency(self.ar_transparency.value())
        self.btn_ar.setChecked(self._ar_overlay_on)
        self.btn_ar.toggled.connect(self._toggle_ar_overlay)
        return panel

    def _set_ar_transparency(self, percent):
        self._ar_opacity = 1.0 - percent / 100.0
        self.lbl_ar_transparency.setText(f"Transparency: {percent}%")
        self._ar_frame_n = 9  # apply on the next display tick

    def _toggle_ar_overlay(self, on):
        self._ar_overlay_on = bool(on)
        self._ar_overlay = None
        self._ar_frame_n = 9
        self.ar_transparency.setEnabled(on)
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
        # `pose_at`, never `latest_pose`. The scan has to be projected with
        # the pose recorded at this frame's own timestamp; pairing the newest
        # odometry sample with a picture that is a frame or two old slides the
        # scan across the image by however far the camera travelled in
        # between. That error is zero while the rig is parked, which is why a
        # single-frame lineup looked perfect and live tracking drifted.
        # A None result (no pose within the pairing tolerance) leaves the last
        # overlay on screen rather than blanking it.
        try:
            self._ar_overlay = matcher.image_overlay(
                self.engine.pose_at, opacity=self._ar_opacity)
        except Exception as exc:
            print(f"[Overlay] error: {exc}")
            self._ar_overlay = None

    def _toggle_gopro(self, on):
        if self._gopro is None:
            return
        if on:
            self._gopro.start()
        else:
            self._gopro.stop(timeout=0)
        self.btn_gopro.setText("Stop GoPro" if on else "Start GoPro")

    def _refresh_gopro(self):
        """Show the newest GoPro frame; it is decoded on the stream's thread."""
        stream = self._gopro
        if stream is None:
            return
        seq, frame = stream.latest()
        if seq != self._gopro_seq:
            self._gopro_seq = seq
            if frame is None:
                self.lbl_gopro.clear()
                self.lbl_gopro.setText("No GoPro video")
            else:
                h, w, ch = frame.shape
                q_img = QImage(frame.data, w, h, ch * w, QImage.Format_BGR888)
                lbl_size = self.lbl_gopro.size()
                self.lbl_gopro.setPixmap(QPixmap.fromImage(q_img).scaled(
                    lbl_size.width(), lbl_size.height(), Qt.KeepAspectRatio,
                    Qt.FastTransformation))
        if stream.status != self._gopro_status:
            self._gopro_status = stream.status
            self.lbl_gopro_status.setText(stream.status)

    def _toggle_projector(self, on):
        # Detect the actual routing, including a dashboard restarted after the
        # operator switched the laser off. Passive/stock launches have no splitter.
        split = any(p.node_name == "vp_realsense_splitter"
                    for p in self.ros_node.get_publishers_info_by_topic(
                        self.cfg["topics"]["image"]))
        self._projector.set_enabled(on, alternating=split)
        self._refresh_projector()

    def _refresh_projector(self):
        self._projector.tick()
        control = self._projector
        state = control.state
        self.btn_projector.blockSignals(True)
        self.btn_projector.setChecked(state.enabled if state is not None else False)
        self.btn_projector.blockSignals(False)
        self.btn_projector.setEnabled(state is not None and not control.busy)
        if control.busy:
            label = "APPLYING…"
        elif state is None:
            label = "UNAVAILABLE"
        elif not state.enabled:
            label = "OFF"
        else:
            label = "ON"
        self.btn_projector.setText("IR PROJECTOR: " + label)
        message = control.error or control.connection_error
        self.lbl_projector_status.setText(message)
        self.lbl_projector_status.setVisible(bool(message))

    def depth_preview_callback(self, msg):
        """IR image paired with active or passive depth from the same exposure."""
        self._queue_depth_preview("image", msg)

    def _queue_depth_preview(self, kind, msg):
        if self._depth_preview_join is None:
            self._depth_preview_join = FrameJoin(("image", "depth"))
        parts = self._depth_preview_join.add(kind, stamp_ns(msg), msg)
        if parts is not None:
            self._depth_preview_sample = (parts["image"], parts["depth"], time.monotonic())

    def _toggle_depth_view(self, on):
        self._depth_view_on = bool(on)
        self._depth_cov = None
        self._depth_drawn = None
        self._depth_preview_analyzed = None
        self._depth_view_next = 0.0
        if self.depth_preview_panel is not None:
            self.depth_preview_panel.setVisible(bool(on))

    def _refresh_depth_view(self):
        """Render an exact image/depth pair independently of scan alignment."""
        if not self._depth_view_on:
            return
        sample = self._depth_preview_sample
        now = time.monotonic()
        K = self.depth_K
        if sample is None or K is None or now - sample[2] > 0.5:
            self._depth_cov = None
            self._depth_preview_frame = None
            self._depth_drawn = None
            self.lbl_depth_preview.clear()
            self.lbl_depth_preview.setText("Waiting for synchronized image and depth")
            self.lbl_depth_readiness.setText("Depth unavailable · Auto lineup cannot assess this view")
            self.lbl_depth_readiness.setStyleSheet("color:#ee9a9a;padding:4px;")
            return
        if now < self._depth_view_next:
            return
        self._depth_view_next = now + 0.1
        lidar = self.cfg["lidar"]
        try:
            if sample is not self._depth_preview_analyzed:
                gray = self.cv_bridge.imgmsg_to_cv2(sample[0], "mono8")
                depth = self.cv_bridge.imgmsg_to_cv2(sample[1], "16UC1")
                if gray.shape != depth.shape:
                    raise ValueError("image and depth must share the same resolution")
                self._depth_cov = depth_coverage(depth, K, lidar["min_range_m"],
                                                lidar["max_range_m"], lidar["observability_eps"])
                self._depth_preview_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                self._depth_preview_analyzed = sample
            need_rank = self.cfg["fusion"].get("min_scan_rank", 3)
            colour, head, tip = verdict(self._depth_cov, need_rank)
            self.lbl_depth_readiness.setText(head + "\n" + tip + " · Near red / far blue")
            b, g, r = colour
            self.lbl_depth_readiness.setStyleSheet(f"color:rgb({r},{g},{b});padding:4px;")
            frame = np.ascontiguousarray(self._with_depth_points(self._depth_preview_frame))
            h, w, ch = frame.shape
            q_img = QImage(frame.data, w, h, frame.strides[0], QImage.Format_BGR888)
            self.lbl_depth_preview.setPixmap(QPixmap.fromImage(q_img).scaled(
                self.lbl_depth_preview.size(), Qt.KeepAspectRatio, Qt.FastTransformation))
        except Exception as exc:
            self._depth_cov = None
            self._depth_preview_frame = None
            self._depth_drawn = None
            self.lbl_depth_preview.clear()
            self.lbl_depth_preview.setText("Depth preview unavailable")
            self.lbl_depth_readiness.setText(f"Depth preview: {exc}")
            self.lbl_depth_readiness.setStyleSheet("color:#ee9a9a;padding:4px;")

    def _with_depth_points(self, image):
        """The camera image with the depth preview drawn on, while it is enabled.

        The overlay branch redisplays the same image every refresh, so the
        drawing is cached until either the image or the analysis changes.
        """
        if not self._depth_view_on or image is None:
            return image
        cached = self._depth_drawn
        if cached is not None and cached[0] is image and cached[1] is self._depth_cov:
            return cached[2]
        lidar = self.cfg["lidar"]
        drawn = draw_depth_points(image, self._depth_cov, lidar["min_range_m"],
                                  lidar["max_range_m"], self.cfg["fusion"].get("min_scan_rank", 3), draw_status=False)
        self._depth_drawn = (image, self._depth_cov, drawn)
        return drawn

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
            stamp=self._stamp_seconds(msg), source="odometry")

    def vo_pose_callback(self, msg):
        """Visual-odometry pose, used when richer odometry is unavailable.

        Both topics carry the same tracking result; odometry is preferred because
        it also carries velocity. Letting both drive the output would mean two
        writers fighting over the stream.
        """
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        self._accept_pose((p.x, p.y, p.z), (o.x, o.y, o.z, o.w),
                          (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
                          stamp=self._stamp_seconds(msg), source="vo")

    @staticmethod
    def _stamp_seconds(msg):
        """Message timestamp in seconds, or None if the header looks unset."""
        try:
            st = msg.header.stamp
            t = st.sec + st.nanosec * 1e-9
            return t if t > 0.0 else None
        except AttributeError:
            return None

    def _accept_pose(self, position, quaternion, linear, angular, stamp=None, *, source=None):
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
        if not self.engine.accept_pose(position, quaternion, linear, angular, stamp, source=source):
            return
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
            self.engine.observe_scan(self.lidar_matcher.last_marker_measurement)
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
        sample = self._marker_detection_sample
        detections = sample[0] if sample and 0 <= time.monotonic() - sample[2] <= .5 else []
        self.map_viewer.persp_view.set_detected_markers(detections)
        records = self._mapped_marker_records()
        if records:
            self.lbl_marker_map.setText(f"{len(records)} assigned Locators · joint matching enabled.\n" + self._marker_support_detail)
        else:
            self.lbl_marker_map.setText(self._marker_map_hint())
        self._poll_marker_lineup()
        self._survey_markers()
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
            self.engine.observe_scan(matcher.last_marker_measurement)
            self._lineup_frame_n += 1
            if self._lineup_frame_n % 30 == 0:      # ~2 Hz at 60 Hz refresh
                try:
                    self._image_lineup = matcher.image_lineup(
                        self.engine.pose_at)
                except Exception as exc:
                    print(f"[LiDAR] lineup error: {exc}")
                    self._image_lineup = None
            if matcher.fix_counter != self._last_fix_counter:
                self._last_fix_counter = matcher.fix_counter
                self._lidar_measurement = matcher.last_measurement
                self._push_map_alignment()

        self._refresh_ar_overlay()
        self._refresh_depth_view()
        self._refresh_gopro()

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
            frame = self.latest_cv_image
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_BGR888)
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
        self._set_status_style(self.lbl_session, self._SESSION_BADGE_CSS.format(bg="#202d3c", fg=colour))
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
        self._scale_sweep = None
        if hasattr(self, '_scale_timer'):
            self._scale_timer.stop()
        if hasattr(self, '_projector_timer'):
            self._projector_timer.stop()
            self._projector.close()
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
        if self._gopro is not None:
            self._gopro.stop()            # takes the camera out of webcam mode
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
