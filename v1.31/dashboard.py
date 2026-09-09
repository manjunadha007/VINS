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
    QApplication, QMainWindow, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QWidget, QFrame, QSizePolicy, QScrollArea, QSplitter
)
from PyQt5.QtGui import QImage, QPixmap, QFont, QPainter, QPen, QColor, QBrush
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QRectF, QPointF

# ──────────────────────────────────────────────────────
# Shared modules. These must be deployed alongside this file;
# launch_isaac.sh copies them into the container workspace.
# ──────────────────────────────────────────────────────
from freed import get_freed_packet
from pose_transform import (PoseGate, PosePredictor, PoseSmoother, PoseTransformer,
                            StationaryHold,
                            invert_transform, make_transform, matrix_to_rpy,
                            matrix_to_quaternion, rpy_to_matrix)
import tracking_config
from map_viewer import MapViewerWidget

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
            painter.setPen(QColor('#00d4ff'))
            painter.setFont(QFont('Arial', 8, QFont.Bold))
            painter.drawText(ox + 4, oy + 14, label)

        # Top-down view: X horizontal, Y vertical
        draw_panel(xs, ys, (x_min, x_max), (y_min, y_max),
                   4, 24, panel_w, panel_h, 'TOP  X→  Y↑', '#00d4ff', '#00d4ff')

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

class GNSSListenerThread(QThread):
    gnss_updated = pyqtSignal(float, float, float, int, float)

    def __init__(self, cfg=None):
        super().__init__()
        cfg = cfg or {}
        self.port = cfg.get("port", "/dev/ttyACM0")
        self.baud = int(cfg.get("baud", 115200))

    def run(self):
        try:
            import serial
        except ImportError:
            print("[GNSS] pyserial not installed — GNSS disabled "
                  "(pip install pyserial, or set gnss.enabled: false)")
            return
        try:
            ser = serial.Serial(self.port, baudrate=self.baud, timeout=1.0)
            print(f"[Dashboard] Connected to RTK GNSS on {self.port}")
            while not self.isInterruptionRequested():
                line = ser.readline().decode('ascii', errors='ignore').strip()
                if line.startswith('$GPGGA') or line.startswith('$GNGGA'):
                    parts = line.split(',')
                    if len(parts) > 9:
                        try:
                            if not parts[2] or not parts[4] or not parts[9]:
                                self.gnss_updated.emit(0.0, 0.0, 0.0, 0, time.time())
                                continue
                                
                            lat_raw = parts[2]
                            lat_deg = float(lat_raw[:2]) + float(lat_raw[2:]) / 60.0
                            if parts[3] == 'S': lat_deg = -lat_deg
                            
                            lon_raw = parts[4]
                            lon_deg = float(lon_raw[:3]) + float(lon_raw[3:]) / 60.0
                            if parts[5] == 'W': lon_deg = -lon_deg
                            
                            fix_quality = int(parts[6]) if parts[6] else 0
                            alt = float(parts[9])
                            
                            self.gnss_updated.emit(lat_deg, lon_deg, alt, fix_quality, time.time())
                        except Exception as e:
                            pass
            ser.close()
        except Exception as e:
            print(f"[Error] GNSS Thread: {e}")

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
# Main Dashboard
# ──────────────────────────────────────────────────────
class HyperFusionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HyperFusion — 6DOF Tracking Dashboard (ROS Edition)")
        self.setGeometry(60, 60, 1280, 720)  # Start smaller but allow resizing
        self.setStyleSheet("background-color: #0d0d1a; color: #e0e0e0;")

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
        self.transformer = PoseTransformer(self.cfg["transform"])
        self.predictor = PosePredictor(
            lead_time_s=trk["lead_time_s"],
            max_extrapolation_s=trk["max_extrapolation_s"])
        self.smoother = PoseSmoother(alpha=trk["smoothing_alpha"])
        self.stale_timeout = float(trk["stale_pose_timeout_s"])
        self.gate = PoseGate(
            enabled=trk.get("gate_enabled", True),
            max_speed_m_s=trk.get("gate_max_speed_m_s", 5.0),
            max_angular_deg_s=trk.get("gate_max_angular_deg_s", 900.0),
            max_jump_m=trk.get("gate_max_jump_m", 0.15),
            max_consecutive_rejects=trk.get("gate_max_consecutive_rejects", 15))
        self.hold = StationaryHold(
            enabled=trk.get("hold_enabled", True),
            gyro_still_rad_s=trk.get("hold_gyro_still_rad_s", 0.05),
            gyro_release_rad_s=trk.get("hold_gyro_release_rad_s", 0.08),
            accel_still_std=trk.get("hold_accel_still_std", 0.15),
            window_samples=trk.get("hold_window_samples", 60))

        self.latest_pose = None          # (position, quaternion) in ROS frame
        self.latest_unreal = None        # last UnrealPose actually sent
        self.depth_K = None
        self.slam_status = "WAITING FOR SLAM"
        self.last_vio_time = 0.0
        self._packets_sent = 0
        self._pose_updates = 0
        self._last_rate_check = time.time()
        self._pose_rate = 0.0
        self._send_rate = 0.0

        # ── ArUco ground zero ────────────────────────────────────────────────
        self.aruco_enabled = bool(aru["enabled"])
        dict_name = aru.get("dictionary", "DICT_4X4_1000")
        dict_id = getattr(aruco, dict_name, aruco.DICT_4X4_1000)

        # OpenCV renamed these between 4.5 (container) and 4.7+ (host).
        if hasattr(aruco, "getPredefinedDictionary"):
            self.aruco_dict = aruco.getPredefinedDictionary(dict_id)
        else:
            self.aruco_dict = aruco.Dictionary_get(dict_id)

        if hasattr(aruco, "DetectorParameters_create"):
            self.aruco_params = aruco.DetectorParameters_create()
        else:
            self.aruco_params = aruco.DetectorParameters()

        if hasattr(aruco, "ArucoDetector"):
            self.aruco_detector = aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        else:
            self.aruco_detector = None

        # Intrinsics start as a config fallback and are replaced by the real
        # values from camera_info. PnP with wrong intrinsics puts ground zero in
        # the wrong place, so this matters.
        fb = aru["fallback_intrinsics"]
        self.aruco_K = np.array([[fb["fx"], 0.0, fb["cx"]],
                                 [0.0, fb["fy"], fb["cy"]],
                                 [0.0, 0.0, 1.0]])
        self.aruco_D = np.zeros((5, 1))
        self._intrinsics_from_camera_info = False
        self.marker_size = float(aru["marker_size_m"])
        self.request_ground_zero = False

        self._build_ui()

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

        self.ros_node = DashboardNode(self)
        self.ros_thread = threading.Thread(
            target=rclpy.spin, args=(self.ros_node,), daemon=True)
        self.ros_thread.start()

    def _start_gnss(self):
        self.gnss_thread = GNSSListenerThread(self.cfg["gnss"])
        self.gnss_thread.gnss_updated.connect(self._cb_gnss)
        self.gnss_thread.start()

    def _build_ui(self):
        root = QHBoxLayout()
        root.setSpacing(0)
        root.setContentsMargins(8, 8, 8, 8)

        # LEFT COLUMN
        left = QVBoxLayout()
        left.setSpacing(6)

        # ── Tracking status: the panel the operator actually watches ──────────
        left.addWidget(self._hdr("📡  Tracking Status", '#00ff88', 12))
        self.lbl_status = QLabel("WAITING FOR SLAM\n—\n—")
        self.lbl_status.setFont(QFont("Monospace", 9))
        self.lbl_status.setStyleSheet(
            "background:#111; color:#ffaa00; padding:8px; "
            "border:1px solid #333; border-radius:4px;")
        left.addWidget(self.lbl_status)

        left.addWidget(self._hdr("🔍  Camera View  (ArUco Anchor)"))
        self.lbl_track = self._cambox(320, 200, "Waiting for Camera...")
        left.addWidget(self.lbl_track)

        # GNSS is opt-in via tracking.yaml; hide the whole block when it is off
        # so a studio setup is not cluttered by a permanently red panel.
        if self.gnss_enabled:
            left.addWidget(self._hdr("🛰️  RTK GNSS Status", '#ff0055'))

            self.btn_gnss = QPushButton("STOP RTK GNSS")
            self.btn_gnss.setStyleSheet("background:#550000; color:white; font-weight:bold; padding:4px;")
            self.btn_gnss.setCursor(Qt.PointingHandCursor)
            self.btn_gnss.clicked.connect(self._toggle_gnss)
            left.addWidget(self.btn_gnss)

            self.lbl_gnss = QLabel(f"Waiting for {self.cfg['gnss']['port']}...")
            self.lbl_gnss.setFont(QFont("Monospace", 9))
            self.lbl_gnss.setStyleSheet("background:#111; padding:8px; border:1px solid #333; border-radius:4px;")
            left.addWidget(self.lbl_gnss)
        else:
            self.btn_gnss = None
            self.lbl_gnss = None

        left.addWidget(self._hdr("⏱️  TC-NEO Timecode Sync", '#00ffaa'))
        self.lbl_tc = QLabel("TC: NO SIGNAL (Software Fallback)")
        self.lbl_tc.setFont(QFont("Monospace", 9))
        self.lbl_tc.setStyleSheet("background:#111; padding:8px; border:1px solid #333; border-radius:4px;")
        left.addWidget(self.lbl_tc)

        left.addWidget(self._hdr("📐  IMU Telemetry", '#ffaa00'))
        
        self.lbl_imu = QLabel("Waiting…")
        self.lbl_imu.setFont(QFont("Monospace", 9))
        self.lbl_imu.setStyleSheet("background:#111; padding:8px; border:1px solid #333; border-radius:4px;")
        left.addWidget(self.lbl_imu)

        left.addWidget(self._hdr("📍  Live 6DOF Pose (as sent to Unreal)", '#00ff88'))
        self.lbl_pose = QLabel("X: --  Y: --  Z: --\nPan: --  Tilt: --  Roll: --")
        self.lbl_pose.setFont(QFont("Monospace", 9))
        self.lbl_pose.setStyleSheet("background:#111; padding:8px; border:1px solid #333; border-radius:4px;")
        left.addWidget(self.lbl_pose)

        self.btn_aruco = QPushButton("🎯  ANCHOR TO ARUCO GROUND ZERO")
        self.btn_aruco.setFont(QFont("Arial", 11, QFont.Bold))
        self.btn_aruco.setFixedHeight(40)
        self.btn_aruco.setCursor(Qt.PointingHandCursor)
        self.btn_aruco.setStyleSheet("QPushButton{background:#303f9f;color:#fff;border-radius:4px;border:1px solid #3f51b5;} QPushButton:hover{background:#3f51b5;}")
        self.btn_aruco.clicked.connect(self._anchor_aruco)
        left.addWidget(self.btn_aruco)

        self.btn_udp = QPushButton("UDP STREAM TO UNREAL ENGINE: ON  ▶")
        self.btn_udp.setFont(QFont("Arial", 13, QFont.Bold))
        self.btn_udp.setFixedHeight(52)
        self.btn_udp.setCursor(Qt.PointingHandCursor)
        self._style_btn(True)
        self.btn_udp.clicked.connect(self._toggle_stream)
        left.addWidget(self.btn_udp)
        left.addStretch()

        # DIVIDER
        div = QFrame()
        div.setFrameShape(QFrame.VLine)
        div.setStyleSheet("color:#2a2a3a; margin:0 6px;")

        # RIGHT COLUMN
        right = QVBoxLayout()
        right.setSpacing(6)

        right.addWidget(self._hdr("🗺️  LiDAR Map & AR Alignment", '#ff00aa', 13))
        self.map_viewer = MapViewerWidget()
        right.addWidget(self.map_viewer, stretch=1)

        # Use a QSplitter as the main layout so left/right columns are resizable
        main_split = QSplitter(Qt.Horizontal)
        main_split.setHandleWidth(5)
        main_split.setStyleSheet(
            "QSplitter::handle{background:#555566;} QSplitter{background:#0d0d1a;}")

        left_widget = QWidget()
        left_widget.setLayout(left)
        left_widget.setStyleSheet("background:#0d0d1a;")

        right_widget = QWidget()
        right_widget.setLayout(right)
        right_widget.setStyleSheet("background:#0d0d1a;")

        main_split.addWidget(left_widget)
        main_split.addWidget(right_widget)
        main_split.setStretchFactor(0, 35)
        main_split.setStretchFactor(1, 65)

        self.setCentralWidget(main_split)

    def _hdr(self, text, color='#00d4ff', size=10):
        l = QLabel(text)
        l.setFont(QFont("Arial", size, QFont.Bold))
        l.setStyleSheet(f"color:{color}; padding:2px 0;")
        return l

    def _cambox(self, min_w, min_h, placeholder):
        l = QLabel(placeholder)
        l.setAlignment(Qt.AlignCenter)
        l.setMinimumSize(min_w, min_h)
        l.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        l.setStyleSheet("background:#000; border:1px solid #2a2a3a; border-radius:4px;")
        return l

    def _style_btn(self, on):
        if on:
            s = ("QPushButton{background:#00c853;color:#000;border-radius:6px;"
                 "padding:10px;border:2px solid #00e676;}"
                 "QPushButton:hover{background:#00e676;}")
        else:
            s = ("QPushButton{background:#b71c1c;color:#fff;border-radius:6px;"
                 "padding:10px;border:2px solid #d50000;}"
                 "QPushButton:hover{background:#d50000;}")
        self.btn_udp.setStyleSheet(s)

    def _toggle_stream(self):
        self.streaming_enabled = not self.streaming_enabled
        state = "ON  ▶" if self.streaming_enabled else "OFF  ■"
        self.btn_udp.setText(f"UDP STREAM TO UNREAL ENGINE: {state}")
        self._style_btn(self.streaming_enabled)

    def _anchor_aruco(self):
        """Request ArUco ground zero anchor on next camera frame."""
        if not self.aruco_enabled:
            print("[ArUco] disabled in tracking.yaml")
            return
        self.request_ground_zero = True
        self.btn_aruco.setText("⏳  SCANNING FOR ARUCO MARKER...")
        self.btn_aruco.setStyleSheet("QPushButton{background:#ff9800;color:#000;border-radius:4px;border:1px solid #ffa000;font-weight:bold;}")
        print("[ArUco] Ground Zero anchor requested — point camera at marker.")

    def _toggle_gnss(self):
        if self.btn_gnss is None:
            return
        if self.gnss_thread is not None:
            self.btn_gnss.setText("START RTK GNSS")
            self.btn_gnss.setStyleSheet("background:#005500; color:white; font-weight:bold; padding:4px;")
            self.lbl_gnss.setText("GNSS Disabled by User")
            self.gnss_thread.requestInterruption()
            self.gnss_thread.wait(2000)
            self.gnss_thread = None
        else:
            self.btn_gnss.setText("STOP RTK GNSS")
            self.btn_gnss.setStyleSheet("background:#550000; color:white; font-weight:bold; padding:4px;")
            self.lbl_gnss.setText(f"Waiting for {self.cfg['gnss']['port']}...")
            self._start_gnss()

    # --- ROS Callbacks ---
    def image_callback(self, msg):
        """Throttled ~5fps camera view for ArUco anchoring + visual lag check."""
        try:
            self._cam_frame_n += 1
            # Skip 2 of every 3 frames (~5fps) unless actively anchoring
            if self._cam_frame_n % 3 != 0 and not self.request_ground_zero:
                return

            # Decode grayscale infra1
            if msg.encoding in ("mono8", "8UC1"):
                gray = self.cv_bridge.imgmsg_to_cv2(msg, "mono8")
            elif msg.encoding in ("mono16", "16UC1"):
                gray = (self.cv_bridge.imgmsg_to_cv2(msg, "16UC1") >> 8).astype(np.uint8)
            else:
                gray = self.cv_bridge.imgmsg_to_cv2(msg, "mono8")

            # Downscale display frame (320x200)
            disp = cv2.resize(gray, (320, 200))
            cv_img = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
            if hasattr(self, 'map_viewer'):
                # Pass full res gray for AR overlay (or color if we had it), but disp is fine for UI
                self.map_viewer.update_camera_frame(cv_img)

            # ArUco detection on full-res when anchoring or every 15 frames
            if (self.request_ground_zero or self._cam_frame_n % 15 == 0) and (not hasattr(self, 'map_viewer') or self.map_viewer.is_pose_stored()):
                if self.aruco_detector:
                    corners, ids, _ = self.aruco_detector.detectMarkers(gray)
                else:
                    corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict,
                                                           parameters=self.aruco_params)
                self.last_aruco_corners = corners
                self.last_aruco_ids = ids
            else:
                corners = getattr(self, 'last_aruco_corners', None)
                ids = getattr(self, 'last_aruco_ids', None)

            if ids is not None and len(ids) > 0:
                scale_x, scale_y = 320 / msg.width, 200 / msg.height
                disp_corners = [c * np.array([[scale_x, scale_y]]) for c in corners]
                aruco.drawDetectedMarkers(cv_img, disp_corners, ids)

                if self.request_ground_zero and self.latest_pose is not None:
                    s = self.marker_size / 2.0
                    obj_pts = np.array([[-s,s,0],[s,s,0],[s,-s,0],[-s,-s,0]], dtype=np.float32)
                    ok, rvec, tvec = cv2.solvePnP(obj_pts, corners[0][0], self.aruco_K, self.aruco_D)
                    if ok:
                        position, quat = self.latest_pose
                        T_world_cam = make_transform(position, quat)

                        # solvePnP returns the marker in the camera OPTICAL frame
                        # (x right, y down, z forward), but the SLAM pose is
                        # camera_link (x forward, y left, z up). Composing them
                        # directly -- as this used to -- mixes two frames and
                        # anchors ground zero in the wrong place.
                        T_body_optical = np.array([
                            [0.0,  0.0, 1.0, 0.0],
                            [-1.0, 0.0, 0.0, 0.0],
                            [0.0, -1.0, 0.0, 0.0],
                            [0.0,  0.0, 0.0, 1.0],
                        ])
                        R_c_m, _ = cv2.Rodrigues(rvec)
                        T_optical_marker = np.eye(4)
                        T_optical_marker[:3, :3] = R_c_m
                        T_optical_marker[:3, 3] = tvec.squeeze()

                        T_world_marker = T_world_cam @ T_body_optical @ T_optical_marker

                        # Keep only position + heading: a marker taped down
                        # slightly off-level should not tilt the whole virtual
                        # world, so ground zero stays horizon-level.
                        marker_fwd = T_world_marker[:3, :3][:, 1]
                        yaw = math.atan2(marker_fwd[1], marker_fwd[0])
                        T_level = np.eye(4)
                        T_level[:3, :3] = rpy_to_matrix(0.0, 0.0, yaw)
                        T_level[:3, 3] = T_world_marker[:3, 3]

                        self.transformer.set_ground_zero_from_transform(T_level)
                        self.request_ground_zero = False
                        self.smoother.reset()
                        self.btn_aruco.setText("✅  ANCHORED TO GROUND ZERO")
                        self.btn_aruco.setStyleSheet(
                            "QPushButton{background:#00c853;color:#000;"
                            "border-radius:4px;border:2px solid #00e676;font-weight:bold;}")
                        print(f"[ArUco] ✅ Anchored to Ground Zero at "
                              f"{T_level[:3, 3].round(3)} heading {math.degrees(yaw):+.1f}°")
                        if hasattr(self, 'map_viewer') and hasattr(self, 'localizer'):
                            from pose_transform import invert_transform
                            # T_level is Marker in VIO frame.
                            # map_viewer.T_align is Map in VIO frame.
                            # Marker in Map = inv(T_align) * T_level
                            T_map_marker = invert_transform(self.map_viewer.T_align) @ T_level
                            print(f"[LiDAR] Automated Virtual Anchor placed at Map Coordinates: {T_map_marker[:3, 3].round(3)}")
                            # We can store this if needed, but the visual lineup (T_align) is already perfect!

            st = "ANCHORED ✓" if self.transformer.is_zeroed else (
                 "MARKER DETECTED" if (ids is not None and len(ids) > 0) else "SCANNING...")
            col = (0,255,80) if self.transformer.is_zeroed else (
                  (0,255,255) if (ids is not None and len(ids) > 0) else (150,150,150))
            cv2.putText(cv_img, st, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
            self.latest_cv_image = cv_img.copy()
        except Exception:
            pass



    def camera_info_callback(self, msg):
        """Take the real intrinsics from the camera rather than trusting config.

        ArUco ground zero is solved with PnP; wrong intrinsics put the anchor in
        the wrong place, and the hardcoded fallback only matched one particular
        D455 at one particular resolution.
        """
        if self._intrinsics_from_camera_info:
            return
        k = list(msg.k) if hasattr(msg, "k") else list(msg.K)
        if len(k) < 9 or k[0] == 0.0:
            return
        self.aruco_K = np.array(k, dtype=float).reshape(3, 3)
        d = list(msg.d) if hasattr(msg, "d") else list(msg.D)
        self.aruco_D = (np.array(d, dtype=float).reshape(-1, 1)
                        if d else np.zeros((5, 1)))
        self._intrinsics_from_camera_info = True
        print(f"[Intrinsics] from camera_info: fx={self.aruco_K[0,0]:.2f} "
              f"fy={self.aruco_K[1,1]:.2f} cx={self.aruco_K[0,2]:.2f} "
              f"cy={self.aruco_K[1,2]:.2f}")

    def imu_callback(self, msg):
        """Telemetry only.

        Isaac ROS consumes this same IMU internally (enable_imu_fusion:=True) and
        publishes an already-fused pose. This bridge used to integrate the gyro
        here and write its own estimate into latest_pose, racing Isaac's output
        and injecting jitter. Now it only feeds the on-screen readout.
        """
        g = msg.angular_velocity
        a = msg.linear_acceleration

        # The IMU now has a real job besides the readout: deciding whether the
        # rig is physically still, so stationary drift can be held out.
        self.hold.update_imu((g.x, g.y, g.z), (a.x, a.y, a.z))

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
        """Single entry point for every incoming pose."""
        now = time.time()

        # Absorb tracker snaps so the virtual camera never teleports. Measured on
        # recorded hardware data: this cut peak excursion on a diverging run from
        # 138.9 m to 1.08 m and the worst single-frame step from 106.9 m to 146 mm.
        #
        # The gate is fed the SENSOR timestamp, not arrival time: Isaac's stamps
        # are clean (11.1 ms +/- 0.006 at 90 Hz) whereas arrival times bunch up
        # when messages are delivered in a batch, which makes ordinary motion
        # look infinitely fast and gets it wrongly rejected.
        position, quaternion, snapped, reason = self.gate.filter(
            position, quaternion, stamp if stamp is not None else now)
        if snapped:
            # Do not extrapolate across a discontinuity.
            linear = (0.0, 0.0, 0.0)
            angular = (0.0, 0.0, 0.0)
            if self.gate.consecutive_rejects <= 1 or self.gate.consecutive_rejects % 100 == 0:
                pass # Removed print to prevent UI freeze from console spam

        # Pin the output while the IMU says the rig is not moving.
        position, quaternion, held = self.hold.apply(position, quaternion)
        if held:
            linear = (0.0, 0.0, 0.0)
            angular = (0.0, 0.0, 0.0)

        with self._pose_lock:
            self.latest_pose = (position, quaternion)
            self.predictor.update(position, quaternion, linear, angular, now)
            self.last_vio_time = now
            self._pose_updates += 1

            if hasattr(self, 'map_viewer'):
                self.map_viewer.update_camera_pose(position, quaternion)

        if self.slam_status != "TRACKING":
            self.slam_status = "TRACKING"
            print("[SLAM] ✅ Tracking — streaming 6DOF to Unreal Engine.")

        if self.send_on_pose:
            self._stream_freed()



    def camera_pose_now(self):
        """The pose to send right now: predicted, converted, smoothed.

        Returns (UnrealPose, is_valid). On tracking loss the last good pose is
        held and is_valid goes False -- freezing the virtual camera is recoverable
        on set, teleporting it is not.
        """
        now = time.time()
        with self._pose_lock:
            if not self.predictor.has_pose:
                return None, False
            fresh = self.predictor.age(now) <= self.stale_timeout
            predicted = self.predictor.predict(now) if fresh else None
            fallback = self.latest_pose

        if predicted is not None:
            position, quaternion = predicted
        else:
            position, quaternion = fallback

        pose = self.transformer.ros_to_unreal(position, quaternion)
        if fresh:
            pose = self.smoother.apply(pose)
        return pose, fresh

    # --- Hardware Callbacks ---
    def _cb_tc(self, tc_str):
        self.latest_timecode = tc_str
        self.lbl_tc.setText(f"{tc_str} (Hardware Sync Locked)")

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
        """Send one FreeD packet to Unreal Engine."""
        if not self.streaming_enabled:
            return

        pose, valid = self.camera_pose_now()
        if pose is None:
            return   # nothing tracked yet: stay silent rather than assert an origin

        if not valid and self.slam_status == "TRACKING":
            self.slam_status = "TRACKING LOST — HOLDING"
            print("[SLAM] ⚠️  Pose stale — holding last known position.")

        pkt = get_freed_packet(CAMERA_ID,
                               pose.tilt, pose.pan, pose.roll,
                               pose.x, pose.y, pose.z)
        try:
            self.sock.sendto(pkt, (UDP_IP, UDP_PORT))
            self._packets_sent += 1
            self.latest_unreal = pose
        except Exception as exc:
            if self._packets_sent == 0:
                print(f"[FreeD] send failed to {UDP_IP}:{UDP_PORT} — {exc}")

    def _refresh(self):
        self.lbl_imu.setText(self.latest_imu)

        # Thread-safe GUI updates for images
        if hasattr(self, 'latest_cv_image') and self.latest_cv_image is not None:
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
            self._pose_rate = self._pose_updates / elapsed
            self._send_rate = self._packets_sent / elapsed
            self._pose_updates = 0
            self._packets_sent = 0
            self._last_rate_check = now

        # ── Pose readout, in the same numbers Unreal receives ────────────────
        pose = self.latest_unreal
        if pose is not None:
            self.lbl_pose.setText(
                f"X: {pose.x:+8.3f} m   Y: {pose.y:+8.3f} m   Z: {pose.z:+8.3f} m\n"
                f"Pan: {pose.pan:+7.2f}°  Tilt: {pose.tilt:+7.2f}°  "
                f"Roll: {pose.roll:+7.2f}°"
            )

        fresh = (now - self.last_vio_time) < self.stale_timeout if self.last_vio_time else False
        if not self.last_vio_time:
            state, colour = "WAITING FOR SLAM", "#ffaa00"
        elif fresh:
            state, colour = "TRACKING", "#00ff88"
        else:
            state, colour = "TRACKING LOST — HOLDING POSE", "#ff3355"

        anchor = "ANCHORED" if self.transformer.is_zeroed else "not anchored"
        hold = "  |  HOLD (still)" if self.hold.is_stationary else ""
        gate = ""
        if self.gate.enabled and (self.gate.total_rejects or self.gate.resyncs):
            gate = (f"\nGate: {self.gate.total_rejects} rejected, "
                    f"{self.gate.resyncs} resync  ({self.gate.last_reason})")
        self.lbl_status.setText(
            f"{state}\n"
            f"SLAM in: {self._pose_rate:5.1f} Hz   →   FreeD out: {self._send_rate:5.1f} Hz\n"
            f"Unreal: {UDP_IP}:{UDP_PORT} cam {CAMERA_ID}   |   {anchor}{hold}{gate}"
        )
        self.lbl_status.setStyleSheet(
            f"background:#111; color:{colour}; padding:8px; "
            f"border:1px solid #333; border-radius:4px;")

    def closeEvent(self, event):
        if hasattr(self, 'gnss_thread') and self.gnss_thread:
            self.gnss_thread.requestInterruption()
            self.gnss_thread.wait(2000)
        if hasattr(self, 'ltc_thread') and self.ltc_thread:
            self.ltc_thread.requestInterruption()
            self.ltc_thread.wait(2000)
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
