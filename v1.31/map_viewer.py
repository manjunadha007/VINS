"""
MapViewerWidget — 3DEqualizer-style viewport editor for LiDAR alignment.
Uses pure QPainter rendering (no pyqtgraph) for rock-solid resize/maximize.
"""

import numpy as np
import math
import cv2
import collections
import threading
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QDoubleSpinBox, QGroupBox, QPushButton, QLineEdit,
    QSplitter, QSizePolicy, QFileDialog
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QPointF, QRectF
from PyQt5.QtGui import (QImage, QPixmap, QPainter, QColor, QPen, QBrush,
                          QFont, QPolygonF)


# ─────────────────────────────────────────────────
# Pure QPainter viewport widgets (no pyqtgraph)
# ─────────────────────────────────────────────────

class OrthoView(QWidget):
    """2D orthographic view (Top or Side) with grid, point cloud, trail, camera dot."""

    def __init__(self, title="View", axis_h="X", axis_v="Y"):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(100, 80)
        self._title = title
        self._ax_h, self._ax_v = axis_h, axis_v
        self._pts = None       # Nx2
        self._trail = []       # list of (h, v)
        self._cam = None       # (h, v)
        self._lock = threading.Lock()
        # View transform
        self._center = np.array([0., 0.])
        self._scale = 200.0  # 200 pixels per metre — a 2.5 m room fills 500 px
        self._auto_center_frames = 50 # Auto-center on camera for the first few updates
        self._drag_start = None

    def set_points(self, h_arr, v_arr):
        with self._lock:
            self._pts = list(zip(h_arr, v_arr))
        self.update()

    def set_trail(self, h_arr, v_arr):
        with self._lock:
            self._trail = list(zip(h_arr, v_arr))
        self.update()

    def set_cam(self, h, v):
        with self._lock:
            self._cam = (h, v)
            if self._auto_center_frames > 0:
                self._center = np.array([h, v], dtype=float)
                self._auto_center_frames -= 1
        self.update()

    def _world_to_screen(self, wh, wv):
        W, H = self.width(), self.height()
        sx = W / 2 + (wh - self._center[0]) * self._scale
        sy = H / 2 - (wv - self._center[1]) * self._scale
        return sx, sy

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        W, H = self.width(), self.height()
        p.fillRect(0, 0, W, H, QColor('#111122'))

        # Grid
        p.setPen(QPen(QColor('#1a1a2e'), 1))
        grid_step = max(1.0, 10 ** math.floor(math.log10(max(W, H) / self._scale / 5 + 0.01)))
        cx, cy = self._center
        for i in range(-50, 51):
            gw = round(cx / grid_step) * grid_step + i * grid_step
            sx, _ = self._world_to_screen(gw, 0)
            if 0 <= sx <= W:
                col = QColor('#333355') if abs(gw) < grid_step * 0.01 else QColor('#1a1a2e')
                p.setPen(QPen(col, 1 if abs(gw) > grid_step * 0.01 else 2))
                p.drawLine(int(sx), 0, int(sx), H)
            _, sy = self._world_to_screen(0, gw)
            if 0 <= sy <= H:
                col = QColor('#333355') if abs(gw) < grid_step * 0.01 else QColor('#1a1a2e')
                p.setPen(QPen(col, 1 if abs(gw) > grid_step * 0.01 else 2))
                p.drawLine(0, int(sy), W, int(sy))

        with self._lock:
            pts = self._pts
            trail = list(self._trail)
            cam = self._cam

        # Points
        if pts is not None and len(pts) > 0:
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(QColor(100, 180, 255, 80)))
            for wh, wv in pts[::max(1, len(pts) // 5000)]:
                sx, sy = self._world_to_screen(wh, wv)
                if 0 <= sx <= W and 0 <= sy <= H:
                    p.drawEllipse(QPointF(sx, sy), 1.5, 1.5)

        # Trail
        if len(trail) > 1:
            p.setPen(QPen(QColor('#ff6600'), 1.5))
            for i in range(1, len(trail)):
                x0, y0 = self._world_to_screen(*trail[i-1])
                x1, y1 = self._world_to_screen(*trail[i])
                p.drawLine(QPointF(x0, y0), QPointF(x1, y1))

        # Camera dot
        if cam is not None:
            sx, sy = self._world_to_screen(*cam)
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(QColor(255, 50, 50, 60)))
            p.drawEllipse(QPointF(sx, sy), 12, 12)
            p.setBrush(QBrush(QColor('#ff4444')))
            p.drawEllipse(QPointF(sx, sy), 6, 6)

        # Title
        p.setPen(QColor('#00d4ff'))
        p.setFont(QFont('Arial', 9, QFont.Bold))
        p.drawText(6, 16, f"{self._title}  {self._ax_h}→  {self._ax_v}↑")
        p.end()

    def wheelEvent(self, e):
        delta = e.angleDelta().y()
        self._scale *= 1.15 if delta > 0 else 0.87
        self._scale = max(10.0, min(10000.0, self._scale))
        self.update()

    def mousePressEvent(self, e):
        if e.button() == Qt.MiddleButton:
            self._drag_start = e.pos()

    def mouseMoveEvent(self, e):
        if self._drag_start and e.buttons() & Qt.MiddleButton:
            dx = e.pos().x() - self._drag_start.x()
            dy = e.pos().y() - self._drag_start.y()
            self._center[0] -= dx / self._scale
            self._center[1] += dy / self._scale
            self._drag_start = e.pos()
            self.update()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MiddleButton:
            self._drag_start = None


class PerspView(QWidget):
    """3D perspective view with orbit/pan/zoom, drawn with QPainter."""

    def __init__(self):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(100, 80)
        self._pts = None       # Nx3
        self._trail = []       # list of (x,y,z)
        self._cam_pos = np.array([0., 0., 0.])
        self._lock = threading.Lock()
        self._orb_center = np.array([0., 0., 0.])
        self._orb_dist = 5.0   # 5 metres back — fits a small room
        self._orb_yaw = 45.0
        self._orb_pitch = 30.0
        self._auto_center_frames = 50
        self._drag_start = None
        self._drag_btn = None
        self._overlay_pix = None
        self._show_overlay = False

    def set_points(self, pts_3d):
        with self._lock:
            self._pts = pts_3d
        self.update()

    def set_trail(self, pts_3d):
        with self._lock:
            self._trail = pts_3d
        self.update()

    def set_cam(self, pt_3d):
        with self._lock:
            self._cam_pos = np.array(pt_3d)
            if self._auto_center_frames > 0:
                self._orb_center = self._cam_pos.copy()
                self._auto_center_frames -= 1
        self.update()

    def set_overlay(self, cv_img):
        if cv_img is None:
            return
        h, w, ch = cv_img.shape
        qimg = QImage(cv_img.data, w, h, ch * w, QImage.Format_BGR888)
        self._overlay_pix = QPixmap.fromImage(qimg)
        if self._show_overlay:
            self.update()

    def toggle_overlay(self, on):
        self._show_overlay = on
        self.update()

    def _eye_basis(self):
        yr = math.radians(self._orb_yaw)
        pr = math.radians(self._orb_pitch)
        d = self._orb_dist
        eye = self._orb_center + np.array([
            d * math.cos(pr) * math.cos(yr),
            d * math.cos(pr) * math.sin(yr),
            d * math.sin(pr)])
        fwd = self._orb_center - eye
        fwd /= (np.linalg.norm(fwd) + 1e-12)
        world_up = np.array([0., 0., 1.])
        right = np.cross(fwd, world_up)
        rn = np.linalg.norm(right)
        right = right / rn if rn > 1e-6 else np.array([1., 0., 0.])
        up = np.cross(right, fwd)
        return eye, fwd, right, up

    def _project(self, pts_3d):
        """Project Nx3 world points to Nx2 screen coords + validity mask."""
        eye, fwd, right, up = self._eye_basis()
        W, H = self.width(), self.height()
        f = min(W, H) * 0.8
        rel = pts_3d - eye
        z = rel @ (-fwd)
        ok = z > 0.1
        sx_arr = rel @ right
        sy_arr = rel @ up
        screen = np.zeros((len(pts_3d), 2))
        screen[ok, 0] = W / 2 + f * sx_arr[ok] / z[ok]
        screen[ok, 1] = H / 2 - f * sy_arr[ok] / z[ok]
        return screen, ok

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        W, H = self.width(), self.height()
        p.fillRect(0, 0, W, H, QColor('#0a0a18'))

        # Camera overlay (semi-transparent, behind everything else)
        if self._show_overlay and self._overlay_pix is not None:
            scaled = self._overlay_pix.scaled(W, H, Qt.KeepAspectRatio, Qt.FastTransformation)
            ox = (W - scaled.width()) // 2
            oy = (H - scaled.height()) // 2
            p.setOpacity(0.35)
            p.drawPixmap(ox, oy, scaled)
            p.setOpacity(1.0)

        # Grid
        grid_segs = []
        for i in range(-10, 11):
            grid_segs.append(np.array([[i, -10, 0], [i, 10, 0]], dtype=float))
            grid_segs.append(np.array([[-10, i, 0], [10, i, 0]], dtype=float))
        p.setPen(QPen(QColor('#222244'), 1))
        for seg in grid_segs:
            scr, ok = self._project(seg)
            if np.all(ok):
                p.drawLine(QPointF(scr[0, 0], scr[0, 1]), QPointF(scr[1, 0], scr[1, 1]))

        # Axes
        for seg, col in [
            (np.array([[0,0,0],[2,0,0]], dtype=float), '#ff3333'),
            (np.array([[0,0,0],[0,2,0]], dtype=float), '#33ff33'),
            (np.array([[0,0,0],[0,0,2]], dtype=float), '#3333ff'),
        ]:
            scr, ok = self._project(seg)
            if np.all(ok):
                p.setPen(QPen(QColor(col), 2))
                p.drawLine(QPointF(scr[0, 0], scr[0, 1]), QPointF(scr[1, 0], scr[1, 1]))

        with self._lock:
            pts = self._pts
            trail = list(self._trail)
            cam = self._cam_pos.copy()

        # Points
        if pts is not None and len(pts) > 0:
            subset = pts[::max(1, len(pts) // 5000)]
            scr, ok = self._project(subset)
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(QColor(80, 200, 255, 80)))
            for i in range(len(subset)):
                if ok[i] and 0 <= scr[i, 0] <= W and 0 <= scr[i, 1] <= H:
                    p.drawEllipse(QPointF(scr[i, 0], scr[i, 1]), 1.5, 1.5)

        # Trail
        if len(trail) > 1:
            trail_arr = np.array(trail)
            scr, ok = self._project(trail_arr)
            p.setPen(QPen(QColor('#ff6600'), 1.5))
            for i in range(1, len(trail)):
                if ok[i-1] and ok[i]:
                    p.drawLine(QPointF(scr[i-1, 0], scr[i-1, 1]),
                               QPointF(scr[i, 0], scr[i, 1]))

        # Camera dot
        scr, ok = self._project(cam.reshape(1, 3))
        if ok[0]:
            sx, sy = scr[0]
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(QColor(255, 50, 50, 60)))
            p.drawEllipse(QPointF(sx, sy), 14, 14)
            p.setBrush(QBrush(QColor('#ff4444')))
            p.drawEllipse(QPointF(sx, sy), 7, 7)

        # Title
        p.setPen(QColor('#666'))
        p.setFont(QFont('Arial', 9))
        p.drawText(6, 16, "3D Perspective  MMB=orbit  RMB=pan  Scroll=zoom")
        p.end()

    def wheelEvent(self, e):
        delta = e.angleDelta().y()
        self._orb_dist *= 0.9 if delta > 0 else 1.1
        self._orb_dist = max(0.2, min(100.0, self._orb_dist))
        self.update()

    def mousePressEvent(self, e):
        if e.button() in (Qt.MiddleButton, Qt.RightButton):
            self._drag_start = e.pos()
            self._drag_btn = e.button()

    def mouseMoveEvent(self, e):
        if self._drag_start:
            dx = e.pos().x() - self._drag_start.x()
            dy = e.pos().y() - self._drag_start.y()
            if self._drag_btn == Qt.MiddleButton:
                self._orb_yaw += dx * 0.5
                self._orb_pitch = max(-89, min(89, self._orb_pitch - dy * 0.5))
            elif self._drag_btn == Qt.RightButton:
                _, _, right, up = self._eye_basis()
                self._orb_center -= right * dx * 0.005 * self._orb_dist
                self._orb_center -= up * dy * 0.005 * self._orb_dist
            self._drag_start = e.pos()
            self.update()

    def mouseReleaseEvent(self, e):
        if e.button() in (Qt.MiddleButton, Qt.RightButton):
            self._drag_start = None
            self._drag_btn = None


# ─────────────────────────────────────────────────
# Main widget
# ─────────────────────────────────────────────────

class MapViewerWidget(QWidget):
    alignment_changed = pyqtSignal(np.ndarray)
    file_loaded = pyqtSignal(str)
    TRAIL_MAX = 2000

    def __init__(self):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.tx = self.ty = self.tz = 0.0
        self.roll = self.pitch = self.yaw = 0.0
        self.scale = 1.0
        self.pose_stored = False
        self.map_points_3d = None
        self.cam_pose = np.eye(4)
        self.camera_K = None
        self.latest_frame = None
        self.T_align = np.eye(4)
        self._trail = collections.deque(maxlen=self.TRAIL_MAX)
        self._aligned_cache = None
        self._build_ui()
        self._update_transform()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(2)

        # ── Toolbar ──
        tb = QHBoxLayout()
        tb.setSpacing(3)
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Paste .obj path or browse →")
        self.path_edit.setStyleSheet("background:#1a1a2e;color:white;padding:3px;border:1px solid #333;")
        self.path_edit.setFixedWidth(180)
        self.path_edit.returnPressed.connect(self._load_from_path)
        tb.addWidget(self.path_edit)
        btn_br = QPushButton("📂")
        btn_br.setFixedWidth(30); btn_br.setStyleSheet("padding:3px;")
        btn_br.clicked.connect(self._browse_file)
        tb.addWidget(btn_br)
        btn_ld = QPushButton("Load")
        btn_ld.setFixedWidth(40); btn_ld.setStyleSheet("padding:3px;")
        btn_ld.clicked.connect(self._load_from_path)
        tb.addWidget(btn_ld)

        def _spin(lbl, lo, hi, step, val):
            grp = QGroupBox(lbl)
            grp.setStyleSheet("QGroupBox{font-size:9px;padding:0;margin:0;}")
            lay = QHBoxLayout(grp); lay.setContentsMargins(1,1,1,1)
            sb = QDoubleSpinBox()
            sb.setRange(lo, hi); sb.setSingleStep(step); sb.setValue(val)
            sb.setFixedWidth(50); sb.valueChanged.connect(self._on_ui_change)
            lay.addWidget(sb)
            return grp, sb

        self.grp_tx,self.sb_tx = _spin("X",-100,100,0.1,0)
        self.grp_ty,self.sb_ty = _spin("Y",-100,100,0.1,0)
        self.grp_tz,self.sb_tz = _spin("Z",-100,100,0.1,0)
        self.grp_r,self.sb_r   = _spin("R",-180,180,1,0)
        self.grp_p,self.sb_p   = _spin("P",-180,180,1,0)
        self.grp_y,self.sb_y   = _spin("Yw",-180,180,1,0)
        self.grp_s,self.sb_s   = _spin("S",0.01,100,0.1,1)
        for w in (self.grp_tx,self.grp_ty,self.grp_tz,self.grp_r,self.grp_p,self.grp_y,self.grp_s):
            tb.addWidget(w)

        self.btn_store = QPushButton("🔒 STORE")
        self.btn_store.setStyleSheet("background:#00aa00;color:white;font-weight:bold;padding:3px;")
        self.btn_store.clicked.connect(self._store_pose)
        tb.addWidget(self.btn_store)
        self.btn_reset = QPushButton("🔓 RESET")
        self.btn_reset.setStyleSheet("background:#aa0000;color:white;font-weight:bold;padding:3px;")
        self.btn_reset.clicked.connect(self._reset_pose)
        self.btn_reset.setEnabled(False)
        tb.addWidget(self.btn_reset)
        tb.addStretch()
        root.addLayout(tb)

        _ss = "QSplitter::handle{background:#555566;} QSplitter{background:#0d0d1a;}"

        # Views
        self.top_view = OrthoView("TOP", "X", "Y")
        self.side_view = OrthoView("SIDE", "X", "Z")
        self.persp_view = PerspView()

        # Overlay toggle
        bot = QHBoxLayout()
        self.btn_overlay = QPushButton("📷 Overlay: OFF")
        self.btn_overlay.setCheckable(True)
        self.btn_overlay.setStyleSheet("padding:3px;font-weight:bold;background:#222233;color:#888;")
        self.btn_overlay.clicked.connect(self._toggle_overlay)
        bot.addWidget(self.btn_overlay)
        bot.addStretch()

        persp_w = QWidget()
        persp_lay = QVBoxLayout(persp_w)
        persp_lay.setContentsMargins(0,0,0,0); persp_lay.setSpacing(0)
        persp_lay.addWidget(self.persp_view, stretch=1)
        persp_lay.addLayout(bot)

        self.h_split = QSplitter(Qt.Horizontal)
        self.h_split.setHandleWidth(5); self.h_split.setStyleSheet(_ss)
        self.h_split.addWidget(self.top_view)
        self.h_split.addWidget(self.side_view)

        self.v_split = QSplitter(Qt.Vertical)
        self.v_split.setHandleWidth(5); self.v_split.setStyleSheet(_ss)
        self.v_split.addWidget(self.h_split)
        self.v_split.addWidget(persp_w)
        self.v_split.setStretchFactor(0, 35)
        self.v_split.setStretchFactor(1, 65)
        root.addWidget(self.v_split, stretch=1)

    def _toggle_overlay(self):
        on = self.btn_overlay.isChecked()
        self.persp_view.toggle_overlay(on)
        if on:
            self.btn_overlay.setText("📷 Overlay: ON")
            self.btn_overlay.setStyleSheet("padding:3px;font-weight:bold;background:#335533;color:white;")
        else:
            self.btn_overlay.setText("📷 Overlay: OFF")
            self.btn_overlay.setStyleSheet("padding:3px;font-weight:bold;background:#222233;color:#888;")

    # ── Public API ──
    def is_pose_stored(self): return self.pose_stored

    def _store_pose(self):
        self.pose_stored = True
        self.btn_store.setEnabled(False); self.btn_reset.setEnabled(True)
        self.btn_store.setStyleSheet("background:#333;color:#888;font-weight:bold;padding:3px;")
        self.btn_reset.setStyleSheet("background:#ff3333;color:white;font-weight:bold;padding:3px;")
        for w in (self.sb_tx,self.sb_ty,self.sb_tz,self.sb_r,self.sb_p,self.sb_y,self.sb_s):
            w.setEnabled(False)

    def _reset_pose(self):
        self.pose_stored = False; self._trail.clear()
        self.btn_store.setEnabled(True); self.btn_reset.setEnabled(False)
        self.btn_store.setStyleSheet("background:#00aa00;color:white;font-weight:bold;padding:3px;")
        self.btn_reset.setStyleSheet("background:#550000;color:white;font-weight:bold;padding:3px;")
        for w in (self.sb_tx,self.sb_ty,self.sb_tz,self.sb_r,self.sb_p,self.sb_y,self.sb_s):
            w.setEnabled(True)

    def _browse_file(self):
        import os
        from PyQt5.QtWidgets import QComboBox, QMessageBox
        dialog = QFileDialog(self, "Load LiDAR Map", "", "3D Files (*.obj *.ply *.pcd);;All (*)")
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        
        # Make "Look in:" combo box editable so users can paste paths
        for cb in dialog.findChildren(QComboBox):
            if cb.objectName() == "lookInCombo":
                cb.setEditable(True)
                def on_enter(cb=cb):
                    p = cb.currentText()
                    if os.path.isdir(p): 
                        dialog.setDirectory(p)
                    else:
                        QMessageBox.warning(self, "Path Not Found", 
                            f"The path doesn't exist inside the Docker container:\n{p}\n\n"
                            "Please move your LiDAR scans to the Isaac ROS workspace folder:\n"
                            "/home/kiransai/workspaces/isaac_ros-dev/\n\n"
                            "Inside this window, they will appear under:\n/workspaces/isaac_ros-dev/")
                cb.lineEdit().returnPressed.connect(on_enter)
                break
                
        if dialog.exec_():
            files = dialog.selectedFiles()
            if files:
                self.path_edit.setText(files[0])
                self.file_loaded.emit(files[0])

    def _load_from_path(self):
        path = self.path_edit.text().strip()
        if path: self.file_loaded.emit(path)

    def _on_ui_change(self):
        self.tx,self.ty,self.tz = self.sb_tx.value(),self.sb_ty.value(),self.sb_tz.value()
        self.roll,self.pitch,self.yaw = self.sb_r.value(),self.sb_p.value(),self.sb_y.value()
        self.scale = self.sb_s.value()
        self._update_transform()

    def _update_transform(self):
        from pose_transform import rpy_to_matrix
        T = np.eye(4)
        R = rpy_to_matrix(math.radians(self.roll), math.radians(self.pitch), math.radians(self.yaw))
        T[:3,:3] = R * self.scale
        T[0,3],T[1,3],T[2,3] = self.tx,self.ty,self.tz
        self.T_align = T
        self.alignment_changed.emit(self.T_align)
        self._aligned_cache = None
        self._redraw_map()

    def set_map_points(self, points):
        if len(points) > 50000:
            idx = np.random.choice(len(points), 50000, replace=False)
            self.map_points_3d = points[idx]
        else:
            self.map_points_3d = points.copy()
        self._aligned_cache = None
        self._redraw_map()

    def set_camera_intrinsics(self, K):
        self.camera_K = K

    def update_camera_frame(self, frame_bgr):
        self.latest_frame = frame_bgr
        self.persp_view.set_overlay(frame_bgr)

    def update_camera_pose(self, position, quaternion):
        from pose_transform import make_transform
        self.cam_pose = make_transform(position, quaternion)
        px, py, pz = float(position[0]), float(position[1]), float(position[2])
        self._trail.append((px, py, pz))
        self.top_view.set_cam(px, py)
        self.side_view.set_cam(px, pz)
        self.persp_view.set_cam((px, py, pz))
        if len(self._trail) > 1:
            t = list(self._trail)
            th = [p[0] for p in t]
            ty = [p[1] for p in t]
            tz = [p[2] for p in t]
            self.top_view.set_trail(th, ty)
            self.side_view.set_trail(th, tz)
            self.persp_view.set_trail(t)

    def _get_aligned(self):
        if self.map_points_3d is None: return None
        if self._aligned_cache is None:
            hom = np.column_stack((self.map_points_3d, np.ones(len(self.map_points_3d))))
            self._aligned_cache = (self.T_align @ hom.T).T[:, :3]
        return self._aligned_cache

    def _redraw_map(self):
        pts = self._get_aligned()
        if pts is None: return
        self.top_view.set_points(pts[:, 0], pts[:, 1])
        self.side_view.set_points(pts[:, 0], pts[:, 2])
        self.persp_view.set_points(pts)
