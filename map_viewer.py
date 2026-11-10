"""
MapViewerWidget — 3DEqualizer-style viewport editor for LiDAR alignment.
Overlays use pure QPainter. The 3D perspective viewport renders the imported mesh with real OpenGL (depth-buffered, GPU
rasterized) so every polygon draws every frame regardless of view angle or
mesh density -- QPainter has no z-buffer and can't do this correctly for
dense, concave scanned geometry. Grid/trail/HUD overlays stay QPainter,
composited on top via beginNativePainting()/endNativePainting().
"""

import ctypes
import time
import numpy as np
import math
import cv2
import collections
import threading
from OpenGL import GL as gl
from OpenGL.raw.GL.VERSION.GL_2_0 import glVertexAttribPointer as _raw_vertex_attrib_pointer
from PyQt5.QtWidgets import (
    QWidget, QOpenGLWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton,
    QSizePolicy, QFileDialog, QListWidget,
    QDoubleSpinBox, QGridLayout, QScrollArea
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QPointF, QRectF, QLineF
from PyQt5.QtGui import (QImage, QPixmap, QPainter, QColor, QPen, QBrush,
                          QFont, QCursor, QSurfaceFormat, QOpenGLContext)

# Give every QOpenGLWidget in this app a real depth buffer. Must be set
# before the first QOpenGLWidget's native window is created; setting it at
# module import time (map_viewer is imported before QApplication/any
# widget is constructed) is the earliest safe point.
_fmt = QSurfaceFormat()
_fmt.setDepthBufferSize(24)
# Do not block the GUI thread on the vertical retrace.
#
# A QOpenGLWidget inside a widget hierarchy is presented through
# QPlatformBackingStore::composeAndFlush, which ends in a GLX buffer swap. At
# the default swap interval of 1 that swap waits for the compositor's Present
# event -- captured on a frozen dashboard as the main thread parked in
# xcb_wait_for_special_event, inside composeAndFlush, below QEventLoop::exec.
# While it sits there it is not in the event loop, so it stops answering X
# messages: input dies and the window manager reports "not responding", even
# though timers still fire in the gaps and the viewport still repaints. It
# showed up when the window was *maximized* specifically -- an exactly
# screen-sized window takes a different presentation path from one merely
# dragged to nearly that size.
#
# Nothing here needs the retrace: repaints are already rate-limited
# (MapViewerWidget.VIEW_REFRESH_HZ), and tearing in a monitoring viewport
# costs nothing next to an unresponsive UI.
_fmt.setSwapInterval(0)
QSurfaceFormat.setDefaultFormat(_fmt)


_MESH_VERTEX_SHADER = """
#version 120
attribute vec3 aPos;
uniform mat4 uMVP;
uniform mat4 uView;
varying vec3 vViewPos;
void main() {
    vViewPos = (uView * vec4(aPos, 1.0)).xyz;
    gl_Position = uMVP * vec4(aPos, 1.0);
}
"""

_MESH_FRAGMENT_SHADER = """
#version 120
varying vec3 vViewPos;
void main() {
    // Flat per-face normal reconstructed from screen-space derivatives of
    // the (planar-per-triangle) view-space position -- no normal buffer
    // needed, and works regardless of the source mesh's own winding/
    // normals. Forced to face the camera so every polygon shades sensibly
    // double-sided, matching Maya/Blender's viewport (a LiDAR-reconstructed
    // mesh rarely has the consistent winding a backface cull would need).
    vec3 n = normalize(cross(dFdx(vViewPos), dFdy(vViewPos)));
    if (n.z < 0.0) n = -n;
    float shade = clamp(n.z * 0.85 + 0.25, 0.12, 1.0);
    vec3 base = vec3(90.0, 145.0, 200.0) / 255.0;
    gl_FragColor = vec4(base * shade, 1.0);
}
"""


# ─────────────────────────────────────────────────
# Pure QPainter viewport widgets (no pyqtgraph)
# ─────────────────────────────────────────────────

class _VerticalGrip(QWidget):
    """A drag handle that resizes the overlay panel it sits at the bottom of.

    The viewport's panels are absolutely positioned children floating over the
    GL surface rather than items in a layout, so there is no splitter to put
    between them, and QSizeGrip only resizes top-level windows. This is the
    small amount of hand-rolled hit-testing that leaves.
    """

    HEIGHT = 8

    def __init__(self, parent, on_drag):
        super().__init__(parent)
        self._on_drag = on_drag
        self._press_y = None
        self.setFixedHeight(self.HEIGHT)
        self.setCursor(Qt.SizeVerCursor)
        self.setToolTip("Drag to resize the Outliner")
        # The panel sets a background and border on itself and every
        # descendant; opt out like the other children do.
        self.setStyleSheet("border:none; background:transparent;")

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._press_y = e.globalPos().y()
            e.accept()

    def mouseMoveEvent(self, e):
        if self._press_y is None:
            return
        y = e.globalPos().y()
        self._on_drag(y - self._press_y)
        self._press_y = y
        e.accept()

    def mouseReleaseEvent(self, e):
        self._press_y = None
        e.accept()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setPen(QPen(QColor(110, 130, 150), 1))
        cy = self.height() // 2
        cx = self.width() // 2
        for dx in (-10, -2, 6):
            p.drawLine(cx + dx, cy, cx + dx + 4, cy)
        p.end()


class TransformChannelBox(QWidget):
    """Numeric channels for the active outliner item; refreshing never edits it."""
    channel_edited = pyqtSignal(int, float)

    def __init__(self):
        super().__init__()
        self.setStyleSheet(
            "QWidget { background:#151e29; color:#dce4ef; border:none; }"
            "QLabel { background:transparent; }"
            "QDoubleSpinBox { background:#0c131d; color:#e4edf7; border:1px solid #344356;"
            " border-radius:3px; padding:3px; font-size:11px; }"
            "QDoubleSpinBox:focus { border-color:#7dbce9; }"
            "QDoubleSpinBox:disabled { color:#64748b; }")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(5)
        title = QLabel("CHANNEL BOX")
        title.setStyleSheet("color:#7dbce9;font-size:10px;font-weight:bold;")
        layout.addWidget(title)
        self.item_label = QLabel("Select an item")
        self.item_label.setWordWrap(True)
        layout.addWidget(self.item_label)
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(4)
        self.fields = {}
        for index, key in enumerate(("tx", "ty", "tz", "rx", "ry", "rz")):
            axis = index % 3
            label = QLabel(("Position " if index < 3 else "Rotation ") + "XYZ"[axis])
            label.setStyleSheet(f"color:{('#ed8b8b', '#91d5a3', '#94bdf2')[axis]};")
            field = QDoubleSpinBox()
            field.setObjectName("channel_" + key)
            field.setDecimals(4 if index < 3 else 3)
            field.setRange(-1_000_000, 1_000_000)
            field.setSingleStep(0.01 if index < 3 else 1.0)
            field.setSuffix(" m" if index < 3 else "°")
            field.setKeyboardTracking(False)
            field.setAccelerated(True)
            field.setMinimumWidth(118)
            field.setToolTip("Type a value and press Enter or Tab to commit.")
            field.valueChanged.connect(
                lambda value, channel=index: self.channel_edited.emit(channel, value))
            self.fields[key] = field
            grid.addWidget(label, index, 0)
            grid.addWidget(field, index, 1)
        layout.addLayout(grid)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color:#9eafc2;font-size:10px;")
        layout.addWidget(self.status)
        layout.addStretch()

    def refresh(self, name, position, rotation, enabled, note, force=False):
        self.item_label.setText(name)
        self.status.setText(note)
        for field, value in zip(self.fields.values(), (*position, *rotation)):
            field.setEnabled(enabled)
            # Live camera updates may continue while an operator types. Keep
            # that unfinished channel, and refresh the other five normally.
            if not force and field.hasFocus() and field.lineEdit().isModified():
                continue
            blocked = field.blockSignals(True)
            if force or abs(field.value() - value) >= 0.5 * 10 ** -field.decimals():
                field.setValue(float(value))
                field.lineEdit().setModified(False)
            field.blockSignals(blocked)


class PerspView(QOpenGLWidget):
    """3D perspective viewport — Unreal Engine-style navigation, Maya-style grid.

    Navigation (matches Unreal Engine defaults):
        LMB drag      : Orbit around focus point
        MMB drag      : Pan (truck/pedestal)
        RMB drag      : Look around (FPS-style)
        RMB + WASD/QE : Fly through the scene (hold RMB, then use keys)
        RMB + Scroll  : Adjust fly speed
        Scroll        : Dolly zoom (move closer/further from focus)
        F             : Focus on selected outliner item
        Del           : Delete the selected scan (Point Cloud)
        W / E         : Switch the gizmo between Move and Rotate (not while
                        flying, where W and E are forward and up)

    The gizmo follows Unreal's model: one manipulator at a time on whatever
    the Outliner has selected. Move draws the X/Y/Z arrows plus a centre
    handle that slides in the ground plane; Rotate draws three world-aligned
    rings that turn about world X, Y and Z.

    The cursor is re-centered on every drag move (LMB/MMB/RMB), so orbiting,
    panning, and looking around never run out of screen room — this is what
    gives the "infinite walk-around" feel instead of clamping at the monitor
    edge. The ground grid is near-plane clipped and radially faded so it
    reads as an endless plane rather than a visible finite tile.
    """

    #: screen-space pick radius for a locator gizmo handle, in pixels.
    LOCATOR_PICK_PX = 14

    #: Outliner overlay panel. Width is fixed; the height is a starting
    #: point the operator drags from, bounded so it can neither collapse to
    #: nothing nor run off the viewport.
    #: Gizmo modes, Unreal-style: one manipulator at a time, never both.
    #: Translate draws the X/Y/Z arrows and the planar centre handle; Rotate
    #: draws the three world-aligned rings. W and E switch, as in Unreal.
    GIZMO_MODES = ('translate', 'rotate')

    OUTLINER_W = 260
    OUTLINER_DEFAULT_H = 520     # outliner, manipulator controls, numeric channels
    OUTLINER_MIN_H = 120         # a floor; the layout's own minimum wins if larger
    OUTLINER_MAX_H = 900

    locator_changed = pyqtSignal()
    camera_moved = pyqtSignal()          # the camera gizmo was dragged
    mesh_moved = pyqtSignal()            # the point-cloud gizmo was dragged
    delete_requested = pyqtSignal()      # Del pressed with the scan selected
    trail_changed = pyqtSignal()         # display offset, never recorded samples

    def __init__(self):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(100, 80)
        self.setFocusPolicy(Qt.StrongFocus)  # needed for WASD
        self._pts = None       # Nx3
        self._mesh_verts = None    # Nx3, set together with _mesh_faces
        self._mesh_faces = None    # Mx3 int, triangle vertex indices
        self._trail = []       # list of (x,y,z)
        self._cam_pos = np.array([0., 0., 0.])
        self._lock = threading.Lock()

        # Locator ("tracking origin" -- BP_TrackingOrigin's replacement).
        # Every ArUco anchor maps the marker onto this transform. Translate
        # X/Y/Z + yaw only -- matches try_anchor()'s own level_horizon
        # convention, which already discards the marker's own pitch/roll.
        self._locator_pos = np.array([0., 0., 0.])
        # Orientation is (roll, pitch, yaw) in degrees. It used to be a lone
        # yaw, which is all the translate-only gizmo could produce.
        self._locator_rpy = np.zeros(3)
        self._locator_marker_size = None
        self._active_locator_id = "locator"
        self._locators = {"locator": dict(name="Locator", transform=np.eye(4), marker=None)}
        self._locator_counter = 1
        self._camera_edit_locked = False
        self._active_gizmo = None   # None|x|y|z|plane_xy|rot_x|rot_y|rot_z
        self._gizmo_drag_screen0 = None
        self._gizmo_drag_pos0 = None
        self._gizmo_drag_yaw0 = 0.0
        self._gizmo_drag_axis_dir = None    # cached at drag start
        self._gizmo_drag_px_per_unit = None
        self._gizmo_drag_plane_Jinv = None
        self._gizmo_drag_origin_scr = None
        self._locator_selected = False
        self._camera_selected = False
        # One manipulator and channel box share the active outliner target.
        self._gizmo_target = 'locator'   # replaced by the outliner's default
        self._gizmo_enabled = True
        self._cam_rpy = np.zeros(3)
        # Where the scan's gizmo sits and how it is turned. Pushed down by
        # MapViewerWidget, which owns the mesh-alignment transform.
        self._mesh_gizmo_pos = np.array([0., 0., 0.])
        self._mesh_gizmo_rpy = np.zeros(3)
        self._mesh_edit_locked = False
        self._trail_pos = np.zeros(3)
        self._trail_rpy = np.zeros(3)
        self._world_pos = np.zeros(3)
        self._world_rpy = np.zeros(3)

        # Camera state (orbit mode)
        self._orb_center = np.array([0., 0., 0.])
        self._orb_dist = 5.0
        self._orb_yaw = 45.0
        self._orb_pitch = 30.0

        # Camera near/far clip planes. The viewport's Clip Planes slider
        # panel was removed, so these are simply the range everything
        # projects against; set_near_clip()/set_far_clip() still adjust them
        # programmatically and keep near < far.
        self._near = 0.05
        self._far = 10000.0

        # GPU mesh resources (compiled/allocated lazily on first paint, see
        # _ensure_gl_ready/_gl_sync_mesh -- not here, since __init__ runs
        # before this widget has a GL context to compile shaders against).
        self._gl_prog = None
        self._gl_vbo = None
        self._gl_ebo = None
        self._gl_u_mvp = None
        self._gl_u_view = None
        self._gl_index_count = 0
        self._gl_mesh_dirty = False

        # Fly mode state
        self._fly_speed = 2.0    # metres per second base speed
        self._keys_held = set()
        self._fly_timer = QTimer(self)
        self._fly_timer.setInterval(16)  # ~60 Hz
        self._fly_timer.timeout.connect(self._fly_tick)

        self._drag_start = None
        self._drag_btn = None
        self._last_global = None
        self._overlay_pix = None
        self._show_overlay = False
        self._gizmo_mode = 'translate'
        self._outliner_h = self.OUTLINER_DEFAULT_H
        self._setup_outliner()

    def _setup_outliner(self):
        self.outliner_widget = QWidget(self)
        self.outliner_widget.setStyleSheet("background:rgba(15, 15, 25, 200); border-radius:4px; border:1px solid #334;")

        lay = QVBoxLayout(self.outliner_widget)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)

        lbl = QLabel("Outliner")
        lbl.setStyleSheet("color:#00d4ff; font-weight:bold; font-size:10px; border:none; background:transparent;")
        lay.addWidget(lbl)

        self.outliner_list = QListWidget()
        self.outliner_list.setStyleSheet(
            "QListWidget { background:transparent; border:none; color:#e0e0e0; outline:none; font-size:11px; }"
            "QListWidget::item { padding:2px; }"
            "QListWidget::item:selected { background:#303f9f; border-radius:2px; color:white; }"
            # The panel's own background/border cascades into the scrollbar,
            # which otherwise renders as a bright slab across the panel.
            "QScrollBar:vertical { background:#1a1a2e; width:8px; border:none; margin:0; }"
            "QScrollBar::handle:vertical { background:#555; border-radius:4px; min-height:18px; }"
            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height:0; border:none; }"
            "QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background:transparent; }"
        )
        # Fixed-width overlay: there is nowhere to scroll sideways to, so
        # elide the long entry instead of showing a horizontal bar.
        self.outliner_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.outliner_list.setTextElideMode(Qt.ElideRight)
        # The panel is drag-resizable, so the list takes the slack rather
        # than being pinned at a height taller than the panel showing it --
        # which is what clipped 'World Origin' and the locator behind the
        # buttons. OUTLINER_DEFAULT_H shows all five rows; this floor is what
        # the panel can be dragged down to.
        self.outliner_list.setMinimumHeight(54)
        self.outliner_list.setMaximumHeight(110)
        self.outliner_list.addItems(["Camera", "Point Cloud", "Trail",
                                     "World Origin", "Locator (Tracking Origin)"])
        self.outliner_list.item(4).setData(Qt.UserRole, "locator")
        # Connect *before* choosing the default row, so the handler actually
        # runs and `_gizmo_target` agrees with what is highlighted. The other
        # order left the list showing "Camera" while the gizmo was still the
        # locator's -- so the first drag on the gizmo moved the locator, and
        # the outliner then jumped to "Locator" to match.
        self.outliner_list.currentRowChanged.connect(self._on_outliner_selection_changed)
        self._select_gizmo_row('locator')
        lay.addWidget(self.outliner_list)

        locator_buttons = QHBoxLayout()
        self.btn_duplicate_locator = QPushButton("Duplicate Locator")
        self.btn_duplicate_locator.clicked.connect(self.duplicate_locator)
        locator_buttons.addWidget(self.btn_duplicate_locator)
        self.btn_remove_locator = QPushButton("Remove")
        self.btn_remove_locator.clicked.connect(self.remove_locator)
        locator_buttons.addWidget(self.btn_remove_locator)
        lay.addLayout(locator_buttons)
        from PyQt5.QtWidgets import QComboBox
        self.detected_marker = QComboBox()
        self.detected_marker.addItem("Show a marker in the live camera", None)
        self.detected_marker.setToolTip("Select the detected ID to assign to the selected Locator.")
        lay.addWidget(self.detected_marker)
        assignment = QHBoxLayout()
        self.locator_width = QDoubleSpinBox()
        self.locator_width.setRange(.005, 5.)
        self.locator_width.setDecimals(4)
        self.locator_width.setValue(.1)
        self.locator_width.setSuffix(" m")
        self.locator_width.setToolTip("Measured printed square width, excluding paper margin.")
        assignment.addWidget(self.locator_width)
        self.btn_assign_marker = QPushButton("Assign ID")
        self.btn_assign_marker.clicked.connect(self._assign_detected_marker)
        assignment.addWidget(self.btn_assign_marker)
        self.btn_clear_marker = QPushButton("Clear ID")
        self.btn_clear_marker.clicked.connect(self.clear_locator_marker)
        assignment.addWidget(self.btn_clear_marker)
        lay.addLayout(assignment)
        self.locator_assignment_status = QLabel("+X: printed top. +Z: out of marker face.")
        self.locator_assignment_status.setWordWrap(True)
        self.locator_assignment_status.setStyleSheet(
            "color:#aebfd2;font-size:10px;border:none;background:transparent;")
        lay.addWidget(self.locator_assignment_status)
        for button in (self.btn_duplicate_locator, self.btn_remove_locator,
                       self.btn_assign_marker, self.btn_clear_marker):
            button.setStyleSheet(
                "QPushButton {color:#dce7f4;background:#233547;border:1px solid #3e586d;"
                "border-radius:3px;padding:4px;font-size:10px;}"
                "QPushButton:hover {background:#304a61;}")
            button.setCursor(Qt.PointingHandCursor)
        self.detected_marker.setStyleSheet(
            "QComboBox {color:#dce7f4;background:#172431;border:1px solid #3e586d;"
            "padding:4px;font-size:10px;}"
            "QComboBox QAbstractItemView {color:#dce7f4;background:#172431;"
            "selection-background-color:#303f9f;}")
        self.locator_width.setStyleSheet(
            "QDoubleSpinBox {color:#dce7f4;background:#172431;border:1px solid #3e586d;"
            "padding:3px;font-size:10px;}")

        self.btn_gizmo = QPushButton("✥ Gizmo: ON")
        self.btn_gizmo.setCheckable(True)
        self.btn_gizmo.setChecked(True)
        self.btn_gizmo.setCursor(Qt.PointingHandCursor)
        self.btn_gizmo.setStyleSheet(
            "QPushButton { background:#1b3a4b; color:#7fdbff; border:1px solid #2a5a70;"
            " border-radius:3px; font-size:10px; padding:3px; }"
            "QPushButton:!checked { background:#2a2a2a; color:#888;"
            " border:1px solid #444; }")
        self.btn_gizmo.setToolTip(
            "Show and allow dragging the gizmo on the selected object. "
            "Turn it off to navigate without catching a handle.")
        self.btn_gizmo.toggled.connect(self._on_gizmo_toggled)
        lay.addWidget(self.btn_gizmo)

        # Manipulator mode, Unreal-style: one at a time, W and E to switch.
        mode_row = QHBoxLayout()
        mode_row.setContentsMargins(0, 0, 0, 0)
        mode_row.setSpacing(3)
        mode_ss = (
            "QPushButton { background:#2a2a2a; color:#888; border:1px solid #444;"
            " border-radius:3px; font-size:10px; padding:3px; }"
            "QPushButton:checked { background:#1b3a4b; color:#7fdbff;"
            " border:1px solid #2a5a70; font-weight:bold; }")
        self.btn_move = QPushButton("⤧ Move")
        self.btn_move.setToolTip("Translate along X / Y / Z, or drag the "
                                 "centre to slide in the ground plane  (W)")
        self.btn_rotate = QPushButton("⟳ Rotate")
        self.btn_rotate.setToolTip("Turn about world X / Y / Z on the "
                                   "matching ring  (E)")
        for b, mode in ((self.btn_move, 'translate'), (self.btn_rotate, 'rotate')):
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            b.setStyleSheet(mode_ss)
            b.clicked.connect(lambda _checked, m=mode: self.set_gizmo_mode(m))
            mode_row.addWidget(b)
        self.btn_move.setChecked(True)
        lay.addLayout(mode_row)

        self.btn_center = QPushButton("\U0001f3af Center View")
        self.btn_center.setStyleSheet("QPushButton{background:#00d4ff;color:#000;border-radius:2px;border:none;padding:4px;font-weight:bold;font-size:10px;} QPushButton:hover{background:#00aadd;}")
        self.btn_center.setCursor(Qt.PointingHandCursor)
        self.btn_center.clicked.connect(self._center_on_selection)
        lay.addWidget(self.btn_center)

        self.channel_box = TransformChannelBox()
        self.channel_box.channel_edited.connect(self._edit_channel)
        self.channel_scroll = QScrollArea()
        self.channel_scroll.setWidgetResizable(True)
        self.channel_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.channel_scroll.setMinimumHeight(70)
        self.channel_scroll.setWidget(self.channel_box)
        lay.addWidget(self.channel_scroll, 1)
        self._sync_channels(force=True)

        self.outliner_grip = _VerticalGrip(self.outliner_widget,
                                           self._drag_outliner_height)
        lay.addWidget(self.outliner_grip)

    def _drag_outliner_height(self, dy):
        """Grip callback: grow or shrink the Outliner by a drag delta."""
        self._outliner_h += dy
        self._layout_outliner()

    def _layout_outliner(self):
        """Place the Outliner, clamping its height to what the viewport can
        show. The clamp is applied here rather than in the drag handler so a
        panel dragged tall in a big window still fits when the window shrinks,
        without forgetting the height the operator asked for."""
        # Never below what the contents actually need -- squashing the panel
        # past the layout's own minimum is what makes children overlap.
        floor = max(self.OUTLINER_MIN_H,
                    self.outliner_widget.minimumSizeHint().height())
        ceiling = max(floor, self.height() - 20)
        self._outliner_h = max(floor, min(self._outliner_h, self.OUTLINER_MAX_H))
        h = min(self._outliner_h, ceiling)
        width = min(self.OUTLINER_W, max(100, self.width() - 20))
        self.outliner_widget.setGeometry(
            max(0, self.width() - width - 10), 10, width, h)

    def set_near_clip(self, value):
        self._near = max(1e-4, float(value))
        if self._near >= self._far:
            self._far = self._near * 10
        self.update()

    def set_far_clip(self, value):
        self._far = max(self._near * 1.01, float(value))
        self.update()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._layout_outliner()

    def _center_on_selection(self):
        item = self.outliner_list.currentItem()
        if not item: return
        name = item.text()

        with self._lock:
            cloud = self._pts if self._pts is not None else self._mesh_verts
            if name == "Camera":
                self._orb_center = self._cam_pos.copy()
            elif name == "Point Cloud" and cloud is not None and len(cloud) > 0:
                self._orb_center = np.mean(cloud, axis=0)
            elif name == "Trail" and self._trail:
                self._orb_center = self.display_points("trail", [self._trail[-1]])[0]
            elif name == "World Origin":
                self._orb_center = self._world_pos.copy()
            elif name.startswith("Locator"):
                self._orb_center = self._locator_pos.copy()
        self.update()

    def _on_gizmo_toggled(self, on):
        self.btn_gizmo.setText(f"✥ Gizmo: {'ON' if on else 'OFF'}")
        self.set_gizmo_enabled(on)

    def _select_gizmo_row(self, target):
        """Point the outliner at `target`'s row.

        Skips the call when that row is already current: the selection
        handler clears `_active_gizmo`, so firing it while a drag is being
        set up would cancel the drag it was called to accompany.
        """
        spec = self.GIZMO_TARGETS.get(target)
        if spec is None:
            return
        for row in range(self.outliner_list.count()):
            if target == 'locator' and self.outliner_list.item(row).data(Qt.UserRole) != self._active_locator_id:
                continue
            if self.outliner_list.item(row).text().startswith(spec['prefix']):
                if self.outliner_list.currentRow() != row:
                    self.outliner_list.setCurrentRow(row)
                break

    def select_scan(self):
        """Point the outliner (and so the manipulator) at the scan."""
        self._select_gizmo_row('mesh')

    def select_locator(self):
        """Point the outliner (and so the manipulator) at the tracking origin."""
        self._select_gizmo_row('locator')

    def _on_outliner_selection_changed(self, row):
        item = self.outliner_list.item(row)
        if item is not None and item.data(Qt.UserRole) in self._locators:
            self._locators[self._active_locator_id]["transform"] = self.get_locator_transform()
            self._active_locator_id = item.data(Qt.UserRole)
            self.set_locator_transform(self._locators[self._active_locator_id]["transform"])
            marker = self._locators[self._active_locator_id]["marker"]
            self._locator_marker_size = marker["size_m"] if marker else None
            if marker and hasattr(self, "locator_width"):
                self.locator_width.setValue(marker["size_m"])
            if hasattr(self, "locator_assignment_status"):
                assigned = f"{marker['dictionary']} / ID {marker['id']}" if marker else "Unassigned"
                self.locator_assignment_status.setText(assigned + " · +X: printed top; +Z: out of face.")
        name = item.text() if item else ""
        self._gizmo_target = None
        for key, spec in self.GIZMO_TARGETS.items():
            if name.startswith(spec['prefix']):
                self._gizmo_target = key
                break
        self._locator_selected = self._gizmo_target == 'locator'
        self._camera_selected = self._gizmo_target == 'camera'
        self._active_gizmo = None
        self._sync_channels(force=True)
        self.update()

    def _channel_state(self):
        target = self._gizmo_target
        if target is None:
            return False, "Select an item in the outliner."
        if target == "mesh":
            points = self._pts if self._pts is not None else self._mesh_verts
            if points is None or len(points) == 0:
                return False, "Import a scan to edit its transform."
            if self._mesh_edit_locked:
                return False, "Scan alignment is locked."
            return True, "World position of scan centre. Rotation pivots around that centre."
        if target == "camera":
            if self._camera_edit_locked:
                return False, "Live tracked camera. Position and rotation follow the physical camera."
            return True, "World pose. With a loaded scan, edits preview a manual lineup; Apply commits it."
        if target == "trail":
            return True, "Display offset of the trail; recorded camera samples stay in their original coordinates."
        if target == "world":
            return True, "Display transform of the 3D reference grid and axes."
        return True, "World pose of the tracking origin."

    def _sync_channels(self, force=False):
        if not hasattr(self, "channel_box"):
            return
        spec = self.GIZMO_TARGETS.get(self._gizmo_target)
        enabled, note = self._channel_state()
        self.channel_box.refresh(
            spec["prefix"] if spec else "Select an item",
            self._gizmo_pos() if spec else np.zeros(3),
            self._gizmo_rpy() if spec else np.zeros(3),
            enabled, note, force=force)

    def _edit_channel(self, index, value):
        enabled, _ = self._channel_state()
        if not enabled or not np.isfinite(value):
            self._sync_channels(force=True)
            return
        # Change just this channel using the latest pose. Rounded readouts
        # in the other fields must not quantise or overwrite live values.
        if index < 3:
            pos = self._gizmo_pos().copy()
            pos[index] = value
            self._gizmo_set_pos(pos)
        else:
            rpy = self._gizmo_rpy().copy()
            rpy[index - 3] = (value + 180.0) % 360.0 - 180.0
            self._gizmo_set_rpy(rpy)
        self._gizmo_changed()
        self._sync_channels(force=True)
        self.update()

    def set_camera_edit_locked(self, locked):
        self._camera_edit_locked = bool(locked)
        if locked and self._gizmo_target == "camera":
            self._active_gizmo = None
        self._sync_channels(force=True)
        self.update()

    def set_mesh_edit_locked(self, locked):
        self._mesh_edit_locked = bool(locked)
        self._sync_channels(force=True)
        self.update()

    def get_display_transform(self, target):
        """Transform for the trail or the 3D reference grid/axes."""
        if target not in ("trail", "world"):
            raise ValueError("Display transform must be trail or world")
        from pose_transform import rpy_to_matrix
        T = np.eye(4)
        T[:3, :3] = rpy_to_matrix(*np.radians(getattr(self, "_" + target + "_rpy")))
        T[:3, 3] = getattr(self, "_" + target + "_pos")
        return T

    def set_display_transform(self, target, T):
        """Set the trail or reference-grid transform in code (File > Open Scene)."""
        if target not in ("trail", "world"):
            raise ValueError("Display transform must be trail or world")
        from pose_transform import matrix_to_rpy
        T = np.asarray(T, dtype=float)
        setattr(self, "_" + target + "_pos", T[:3, 3].copy())
        setattr(self, "_" + target + "_rpy",
                np.degrees(np.asarray(matrix_to_rpy(T[:3, :3]), dtype=float)))
        if target == "trail":
            self.trail_changed.emit()
        self._sync_channels(force=True)
        self.update()

    def display_points(self, target, points):
        T = self.get_display_transform(target)
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        return np.einsum("ij,nj->ni", T[:3, :3], points) + T[:3, 3]

    # -- Data setters --

    def set_points(self, pts_3d):
        with self._lock:
            self._pts = pts_3d
            self._mesh_verts = None
            self._mesh_faces = None
        self._gl_mesh_dirty = True
        self._sync_channels()
        self.update()

    def set_mesh(self, verts, faces):
        with self._lock:
            self._pts = None
            self._mesh_verts = verts
            self._mesh_faces = np.asarray(faces, dtype=np.int64)
        self._gl_mesh_dirty = True
        self._sync_channels()
        self.update()

    def clear_map(self):
        """Drop the scan from the viewport.

        Nulling both the point and mesh sides marks the GL buffers dirty, and
        the next sync uploads an empty index range -- so the mesh stops being
        drawn instead of lingering as a stale VBO the shader still walks.
        """
        with self._lock:
            self._pts = None
            self._mesh_verts = None
            self._mesh_faces = None
        self._gl_mesh_dirty = True
        self._sync_channels()
        self.update()

    def set_trail(self, pts_3d):
        with self._lock:
            self._trail = pts_3d
        self.update()

    def set_cam(self, pt_3d, yaw_deg=None, rpy_deg=None):
        with self._lock:
            # Do not fight the operator: while the camera gizmo is being
            # dragged, the live pose must not yank it back.
            if self._camera_edit_locked or not (self._gizmo_target == 'camera'
                    and self._active_gizmo is not None):
                self._cam_pos = np.array(pt_3d)
                if rpy_deg is not None:
                    self._cam_rpy = np.asarray(rpy_deg, dtype=float).copy()
                elif yaw_deg is not None:
                    self._cam_rpy[2] = float(yaw_deg)
        self._sync_channels()
        self.update()

    def get_camera_gizmo(self):
        """(position, yaw degrees) of the camera gizmo, as dragged."""
        return np.asarray(self._cam_pos, dtype=float).copy(), float(self._cam_rpy[2])

    def get_camera_gizmo_rpy(self):
        """Position and full (roll, pitch, yaw) in degrees."""
        return (np.asarray(self._cam_pos, dtype=float).copy(),
                self._cam_rpy.copy())

    # -- Locator (tracking origin) --

    def set_locator_marker_size(self, size_m):
        """Show the floor/table marker footprint with +X toward printed top."""
        self._locator_marker_size = float(size_m) if size_m is not None else None
        self.update()

    def _draw_marker_target(self, painter, basis):
        # All assigned locators stay visible; only the selected one has a gizmo.
        for key, entry in self.locator_entries().items():
            if key == self._active_locator_id:
                continue
            T = entry["transform"]
            marker = entry["marker"]
            s = (marker["size_m"] if marker else .1) / 2
            local = np.array([[-s,-s,0], [s,-s,0], [s,s,0], [-s,s,0], [-s,-s,0]])
            points = local @ T[:3,:3].T + T[:3,3]
            painter.setPen(QPen(QColor("#c4a86c"), 2))
            for a, b in zip(points[:-1], points[1:]):
                screen = self._project_segment(a, b, basis)
                if screen is not None:
                    painter.drawLine(QPointF(*screen[0]), QPointF(*screen[1]))
            screen, visible = self._project_pts(T[:3,3].reshape(1,3), basis)
            if visible[0]:
                label = entry["name"] + (f" · ID {marker['id']}" if marker else " · unassigned")
                painter.drawText(QPointF(*screen[0]) + QPointF(5,-5), label)
        if self._locator_marker_size is None:
            return
        s = self._locator_marker_size / 2
        T = self.get_locator_transform()
        local = np.array([[-s, -s, 0.], [s, -s, 0.],
                          [s, s, 0.], [-s, s, 0.], [0., 0., 0.],
                          [s * 2, 0., 0.]])
        points = local @ T[:3, :3].T + T[:3, 3]
        painter.setPen(QPen(QColor("#f0c17b"), 2))
        for a, b in ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5)):
            screen = self._project_segment(points[a], points[b], basis)
            if screen is not None:
                painter.drawLine(QPointF(*screen[0]), QPointF(*screen[1]))
        screen, visible = self._project_pts(points[5:6], basis)
        if visible[0]:
            painter.setFont(QFont("DejaVu Sans", 9))
            marker = self._locators[self._active_locator_id]["marker"]
            label = (f"ID {marker['id']} · " if marker else "") + "MARKER TOP (+X)"
            painter.drawText(QPointF(*screen[0]) + QPointF(5, -5), label)

    def get_locator_transform(self, locator_id=None):
        """4x4 in the same world/ROS frame as the marker solve and camera dot."""
        if locator_id is not None and locator_id != self._active_locator_id:
            return self._locators[locator_id]["transform"].copy()
        from pose_transform import rpy_to_matrix
        T = np.eye(4)
        r, pi, y = np.radians(self._locator_rpy)
        T[:3, :3] = rpy_to_matrix(r, pi, y)
        T[:3, 3] = self._locator_pos
        return T

    def locator_entries(self):
        """Independent poses and assignments, including edits in the channel box."""
        return {key: dict(name=entry["name"], transform=self.get_locator_transform(key),
                          marker=None if entry["marker"] is None else dict(entry["marker"]))
                for key, entry in self._locators.items()}

    def set_locator_entries(self, entries, active=None):
        """Replace every Locator with saved ones (File > Open Scene).

        `entries` is shaped like `locator_entries()`; `active` is the one to
        select, and so the anchor target, the first by default. The outliner
        is rebuilt with its signals held: the selection handler writes the
        old selection's pose into its entry, which would overwrite whichever
        restored Locator shares that key.
        """
        from PyQt5.QtWidgets import QListWidgetItem
        from marker_map import MAX_MARKERS
        if not 1 <= len(entries) <= MAX_MARKERS:
            raise ValueError(f"A scene holds 1 to {MAX_MARKERS} Locators.")
        active = active if active in entries else next(iter(entries))
        was = self.outliner_list.blockSignals(True)
        try:
            for row in reversed(range(self.outliner_list.count())):
                if self.outliner_list.item(row).data(Qt.UserRole) in self._locators:
                    self.outliner_list.takeItem(row)
            self._locators = {
                key: dict(name=entry["name"], transform=np.asarray(entry["transform"], dtype=float).copy(),
                          marker=None if entry["marker"] is None else dict(entry["marker"]))
                for key, entry in entries.items()}
            # Duplicates are numbered on from here, so none reuses a key.
            numbers = [int(key.rsplit("_", 1)[-1]) for key in self._locators
                       if key.rsplit("_", 1)[-1].isdigit()]
            self._locator_counter = max([len(self._locators), *numbers])
            for key, entry in self._locators.items():
                marker = entry["marker"]
                text = (f"{entry['name']} · ID {marker['id']}" if marker
                        else "Locator (Tracking Origin)" if key == "locator"
                        else entry["name"] + " · unassigned")
                item = QListWidgetItem(text)
                item.setData(Qt.UserRole, key)
                self.outliner_list.addItem(item)
                if key == active:
                    row = self.outliner_list.count() - 1
            self._active_locator_id = active
            self.set_locator_transform(self._locators[active]["transform"])
            self.outliner_list.setCurrentRow(row)
        finally:
            self.outliner_list.blockSignals(was)
        self._on_outliner_selection_changed(row)

    def duplicate_locator(self):
        from marker_map import MAX_MARKERS
        if self._gizmo_target != 'locator' or len(self._locators) >= MAX_MARKERS:
            self.locator_assignment_status.setText(f"Select a Locator to duplicate (up to {MAX_MARKERS}).")
            return None
        from PyQt5.QtWidgets import QListWidgetItem
        self._locator_counter += 1
        key = f"locator_{self._locator_counter}"
        self._locators[key] = dict(name=f"Locator {self._locator_counter}",
                                  transform=self.get_locator_transform(), marker=None)
        item = QListWidgetItem(self._locators[key]["name"] + " · unassigned")
        item.setData(Qt.UserRole, key)
        self.outliner_list.addItem(item)
        self.outliner_list.setCurrentItem(item)
        self.locator_assignment_status.setText("Move this duplicate to its marker, then assign a unique ID.")
        return key

    def add_marker_locator(self, T, family, fid, size_m):
        """A new Locator for an automatically placed marker; the selection stays put."""
        from PyQt5.QtWidgets import QListWidgetItem
        from marker_map import MAX_MARKERS
        if len(self._locators) >= MAX_MARKERS:
            raise ValueError(f"Already {MAX_MARKERS} Locators.")
        self._locator_counter += 1
        key = f"locator_{self._locator_counter}"
        self._locators[key] = dict(name=f"Locator {self._locator_counter}",
                                   transform=np.asarray(T, dtype=float).copy(), marker=None)
        item = QListWidgetItem(self._locators[key]["name"])
        item.setData(Qt.UserRole, key)
        self.outliner_list.addItem(item)
        try:
            self.assign_marker_to_locator(key, family, fid, size_m)
        except ValueError:
            del self._locators[key]
            self.outliner_list.takeItem(self.outliner_list.row(item))
            raise
        return key

    def assign_marker_to_locator(self, key, family, fid, size_m):
        """Assign an ID to one Locator, selected or not."""
        if any(k != key and e["marker"] is not None
               and (e["marker"]["dictionary"], e["marker"]["id"]) == (family, fid)
               for k, e in self._locators.items()):
            raise ValueError(f"ID {fid} already belongs to another Locator.")
        from marker_map import validate_layout
        validate_layout([dict(dictionary=family, id=fid, size_m=size_m,
                              position_m=[0,0,0], rotation_deg=[0,0,0])])
        entry = self._locators[key]
        entry["marker"] = dict(dictionary=family, id=fid, size_m=float(size_m))
        for i in range(self.outliner_list.count()):
            item = self.outliner_list.item(i)
            if item.data(Qt.UserRole) == key:
                item.setText(f"{entry['name']} · ID {fid}")
        if key == self._active_locator_id:
            self.locator_width.setValue(size_m)
            self.set_locator_marker_size(size_m)
        self.update()

    def remove_locator(self):
        if self._gizmo_target != 'locator' or len(self._locators) <= 1:
            return
        key = self._active_locator_id
        row = self.outliner_list.currentRow()
        # Select the survivor while the old transform is still available.
        survivor = next(k for k in self._locators if k != key)
        for i in range(self.outliner_list.count()):
            if self.outliner_list.item(i).data(Qt.UserRole) == survivor:
                self.outliner_list.setCurrentRow(i)
                break
        del self._locators[key]
        self.outliner_list.takeItem(row)
        self.update()

    def set_detected_markers(self, detections):
        keys = sorted({(family, int(fid)) for family, fid, _ in detections})
        if keys == getattr(self, "_detected_marker_keys", None):
            return
        self._detected_marker_keys = keys
        current = self.detected_marker.currentData()
        self.detected_marker.clear()
        self.detected_marker.addItem("Select detected marker" if keys else "No markers visible", None)
        for family, fid in keys:
            self.detected_marker.addItem(f"{family} · ID {fid}", (family, fid))
        index = self.detected_marker.findData(current) if current is not None else -1
        self.detected_marker.setCurrentIndex(index if index >= 0 else (1 if len(keys) == 1 else 0))

    def assign_locator_marker(self, family, fid, size_m):
        if self._gizmo_target != 'locator':
            raise ValueError("Select a Locator in the outliner first.")
        self.assign_marker_to_locator(self._active_locator_id, family, fid, size_m)
        entry = self._locators[self._active_locator_id]
        self.locator_assignment_status.setText(f"{entry['name']} assigned to {family} / ID {fid}.")

    def _assign_detected_marker(self):
        key = self.detected_marker.currentData()
        if key is None:
            self.locator_assignment_status.setText("Choose an ID detected in the live camera.")
            return
        try:
            self.assign_locator_marker(*key, self.locator_width.value())
        except ValueError as exc:
            self.locator_assignment_status.setText(str(exc))

    def clear_locator_marker(self):
        if self._gizmo_target != 'locator':
            return
        entry = self._locators[self._active_locator_id]
        entry["marker"] = None
        self.outliner_list.currentItem().setText(entry["name"] + " · unassigned")
        self.set_locator_marker_size(None)
        self.locator_assignment_status.setText("Unassigned · +X: printed top; +Z: out of face.")

    def set_locator_transform(self, T, locator_id=None):
        """Set the locator from a 4x4, orientation included.

        Roll and pitch used to be discarded here (only the heading survived),
        which the rotate gizmo would have silently thrown away on the next
        external set. Does not emit locator_changed: this is a programmatic
        set, not a user drag.
        """
        from pose_transform import matrix_to_rpy
        T = np.asarray(T, dtype=float)
        if locator_id is not None and locator_id != self._active_locator_id:
            self._locators[locator_id]["transform"] = T.copy()
            self.update()
            return
        self._locator_pos = T[:3, 3].copy()
        self._locator_rpy = np.degrees(np.asarray(matrix_to_rpy(T[:3, :3]),
                                                  dtype=float))
        self._sync_channels()
        self.update()

    # -- One manipulator for all outliner targets --

    #: Every outliner item has channels and a manipulator. Trail and World
    #: Origin edit display transforms, separate from tracking coordinates.
    GIZMO_TARGETS = {
        'camera':  {'prefix': 'Camera',      'label': 'CAMERA',
                    'tint': (0, 229, 255), 'attrs': ('_cam_pos', '_cam_rpy')},
        'locator': {'prefix': 'Locator',     'label': 'TRACKING ORIGIN',
                    'tint': (255, 204, 0), 'attrs': ('_locator_pos', '_locator_rpy')},
        'mesh':    {'prefix': 'Point Cloud', 'label': 'SCAN ALIGNMENT',
                    'tint': (124, 252, 0), 'attrs': ('_mesh_gizmo_pos', '_mesh_gizmo_rpy')},
        'trail':   {'prefix': 'Trail', 'label': 'TRAIL DISPLAY',
                    'tint': (255, 150, 70), 'attrs': ('_trail_pos', '_trail_rpy')},
        'world':   {'prefix': 'World Origin', 'label': 'REFERENCE GRID',
                    'tint': (190, 180, 255), 'attrs': ('_world_pos', '_world_rpy')},
    }

    @property
    def gizmo_active(self):
        """Is there a gizmo to draw and pick right now?"""
        return (self._gizmo_enabled and self._gizmo_target is not None
                and not (self._gizmo_target == 'mesh' and self._mesh_edit_locked)
                and not (self._gizmo_target == 'camera' and self._camera_edit_locked))

    def set_gizmo_enabled(self, on):
        self._gizmo_enabled = bool(on)
        if not self._gizmo_enabled:
            self._active_gizmo = None
        self.update()

    def _gizmo_pos(self):
        return getattr(self, self.GIZMO_TARGETS[self._gizmo_target]['attrs'][0])

    def _gizmo_set_pos(self, pos):
        setattr(self, self.GIZMO_TARGETS[self._gizmo_target]['attrs'][0],
                np.asarray(pos, dtype=float).copy())

    def _gizmo_rpy(self):
        """(roll, pitch, yaw) in degrees for whatever the gizmo is on."""
        return getattr(self, self.GIZMO_TARGETS[self._gizmo_target]['attrs'][1])

    def _gizmo_set_rpy(self, rpy):
        setattr(self, self.GIZMO_TARGETS[self._gizmo_target]['attrs'][1],
                np.asarray(rpy, dtype=float).copy())

    def _gizmo_yaw(self):
        return float(self._gizmo_rpy()[2])

    def _gizmo_set_yaw(self, yaw_deg):
        rpy = self._gizmo_rpy().copy()
        rpy[2] = float(yaw_deg)
        self._gizmo_set_rpy(rpy)

    def set_gizmo_mode(self, mode):
        """Switch between the translate and rotate manipulators.

        Only one is ever drawn or pickable -- Unreal's behaviour, and the
        reason the old gizmo was confusing: it showed the heading ring and
        the translate arrows at once, so a drag near the centre could move
        or turn the object depending on which handle happened to be nearer.
        """
        if mode not in self.GIZMO_MODES:
            return
        self._gizmo_mode = mode
        self._active_gizmo = None
        if hasattr(self, 'btn_move'):
            for b, m in ((self.btn_move, 'translate'), (self.btn_rotate, 'rotate')):
                was = b.blockSignals(True)
                b.setChecked(m == mode)
                b.blockSignals(was)
        self.update()

    def gizmo_mode(self):
        return self._gizmo_mode

    def _gizmo_changed(self):
        if self._gizmo_target == 'camera':
            self.camera_moved.emit()
        elif self._gizmo_target == 'mesh':
            self.mesh_moved.emit()
        elif self._gizmo_target == 'locator':
            self.locator_changed.emit()
        elif self._gizmo_target == 'trail':
            self.trail_changed.emit()
        self._sync_channels()

    def _gizmo_is_selected(self):
        return self._gizmo_target is not None

    def _gizmo_label(self):
        spec = self.GIZMO_TARGETS.get(self._gizmo_target)
        return spec['label'] if spec else ''

    def _gizmo_tint(self):
        spec = self.GIZMO_TARGETS.get(self._gizmo_target)
        return spec['tint'] if spec else (200, 200, 200)

    def set_mesh_gizmo(self, pos, rot):
        """Where the scan's gizmo sits, pushed down by the owning widget.

        `rot` is either a lone yaw in degrees (the old signature) or a full
        (roll, pitch, yaw) triple.
        """
        self._mesh_gizmo_pos = np.asarray(pos, dtype=float).copy()
        rot = np.atleast_1d(np.asarray(rot, dtype=float))
        self._mesh_gizmo_rpy = (rot.copy() if rot.size == 3
                                else np.array([0., 0., float(rot[0])]))
        self._sync_channels()
        if self._gizmo_target == 'mesh':
            self.update()

    def get_mesh_gizmo(self):
        return self._mesh_gizmo_pos.copy(), float(self._mesh_gizmo_rpy[2])

    def get_mesh_gizmo_rpy(self):
        """Position and full (roll, pitch, yaw) in degrees."""
        return self._mesh_gizmo_pos.copy(), self._mesh_gizmo_rpy.copy()

    def _locator_gizmo_world_points(self):
        """World-space origin + arrow-tip points for the active gizmo."""
        axis_len = max(self._orb_dist * 0.12, 0.3)
        o = self._gizmo_pos()
        return {
            'origin': o,
            'x': o + np.array([axis_len, 0., 0.]),
            'y': o + np.array([0., axis_len, 0.]),
            'z': o + np.array([0., 0., axis_len]),
        }, axis_len

    #: world axis index and colour for each rotation ring
    ROT_RINGS = (('rot_x', 0, np.array([1., 0., 0.]), '#ff5555'),
                 ('rot_y', 1, np.array([0., 1., 0.]), '#55ff55'),
                 ('rot_z', 2, np.array([0., 0., 1.]), '#5599ff'))

    def _rotation_ring_points(self, axis_idx, n=48):
        """World-space circle around one axis through the gizmo origin.

        World-aligned, not object-aligned: the rings always describe rotation
        about world X/Y/Z, which is what the alignment boxes and the pose
        readout are expressed in, so what you drag is what the numbers say.
        """
        pts, axis_len = self._locator_gizmo_world_points()
        o = pts['origin']
        r = axis_len * 0.8
        t = np.linspace(0, 2 * math.pi, n, endpoint=False)
        c, sn = r * np.cos(t), r * np.sin(t)
        ring = np.zeros((n, 3))
        a, b = [(1, 2), (2, 0), (0, 1)][axis_idx]
        ring[:, a] = c
        ring[:, b] = sn
        return o + ring

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

    # -- Camera math --

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

    @staticmethod
    def _dot3(pts, v):
        """`pts @ v` for an (N, 3) array against a 3-vector, without BLAS.

        numpy sends `@` to OpenBLAS, which for a matrix this skinny wakes its
        entire thread pool -- one thread per core -- to do a few thousand
        multiply-adds. Measured at the sizes this viewport uses (4000 rows):
        722 % CPU for `pts @ v` versus 99 % for the explicit form, to save
        0.011 ms of wall time. The viewport projects on every paint, so that
        pool never settles: it pegs every core and leaves nothing for the GUI
        thread, which reads as the whole dashboard freezing. Same arithmetic,
        same results, one core.
        """
        return pts[:, 0] * v[0] + pts[:, 1] * v[1] + pts[:, 2] * v[2]

    def _project_pts(self, pts_3d, basis):
        """Project Nx3 world points to Nx2 screen coords + validity mask,
        given a precomputed (eye, fwd, right, up) camera basis. Points
        outside the [near, far] clip range are marked invalid, matching
        Maya's camera near/far clip planes."""
        eye, fwd, right, up = basis
        W, H = self.width(), self.height()
        f = min(W, H) * 0.8
        rel = pts_3d - eye
        z = self._dot3(rel, fwd)
        ok = (z > self._near) & (z < self._far)
        sx_arr = self._dot3(rel, right)
        sy_arr = self._dot3(rel, up)
        screen = np.zeros((len(pts_3d), 2))
        screen[ok, 0] = W / 2 + f * sx_arr[ok] / z[ok]
        screen[ok, 1] = H / 2 - f * sy_arr[ok] / z[ok]
        return screen, ok

    def _project(self, pts_3d):
        """Project Nx3 world points to Nx2 screen coords + validity mask."""
        return self._project_pts(pts_3d, self._eye_basis())

    def _project_segment(self, a, b, basis):
        """Project a world-space line segment, clipping the part that falls
        outside the [near, far] clip range so lines crossing a clip plane
        (common while flying close to the ground grid, or with a tightened
        far clip) still draw their visible portion instead of vanishing
        outright."""
        eye, fwd, _, _ = basis
        za = float(np.dot(a - eye, fwd))
        zb = float(np.dot(b - eye, fwd))
        # Clip a hair inside [near, far] (not exactly on the boundary) so
        # the clipped endpoint reliably passes the strict visibility test
        # in _project_pts instead of landing on the boundary and getting
        # zeroed out by float rounding in the interpolate-then-reproject
        # round trip.
        near_t = self._near * 1.001 + 1e-4
        far_t = self._far * 0.999 - 1e-4
        if far_t <= near_t:
            far_t = near_t + 1e-4

        dz = zb - za
        if abs(dz) < 1e-12:
            if za < near_t or za > far_t:
                return None
            t_lo, t_hi = 0.0, 1.0
        else:
            t1 = (near_t - za) / dz
            t2 = (far_t - za) / dz
            t_lo = max(0.0, min(t1, t2))
            t_hi = min(1.0, max(t1, t2))
            if t_lo >= t_hi:
                return None

        a2 = a + t_lo * (b - a)
        b2 = a + t_hi * (b - a)
        screen, _ = self._project_pts(np.array([a2, b2]), basis)
        return screen

    def _project_segments(self, A, B, basis):
        """`_project_segment` over many segments at once.

        Same maths -- clip the segment to [near, far] in view depth, then
        perspective-divide the surviving endpoints -- but as whole-array
        numpy instead of one Python call per segment. The per-call version
        costs ~13 us, which is nothing for one line and 26 ms for the
        2000-point trail; that alone put the viewport under 30 fps while
        the UI timer asked for 60, so the GUI thread never caught up.

        Returns `(screen, keep)`: an (N, 2, 2) array of screen-space
        endpoints, and a bool mask of which segments survived clipping.
        Only entries where `keep` is set are meaningful.
        """
        A = np.asarray(A, dtype=float).reshape(-1, 3)
        B = np.asarray(B, dtype=float).reshape(-1, 3)
        n = len(A)
        screen = np.zeros((n, 2, 2))
        if n == 0:
            return screen, np.zeros(0, dtype=bool)

        eye, fwd, _, _ = basis
        za = self._dot3(A - eye, fwd)
        zb = self._dot3(B - eye, fwd)

        # Clip a hair inside [near, far] -- see _project_segment for why.
        near_t = self._near * 1.001 + 1e-4
        far_t = self._far * 0.999 - 1e-4
        if far_t <= near_t:
            far_t = near_t + 1e-4

        dz = zb - za
        flat = np.abs(dz) < 1e-12          # segment parallel to the view plane
        denom = np.where(flat, 1.0, dz)    # avoid a divide warning on those
        t1 = (near_t - za) / denom
        t2 = (far_t - za) / denom
        t_lo = np.where(flat, 0.0, np.maximum(0.0, np.minimum(t1, t2)))
        t_hi = np.where(flat, 1.0, np.minimum(1.0, np.maximum(t1, t2)))
        # A parallel segment is wholly in or wholly out; the rest survive
        # only if the clipped span is non-empty.
        keep = np.where(flat, (za >= near_t) & (za <= far_t), t_lo < t_hi)
        if not keep.any():
            return screen, keep

        d = B - A
        a2 = A[keep] + t_lo[keep, None] * d[keep]
        b2 = A[keep] + t_hi[keep, None] * d[keep]
        m = len(a2)
        scr, ok = self._project_pts(np.concatenate([a2, b2]), basis)
        # After clipping both endpoints are inside the range by construction;
        # drop any that still fail rather than drawing them to (0, 0).
        both = ok[:m] & ok[m:]
        idx = np.flatnonzero(keep)[both]
        screen[idx, 0, :] = scr[:m][both]
        screen[idx, 1, :] = scr[m:][both]
        keep = np.zeros(n, dtype=bool)
        keep[idx] = True
        return screen, keep

    def _polyline_lines(self, pts, basis):
        """Clipped screen-space QLineFs for a world-space polyline, ready
        for a single QPainter.drawLines()."""
        P = np.asarray(pts, dtype=float).reshape(-1, 3)
        if len(P) < 2:
            return []
        screen, keep = self._project_segments(P[:-1], P[1:], basis)
        return [QLineF(screen[i, 0, 0], screen[i, 0, 1],
                       screen[i, 1, 0], screen[i, 1, 1])
                for i in np.flatnonzero(keep)]

    # -- Maya-style multi-level grid --

    def _draw_grid(self, p, basis):
        """Draw a Maya/UE-style infinite ground grid with multiple subdivision
        levels. Lines are near-plane clipped (so they don't vanish crossing
        behind the camera while flying low) and radially faded toward the
        tile edge so the grid reads as endless rather than a visible finite
        square that pops as you move."""
        dist = self._orb_dist
        base = 10 ** math.floor(math.log10(max(dist * 0.3, 0.01)))
        levels = [
            (base * 0.1, QColor(30, 30, 45, 40), 1),
            (base,       QColor(45, 45, 65, 80), 1),
            (base * 10,  QColor(65, 65, 90, 140), 1),
        ]
        reference = self.get_display_transform('world')
        local_center = reference[:3, :3].T @ (self._orb_center - reference[:3, 3])
        cx, cy = local_center[:2]

        # 3 levels x 81 lines x 2 directions = 486 segments a frame. Projected
        # in one batch per level (see _project_segments); the pen still has to
        # be set per line because each carries its own radial fade alpha.
        for grid_step, color, width in levels:
            if grid_step < 0.001:
                continue
            half_extent = grid_step * 40
            fade_start = half_extent * 0.5
            i = np.arange(-40, 41)
            d = np.abs(i * grid_step)
            fade = 1.0 - np.clip((d - fade_start) / (half_extent - fade_start),
                                 0.0, 1.0)
            vis = fade > 0.02
            if not vis.any():
                continue
            iv, fv = i[vis], fade[vis]
            n = len(iv)
            zeros = np.zeros(n)
            gw = np.round(cx / grid_step) * grid_step + iv * grid_step
            gw2 = np.round(cy / grid_step) * grid_step + iv * grid_step
            # Interleaved so the draw order matches the old loop exactly:
            # the line along +Y at gw, then the one along +X at gw2.
            A = np.empty((2 * n, 3))
            B = np.empty((2 * n, 3))
            A[0::2] = np.stack([gw, np.full(n, cy - half_extent), zeros], axis=1)
            B[0::2] = np.stack([gw, np.full(n, cy + half_extent), zeros], axis=1)
            A[1::2] = np.stack([np.full(n, cx - half_extent), gw2, zeros], axis=1)
            B[1::2] = np.stack([np.full(n, cx + half_extent), gw2, zeros], axis=1)

            A = np.einsum('ij,nj->ni', reference[:3, :3], A) + reference[:3, 3]
            B = np.einsum('ij,nj->ni', reference[:3, :3], B) + reference[:3, 3]
            screen, keep = self._project_segments(A, B, basis)
            alphas = np.repeat((color.alpha() * fv).astype(int), 2)
            for k in np.flatnonzero(keep):
                c = QColor(color)
                c.setAlpha(int(alphas[k]))
                p.setPen(QPen(c, width))
                p.drawLine(QPointF(screen[k, 0, 0], screen[k, 0, 1]),
                           QPointF(screen[k, 1, 0], screen[k, 1, 1]))

        # World-origin axis lines
        half = max(dist * 50, 1000)
        for seg, col in [
            ((np.array([0., -half, 0.]), np.array([0., half, 0.])), QColor(80, 80, 100, 180)),
            ((np.array([-half, 0., 0.]), np.array([half, 0., 0.])), QColor(80, 80, 100, 180)),
        ]:
            ends = self.display_points('world', seg)
            scr = self._project_segment(ends[0], ends[1], basis)
            if scr is not None:
                p.setPen(QPen(col, 2))
                p.drawLine(QPointF(scr[0, 0], scr[0, 1]), QPointF(scr[1, 0], scr[1, 1]))

    # -- Mesh rendering (imported .obj/.stl/etc, faces preserved as-is) --
    #
    # Real OpenGL, not QPainter: a CPU painter's-algorithm (sort triangles
    # by centroid depth, draw back-to-front) cannot correctly resolve a
    # dense, concave, self-overlapping scanned mesh -- there is no single
    # global sort order for triangles that interpenetrate or wrap around
    # the camera, which is exactly the situation for a room-scale interior
    # scan viewed from inside. A real depth buffer resolves overlap
    # per-pixel and is correct regardless of triangle count or concavity,
    # so every polygon draws every frame at full detail, like Maya/Blender.

    def initializeGL(self):
        # Qt may recreate the context when the viewport changes top-level
        # windows. Buffer IDs belong to that context, never to the widget.
        self._gl_prog = None
        self._gl_vbo = None
        self._gl_ebo = None
        self._gl_u_mvp = None
        self._gl_u_view = None
        self._gl_index_count = 0
        self._gl_mesh_dirty = True

    def _ensure_gl_ready(self):
        """Compile lazily inside paintGL, with Qt's framebuffer bound."""
        if self._gl_prog is not None:
            return
        vs = gl.glCreateShader(gl.GL_VERTEX_SHADER)
        gl.glShaderSource(vs, _MESH_VERTEX_SHADER)
        gl.glCompileShader(vs)
        if not gl.glGetShaderiv(vs, gl.GL_COMPILE_STATUS):
            raise RuntimeError(gl.glGetShaderInfoLog(vs).decode())
        fs = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)
        gl.glShaderSource(fs, _MESH_FRAGMENT_SHADER)
        gl.glCompileShader(fs)
        if not gl.glGetShaderiv(fs, gl.GL_COMPILE_STATUS):
            raise RuntimeError(gl.glGetShaderInfoLog(fs).decode())
        prog = gl.glCreateProgram()
        gl.glAttachShader(prog, vs)
        gl.glAttachShader(prog, fs)
        gl.glBindAttribLocation(prog, 0, "aPos")
        gl.glLinkProgram(prog)
        if not gl.glGetProgramiv(prog, gl.GL_LINK_STATUS):
            raise RuntimeError(gl.glGetProgramInfoLog(prog).decode())
        gl.glDeleteShader(vs)
        gl.glDeleteShader(fs)

        self._gl_prog = prog
        self._gl_u_mvp = gl.glGetUniformLocation(prog, "uMVP")
        self._gl_u_view = gl.glGetUniformLocation(prog, "uView")
        self._gl_vbo = gl.glGenBuffers(1)
        self._gl_ebo = gl.glGenBuffers(1)

    def _gl_sync_mesh(self):
        """Upload the current mesh to the GPU if set_mesh()/set_points()
        changed it since the last paint. Done here (GUI thread, GL context
        guaranteed current) rather than in the setter, since a setter could
        in principle be called off-thread."""
        if not self._gl_mesh_dirty:
            return
        with self._lock:
            verts, faces = self._mesh_verts, self._mesh_faces
        self._gl_mesh_dirty = False
        if verts is None or faces is None or len(faces) == 0:
            self._gl_index_count = 0
            return
        v = np.ascontiguousarray(verts, dtype=np.float32)
        idx = np.ascontiguousarray(faces, dtype=np.uint32).reshape(-1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._gl_vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, v.nbytes, v, gl.GL_DYNAMIC_DRAW)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self._gl_ebo)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, idx.nbytes, idx, gl.GL_DYNAMIC_DRAW)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)
        self._gl_index_count = idx.size

    def _build_view_matrix(self, basis):
        """World -> view/camera space, using this widget's own (eye, fwd,
        right, up) orbit basis directly as the view frame -- (right, up,
        -fwd) is right-handed (cross(right, up) == -fwd here), matching
        standard OpenGL view space where the camera looks down -Z."""
        eye, fwd, right, up = basis
        V = np.eye(4, dtype=np.float64)
        V[0, :3] = right
        V[1, :3] = up
        V[2, :3] = -fwd
        V[0, 3] = -np.dot(right, eye)
        V[1, 3] = -np.dot(up, eye)
        V[2, 3] = np.dot(fwd, eye)
        return V

    def _build_proj_matrix(self, W, H):
        """Perspective projection matched exactly to this widget's existing
        pinhole formula (screen_x = W/2 + f*x/z, screen_y = H/2 - f*y/z
        with f = min(W,H)*0.8, used throughout for the grid/gizmo/trail),
        so the GPU-rendered mesh lines up pixel-for-pixel with the
        QPainter-drawn overlays. Only the Z row is the standard OpenGL
        near/far depth mapping -- the CPU code has no equivalent to match
        since it never used a depth buffer."""
        foc = min(W, H) * 0.8
        near, far = self._near, self._far
        P = np.zeros((4, 4), dtype=np.float64)
        P[0, 0] = 2.0 * foc / W
        P[1, 1] = 2.0 * foc / H
        P[2, 2] = -(far + near) / (far - near)
        P[2, 3] = -2.0 * far * near / (far - near)
        P[3, 2] = -1.0
        return P

    def _gl_draw_mesh(self, basis):
        """Issue the native-GL mesh draw. Must be called between a
        QPainter's beginNativePainting()/endNativePainting() (see
        paintGL) so it composites correctly with the QPainter-drawn
        background/grid/overlays sharing the same paint."""
        context = QOpenGLContext.currentContext()
        if context is None or not context.isValid():
            return
        self._ensure_gl_ready()
        self._gl_sync_mesh()
        if self._gl_index_count == 0:
            return

        W, H = self.width(), self.height()
        dpr = self.devicePixelRatioF()
        gl.glViewport(0, 0, max(1, int(W * dpr)), max(1, int(H * dpr)))
        gl.glClear(gl.GL_DEPTH_BUFFER_BIT)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthFunc(gl.GL_LESS)
        # No backface cull -- double-sided, like Maya/Blender's viewport
        # (see the fragment shader for why: LiDAR mesh winding is unreliable).
        gl.glDisable(gl.GL_CULL_FACE)

        view = self._build_view_matrix(basis)
        proj = self._build_proj_matrix(W, H)
        mvp = proj @ view

        gl.glUseProgram(self._gl_prog)
        # glUniformMatrix4fv expects column-major data when transpose=False;
        # these numpy matrices are built row-major (math convention), so
        # transpose before upload instead of relying on driver support for
        # transpose=True.
        gl.glUniformMatrix4fv(self._gl_u_mvp, 1, gl.GL_FALSE,
                               np.ascontiguousarray(mvp.T, dtype=np.float32))
        gl.glUniformMatrix4fv(self._gl_u_view, 1, gl.GL_FALSE,
                               np.ascontiguousarray(view.T, dtype=np.float32))

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._gl_vbo)
        gl.glEnableVertexAttribArray(0)
        # This is a VBO byte offset, not a Python array. The high-level
        # wrapper tries to retain it in a GLX context cache, which fails when
        # Qt owns an EGL context (including offscreen rendering).
        _raw_vertex_attrib_pointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, ctypes.c_void_p(0))
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self._gl_ebo)
        gl.glDrawElements(gl.GL_TRIANGLES, self._gl_index_count, gl.GL_UNSIGNED_INT, ctypes.c_void_p(0))
        gl.glDisableVertexAttribArray(0)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)
        gl.glUseProgram(0)

        # Leave depth testing off for the QPainter 2D drawing (trail, HUD
        # text, camera dot) that resumes right after endNativePainting().
        gl.glDisable(gl.GL_DEPTH_TEST)

    # -- Painting --

    def paintGL(self):
        # Keep QOpenGLWidget.paintEvent intact: it owns framebuffer
        # recreation and composition during maximize/restore. Native GL
        # mixed with QPainter belongs here, where Qt binds this widget's
        # context and its current framebuffer before invoking us.
        # https://doc.qt.io/archives/qt-5.15/qopenglwidget.html#painting-techniques
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        W, H = self.width(), self.height()
        basis = self._eye_basis()

        # Unreal Engine-style dark gradient background
        from PyQt5.QtGui import QLinearGradient
        grad = QLinearGradient(0, 0, 0, H)
        grad.setColorAt(0.0, QColor(18, 18, 30))
        grad.setColorAt(0.5, QColor(12, 12, 22))
        grad.setColorAt(1.0, QColor(8, 8, 16))
        p.fillRect(0, 0, W, H, grad)

        # Camera overlay
        if self._show_overlay and self._overlay_pix is not None:
            scaled = self._overlay_pix.scaled(W, H, Qt.KeepAspectRatio, Qt.FastTransformation)
            ox = (W - scaled.width()) // 2
            oy = (H - scaled.height()) // 2
            p.setOpacity(0.35)
            p.drawPixmap(ox, oy, scaled)
            p.setOpacity(1.0)

        # Maya-style multi-level grid
        self._draw_grid(p, basis)

        # World axes gizmo
        axis_len = max(self._orb_dist * 0.15, 0.5)
        for seg, col, label in [
            ((np.array([0., 0., 0.]), np.array([axis_len, 0., 0.])), '#ff3333', 'X'),
            ((np.array([0., 0., 0.]), np.array([0., axis_len, 0.])), '#33ff33', 'Y'),
            ((np.array([0., 0., 0.]), np.array([0., 0., axis_len])), '#3333ff', 'Z'),
        ]:
            ends = self.display_points('world', seg)
            scr = self._project_segment(ends[0], ends[1], basis)
            if scr is not None:
                p.setPen(QPen(QColor(col), 2))
                p.drawLine(QPointF(scr[0, 0], scr[0, 1]), QPointF(scr[1, 0], scr[1, 1]))
                p.setFont(QFont('Arial', 10, QFont.Bold))
                p.drawText(QPointF(scr[1, 0] + 4, scr[1, 1] - 4), label)

        with self._lock:
            pts = self._pts
            mesh_faces = self._mesh_faces
            trail = list(self._trail)
            cam = self._cam_pos.copy()

        # Mesh (imported .obj/.stl/etc, rendered as shaded faces via real
        # OpenGL -- every polygon, every frame, depth-correct).
        if mesh_faces is not None and len(mesh_faces) > 0:
            p.beginNativePainting()
            self._gl_draw_mesh(basis)
            p.endNativePainting()

        # Points (plain point-cloud imports, e.g. .ply/.pcd with no faces)
        elif pts is not None and len(pts) > 0:
            subset = pts[::max(1, len(pts) // 8000)]
            scr, ok = self._project_pts(subset, basis)
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(QColor(80, 200, 255, 90)))
            for i in range(len(subset)):
                if ok[i] and 0 <= scr[i, 0] <= W and 0 <= scr[i, 1] <= H:
                    p.drawEllipse(QPointF(scr[i, 0], scr[i, 1]), 1.5, 1.5)

        # Trail (per-segment clipped so it doesn't vanish crossing near the
        # camera). Batched: TRAIL_MAX is 2000, and a Python loop of
        # _project_segment over that is 26 ms a frame on its own.
        if len(trail) > 1:
            lines = self._polyline_lines(self.display_points('trail', trail), basis)
            if lines:
                p.setPen(QPen(QColor('#ff6600'), 1.5))
                p.drawLines(lines)

        self._draw_marker_target(p, basis)

        # Locator ("tracking origin" -- every ArUco anchor snaps here)
        if self.gizmo_active:
            self._draw_gizmo(p, basis)

        # Camera dot (glow + solid core)
        scr, ok = self._project_pts(cam.reshape(1, 3), basis)
        if ok[0]:
            sx, sy = scr[0]
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(QColor(255, 50, 50, 40)))
            p.drawEllipse(QPointF(sx, sy), 18, 18)
            p.setBrush(QBrush(QColor(255, 50, 50, 80)))
            p.drawEllipse(QPointF(sx, sy), 10, 10)
            p.setBrush(QBrush(QColor('#ff4444')))
            p.drawEllipse(QPointF(sx, sy), 5, 5)

        # Nav hint
        p.setPen(QColor(80, 80, 100))
        p.setFont(QFont('Arial', 8))
        p.drawText(6, H - 8,
                   "LMB: Orbit  |  MMB: Pan  |  RMB: Look  |  W/E: Move or Rotate gizmo  |  "
                   "RMB+WASD: Fly  |  Scroll: Zoom (RMB+Scroll: Fly Speed)  |  F: Focus")

        # Grid scale indicator
        base = 10 ** math.floor(math.log10(max(self._orb_dist * 0.3, 0.01)))
        unit = f"{base:.0f}m" if base >= 1 else f"{base*100:.0f}cm"
        p.setPen(QColor(100, 100, 130))
        p.setFont(QFont('Arial', 9))
        p.drawText(6, 16, f"Grid: {unit}")

        if self._drag_btn == Qt.RightButton:
            p.setPen(QColor(0, 212, 255))
            p.setFont(QFont('Arial', 9, QFont.Bold))
            p.drawText(6, 32, f"Fly speed: {self._fly_speed:.2f} m/s")

        p.end()

    def _draw_gizmo(self, p, basis):
        """The active manipulator, drawn over the mesh (depth test is off by
        the time this runs).

        One mode at a time: translate draws the X/Y/Z arrows and the planar
        centre handle, rotate draws the three world-aligned rings. Never both
        -- that was the old behaviour and it made a click near the centre
        ambiguous between moving and turning.
        """
        pts, axis_len = self._locator_gizmo_world_points()
        o_scr, o_ok = self._project_pts(pts['origin'].reshape(1, 3), basis)
        if not o_ok[0]:
            return
        ox, oy = o_scr[0]
        tint = self._gizmo_tint()
        selected = self._gizmo_is_selected()

        if self._gizmo_mode == 'rotate':
            self._draw_rotate_gizmo(p, basis, ox, oy, tint, selected)
        else:
            self._draw_translate_gizmo(p, basis, pts, ox, oy, tint, selected)

        p.setPen(QColor(tint[0], tint[1], tint[2]))
        p.setFont(QFont('Arial', 9, QFont.Bold))
        p.drawText(QPointF(ox + 10, oy - 10),
                   f"{self._gizmo_label()}  [{self._gizmo_mode.upper()}]")

    def _draw_rotate_gizmo(self, p, basis, ox, oy, tint, selected):
        for key, idx, _axis, col in self.ROT_RINGS:
            ring = self._rotation_ring_points(idx)
            scr, ok = self._project_pts(ring, basis)
            active = self._active_gizmo == key
            colour = QColor('#ffffff') if active else QColor(col)
            p.setPen(QPen(colour, 4 if active else (3 if selected else 2)))
            n = len(ring)
            for i in range(n):
                j = (i + 1) % n
                if ok[i] and ok[j]:
                    p.drawLine(QPointF(scr[i, 0], scr[i, 1]),
                               QPointF(scr[j, 0], scr[j, 1]))
        # A short spoke showing the current heading, so a yaw is legible even
        # looking straight down the Z ring.
        _, axis_len = self._locator_gizmo_world_points()
        yaw = math.radians(self._gizmo_yaw())
        spoke = (self._gizmo_pos()
                 + axis_len * 0.8 * np.array([math.cos(yaw), math.sin(yaw), 0.]))
        sscr, sok = self._project_pts(spoke.reshape(1, 3), basis)
        if sok[0]:
            p.setPen(QPen(QColor('#ffee88'), 2))
            p.drawLine(QPointF(ox, oy), QPointF(sscr[0, 0], sscr[0, 1]))
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(QColor(tint[0], tint[1], tint[2], 220)))
        p.drawEllipse(QPointF(ox, oy), 5, 5)

    def _draw_translate_gizmo(self, p, basis, pts, ox, oy, tint, selected):

        # Translate arrows (X/Y/Z)
        for key, tip_world, col in (('x', pts['x'], '#ff5555'),
                                    ('y', pts['y'], '#55ff55'),
                                    ('z', pts['z'], '#5599ff')):
            tscr, tok = self._project_pts(tip_world.reshape(1, 3), basis)
            if not tok[0]:
                continue
            active = self._active_gizmo == key
            width = 4 if active else (3 if selected else 2)
            color = QColor('#ffffff') if active else QColor(col)
            p.setPen(QPen(color, width))
            p.drawLine(QPointF(ox, oy), QPointF(tscr[0, 0], tscr[0, 1]))
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(color))
            p.drawEllipse(QPointF(tscr[0, 0], tscr[0, 1]), 5, 5)

        # Centre / planar-move handle
        plane_active = self._active_gizmo == 'plane_xy'
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(QColor('#ffffff') if plane_active
                          else QColor(tint[0], tint[1], tint[2], 220)))
        p.drawEllipse(QPointF(ox, oy), 7 if selected else 5,
                      7 if selected else 5)

    # -- Locator picking + dragging --

    def _pick_gizmo_handle(self, pos):
        """Nearest locator gizmo handle within LOCATOR_PICK_PX of `pos`, or
        None. Checked on every LMB press regardless of outliner selection --
        clicking a handle auto-selects the locator, like picking any object."""
        basis = self._eye_basis()
        pts, _axis_len = self._locator_gizmo_world_points()
        best, best_d = None, float(self.LOCATOR_PICK_PX)

        if self._gizmo_mode == 'rotate':
            # Rings are sampled: a ring is not a single point, and which part
            # of it is nearest the cursor depends on the view.
            for key, idx, _axis, _col in self.ROT_RINGS:
                ring = self._rotation_ring_points(idx, n=64)
                scr, ok = self._project_pts(ring, basis)
                for i in range(len(ring)):
                    if not ok[i]:
                        continue
                    d = math.hypot(scr[i, 0] - pos.x(), scr[i, 1] - pos.y())
                    if d < best_d:
                        best_d, best = d, key
            return best

        world_pts = np.array([pts['x'], pts['y'], pts['z'], pts['origin']])
        labels = ['x', 'y', 'z', 'plane_xy']
        scr, ok = self._project_pts(world_pts, basis)
        for i, lbl in enumerate(labels):
            if not ok[i]:
                continue
            d = math.hypot(scr[i, 0] - pos.x(), scr[i, 1] - pos.y())
            if d < best_d:
                best_d, best = d, lbl
        return best

    def _axis_screen_dir(self, axis_char, basis):
        """Unit screen-space direction of a small step along a world axis at
        the locator's current position, plus the resulting pixels-per-world-
        unit scale. None if the locator isn't currently on screen."""
        unit = {'x': np.array([1., 0., 0.]),
                'y': np.array([0., 1., 0.]),
                'z': np.array([0., 0., 1.])}[axis_char]
        step = max(self._orb_dist * 0.05, 0.05)
        # At the gizmo's own position -- this used to measure at the locator
        # whichever object was selected, so dragging the camera or the scan
        # scaled by the wrong point whenever they were apart.
        origin = np.asarray(self._gizmo_pos(), dtype=float)
        p0, ok0 = self._project_pts(origin.reshape(1, 3), basis)
        p1, ok1 = self._project_pts((origin + unit * step).reshape(1, 3), basis)
        if not (ok0[0] and ok1[0]):
            return None, 0.0
        d = p1[0] - p0[0]
        n = float(np.linalg.norm(d))
        if n < 1e-6:
            return None, 0.0
        return d / n, n / step

    def _begin_locator_drag(self, kind, screen_pos):
        basis = self._eye_basis()
        self._active_gizmo = kind
        self._gizmo_drag_screen0 = screen_pos
        self._gizmo_drag_pos0 = np.asarray(self._gizmo_pos(), dtype=float).copy()
        self._gizmo_drag_rpy0 = self._gizmo_rpy().copy()

        if kind in ('x', 'y', 'z'):
            self._gizmo_drag_axis_dir, self._gizmo_drag_px_per_unit = \
                self._axis_screen_dir(kind, basis)
        elif kind == 'plane_xy':
            dx_dir, dx_ppu = self._axis_screen_dir('x', basis)
            dy_dir, dy_ppu = self._axis_screen_dir('y', basis)
            self._gizmo_drag_plane_Jinv = None
            if dx_dir is not None and dy_dir is not None:
                J = np.array([[dx_dir[0] * dx_ppu, dy_dir[0] * dy_ppu],
                              [dx_dir[1] * dx_ppu, dy_dir[1] * dy_ppu]])
                if abs(np.linalg.det(J)) > 1e-9:
                    self._gizmo_drag_plane_Jinv = np.linalg.inv(J)
        elif kind in ('rot_x', 'rot_y', 'rot_z'):
            o_scr, o_ok = self._project_pts(
                np.asarray(self._gizmo_pos(), dtype=float).reshape(1, 3), basis)
            self._gizmo_drag_origin_scr = o_scr[0] if o_ok[0] else None
            # Which way the ring turns on screen depends on whether its axis
            # points towards the viewer or away: without this, half the rings
            # would follow the cursor and half would run against it.
            _eye, fwd, _r, _u = basis
            axis = dict((k, a) for k, _i, a, _c in self.ROT_RINGS)[kind]
            self._gizmo_drag_rot_sign = 1.0 if float(np.dot(axis, fwd)) > 0 else -1.0
            self._gizmo_drag_rot_idx = dict(
                (k, i) for k, i, _a, _c in self.ROT_RINGS)[kind]

    def _drag_locator(self, screen_pos):
        dx = screen_pos.x() - self._gizmo_drag_screen0.x()
        dy = screen_pos.y() - self._gizmo_drag_screen0.y()

        if self._active_gizmo in ('x', 'y', 'z'):
            dirn, ppu = self._gizmo_drag_axis_dir, self._gizmo_drag_px_per_unit
            if dirn is not None and ppu > 1e-6:
                delta = float(dx * dirn[0] + dy * dirn[1]) / ppu
                new_pos = self._gizmo_drag_pos0.copy()
                idx = {'x': 0, 'y': 1, 'z': 2}[self._active_gizmo]
                new_pos[idx] += delta
                self._gizmo_set_pos(new_pos)
                self._gizmo_changed()

        elif self._active_gizmo == 'plane_xy':
            if self._gizmo_drag_plane_Jinv is not None:
                d_world = self._gizmo_drag_plane_Jinv @ np.array([dx, dy])
                new_pos = self._gizmo_drag_pos0.copy()
                new_pos[0] += float(d_world[0])
                new_pos[1] += float(d_world[1])
                self._gizmo_set_pos(new_pos)
                self._gizmo_changed()

        elif self._active_gizmo in ('rot_x', 'rot_y', 'rot_z'):
            if self._gizmo_drag_origin_scr is not None:
                ox, oy = self._gizmo_drag_origin_scr
                ang0 = math.atan2(self._gizmo_drag_screen0.y() - oy,
                                  self._gizmo_drag_screen0.x() - ox)
                ang1 = math.atan2(screen_pos.y() - oy, screen_pos.x() - ox)
                delta_deg = (math.degrees(ang1 - ang0)
                             * self._gizmo_drag_rot_sign)
                idx = self._gizmo_drag_rot_idx
                rpy = self._gizmo_drag_rpy0.copy()
                # Wrapped, not accumulated: angles are periodic, and letting
                # one run past +/-180 puts it outside what the alignment
                # boxes can hold.
                rpy[idx] = ((rpy[idx] + delta_deg + 180.0) % 360.0) - 180.0
                self._gizmo_set_rpy(rpy)
                self._gizmo_changed()

        self.update()

    # -- Unreal Engine-style navigation --

    def wheelEvent(self, e):
        delta = e.angleDelta().y()
        if self._drag_btn == Qt.RightButton:
            # Unreal Engine behaviour: scrolling while flying (RMB held)
            # changes fly speed instead of the orbit dolly distance.
            factor = 1.15 if delta > 0 else 1 / 1.15
            self._fly_speed = max(0.05, min(50.0, self._fly_speed * factor))
        else:
            factor = 0.85 if delta > 0 else 1.18
            self._orb_dist = max(0.001, self._orb_dist * factor)
        self.update()

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            hit = self._pick_gizmo_handle(e.pos()) if self.gizmo_active else None
            if hit is not None:
                # Keep the outliner on whatever the gizmo belongs to, then
                # start the constrained drag -- no orbit. This used to select
                # the Locator row unconditionally, so grabbing *any* handle
                # switched _gizmo_target to the locator: dragging the camera's
                # gizmo moved the locator instead, and the outliner silently
                # jumped to "Locator" so the next drag was on the wrong object.
                self._select_gizmo_row(self._gizmo_target)
                self._begin_locator_drag(hit, e.pos())
                self.update()
                return

        self._drag_start = e.pos()
        self._drag_btn = e.button()
        self._last_global = e.globalPos()
        if e.button() in (Qt.LeftButton, Qt.MiddleButton, Qt.RightButton):
            self.setCursor(Qt.BlankCursor)
        if e.button() == Qt.RightButton:
            self._fly_timer.start()

    def mouseMoveEvent(self, e):
        if self._active_gizmo is not None:
            self._drag_locator(e.pos())
            return
        if self._drag_btn is None:
            return
        gpos = e.globalPos()
        dx = gpos.x() - self._last_global.x()
        dy = gpos.y() - self._last_global.y()
        if dx == 0 and dy == 0:
            return  # synthetic event from the cursor re-center below

        if self._drag_btn == Qt.LeftButton:
            self._orb_yaw += dx * 0.4
            self._orb_pitch = max(-89, min(89, self._orb_pitch - dy * 0.4))
        elif self._drag_btn == Qt.MiddleButton:
            _, _, right, up = self._eye_basis()
            speed = self._orb_dist * 0.003
            self._orb_center -= right * dx * speed
            self._orb_center -= up * dy * speed
        elif self._drag_btn == Qt.RightButton:
            self._orb_yaw += dx * 0.3
            self._orb_pitch = max(-89, min(89, self._orb_pitch - dy * 0.3))

        # Re-center the cursor after every move so orbit/pan/look never run
        # out of screen room — this is what makes navigation feel infinite
        # instead of clamping at the monitor edge, matching Maya/UE viewports.
        center_global = self.mapToGlobal(self.rect().center())
        QCursor.setPos(center_global)
        self._last_global = center_global
        self.update()

    def mouseReleaseEvent(self, e):
        if self._active_gizmo is not None:
            self._active_gizmo = None
            self._gizmo_drag_screen0 = None
            self.update()
            return
        self._drag_start = None
        self._drag_btn = None
        self.setCursor(Qt.ArrowCursor)
        self._fly_timer.stop()

    def keyPressEvent(self, e):
        key = e.key()
        if key == Qt.Key_F:
            self._center_on_selection()
            return
        # Del removes the selected object, as in any DCC outliner. The scan is
        # the only entry that can be deleted -- the camera, trail, origin and
        # locator are the tracking session itself, not scene content.
        if key in (Qt.Key_Delete, Qt.Key_Backspace):
            if self._gizmo_target == 'mesh':
                self.delete_requested.emit()
            return
        # W/E switch manipulator, like Unreal -- but only when not flying:
        # with RMB held they are fly controls (forward / up) and must stay so.
        if self._drag_btn != Qt.RightButton:
            if key == Qt.Key_W:
                self.set_gizmo_mode('translate')
                return
            if key == Qt.Key_E:
                self.set_gizmo_mode('rotate')
                return
        self._keys_held.add(key)

    def keyReleaseEvent(self, e):
        self._keys_held.discard(e.key())

    def _fly_tick(self):
        """WASD+QE fly-through while RMB is held (Unreal Engine style)."""
        if not self._keys_held:
            return
        _, fwd, right, up = self._eye_basis()
        speed = self._fly_speed * self._orb_dist * 0.02
        move = np.array([0., 0., 0.])
        if Qt.Key_W in self._keys_held: move += fwd * speed
        if Qt.Key_S in self._keys_held: move -= fwd * speed
        if Qt.Key_D in self._keys_held: move += right * speed
        if Qt.Key_A in self._keys_held: move -= right * speed
        if Qt.Key_Q in self._keys_held: move -= np.array([0., 0., 1.]) * speed
        if Qt.Key_E in self._keys_held: move += np.array([0., 0., 1.]) * speed
        if Qt.Key_Shift in self._keys_held:
            move *= 3.0
        self._orb_center += move
        self.update()


# ─────────────────────────────────────────────────
# Main widget
# ─────────────────────────────────────────────────

class MapViewerWidget(QWidget):
    alignment_changed = pyqtSignal(np.ndarray)
    file_loaded = pyqtSignal(str)
    locator_changed = pyqtSignal(np.ndarray)   # 4x4, ROS world frame
    camera_moved = pyqtSignal(np.ndarray)      # 4x4, ROS world frame
    delete_requested = pyqtSignal()            # operator asked to delete the scan
    TRAIL_MAX = 2000

    #: Rate at which camera pose is pushed into the 3D view. Every push
    #: repaints it, and each repaint costs milliseconds of GUI thread. The
    #: UI timer runs at 60 Hz because the FreeD stream needs that cadence --
    #: a monitoring viewport does not, and painting views 60 times a second
    #: is most of what saturated the GUI thread and left the dashboard unable
    #: to answer the window manager.
    VIEW_REFRESH_HZ = 30

    #: A pose closer than this to the last trail point adds nothing to look
    #: at but costs a segment on every frame afterwards. Held
    #: still, the camera otherwise fills all TRAIL_MAX points with the same
    #: position -- 2000 segments drawn on top of each other.
    TRAIL_MIN_STEP_M = 0.002

    #: Bounds the alignment is kept inside. These were the ranges of the
    #: toolbar spin boxes; the boxes are gone but the clamps still are the
    #: only thing standing between a runaway gizmo drag and a scan placed a
    #: thousand kilometres from the stage.
    TRANSLATION_LIMIT_M = 1_000_000.0
    SCALE_MIN = 0.0001
    SCALE_MAX = 100.0

    def __init__(self):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.tx = self.ty = self.tz = 0.0
        self.roll = self.pitch = self.yaw = 0.0
        self.scale = 1.0
        self.pose_stored = False
        self.map_points_3d = None
        self.map_mesh_faces = None
        self.cam_pose = np.eye(4)
        self.camera_K = None
        self.latest_frame = None
        self.T_align = np.eye(4)
        self._trail = collections.deque(maxlen=self.TRAIL_MAX)
        self._trail_dirty = False
        self._last_view_push = 0.0
        self._aligned_cache = None
        self._build_ui()
        self._update_transform()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(2)

        self.persp_view = PerspView()
        self.persp_view.locator_changed.connect(self._on_persp_locator_changed)
        self.persp_view.camera_moved.connect(self._on_persp_camera_moved)
        self.persp_view.mesh_moved.connect(self._on_persp_mesh_moved)
        self.persp_view.trail_changed.connect(self._push_trail)
        self.persp_view.delete_requested.connect(self.delete_requested)

        # Overlay toggle
        bot = QHBoxLayout()
        self.btn_overlay = QPushButton("📷 Overlay: OFF")
        self.btn_overlay.setCheckable(True)
        self.btn_overlay.setStyleSheet("padding:3px;font-weight:bold;background:#222233;color:#888;")
        self.btn_overlay.clicked.connect(self._toggle_overlay)
        bot.addWidget(self.btn_overlay)
        bot.addStretch()

        root.addWidget(self.persp_view, stretch=1)
        root.addLayout(bot)

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
        """Lock the alignment: the scan gizmo stops editing it."""
        self.pose_stored = True
        self.persp_view.set_mesh_edit_locked(True)

    def _reset_pose(self):
        """Unlock the alignment and drop the trail drawn under the old one."""
        self.pose_stored = False
        self.persp_view.set_mesh_edit_locked(False)
        self._trail.clear()
        self._push_trail()

    def import_map(self):
        """Open the file dialog and emit `file_loaded` for the chosen scan.

        Reached from the dashboard's File > Import; the toolbar button that
        used to call it is gone, but this is the same dialog and the same
        signal, so the import path behind it is unchanged.
        """
        import os
        from PyQt5.QtWidgets import QComboBox, QMessageBox, QFileDialog
        dialog = QFileDialog(self, "Import LiDAR Map", "", "3D Files (*.obj *.ply *.pcd);;All (*)")
        dialog.setLabelText(QFileDialog.Accept, "📥 Import")
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
                self.file_loaded.emit(files[0])

    #: Former toolbar handler name, kept so existing callers keep working.
    _browse_file = import_map

    def load_path(self, path):
        """Import a scan by path (same route as the file dialog)."""
        path = str(path).strip()
        if path:
            self.file_loaded.emit(path)

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
        self._sync_mesh_gizmo()

    def set_alignment_scale(self, scale):
        """Set the mesh-alignment scale (metres per map-file unit) in code.

        Scale is how a centimetre scan gets reconciled with a metre world, and
        it starts at 1.0. `lidar.map_scale` in tracking.yaml and the extent
        auto-detection both come through here, so the viewer, the matcher and
        the config agree on one number -- a 309 m room matched against a 3 m
        depth cloud is what a disagreement used to look like.

        Clamped to the range the alignment is meaningful over: a scale of zero
        collapses the scan to a point, and a negative one mirrors it.
        """
        scale = float(scale)
        if scale <= 0.0:
            return False
        scale = min(max(scale, self.SCALE_MIN), self.SCALE_MAX)
        if abs(scale - self.scale) < 1e-12:
            return False
        self.scale = scale
        self._update_transform()
        return True

    def set_alignment(self, tx, ty, tz, roll, pitch, yaw, scale, stored=False):
        """Put the scan where a saved scene had it (File > Open Scene).

        One transform update for the lot, so the fusion is rebased once
        rather than once per channel. Clamped, wrapped and rounded the way
        a drag is.
        """
        scale = float(scale)
        if not scale > 0.0:
            raise ValueError("The alignment scale must be positive.")
        self.tx, self.ty, self.tz = float(tx), float(ty), float(tz)
        self.roll, self.pitch, self.yaw = (((float(a) + 180.0) % 360.0) - 180.0
                                           for a in (roll, pitch, yaw))
        self.scale = min(max(scale, self.SCALE_MIN), self.SCALE_MAX)
        self._quantise_alignment()
        self._update_transform()
        self.pose_stored = bool(stored)
        self.persp_view.set_mesh_edit_locked(self.pose_stored)

    @property
    def alignment_scale(self):
        return self.scale

    def set_map_points(self, points):
        self.map_points_3d = points.copy()
        self.map_mesh_faces = None
        self._aligned_cache = None
        self._redraw_map()
        self._sync_mesh_gizmo()
        self._select_scan()

    def set_map_mesh(self, vertices, faces):
        """Load a mesh as-is (vertices + triangle faces) instead of a
        sampled point cloud -- the 3D viewer renders actual shaded faces."""
        self.map_points_3d = vertices.copy()
        self.map_mesh_faces = faces.copy()
        self._aligned_cache = None
        self._redraw_map()
        self._sync_mesh_gizmo()
        self._select_scan()

    def _select_scan(self):
        """Hand the manipulator to the scan that was just loaded.

        The gizmo starts on the tracking origin, so a freshly imported scan
        arrived with the manipulator pointed at something else: dragging it
        moved the locator and left the scan sitting where the file put it,
        which reads as "the scan cannot be moved". Placing a scan is the first
        thing anyone does after importing one, so that is what the manipulator
        selects.
        """
        self.persp_view.select_scan()

    # -- Locator (tracking origin) --

    def _aligned_centroid(self):
        """Centre of the scan as displayed, or the alignment's own origin."""
        pts = self._get_aligned()
        if pts is None or len(pts) == 0:
            return np.array([self.tx, self.ty, self.tz], dtype=float)
        return np.asarray(pts, dtype=float).mean(axis=0)

    def _sync_mesh_gizmo(self):
        """Put the scan's gizmo on the scan, wherever the alignment has it."""
        self.persp_view.set_mesh_gizmo(self._aligned_centroid(),
                                       (self.roll, self.pitch, self.yaw))

    def _on_persp_mesh_moved(self):
        """The operator dragged the scan itself in the 3D view.

        The gizmo rides on the scan's centroid, but what actually gets edited
        is the mesh-alignment transform behind the X/Y/Z/Yaw boxes -- so a drag
        is turned into a delta on the alignment's own translation and
        rotation, which stay the single source of truth for the scan's place.
        """
        if self.pose_stored:
            self._sync_mesh_gizmo()      # alignment locked; snap the gizmo back
            return
        from pose_transform import rpy_to_matrix
        target, target_rpy = self.persp_view.get_mesh_gizmo_rpy()
        centroid = self._aligned_centroid()          # under the current T_align

        # Angles are periodic, so wrap into +/-180 rather than letting them
        # clamp -- a clamp at 180 deg would leave the rotation
        # and the pivot translation describing different things, and the scan
        # would swing off across the room.
        old = np.array([self.roll, self.pitch, self.yaw], dtype=float)
        new = ((np.asarray(target_rpy, dtype=float) + 180.0) % 360.0) - 180.0
        self.roll, self.pitch, self.yaw = (float(new[0]), float(new[1]),
                                           float(new[2]))
        if np.any(np.abs(new - old) > 1e-9):
            # Turn about the centroid so the scan spins in place instead of
            # swinging around the world origin; the centroid stays put, so the
            # translation delta below still measures from the right point.
            # The delta rotation is what actually gets applied -- composing
            # the new orientation onto the old one directly would double it.
            R_old = rpy_to_matrix(*np.radians(old))
            R_new = rpy_to_matrix(*np.radians(new))
            R_delta = R_new @ R_old.T
            t = R_delta @ (np.array([self.tx, self.ty, self.tz]) - centroid) + centroid
            self.tx, self.ty, self.tz = float(t[0]), float(t[1]), float(t[2])

        delta = np.asarray(target, dtype=float) - centroid
        self.tx += float(delta[0])
        self.ty += float(delta[1])
        self.tz += float(delta[2])

        self._quantise_alignment()        # rounds to the precision we record
        self._update_transform()          # ... then builds T_align from that
        self._sync_mesh_gizmo()

    def _quantise_alignment(self):
        """Round the alignment to the precision the scene file records.

        A drag produces full float precision; what gets saved and reported is
        0.1 mm of translation and a thousandth of a degree. Rounding here means
        the transform matches what a saved scene would restore, rather than
        drifting by a rounding step the first time one is loaded back.
        """
        lim = self.TRANSLATION_LIMIT_M
        self.tx = min(max(round(self.tx, 4), -lim), lim)
        self.ty = min(max(round(self.ty, 4), -lim), lim)
        self.tz = min(max(round(self.tz, 4), -lim), lim)
        self.roll = round(self.roll, 3)
        self.pitch = round(self.pitch, 3)
        self.yaw = round(self.yaw, 3)

    def _on_persp_camera_moved(self):
        """The operator dragged the camera dot in the 3D view."""
        from pose_transform import rpy_to_matrix
        pos, rpy = self.persp_view.get_camera_gizmo_rpy()
        T = np.eye(4)
        T[:3, :3] = rpy_to_matrix(*np.radians(rpy))
        T[:3, 3] = pos
        self.cam_pose = T.copy()
        self.camera_moved.emit(T)

    def _on_persp_locator_changed(self):
        """User dragged the locator gizmo in the 3D viewport."""
        self.locator_changed.emit(self.persp_view.get_locator_transform())

    def clear_map(self):
        """Delete the loaded scan from the view.

        The alignment is reset with it: an alignment is a statement about
        where a particular scan sits, so keeping the numbers after the scan is
        gone would silently apply them to whatever gets imported next.
        """
        self.map_points_3d = None
        self.map_mesh_faces = None
        self._aligned_cache = None
        self.tx = self.ty = self.tz = 0.0
        self.roll = self.pitch = self.yaw = 0.0
        self.scale = 1.0
        self.pose_stored = False
        self.persp_view.set_mesh_edit_locked(False)
        self.persp_view.clear_map()
        # Hand the manipulator back to the tracking origin: leaving it on a
        # scan that no longer exists would draw a gizmo for nothing.
        self.persp_view.select_locator()
        self._update_transform()      # emits the identity alignment downstream

    def has_map(self):
        """True while a scan is loaded."""
        return self.map_points_3d is not None and len(self.map_points_3d) > 0

    def get_locator_transform(self, locator_id=None):
        return self.persp_view.get_locator_transform(locator_id)

    def set_locator_transform(self, T, locator_id=None):
        """Programmatic/external set (does not emit locator_changed)."""
        self.persp_view.set_locator_transform(T, locator_id)

    def set_camera_intrinsics(self, K):
        self.camera_K = K

    def update_camera_frame(self, frame_bgr):
        self.latest_frame = frame_bgr
        self.persp_view.set_overlay(frame_bgr)

    def update_camera_pose(self, position, quaternion):
        """Take a new camera pose from the dashboard's 60 Hz refresh.

        The pose itself is kept every time it arrives; what is rate-limited
        is *pushing it into the view*, because each push repaints it
        and that is what the GUI thread cannot afford sixty times a second.
        """
        from pose_transform import make_transform
        self.cam_pose = make_transform(position, quaternion)
        px, py, pz = float(position[0]), float(position[1]), float(position[2])

        last = self._trail[-1] if self._trail else None
        if last is None or ((px - last[0]) ** 2 + (py - last[1]) ** 2
                            + (pz - last[2]) ** 2) >= self.TRAIL_MIN_STEP_M ** 2:
            self._trail.append((px, py, pz))
            self._trail_dirty = True

        now = time.monotonic()
        if now - self._last_view_push < 1.0 / self.VIEW_REFRESH_HZ:
            return
        self._last_view_push = now

        from pose_transform import matrix_to_rpy
        rpy_deg = np.degrees(matrix_to_rpy(self.cam_pose[:3, :3]))
        self.persp_view.set_cam((px, py, pz), rpy_deg=rpy_deg)
        if self._trail_dirty:
            self._push_trail()

    def _push_trail(self):
        """Hand the trail to the view. Split out so clearing it reaches
        the view too -- otherwise a reset leaves the old path drawn."""
        self._trail_dirty = False
        self.persp_view.set_trail(list(self._trail))

    def _get_aligned(self):
        if self.map_points_3d is None: return None
        if self._aligned_cache is None:
            hom = np.column_stack((self.map_points_3d, np.ones(len(self.map_points_3d))))
            self._aligned_cache = (self.T_align @ hom.T).T[:, :3]
        return self._aligned_cache

    def _redraw_map(self):
        pts = self._get_aligned()
        if pts is None: return
        if self.map_mesh_faces is not None:
            self.persp_view.set_mesh(pts, self.map_mesh_faces)
        else:
            self.persp_view.set_points(pts)
