"""A lightweight survey-point lineup workspace, inspired by matchmove tools."""
import logging
import numpy as np
from PyQt5.QtCore import Qt, QRectF, QPointF, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QPainter, QPen
from PyQt5.QtWidgets import (QDialog, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QSplitter, QTableWidget, QTableWidgetItem, QHeaderView,
    QCheckBox, QAbstractItemView)
from map_viewer import PerspView
from lineup_solver import pick_mesh_surface, solve_lineup


logger = logging.getLogger(__name__)


class PlateView(QWidget):
    picked = pyqtSignal(object)

    def __init__(self, gray):
        super().__init__()
        self.gray = np.ascontiguousarray(gray)
        self.canvas = self.gray
        self.points = []
        self.projected = None
        self.zoom = 1.0
        self.pan = np.zeros(2)
        self._drag = None
        self.setMinimumSize(400, 280)
        self.setMouseTracking(True)

    def image_rect(self):
        h,w = self.gray.shape[:2]
        scale = min(self.width()/w, self.height()/h) * self.zoom
        return QRectF((self.width()-w*scale)/2+self.pan[0],
                      (self.height()-h*scale)/2+self.pan[1], w*scale, h*scale)

    def to_image(self, pos):
        r = self.image_rect()
        if not r.contains(QPointF(pos)):
            return None
        h,w = self.gray.shape[:2]
        point = np.array(((pos.x()-r.x())/r.width()*w,
                          (pos.y()-r.y())/r.height()*h))
        return point if np.all(point >= 0) and np.all(point < (w,h)) else None

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#080d14"))
        array = np.ascontiguousarray(self.canvas)
        h,w = array.shape[:2]
        fmt = QImage.Format_Grayscale8 if array.ndim == 2 else QImage.Format_BGR888
        q = QImage(array.data,w,h,array.strides[0],fmt)
        rect = self.image_rect()
        p.drawImage(rect,q)
        def screen(point):
            return QPointF(rect.x()+point[0]/self.gray.shape[1]*rect.width(),
                           rect.y()+point[1]/self.gray.shape[0]*rect.height())
        for i, point in enumerate(self.points):
            centre = screen(point)
            p.setPen(QPen(QColor("#f0c17b"),2))
            p.drawEllipse(centre,5,5)
            p.drawText(centre+QPointF(8,-8),str(i+1))
            if self.projected is not None:
                target = screen(self.projected[i])
                p.setPen(QPen(QColor("#7dd3b0"),1))
                p.drawLine(centre,target)
                p.drawLine(target+QPointF(-4,0),target+QPointF(4,0))
                p.drawLine(target+QPointF(0,-4),target+QPointF(0,4))
        p.end()

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._drag = event.pos()
        elif event.button() == Qt.LeftButton:
            point = self.to_image(event.pos())
            if point is not None:
                self.picked.emit(point)

    def mouseMoveEvent(self, event):
        if self._drag is not None:
            delta = event.pos()-self._drag
            self.pan += (delta.x(),delta.y())
            self._drag = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        self._drag = None

    def wheelEvent(self, event):
        before = self.to_image(event.pos())
        self.zoom = float(np.clip(self.zoom * (1.2 if event.angleDelta().y() > 0 else 1/1.2),1,8))
        if before is not None:
            r=self.image_rect()
            current=np.array((r.x()+before[0]/self.gray.shape[1]*r.width(),
                              r.y()+before[1]/self.gray.shape[0]*r.height()))
            self.pan += np.array((event.pos().x(),event.pos().y()))-current
        self.update()


class SurveyView(PerspView):
    picked = pyqtSignal(object)

    def __init__(self, vertices, faces):
        super().__init__()
        self.set_mesh(vertices,faces)
        self._gizmo_enabled = False
        self.outliner_widget.hide()
        self._orb_center = np.mean(vertices,axis=0)
        self._orb_dist = max(float(np.ptp(vertices,axis=0).max())*1.2,1)
        self.survey_points = []
        self.pending = None

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and event.modifiers() & Qt.ShiftModifier:
            eye,forward,right,up = self._eye_basis()
            f = min(self.width(),self.height())*0.8
            ray = forward + right*(event.x()-self.width()/2)/f - up*(event.y()-self.height()/2)/f
            hit = pick_mesh_surface(self._mesh_verts,self._mesh_faces,eye,ray,self._near,self._far)
            if hit is not None:
                self.picked.emit(hit)
            event.accept()
            return
        super().mousePressEvent(event)

    def paintGL(self):
        super().paintGL()
        points = self.survey_points + ([] if self.pending is None else [self.pending])
        if not points:
            return
        p=QPainter(self)
        screen,ok=self._project(np.asarray(points))
        for i,(uv,valid) in enumerate(zip(screen,ok)):
            if valid:
                p.setPen(QPen(QColor("#f0c17b"),2))
                centre=QPointF(*uv)
                p.drawEllipse(centre,5,5)
                p.drawText(centre+QPointF(8,-8),str(i+1))
        p.end()


class LineupDialog(QDialog):
    def __init__(self, gray, K, vertices, faces, T_display_map, apply_lineup,
                 aligner=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Camera Lineup · Lite")
        self.resize(1260,820)
        self.setMinimumSize(960,650)
        self.gray, self.K = np.array(gray,copy=True), np.array(K,copy=True)
        self.T_display_map = np.array(T_display_map,copy=True)
        self.apply_lineup, self.aligner = apply_lineup, aligner
        self.world, self.pixels = [], []
        self.pending = None
        self.solution = None
        root=QVBoxLayout(self)
        title=QLabel("Camera Lineup")
        title.setStyleSheet("font-size:24px;font-weight:600;color:#dce4ef;")
        root.addWidget(title)
        root.addWidget(QLabel("01  FREEZE  →  02  PAIR POINTS  →  03  SOLVE & REVIEW  →  04  APPLY"))
        self.instructions=QLabel("Frame frozen. Shift-click a recognizable corner in the scan, then click the same corner in the camera image. Add 8–12 pairs across different surfaces (6 minimum; avoid points all on one wall or floor).")
        self.instructions.setWordWrap(True)
        self.instructions.setStyleSheet("color:#b6c9dc;padding:10px;background:#1c2a39;")
        root.addWidget(self.instructions)
        split=QSplitter(Qt.Horizontal)
        left=QWidget(); ll=QVBoxLayout(left)
        ll.addWidget(QLabel("FROZEN CAMERA · Scroll to zoom / middle-drag to pan"))
        self.plate=PlateView(self.gray)
        self.plate.picked.connect(self._image_pick)
        ll.addWidget(self.plate,1)
        split.addWidget(left)
        right=QWidget(); rl=QVBoxLayout(right)
        rl.addWidget(QLabel("ROOM SCAN · Shift-click to pick / drag to orbit"))
        verts = np.asarray(vertices) @ self.T_display_map[:3,:3].T + self.T_display_map[:3,3]
        self.scene=SurveyView(verts,faces)
        self.scene.picked.connect(self._scan_pick)
        rl.addWidget(self.scene,1)
        split.addWidget(right)
        root.addWidget(split,1)
        self.table=QTableWidget(0,5)
        self.table.setHorizontalHeaderLabels(["Pair","Image (px)","Scan XYZ (m)","Error (px)","Use"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setMaximumHeight(160)
        root.addWidget(self.table)
        self.summary=QLabel("0 / 6 pairs · Intrinsics locked to the rectified D455 camera")
        self.summary.setWordWrap(True)
        self.summary.setTextFormat(Qt.PlainText)
        self.instructions.setTextFormat(Qt.PlainText)
        root.addWidget(self.summary)
        actions=QHBoxLayout()
        remove=QPushButton("Remove selected")
        remove.clicked.connect(self._remove)
        actions.addWidget(remove)
        undo=QPushButton("Undo last pair")
        undo.clicked.connect(self._undo)
        actions.addWidget(undo)
        self.overlay=QCheckBox("Scan edge overlay")
        self.overlay.toggled.connect(self._overlay)
        actions.addWidget(self.overlay)
        actions.addStretch()
        self.solve_button=QPushButton("Solve camera")
        self.solve_button.setEnabled(False)
        self.solve_button.clicked.connect(self._solve)
        actions.addWidget(self.solve_button)
        self.apply_button=QPushButton("Apply verified lineup")
        self.apply_button.setEnabled(False)
        self.apply_button.clicked.connect(self._apply)
        actions.addWidget(self.apply_button)
        cancel=QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        actions.addWidget(cancel)
        root.addLayout(actions)

    def _scan_pick(self, display_point):
        self.pending = self.T_display_map[:3,:3].T @ (display_point-self.T_display_map[:3,3])
        self.scene.pending=display_point
        self.scene.update()
        self.instructions.setText(f"Point {len(self.world)+1}: now click the same feature in the frozen camera image.")

    def _image_pick(self, point):
        if self.pending is None:
            self.instructions.setText("First Shift-click the corresponding surface in the room scan.")
            return
        self.world.append(self.pending.copy())
        self.pixels.append(np.array(point))
        self.pending=None
        self.scene.pending=None
        self._changed()
        self.instructions.setText("Pair added. Pick another scan feature, then its image point. Use different depths and spread points across the frame.")

    def _changed(self):
        self.solution=None
        self.apply_button.setEnabled(False)
        self.solve_button.setEnabled(len(self.world)>=6)
        self.plate.points=self.pixels
        self.plate.projected=None
        self.plate.canvas=self.gray
        self.scene.survey_points=[self.T_display_map[:3,:3] @ p + self.T_display_map[:3,3] for p in self.world]
        self.scene.update()
        self.plate.update()
        self._table()
        self.summary.setText(f"{len(self.world)} / 6 minimum pairs · Solve and review the reprojection errors")

    def _table(self):
        self.table.setRowCount(len(self.world))
        for i,(world,pixel) in enumerate(zip(self.world,self.pixels)):
            error="—" if self.solution is None else f"{self.solution.residuals_px[i]:.2f}"
            used="Pending" if self.solution is None else ("Inlier" if i in self.solution.inliers else "Rejected")
            values=[str(i+1),f"{pixel[0]:.1f}, {pixel[1]:.1f}",
                    ", ".join(f"{v:.3f}" for v in world),error,used]
            for j,value in enumerate(values):
                self.table.setItem(i,j,QTableWidgetItem(value))

    def _remove(self):
        row=self.table.currentRow()
        if row>=0:
            self.world.pop(row); self.pixels.pop(row)
            self._changed()

    def _undo(self):
        if self.pending is not None:
            self.pending=None; self.scene.pending=None; self.scene.update()
        elif self.world:
            self.world.pop(); self.pixels.pop(); self._changed()

    def _solve(self):
        # A failed retry must not leave an earlier pose or its overlays visible.
        self.solution=None
        self.apply_button.setEnabled(False)
        self.plate.projected=None
        self.plate.canvas=self.gray
        self.plate.update()
        self._table()
        try:
            result=solve_lineup(self.world,self.pixels,self.K,
                                image_size=(self.gray.shape[1],self.gray.shape[0]))
        except Exception as exc:
            logger.exception("Camera lineup solve failed")
            self._solve_failed(f"{type(exc).__name__}: {exc}")
            return
        if not result.ok:
            self._solve_failed(result.reason)
            return
        self.solution=result
        self.plate.projected=result.projected
        self._table()
        self.summary.setText(f"Solved · {len(result.inliers)}/{len(self.world)} inliers · {result.rms_px:.2f} px RMS · Amber = picked / green = projected")
        self.instructions.setText("Camera solved. Review the green projected crosses against your amber clicks and the errors below, then press Apply verified lineup to update tracking.")
        self.apply_button.setEnabled(True)
        self._overlay()

    def _solve_failed(self, reason):
        message=f"Camera solve failed: {reason}"
        self.instructions.setText(message)
        self.summary.setText(message)

    def _overlay(self, checked=None):
        self.plate.canvas=self.gray
        if self.overlay.isChecked() and self.solution is not None and self.aligner is not None:
            import cv2
            frame=self.aligner.overlay(self.gray,self.K,self.solution.T_map_cam)
            if frame is not None:
                self.plate.canvas=cv2.resize(frame,(self.gray.shape[1],self.gray.shape[0]))
        self.plate.update()

    def _apply(self):
        if self.solution is None:
            return
        try:
            ok,reason=self.apply_lineup(self.solution.T_map_cam)
        except Exception as exc:
            self.summary.setText(f"Depth verification unavailable: {exc}")
            return
        if ok:
            self.accept()
        else:
            self.summary.setText(reason)
