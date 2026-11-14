#!/usr/bin/env python3
"""Camera gizmo + LiDAR AR overlay, driven through the real Qt widgets."""
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QPointF
app = QApplication(sys.argv)

from map_viewer import MapViewerWidget
from pose_transform import rpy_to_matrix, make_transform
import math

w = MapViewerWidget()
pv = w.persp_view
fails = []
def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{'  ' + detail if detail else ''}")
    if not ok: fails.append(name)

print("=== gizmo target follows the outliner ===")
names = [w.persp_view.outliner_list.item(i).text()
         for i in range(w.persp_view.outliner_list.count())]
print(f"  outliner: {names}")
check("has a Camera entry", any(n.startswith("Camera") for n in names))
check("defaults to the locator", pv._gizmo_target == 'locator')

for i, n in enumerate(names):
    if n.startswith("Camera"):
        pv._on_outliner_selection_changed(i)
check("selecting Camera hands over the gizmo", pv._gizmo_target == 'camera')
check("camera gizmo is drawn as selected", pv._gizmo_is_selected())
check("label switches", pv._gizmo_label() == "CAMERA", pv._gizmo_label())

print("\n=== dragging the camera gizmo moves the camera, not the locator ===")
w.update_camera_pose(np.array([1.0, 0.5, 1.4]), (0.0, 0.0, 0.0, 1.0))
loc_before = pv._locator_pos.copy()
cam_before = pv._cam_pos.copy()

moved = {}
w.camera_moved.connect(lambda T: moved.update(T=T))
pv._begin_locator_drag('x', QPointF(100.0, 100.0))
pv._drag_locator(QPointF(160.0, 100.0))
pv._active_gizmo = None
check("camera moved", not np.allclose(pv._cam_pos, cam_before),
      f"{cam_before.round(3)} -> {pv._cam_pos.round(3)}")
check("locator untouched", np.allclose(pv._locator_pos, loc_before))
check("camera_moved emitted a 4x4", "T" in moved and moved["T"].shape == (4, 4))
if "T" in moved:
    check("emitted position matches the gizmo",
          np.allclose(moved["T"][:3, 3], pv._cam_pos))

print("\n=== the live pose must not fight an in-progress drag ===")
pv._active_gizmo = 'x'
held = pv._cam_pos.copy()
pv.set_cam((9.0, 9.0, 9.0), 0.0)
check("drag not overridden by the live pose", np.allclose(pv._cam_pos, held))
pv._active_gizmo = None
pv.set_cam((2.0, 0.0, 1.0), 30.0)
check("live pose resumes once the drag ends", np.allclose(pv._cam_pos, [2, 0, 1]))
check("heading stored", abs(pv._cam_rpy[2] - 30.0) < 1e-9)

print("\n=== back to the locator ===")
for i, n in enumerate(names):
    if n.startswith("Locator"):
        pv._on_outliner_selection_changed(i)
check("gizmo returns to the locator", pv._gizmo_target == 'locator')
cam_before = pv._cam_pos.copy()
loc_before = pv._locator_pos.copy()
pv._begin_locator_drag('y', QPointF(100.0, 100.0))
pv._drag_locator(QPointF(100.0, 160.0))
pv._active_gizmo = None
check("locator moved", not np.allclose(pv._locator_pos, loc_before))
check("camera untouched", np.allclose(pv._cam_pos, cam_before))

print("\n=== trail and world reference have editable display transforms ===")
for name, target in (("Trail", "trail"), ("World Origin", "world")):
    i = names.index(name)
    pv._on_outliner_selection_changed(i)
    check(f"{name}: gizmo follows selection", pv._gizmo_target == target and pv.gizmo_active)
    pv.channel_box.fields["tx"].setValue(1.25)
    check(f"{name}: numeric edit moves display", abs(pv._gizmo_pos()[0] - 1.25) < 1e-9)

print("\n=== Point Cloud drives the scan alignment ===")
pts = np.array([[0., 0., 0.], [2., 0., 0.], [2., 2., 0.], [0., 2., 0.],
                [0., 0., 2.], [2., 0., 2.], [2., 2., 2.], [0., 2., 2.]])
w.set_map_points(pts)
i = names.index("Point Cloud")
pv._on_outliner_selection_changed(i)
check("gizmo target is the mesh", pv._gizmo_target == 'mesh')
check("label", pv._gizmo_label() == "SCAN ALIGNMENT", pv._gizmo_label())
gpos, _ = pv.get_mesh_gizmo()
check("gizmo sits on the scan centroid", np.allclose(gpos, pts.mean(axis=0)),
      str(gpos.round(2)))

tx0 = w.tx
pv._begin_locator_drag('x', QPointF(100.0, 100.0))
pv._drag_locator(QPointF(170.0, 100.0))
pv._active_gizmo = None
check("scan translation changed", abs(w.tx - tx0) > 1e-6,
      f"tx {tx0:.3f} -> {w.tx:.3f}")
check("alignment quantised to 0.1 mm", abs(w.tx - round(w.tx, 4)) < 1e-12,
      f"{w.tx!r}")
check("gizmo followed the scan",
      np.allclose(pv.get_mesh_gizmo()[0], w._aligned_centroid()))

yaw0 = w.yaw
centroid0 = w._aligned_centroid().copy()
# 'rot_z' is the Z ring, which is what turns yaw. This used to drag a handle
# called 'yaw' -- a name no gizmo has had for some time, so _begin_locator_drag
# set up nothing and the drag was silently a no-op that this check then read as
# a broken rotation.
pv.set_gizmo_mode('rotate')
pv._begin_locator_drag('rot_z', QPointF(200.0, 100.0))
pv._drag_locator(QPointF(100.0, 200.0))
pv._active_gizmo = None
check("scan yaw changed", abs(w.yaw - yaw0) > 1e-6, f"{yaw0:.1f} -> {w.yaw:.1f}")
# 1 mm, not machine epsilon: the alignment is deliberately quantised to
# 0.1 mm, so the pivot cannot be exact to floating point.
check("scan spun in place, not around the origin",
      np.allclose(w._aligned_centroid(), centroid0, atol=1e-3),
      f"centroid {centroid0.round(4)} -> {w._aligned_centroid().round(4)}")

print("\n=== a stored (locked) alignment refuses to move ===")
w._store_pose()
tx_locked = w.tx
pv._begin_locator_drag('x', QPointF(100.0, 100.0))
pv._drag_locator(QPointF(180.0, 100.0))
pv._active_gizmo = None
check("locked alignment unchanged", abs(w.tx - tx_locked) < 1e-9)
w._reset_pose()

print("\n=== the gizmo toggle ===")
pv._on_outliner_selection_changed(names.index("Camera"))
check("on by default", pv.gizmo_active)
pv.btn_gizmo.setChecked(False)
check("toggle off hides it", not pv.gizmo_active)
check("toggle label", pv.btn_gizmo.text().endswith("OFF"), pv.btn_gizmo.text())
cam_before = pv._cam_pos.copy()
hit = pv._pick_gizmo_handle(QPointF(0.0, 0.0))
check("no picking while off", not pv.gizmo_active)
pv.btn_gizmo.setChecked(True)
check("toggle back on", pv.gizmo_active)

print("\n=== outliner is tall enough for every entry ===")
h = pv.outliner_list.height()
need = pv.outliner_list.count() * 18
check("outliner fits all entries", h >= need, f"{h}px for {pv.outliner_list.count()} items")

print("\n=== it still paints ===")
try:
    pv.resize(400, 300); pv.grab(); painted = True; err = ""
except Exception as e:
    painted = False; err = str(e)
check("renders without error", painted, err)

print("\nRESULT:", "PASS" if not fails else f"FAIL ({len(fails)}): {fails}")
sys.exit(0 if not fails else 1)
