#!/usr/bin/env python3
"""End-to-end smoke: the operator's locator -> LOCK TO SCAN -> live fusion path,
then ArUco Auto lineup, driven through the real dashboard object against the
real Room01 scan."""
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np, open3d as o3d, yaml

import dashboard
from pose_transform import quaternion_to_matrix, matrix_to_quaternion
from lidar_map_matcher import LidarMapMatcher
from pose_transform import (T_BODY_OPTICAL, invert_transform, make_transform,
                            so3_exp, so3_log, decompose_similarity)
from tracking_engine import TrackingEngine
import tracking_config

MESH = "/workspaces/isaac_ros-dev/Room01_Scaled_UE.obj"
SCALE = 0.01
W, H = 640, 360
FX = 320 / np.tan(np.radians(43.5))
K = np.array([[FX, 0, W / 2], [0, FX, H / 2], [0, 0, 1.0]])

mesh = o3d.io.read_triangle_mesh(MESH); mesh.scale(SCALE, center=(0, 0, 0))
scene = o3d.t.geometry.RaycastingScene()
scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

def look_at(eye, tgt, up=(0, 1, 0)):
    eye = np.array(eye, float); tgt = np.array(tgt, float)
    f = tgt - eye; f /= np.linalg.norm(f)
    l = np.cross(up, f); l /= np.linalg.norm(l); u = np.cross(f, l)
    T = np.eye(4); T[:3, :3] = np.column_stack([f, l, u]); T[:3, 3] = eye; return T

def render(T):
    o = T @ T_BODY_OPTICAL
    r = scene.create_rays_pinhole(intrinsic_matrix=o3d.core.Tensor(K),
        extrinsic_matrix=o3d.core.Tensor(invert_transform(o)), width_px=W, height_px=H)
    z = scene.cast_rays(r)["t_hit"].numpy().copy()
    z[~np.isfinite(z)] = 0; z[(z <= 0.2) | (z > 10)] = 0
    return (z * 1000).astype(np.uint16)

# --- a dashboard instance with only the pieces this path touches -------------
cfg = tracking_config.DEFAULTS
gui = dashboard.HyperFusionApp.__new__(dashboard.HyperFusionApp)
gui.cfg = cfg
gui.engine = TrackingEngine(cfg, log=lambda *a: None)
gui.depth_K = K
gui._depth_frame_n = 0
gui._lidar_validation = None
gui._lidar_measurement = None
gui.send_on_pose = False

lcfg = dict(cfg["lidar"]); lcfg["map_file"] = MESH; lcfg["map_scale"] = SCALE
gui.lidar_matcher = LidarMapMatcher(lcfg)

class FakeViewer:
    """Stands in for MapViewerWidget.

    T_align is a *similarity*: the operator's "S" slider (0.01, because the
    scan is in centimetres) times a real rotation, plus a translation. The
    displayed frame -- where the locator and the camera dot live -- is
    rigid_align @ (scale * map_native).
    """
    def __init__(self):
        self.scale = SCALE
        self.T_align = np.eye(4)
        self.T_align[:3, :3] = so3_exp(np.radians([0, 0, 20.0])) * SCALE
        self.T_align[:3, 3] = [1.0, -0.5, 0.2]
        self._loc = np.eye(4)
    def get_locator_transform(self, locator_id=None):
        return self._loc.copy()
gui.map_viewer = FakeViewer()
SCALE_A, RIGID_A = decompose_similarity(gui.map_viewer.T_align)
assert abs(SCALE_A - SCALE) < 1e-12

class FakeBtn:
    def setText(self, t): self.t = t
    def setStyleSheet(self, s): pass
gui.btn_lidar_lock = FakeBtn()

TRUTH = look_at((-0.2, 1.34, 0.19), (1.4, 1.0, 0.19))   # camera in the scan
depth = render(TRUTH)
print(f"depth: {(depth>0).sum()} valid px, median {np.median(depth[depth>0])/1000:.2f} m")

# SLAM has its own arbitrary origin: pretend cuVSLAM's frame is rotated+offset
# from the scan, which is exactly the real situation on power-up.
T_slam_from_map = np.eye(4)
T_slam_from_map[:3, :3] = so3_exp(np.radians([0, 0, 47.0]))
T_slam_from_map[:3, 3] = [3.1, -1.7, 0.4]
T_slam_cam = T_slam_from_map @ TRUTH                     # what SLAM reports

gui.engine.accept_pose(T_slam_cam[:3, 3],
                       matrix_to_quaternion(T_slam_cam[:3, :3]),
                       (0, 0, 0), (0, 0, 0))
gui.engine.set_depth(depth, K)

# TRUTH is a pose in the map's metre frame (the raycast mesh was pre-scaled),
# which the viewport displays at RIGID_A @ TRUTH.
T_disp_true = RIGID_A @ TRUTH

# Operator drags the locator to roughly where the camera is: 25 cm and 3 deg off.
seed_err = np.eye(4)
seed_err[:3, :3] = so3_exp(np.radians([0, 0, 3.0]))
seed_err[:3, 3] = [0.25, -0.15, 0.05]
gui.map_viewer._loc = seed_err @ T_disp_true

def camera_error(label, T_true=None):
    T_true = T_disp_true if T_true is None else T_true
    pos, quat = gui.engine.camera_pose_ros_corrected()
    d_m = np.linalg.norm(pos - T_true[:3, 3])
    d_deg = np.degrees(np.linalg.norm(
        so3_log(T_true[:3, :3].T @ quaternion_to_matrix(quat))))
    print(f"  {label:<34} camera dot off by {d_m*1000:8.1f} mm  {d_deg:6.2f} deg")
    return d_m, d_deg

print("\nbefore locking (no ground zero -- dot sits at the raw SLAM pose):")
camera_error("unlocked")

print("\npress MATCH VIEW TO SCAN (locator hint):")
gui._lock_to_map_from_locator()
print(f"  button now reads: {gui.btn_lidar_lock.t}")
e0 = camera_error("after solve")

# The reported bug: the dot used to land on the locator instead of on the
# camera's real position. Measure the distance to the hint as well as to truth.
hint_pos = gui.map_viewer._loc[:3, 3]
dot_pos, _ = gui.engine.camera_pose_ros_corrected()
to_hint = np.linalg.norm(dot_pos - hint_pos)
hint_err = np.linalg.norm(hint_pos - T_disp_true[:3, 3])
print(f"  hint was {hint_err*1000:.0f} mm from the true camera position")
print(f"  camera dot ended up {to_hint*1000:.0f} mm from the hint "
      f"({'SOLVED, not snapped' if to_hint > 0.1 else 'SNAPPED TO HINT'})")

print("\n20 continuous fusion ticks:")
for i in range(20):
    gui.lidar_matcher.last_icp_time = -1e9
    meas = gui.lidar_matcher.update_depth(depth, K, T_slam_cam, timestamp=i * 1.0)
    if meas is not None:
        gui._push_map_alignment()
e1 = camera_error("after continuous fusion")
print(f"\n{gui._map_fusion_text()}")

# ── Auto lineup from an ArUco marker must behave the same way ──────────────
# The other route to the same place. The marker only gives a hint: PnP plus the
# operator's locator, which marks where the marker sits in the scan. Place that
# locator 20 cm off and check the scan match overrides it rather than pinning
# the camera to it. Driven through the dashboard's own arm -> poll -> worker ->
# finish path, as the "Auto lineup" checkbox does.
print("\n=== ArUco Auto lineup path ===")
import threading, time
from marker_lineup import MarkerObservation, floor_marker_hint

# Forget the lock above so this path has to earn its own answer.
gui.lidar_matcher.is_aligned = False
gui.lidar_matcher.confidence = 0.0
gui.lidar_matcher.T_slam_to_map = np.eye(4)
gui.engine.transformer.clear_ground_zero()

# A 10 cm marker flat on the scan's floor, as the dashboard reads the floor:
# the display frame's +Z (the frame verify_marker_lineup.py uses too).
MARKER_M = 0.10
verts = np.asarray(mesh.vertices)
lo, hi = verts.min(axis=0), verts.max(axis=0)
centre = (lo + hi) / 2
T_map_markerCV = np.eye(4)                      # OpenCV marker axes: right/up/out
T_map_markerCV[:3, :3] = so3_exp(np.radians([0, 0, 37.0]))
to_locator = np.eye(4); to_locator[:3, :3] = so3_exp(np.radians([0, 0, 90.0]))

def cast(origin, direction):
    hit = scene.cast_rays(o3d.core.Tensor(np.r_[origin, direction].astype(np.float32)[None]))
    return float(hit["t_hit"].numpy()[0]), hit["primitive_normals"].numpy()[0]

# The marker must lie on a flat, +Z-facing patch the camera really sees: the
# lineup checks its size against depth, and the scan's minimum-Z plane at the
# centre is hidden behind furniture. Search outward from the centre for one,
# with the camera 1.25 m above it, as before.
for dx, dy in sorted(((x, y) for x in np.arange(-.6, .61, .1) for y in np.arange(-.6, .61, .1)),
                     key=lambda o: np.hypot(*o)):
    above = np.array([centre[0] + dx, centre[1] + dy, lo[2] + 1.25])
    t_down, n_down = cast(above, np.array([0, 0, -1.0]))
    if not np.isfinite(t_down) or n_down[2] < 0.97:
        continue
    marker = above - [0, 0, t_down - 0.002]
    eye = marker + [-0.55, -0.75, 1.25]
    fwd = marker - eye; fwd /= np.linalg.norm(fwd)
    t_view, n_view = cast(eye, fwd)
    if abs(t_view - np.linalg.norm(marker - eye)) < 0.01 and n_view[2] > 0.97:
        break
else:
    raise SystemExit("no visible flat patch for the marker near the scan centre")
T_map_markerCV[:3, 3] = marker
right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right)
T_map_optical = np.eye(4)
T_map_optical[:3, :3] = np.column_stack([right, np.cross(fwd, right), fwd])
T_map_optical[:3, 3] = eye
TRUTH2 = T_map_optical @ invert_transform(T_BODY_OPTICAL)
T_disp_true2 = RIGID_A @ TRUTH2
T_map_locator = T_map_markerCV @ to_locator

# The operator walks the rig over. Steps stay under the pose gate's jump
# limit, so SLAM keeps tracking and no resync cancels the lineup.
stamp = 1000.0
def slam_pose(T_map_body):
    global stamp
    stamp += 0.05
    T = T_slam_from_map @ T_map_body
    assert gui.engine.accept_pose(T[:3, 3], matrix_to_quaternion(T[:3, :3]),
                                  (0, 0, 0), (0, 0, 0), stamp=stamp)
    return stamp
walk = so3_log(TRUTH[:3, :3].T @ TRUTH2[:3, :3])
steps = int(np.ceil(np.linalg.norm(TRUTH2[:3, 3] - TRUTH[:3, 3]) / 0.05))
for i in range(1, steps + 1):
    T = np.eye(4)
    T[:3, :3] = TRUTH[:3, :3] @ so3_exp(walk * i / steps)
    T[:3, 3] = TRUTH[:3, 3] + (TRUTH2[:3, 3] - TRUTH[:3, 3]) * i / steps
    slam_pose(T)
assert gui.engine.gate.consecutive_rejects == 0, "walk tripped the pose gate"

# Camera intrinsics arrive, and the operator places the locator 20 cm and
# 2.5 deg off where the marker really is.
gui.engine.set_intrinsics(K.ravel())
marker_err = np.eye(4)
marker_err[:3, :3] = so3_exp(np.radians([0, 0, -2.5]))
marker_err[:3, 3] = [-0.20, 0.09, -0.04]
gui.map_viewer._loc = marker_err @ RIGID_A @ T_map_locator

class FakeField:
    def __init__(self, v): self.v = v
    def value(self): return self.v
    def currentText(self): return self.v
    def setEnabled(self, on): pass
class FakeCheck(FakeBtn):
    """The Auto lineup checkbox, whose title reports the lineup state."""
    checked = False
    def blockSignals(self, b): return False
    def setChecked(self, on): self.checked = on
    def isChecked(self): return self.checked
    def setProperty(self, k, v): pass
    def styleSheet(self): return ""
class FakePersp:
    """One Locator, which Auto lineup may give the lineup marker's ID."""
    _active_locator_id = "locator"
    _locators = {"locator": None}
    marker = None
    def set_locator_marker_size(self, w): pass
    def select_locator(self): pass
    def locator_entries(self):
        return {"locator": dict(name="Locator", transform=gui.map_viewer._loc.copy(),
                                marker=self.marker)}
    def assign_marker_to_locator(self, key, family, fid, size_m):
        self.marker = dict(dictionary=family, id=fid, size_m=size_m)
class FakeSignal:
    """The worker thread emits here; the test runs the queued slot itself."""
    def __init__(self): self.results = []
    def emit(self, result): self.results.append(result)
gui.map_viewer.persp_view = FakePersp()
gui.map_viewer.set_locator_transform = lambda T, locator_id=None: setattr(gui.map_viewer, "_loc", T.copy())
gui.marker_lineup_finished = FakeSignal()
gui.aruco_enabled = True
gui._lineup = None
gui.marker_family = FakeField("DICT_4X4_250")
gui.marker_id = FakeField(7)
gui.marker_width = FakeField(MARKER_M)
gui.btn_marker_lineup = FakeCheck()
gui.btn_ar = FakeCheck()
gui.lbl_marker_lineup = FakeBtn()
gui._marker_request = gui._marker_lock_context = None
gui._marker_busy = gui._marker_shutdown = gui._marker_failed = False
gui._marker_detection_sample = None
from marker_map import MarkerSurvey
gui._marker_survey = MarkerSurvey()     # after the lock, other markers would be placed

gui._toggle_marker_lineup(True)
print(f"  armed: {gui.btn_marker_lineup.t} -- {gui._marker_detail}")
assert gui._marker_request is not None, "auto lineup did not arm"

# One fresh frame: depth plus the marker's corners, both paired to this pose.
s = slam_pose(TRUTH2)
gui.engine.set_depth(render(TRUTH2), K, stamp=s)
h = MARKER_M / 2
square = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]])
cam = (square @ T_map_markerCV[:3, :3].T + T_map_markerCV[:3, 3] - eye) @ T_map_optical[:3, :3]
corners = (cam @ K.T)[:, :2] / cam[:, 2:3]
gui._marker_detection_sample = ([("DICT_4X4_250", 7, corners)], s,
                                time.monotonic(), (W, H))
gui._poll_marker_lineup()
print(f"  {gui._marker_detail}")
assert gui._marker_busy, "the detection was not accepted: " + gui._marker_detail

# SLAM keeps publishing while the solve runs, as it does on the rig.
deadline = time.monotonic() + 30
while not gui.marker_lineup_finished.results and time.monotonic() < deadline:
    time.sleep(0.05)
    slam_pose(TRUTH2)
assert gui.marker_lineup_finished.results, "marker solve did not finish"
request, observation, measurement, error = gui.marker_lineup_finished.results[0]
gui._finish_marker_lineup(gui.marker_lineup_finished.results[0])
print(f"  {gui.btn_marker_lineup.t} -- {gui._marker_detail}")
locked = gui._marker_lock_context is not None and gui.btn_marker_lineup.checked

e2 = camera_error("after Auto lineup", T_disp_true2)
# Where the camera would sit if the misplaced locator were taken as truth.
T_hint = floor_marker_hint(observation)
T_cam_hint = T_hint @ T_slam_from_map @ TRUTH2
hint_off = np.linalg.norm(T_cam_hint[:3, 3] - TRUTH2[:3, 3])
print(f"  locator was {np.linalg.norm(marker_err[:3, 3])*1000:.0f} mm off; trusting it "
      f"would put the camera {hint_off*1000:.0f} mm off")
if measurement is not None and measurement.ok:
    print(f"  scan match: {measurement.verify_score:.0%} of the view on the scan, "
          f"rank {measurement.observable_rank}")
# The search found the marker itself, so the Locator moves onto it and keeps its ID.
loc_off = np.linalg.norm(gui.map_viewer._loc[:3, 3] - (RIGID_A @ T_map_locator)[:3, 3])
adopted = gui.map_viewer.persp_view.marker
print(f"  Locator now {loc_off*1000:.0f} mm from the real marker; assigned "
      f"{adopted and (adopted['dictionary'], adopted['id'])}")

ok = (e0[0] < 0.05 and e0[1] < 1.0 and e1[0] <= e0[0] + 1e-6
      and to_hint > 0.1 and locked and e2[0] < 0.05 and e2[1] < 1.0
      and hint_off > 0.1 and loc_off < 0.03
      and adopted is not None and adopted["id"] == 7)
print("\nRESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
