#!/usr/bin/env python3
"""End-to-end smoke: the operator's locator -> LOCK TO SCAN -> live fusion path,
driven through the real dashboard object against the real Room01 scan."""
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
    def get_locator_transform(self):
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

def camera_error(label):
    pos, quat = gui.engine.camera_pose_ros_corrected()
    d_m = np.linalg.norm(pos - T_disp_true[:3, 3])
    d_deg = np.degrees(np.linalg.norm(
        so3_log(T_disp_true[:3, :3].T @ quaternion_to_matrix(quat))))
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

# ── the ArUco anchor button must behave the same way ────────────────────────
# It is the other route to the same place, and the one that used to install the
# marker-to-locator mapping verbatim -- pinning the camera exactly on the
# locator. Drive _on_anchored() with a marker sighting that is deliberately
# wrong by 20 cm and check the fusion overrides it.
print("\n=== ArUco anchor path ===")
from tracking_engine import MarkerSolution

gui.lidar_matcher.is_aligned = False
gui.lidar_matcher.confidence = 0.0
gui.engine.transformer.clear_ground_zero()

# Frames: TRUTH is the camera in the map's metre frame; the viewport displays
# that at RIGID_A @ TRUTH; SLAM reports T_slam_from_map @ (map pose).
offset = np.eye(4); offset[0, 3] = 1.2          # marker 1.2 m in front (body +X)
T_map_marker = TRUTH @ offset
T_slam_marker = T_slam_from_map @ T_map_marker  # what the marker solve returns

# The operator's stated marker position in the scan, deliberately 20 cm and
# 2.5 deg wrong -- this is the hint.
marker_err = np.eye(4)
marker_err[:3, :3] = so3_exp(np.radians([0, 0, -2.5]))
marker_err[:3, 3] = [-0.20, 0.09, -0.04]
gui.engine.set_anchor_target_ros(marker_err @ RIGID_A @ T_map_marker,
                                 source="test")
gui.engine.last_marker_solution = MarkerSolution(
    ok=True, T_world_marker=T_slam_marker)

gui._on_anchored()
e2 = camera_error("after ArUco anchor")
d = gui.lidar_matcher.get_diagnostics()
hint_err = np.linalg.norm(marker_err[:3, 3])
print(f"  marker/locator hint was ~{hint_err*1000:.0f} mm wrong")
print(f"  fusion solved {d['hint_offset_mm']:.0f} mm / {d['hint_offset_deg']:.2f}° "
      f"away from that hint, {d['verify_score']:.0%} of the view on the scan")

ok = (e0[0] < 0.05 and e0[1] < 1.0 and e1[0] <= e0[0] + 1e-6
      and to_hint > 0.1 and e2[0] < 0.05 and d["hint_offset_mm"] > 100.0)
print("\nRESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
