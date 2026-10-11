#!/usr/bin/env python3
"""End-to-end manual lineup, driven through the real dashboard object."""
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np, open3d as o3d
import dashboard
from image_alignment import EdgeAligner, LineupSession
from lidar_map_matcher import LidarMapMatcher
from pose_transform import (T_BODY_OPTICAL, invert_transform, make_transform,
                            so3_exp, so3_log, matrix_to_quaternion,
                            decompose_similarity)
from tracking_engine import TrackingEngine
import tracking_config

MESH = "/workspaces/isaac_ros-dev/Room01_Scaled_UE01.obj"; S = 0.01
W, H = 848, 480
FX = 429.948
K = np.array([[FX, 0, 423.5], [0, FX, 241.7], [0, 0, 1.0]])

# RAW file vertices for the matcher (it applies map_scale itself, exactly as
# the dashboard's import path hands it unscaled points); a metre-scaled copy
# for rendering the synthetic frame.
_raw = o3d.io.read_triangle_mesh(MESH)
V, F = np.asarray(_raw.vertices).copy(), np.asarray(_raw.triangles).copy()
mesh = o3d.io.read_triangle_mesh(MESH); mesh.scale(S, center=(0, 0, 0))
scene = o3d.t.geometry.RaycastingScene()
scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
ab = mesh.get_axis_aligned_bounding_box(); lo, hi = ab.min_bound, ab.max_bound

def look_at(e, t, up=(0, 0, 1)):
    e = np.array(e, float); t = np.array(t, float); f = t - e; f /= np.linalg.norm(f)
    if abs(np.dot(f, up)) > 0.99: up = (0, 1, 0)
    l = np.cross(up, f); l /= np.linalg.norm(l); u = np.cross(f, l)
    T = np.eye(4); T[:3, :3] = np.column_stack([f, l, u]); T[:3, 3] = e; return T

def shade(T_body):
    o = T_body @ T_BODY_OPTICAL
    r = scene.create_rays_pinhole(intrinsic_matrix=o3d.core.Tensor(K),
        extrinsic_matrix=o3d.core.Tensor(invert_transform(o)), width_px=W, height_px=H)
    a = scene.cast_rays(r); z = a["t_hit"].numpy(); n = a["primitive_normals"].numpy()
    d = r.numpy()[:, :, 3:6]; d = d / np.linalg.norm(d, axis=2, keepdims=True)
    img = np.clip(np.abs(np.sum(n * (-d), axis=2)) * 200 + 25, 0, 255)
    img[~np.isfinite(z)] = 0
    rng = np.random.default_rng(0)
    return np.clip(img + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)

TRUTH = look_at(np.array([lo[0] + 0.9, lo[1] + 0.9, lo[2] + 1.25]),
                np.array([hi[0] - 0.6, hi[1] - 0.6, lo[2] + 0.55]))
gray = shade(TRUTH)

cfg = tracking_config.DEFAULTS
gui = dashboard.HyperFusionApp.__new__(dashboard.HyperFusionApp)
gui.cfg = cfg
gui.engine = TrackingEngine(cfg, log=lambda *a: None)
gui.depth_K = K; gui._depth_frame_n = 0
gui._lidar_validation = None; gui._lidar_measurement = None
gui.send_on_pose = False; gui._latest_gray = gray; gui._lineup = None
lcfg = dict(cfg["lidar"]); lcfg["map_file"] = MESH; lcfg["map_scale"] = S
gui.lidar_matcher = LidarMapMatcher(lcfg)
gui.lidar_matcher.set_map_mesh(V, F)

class V_:
    def __init__(s_): s_.scale = S; s_.T_align = np.eye(4); s_._loc = np.eye(4)
    def get_locator_transform(s_): return s_._loc.copy()
gui.map_viewer = V_()

# SLAM's arbitrary origin.
T_slam_from_map = np.eye(4)
T_slam_from_map[:3, :3] = so3_exp(np.radians([0, 0, 41.0]))
T_slam_from_map[:3, 3] = [2.9, -1.4, 0.35]
T_slam_cam = T_slam_from_map @ TRUTH
gui.engine.accept_pose(T_slam_cam[:3, 3], matrix_to_quaternion(T_slam_cam[:3, :3]),
                       (0, 0, 0), (0, 0, 0))

def err(T_map_cam):
    R = TRUTH[:3, :3].T @ T_map_cam[:3, :3]
    return (np.linalg.norm(T_map_cam[:3, 3] - TRUTH[:3, 3]) * 1000,
            np.degrees(np.linalg.norm(so3_log(R))))

# Operator's starting guess: 12 cm and 3 deg off, as if placed by hand.
seed = np.eye(4); seed[:3, :3] = so3_exp(np.radians([0, 0, 3.0])); seed[:3, 3] = [0.12, -0.05, 0.03]
gui.map_viewer._loc = seed @ TRUTH

print("=== open a lineup (nothing aligned yet, so it seeds from the locator) ===")
aligner = gui.lidar_matcher.edge_aligner
sess = LineupSession(gray, K, T_slam_cam, invert_transform(np.eye(4)) @ gui.map_viewer._loc,
                     aligner=aligner)
gui._lineup = sess
e0 = err(sess.T_map_cam); sc0 = sess.score()
print(f"  start:  off truth {e0[0]:6.1f}mm {e0[1]:5.2f}°   "
      f"lineup {sc0['score']:.0%} ({sc0['median_px']:.1f}px)")

print("\n=== operator nudges toward a better lineup ===")
best = sess.score()["score"]; improved = 0
for _ in range(40):
    moved = False
    for kw in (dict(forward=1), dict(forward=-1), dict(left=1), dict(left=-1),
               dict(up=1), dict(up=-1), dict(pan=1), dict(pan=-1),
               dict(tilt=1), dict(tilt=-1)):
        sess.nudge(**kw)
        sc = sess.score()["score"]
        if sc > best + 1e-6:
            best = sc; moved = True; improved += 1; break
        sess.undo()
    if not moved:
        if sess.fine: break
        sess.fine = True                      # switch to fine steps and continue
e1 = err(sess.T_map_cam); sc1 = sess.score()
print(f"  after {improved} accepted nudges: off truth {e1[0]:6.1f}mm {e1[1]:5.2f}°   "
      f"lineup {sc1['score']:.0%} ({sc1['median_px']:.1f}px)")

print("\n=== commit, then let continuous fusion carry on ===")
gui._push_map_alignment = lambda: gui.engine.apply_map_alignment(
    *gui.lidar_matcher.map_transform()[:1])
gui.lidar_matcher.set_initial_alignment(sess.slam_to_map(), source="manual lineup")
gui.engine.apply_map_alignment(gui.lidar_matcher.T_slam_to_map)
committed = gui.lidar_matcher.T_slam_to_map @ T_slam_cam
ec = err(committed)
print(f"  committed pose: off truth {ec[0]:6.1f}mm {ec[1]:5.2f}°")

overlay = sess.overlay()
print(f"  overlay: {None if overlay is None else overlay.shape} "
      f"({'drawn' if overlay is not None else 'MISSING'})")

ok = (sc1["score"] >= sc0["score"] and e1[0] <= e0[0] + 1.0
      and abs(ec[0] - e1[0]) < 1e-6)
print("\nRESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
