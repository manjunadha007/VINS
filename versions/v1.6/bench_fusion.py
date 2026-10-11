#!/usr/bin/env python3
"""Benchmark the map fusion against the real Room01 scan with ray-cast depth."""
import sys, time
import numpy as np
import open3d as o3d
import yaml

from lidar_map_matcher import LidarMapMatcher
from pose_transform import (T_BODY_OPTICAL, make_transform, so3_log, so3_exp,
                            invert_transform)

from tracking_config import load_config
CFG = load_config()["lidar"]
MESH = CFG["map_file"]
SCALE = CFG["map_scale"]
CFG["map_file"] = MESH
CFG["map_scale"] = SCALE

W, H = 640, 360
FX = 320.0 / np.tan(np.radians(87.0 / 2))
K = np.array([[FX, 0, W / 2], [0, FX, H / 2], [0, 0, 1.0]])

mesh = o3d.io.read_triangle_mesh(MESH)
mesh.scale(SCALE, center=(0, 0, 0))             # -> metres, same as the matcher
aabb = mesh.get_axis_aligned_bounding_box()
size = aabb.max_bound - aabb.min_bound
print(f"mesh: {len(mesh.triangles)} tris, room {size[0]:.2f} x {size[1]:.2f} x {size[2]:.2f} m")

scene = o3d.t.geometry.RaycastingScene()
scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

def render(T_world_body):
    T_world_opt = T_world_body @ T_BODY_OPTICAL
    rays = scene.create_rays_pinhole(
        intrinsic_matrix=o3d.core.Tensor(K),
        extrinsic_matrix=o3d.core.Tensor(invert_transform(T_world_opt)),
        width_px=W, height_px=H)
    hit = scene.cast_rays(rays)["t_hit"].numpy()
    d = rays.numpy()[:, :, 3:6]
    z = hit.copy()
    z[~np.isfinite(z)] = 0.0
    z[(z <= 0.2) | (z > 10.0)] = 0.0
    # Realistic stereo noise: sigma_z = z^2 * sigma_d / (f * b)
    rng = np.random.default_rng(0)
    m = z > 0
    z[m] += rng.normal(0, z[m] ** 2 * 0.1 / (FX * 0.095))
    return (z * 1000).astype(np.uint16)

t0 = time.time()
m = LidarMapMatcher(CFG)
print(f"map load + pyramid: {time.time() - t0:.2f}s")
if m.map_cloud is None:
    sys.exit("map failed to load")

def look_at(eye, target, world_up=(0.0, 1.0, 0.0)):
    """Body-frame pose (x forward, y left, z up) looking from eye to target."""
    eye = np.asarray(eye, float); target = np.asarray(target, float)
    fwd = target - eye; fwd /= np.linalg.norm(fwd)
    left = np.cross(np.asarray(world_up, float), fwd)
    left /= np.linalg.norm(left)
    up = np.cross(fwd, left)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([fwd, left, up])
    T[:3, 3] = eye
    return T

# The scan is a furnished room, so most of the interior is inside geometry.
# Stand in the open space the probe found and look across the room.
VIEWS = [("across the room", look_at((-0.2, 1.34, 0.19), (1.4, 1.0, 0.19))),
         ("into a corner",   look_at((-0.2, 1.34, 0.19), (1.4, 0.9, 1.3))),
         ("toward -X wall",  look_at((-0.2, 1.34, 0.19), (-1.5, 1.0, 0.0)))]

def rigid(rpy_deg=(0,0,0), t=(0,0,0)):
    T = np.eye(4); T[:3,:3] = so3_exp(np.radians(rpy_deg)); T[:3,3] = t; return T

def err(T_est, T_true):
    ph = np.append(T_true[:3, 3], 1.0)
    return (np.linalg.norm((T_est @ ph)[:3] - T_true[:3, 3]),
            np.degrees(np.linalg.norm(so3_log(T_est[:3, :3]))))

cases = [("120 mm translation", rigid(t=(0.12, -0.06, 0.03))),
         ("2.0 deg rotation",   rigid(rpy_deg=(0, 0, 2.0))),
         ("80 mm + 1.5 deg",    rigid(rpy_deg=(0.5, 0, 1.5), t=(0.08, 0.04, -0.02))),
         ("250 mm + 3 deg",     rigid(rpy_deg=(0, 0, 3.0), t=(0.25, 0.10, 0.05)))]

for view_name, T_true in VIEWS:
    depth = render(T_true)
    valid = int((depth > 0).sum())
    dv = depth[depth > 0] / 1000.0
    print(f"\n=== view: {view_name} — {valid} valid px ({100*valid/(W*H):.0f}%), "
          f"depth {dv.min():.2f}–{dv.max():.2f} m (median {np.median(dv):.2f}) ===")
    print(f"{'injected drift':<22} {'before':>17} {'after 20 fixes':>19}  ICP ms   fit  rank")
    for name, drift in cases:
        m.set_initial_alignment(drift.copy(), source="bench")
        m.confidence = 0.0
        m.last_icp_time = -1e9      # each case is an independent run
        b_m, b_d = err(m.T_slam_to_map, T_true)
        times = []
        for i in range(20):
            t = time.time()
            m.update_depth(depth, K, T_true, timestamp=i * 10.0)
            times.append((time.time() - t) * 1000)
        a_m, a_d = err(m.T_slam_to_map, T_true)
        d = m.get_diagnostics()
        verdict = "✓" if a_m < b_m * 0.4 else ("~" if a_m < b_m * 0.9 else "✗")
        print(f"{name:<22} {b_m*1000:7.1f}mm {b_d:5.2f}° "
              f"{a_m*1000:8.1f}mm {a_d:5.2f}° {verdict} "
              f"{np.median(times):6.0f}  {d['icp_fitness']:.2f}  {d['observable_rank']}/3")

budget = 1000.0 / CFG.get("icp_rate_hz", 5.0)
print(f"\nbudget at {CFG.get('icp_rate_hz')} Hz = {budget:.0f} ms per fix")
