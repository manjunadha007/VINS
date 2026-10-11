"""End-to-end marker lineup against a real scan, in the container.

Renders an actual DICT_4X4_50 bitmap into a synthetic camera image, runs the
real OpenCV detector on it, raycasts depth from the scan mesh, and drives the
real PnP -> hint -> scan-search -> ICP chain. Needs open3d and the scan file,
so it lives outside run_offline_tests.py.

    python3 verify_marker_lineup.py

Synthetic in the ways that flatter it: exact intrinsics, no motion blur or
rolling shutter, no depth noise, no image/depth timestamp skew. Treat the
millimetre figures as an upper bound on what the rig will do.
"""
import os, sys, time
sys.path.insert(0, "/workspaces/isaac_ros-dev/src")
import numpy as np, cv2, open3d as o3d
from pose_transform import (T_BODY_OPTICAL, invert_transform, rpy_to_matrix,
                            matrix_to_rpy)
from lidar_map_matcher import LidarMapMatcher, sample_mesh_surface
from marker_lineup import MarkerObservation, floor_marker_hint, solve_marker_observation
import tracking_config

OBJ = "/workspaces/isaac_ros-dev/Room01_Scaled_UE.obj"
SCALE = 0.01
MARKER_M = 0.10          # the user's 10 cm marker
FAMILY, MID = "DICT_4X4_50", 7
W, H = 848, 480
K = np.array([[425., 0., 424.], [0., 425., 240.], [0., 0., 1.]])
def _dictionary():
    a = cv2.aruco; did = getattr(a, FAMILY)
    return (a.getPredefinedDictionary(did) if hasattr(a, "getPredefinedDictionary")
            else a.Dictionary_get(did))

def _detect(img):
    a = cv2.aruco; d = _dictionary()
    params = (a.DetectorParameters_create() if hasattr(a, "DetectorParameters_create")
              else a.DetectorParameters())
    if hasattr(a, "ArucoDetector"):
        return a.ArucoDetector(d, params).detectMarkers(img)
    return a.detectMarkers(img, d, parameters=params)

R = []
def ck(n, c, x=""):
    R.append(bool(c)); print(("  ok  " if c else "  FAIL"), n, x)

# ── the scan, in metres, exactly as the dashboard loads it ──────────────────
mesh = o3d.io.read_triangle_mesh(OBJ)
verts = np.asarray(mesh.vertices) * SCALE
mesh.vertices = o3d.utility.Vector3dVector(verts)
faces = np.asarray(mesh.triangles)
lo, hi = verts.min(axis=0), verts.max(axis=0)
print(f"scan: {hi-lo} m, floor z={lo[2]:.2f}, {len(faces)} faces")

scene = o3d.t.geometry.RaycastingScene()
scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

def render_depth(T_map_optical):
    """Raycast a z-depth image (uint16 mm) from the scan at this optical pose."""
    extrinsic = invert_transform(T_map_optical)          # world -> camera
    rays = scene.create_rays_pinhole(
        intrinsic_matrix=o3d.core.Tensor(K),
        extrinsic_matrix=o3d.core.Tensor(extrinsic),
        width_px=W, height_px=H)
    ans = scene.cast_rays(rays)
    t = ans['t_hit'].numpy()
    r = rays.numpy()
    hit = r[..., :3] + t[..., None] * r[..., 3:6]        # world-space hit points
    Rwc, twc = T_map_optical[:3, :3], T_map_optical[:3, 3]
    z = ((hit - twc) @ Rwc)[..., 2]                      # z in the optical frame
    z[~np.isfinite(t)] = 0.0
    return np.clip(np.nan_to_num(z) * 1000.0, 0, 65535).astype(np.uint16)

def render_marker_image(T_map_markerCV, T_map_optical):
    """Warp a real generated marker bitmap into the camera image."""
    img = np.full((H, W), 90, np.uint8)
    cv2.randn(img, 90, 6)                                 # a little sensor noise
    s = MARKER_M / 2
    obj = np.array([[-s, s, 0.], [s, s, 0.], [s, -s, 0.], [-s, -s, 0.]])
    world = (T_map_markerCV[:3, :3] @ obj.T).T + T_map_markerCV[:3, 3]
    cam = (world - T_map_optical[:3, 3]) @ T_map_optical[:3, :3]
    if np.any(cam[:, 2] <= 0):
        return None, None
    uv = (cam @ K.T)[:, :2] / cam[:, 2:3]
    d = _dictionary()
    tile = (cv2.aruco.generateImageMarker(d, MID, 400)
            if hasattr(cv2.aruco, "generateImageMarker")
            else cv2.aruco.drawMarker(d, MID, 400))
    b, n0 = 60, tile.shape[0]
    tile = cv2.copyMakeBorder(tile, b, b, b, b, cv2.BORDER_CONSTANT, value=255)
    # Corners of the marker square inside the bordered tile -- mapping the
    # tile's own corners would shrink the printed square inside the quad.
    src = np.array([[b, b], [b + n0, b], [b + n0, b + n0], [b, b + n0]], np.float32)
    Hm = cv2.getPerspectiveTransform(src, uv.astype(np.float32))
    warped = cv2.warpPerspective(tile, Hm, (W, H), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_TRANSPARENT,
                                 dst=img.copy())
    return warped, uv

# ── ground truth: marker flat on the floor, camera looking down at it ───────
floor_z = float(lo[2])
centre = (lo + hi) / 2
yaw = np.radians(37.0)
T_map_markerCV = np.eye(4)                 # OpenCV marker axes: right / up / out
T_map_markerCV[:3, :3] = rpy_to_matrix(0, 0, yaw)
T_map_markerCV[:3, 3] = [centre[0], centre[1], floor_z + 0.002]
# The operator's locator: +X toward the printed top, +Z up out of the face.
T_map_locator = T_map_markerCV @ np.block(
    [[rpy_to_matrix(0, 0, np.pi / 2), np.zeros((3, 1))], [np.zeros((1, 3)), np.ones((1, 1))]])

cam_pos = np.array([centre[0] - 0.55, centre[1] - 0.75, floor_z + 1.25])
fwd = T_map_markerCV[:3, 3] - cam_pos; fwd /= np.linalg.norm(fwd)
right = np.cross(fwd, [0, 0, 1.]); right /= np.linalg.norm(right)
down = np.cross(fwd, right)
T_map_optical = np.eye(4)
T_map_optical[:3, :3] = np.column_stack([right, down, fwd])   # optical: x right, y down, z fwd
T_map_optical[:3, 3] = cam_pos
T_map_body_true = T_map_optical @ invert_transform(T_BODY_OPTICAL)

print("\n=== 1. render + detect the marker ===")
gray, uv_true = render_marker_image(T_map_markerCV, T_map_optical)
ck("marker projects into the image", gray is not None)
cv2.imwrite("/workspaces/isaac_ros-dev/src/_marker_view.png", gray)
corners, ids, _ = _detect(gray)
ck("real detector found the marker", ids is not None and len(ids) == 1,
   f"ids={None if ids is None else ids.ravel().tolist()}")
ck("it is the right ID", ids is not None and int(ids[0][0]) == MID)
found = corners[0].reshape(4, 2)
ck("detected corners match projection", np.max(np.linalg.norm(found - uv_true, axis=1)) < 2.0,
   f"max {np.max(np.linalg.norm(found - uv_true, axis=1)):.2f} px")

print("\n=== 2. depth from the scan ===")
depth = render_depth(T_map_optical)
valid = (depth > 300) & (depth < 6000)
ck("depth image has real returns", valid.mean() > 0.3, f"{valid.mean():.0%} valid")

print("\n=== 3. inject SLAM drift, build the observation ===")
drift = np.eye(4)                       # SLAM frame is offset+rotated from the map
drift[:3, :3] = rpy_to_matrix(0, 0, np.radians(43))
drift[:3, 3] = [2.4, 1.7, 0.1]
truth = invert_transform(drift)         # what T_slam_to_map must come out as
T_slam_image = drift @ T_map_body_true
obs = MarkerObservation(FAMILY, MID, found, K.copy(), MARKER_M,
                        T_slam_image, T_slam_image.copy(), T_map_locator,
                        depth, K.copy(), np.array([0., 0., 1.]), time.monotonic())

hint = floor_marker_hint(obs)
pos_err = np.linalg.norm(hint[:3, 3] - truth[:3, 3])
rot_err = np.degrees(np.arccos(np.clip((np.trace(hint[:3, :3].T @ truth[:3, :3]) - 1) / 2, -1, 1)))
# The hint only has to land inside the capture region (0.50 m / 20°); a 10 cm
# marker at this range carries a few cm of inherent PnP range error, and the
# scan match is what turns it into a millimetre answer.
ck("PnP hint lands inside the capture region", pos_err < 0.25 and rot_err < 5.0,
   f"{pos_err*1000:.1f} mm, {rot_err:.2f}° (region is 500 mm / 20°)")

print("\n=== 4. scan search + ICP from that hint ===")
cfg = dict(tracking_config.load_config(verbose=False).get("lidar", {}))
cfg["map_file"] = ""; cfg["map_scale"] = 1.0     # verts already scaled to metres
m = LidarMapMatcher(cfg)
cloud = o3d.geometry.PointCloud()
cloud.points = o3d.utility.Vector3dVector(sample_mesh_surface(verts, faces, 400000))
m.set_map_cloud(cloud)
ck("matcher ready", m._ready())
t0 = time.monotonic()
meas = solve_marker_observation(obs, m)
dt = (time.monotonic() - t0) * 1000
ck("scan match accepted", meas is not None and meas.ok,
   "" if (meas and meas.ok) else f"reason={meas.reason if meas else 'none'}")
if meas is not None and meas.ok:
    p = np.linalg.norm(meas.T_slam_to_map[:3, 3] - truth[:3, 3])
    r = np.degrees(np.arccos(np.clip(
        (np.trace(meas.T_slam_to_map[:3, :3].T @ truth[:3, :3]) - 1) / 2, -1, 1)))
    ck("solved pose matches ground truth", p < 0.08 and r < 2.0,
       f"{p*1000:.1f} mm, {r:.2f}° (fit {meas.verify_score:.0%}, rank {meas.observable_rank}, {dt:.0f} ms)")
    ck("all three position axes constrained", meas.observable_rank >= 3,
       f"rank {meas.observable_rank}")
    m.accept_hint_solution(meas, source="aruco", captured_at=obs.captured_at)
    ck("fusion is now aligned and continues", m.is_aligned and m._ready())

print("\n=== 5. a wrong locator must be rejected, not silently wrong ===")
bad = obs.__class__(**{**obs.__dict__})
off = np.eye(4); off[:3, 3] = [1.6, 1.2, 0.0]      # locator placed ~2 m off
bad.T_map_locator = off @ T_map_locator
m2 = LidarMapMatcher(cfg); m2.set_map_cloud(cloud)
bad_meas = solve_marker_observation(bad, m2)
ck("a badly placed locator is rejected", bad_meas is None or not bad_meas.ok,
   f"ok={None if bad_meas is None else bad_meas.ok}"
   + ("" if bad_meas is None or not bad_meas.ok else " -- SILENTLY WRONG"))

print("\n=== 6. operating envelope: how far can the rig stand back? ===")
print(f"  {'placement':<26} {'hint err':>10} {'final err':>11} {'fit':>5} {'rank':>4}  verdict")
for label, off, height in (
        ("close, 0.7 m back",   (-0.40, -0.55), 1.05),
        ("normal, 0.9 m back",  (-0.55, -0.75), 1.25),
        ("far, 1.5 m back",     (-0.95, -1.25), 1.60),
        ("shallow angle",       (-1.20, -1.55), 0.55)):
    cp = np.array([centre[0] + off[0], centre[1] + off[1], floor_z + height])
    f = T_map_markerCV[:3, 3] - cp; f /= np.linalg.norm(f)
    r = np.cross(f, [0, 0, 1.]); r /= np.linalg.norm(r)
    To = np.eye(4); To[:3, :3] = np.column_stack([r, np.cross(f, r), f]); To[:3, 3] = cp
    g, _uv = render_marker_image(T_map_markerCV, To)
    cs, idl, _ = _detect(g)
    if idl is None or len(idl) != 1:
        print(f"  {label:<26} {'--':>10} {'--':>11} {'--':>5} {'--':>4}  marker not detected")
        continue
    Tb = To @ invert_transform(T_BODY_OPTICAL)
    o = MarkerObservation(FAMILY, MID, cs[0].reshape(4, 2), K.copy(), MARKER_M,
                          drift @ Tb, drift @ Tb, T_map_locator, render_depth(To),
                          K.copy(), np.array([0., 0., 1.]), time.monotonic())
    try:
        h = floor_marker_hint(o)
        he = np.linalg.norm(h[:3, 3] - truth[:3, 3])
    except ValueError as exc:
        print(f"  {label:<26} rejected: {exc}"); continue
    mm = LidarMapMatcher(cfg); mm.set_map_cloud(cloud)
    ms = solve_marker_observation(o, mm)
    if ms is not None and ms.ok:
        fe = np.linalg.norm(ms.T_slam_to_map[:3, 3] - truth[:3, 3])
        print(f"  {label:<26} {he*1000:>7.0f} mm {fe*1000:>8.1f} mm {ms.verify_score:>5.0%}"
              f" {ms.observable_rank:>4}  accepted")
        R.append(fe < 0.10)
    else:
        print(f"  {label:<26} {he*1000:>7.0f} mm {'--':>11} {'--':>5} {'--':>4}"
              f"  rejected: {(ms.reason if ms else 'no result')[:44]}")

print("\nRESULT:", "PASS" if all(R) else f"FAIL ({R.count(False)})")
sys.exit(0 if all(R) else 1)
