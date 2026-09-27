#!/usr/bin/env python3
"""
Map-relative pose fusion: RealSense D455 depth vs. a pre-scanned LiDAR map.

What this solves
----------------
cuVSLAM gives a pose that is smooth and locally excellent but has an arbitrary
origin and drifts without bound. The LiDAR scan is the real world, surveyed
once, and does not drift. Fusing them gives what virtual production actually
needs: the camera's pose *in the set*, accurate in absolute terms and still
smooth enough to shoot with.

The estimated quantity is a transform, not a position
-----------------------------------------------------
The output is

    T_map_cam(t)  =  T_slam_to_map  @  T_slam_cam(t)
                     └ slow, fused ┘   └ fast, from VIO ┘

`T_slam_cam(t)` arrives from cuVSLAM at 90 Hz and carries every bit of real
camera motion. `T_slam_to_map` is the accumulated VIO drift expressed as a rigid
transform: it changes slowly, by millimetres per second, and ICP against the map
measures it directly. Because all the fast motion lives in the VIO term, the
drift term can be filtered as hard as we like without adding a millisecond of
latency to the actual camera movement -- which is the whole trick.

This is why the previous implementation could not work. It estimated a *position
correction* instead:

  * it took `icp.transformation[:3, 3]` as that correction, but a rigid
    transform moves a point by `(R - I)p + t`, not by `t`. `t` is where the map
    *origin* went. One degree of ICP rotation with the camera 5 m out puts 87 mm
    into the `(R - I)p` term alone -- larger than the whole rejection gate, so
    real corrections were thrown away and the ones that survived pointed
    somewhere arbitrary.
  * it discarded rotation entirely, so orientation drift was never corrected.
  * the caller subtracted the already-applied correction from a measurement
    taken through an already-corrected pose, double-counting the feedback and
    oscillating.
  * it passed `max_correction_m` (0.10) as ICP's correspondence radius, so an
    error bigger than 10 cm could not even be observed, let alone fixed.

Estimating the transform makes all four of those disappear at once: each ICP run
is an absolute measurement of `T_slam_to_map`, and fusion is a geodesic blend on
SE(3) toward it.

Degenerate geometry
-------------------
ICP against a room scan is often under-constrained. Pointed at one flat wall it
is free to slide in the two directions along that wall and will happily do so,
reporting a confident fitness the whole time. So the observability of the
correction is measured from the spread of the matched surface normals, and the
correction is projected onto the subspace the geometry actually pins down. A
camera looking at a single wall gets its distance-to-wall corrected and nothing
else -- which is exactly the information available.

Using the D455 for what it is
-----------------------------
Stereo depth error grows with the square of range: sigma_z = z^2 * sigma_d /
(f * b), and with the D455's 95 mm baseline a point at 6 m is ~40x noisier than
one at 1 m. Feeding all of them to ICP at equal weight lets the far field, which
is mostly noise, outvote the near field, which is mostly signal. Points are
therefore kept with probability proportional to 1/sigma_z^2, which makes point
*density* track information content and gives an inverse-variance-weighted fit
out of an unweighted solver.
"""
import math
import os
import threading
import time

import numpy as np

try:
    import open3d as o3d
    _HAS_O3D = True
except ImportError:
    _HAS_O3D = False

# scipy's cKDTree, not open3d's KDTreeFlann, for the nearest-neighbour scoring.
# Open3D only offers a one-point-per-call API, so scoring 600 points meant a
# 600-iteration Python loop holding the GIL on the ROS callback thread -- which
# starved the GUI thread's 60 Hz FreeD timer down to ~17 Hz. cKDTree.query()
# does the whole batch in C and releases the GIL.
try:
    from scipy.spatial import cKDTree
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

from pose_transform import (T_BODY_OPTICAL, make_transform,
                            se3_interpolate, so3_exp, so3_log)

try:
    from image_alignment import EdgeAligner
    _HAS_EDGE_ALIGN = True
except ImportError:
    _HAS_EDGE_ALIGN = False

#: D455 stereo baseline, metres (Intel datasheet). Sets the depth noise model.
D455_BASELINE_M = 0.095

#: Sub-pixel disparity noise of the D4 ASIC's stereo matcher, in pixels.
#: 0.1 px is Intel's own figure for the subpixel-refined block matcher and
#: reproduces the published "<2 % of range at 4 m" spec.
D455_DISPARITY_SIGMA_PX = 0.1


def sample_mesh_surface(vertices, triangles, n_points, seed=0):
    """Uniformly sample a mesh's surface, reproducibly.

    Open3D's `sample_points_uniformly` takes no seed and is not reproducible
    (verified on 0.19): two calls on the same mesh return different clouds. The
    fusion is then a different problem on every load -- measured on this scan,
    the same lock-on landed at 7 mm on one run and 151 mm on the next, purely
    because the map had been sampled differently. An operator needs the same
    scan to behave the same way, so the sampling is done here instead.

    Area-weighted triangle choice, then a uniform barycentric point within it,
    which is the standard construction for a uniform surface sample.
    """
    vertices = np.asarray(vertices, dtype=float)
    triangles = np.asarray(triangles, dtype=np.int64)
    if len(triangles) == 0 or n_points <= 0:
        return np.empty((0, 3))

    corners = vertices[triangles]                       # (T, 3, 3)
    areas = 0.5 * np.linalg.norm(
        np.cross(corners[:, 1] - corners[:, 0],
                 corners[:, 2] - corners[:, 0]), axis=1)
    total = float(areas.sum())
    if total <= 0.0:
        return np.empty((0, 3))

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(triangles), size=int(n_points), p=areas / total)
    a, b, c = corners[idx, 0], corners[idx, 1], corners[idx, 2]
    # sqrt on the first coordinate is what makes it uniform over the triangle
    # rather than clustered toward one vertex.
    r1 = np.sqrt(rng.random(int(n_points)))[:, None]
    r2 = rng.random(int(n_points))[:, None]
    return (1.0 - r1) * a + r1 * (1.0 - r2) * b + r1 * r2 * c


def depth_sigma_m(z_m, fx_px, baseline_m=D455_BASELINE_M,
                  disparity_sigma_px=D455_DISPARITY_SIGMA_PX):
    """Expected 1-sigma range error of a stereo depth sample, in metres.

    sigma_z = z^2 * sigma_d / (f * b) -- the standard triangulation error, and
    the reason a 6 m point deserves ~1/40th the weight of a 1 m one.
    """
    z = np.asarray(z_m, dtype=float)
    return z * z * disparity_sigma_px / max(fx_px * baseline_m, 1e-9)


class MapMeasurement:
    """One ICP fix: an absolute measurement of T_slam_to_map, and its quality."""

    __slots__ = ("ok", "reason", "T_slam_to_map", "fitness", "rmse_m",
                 "n_inliers", "translation_m", "rotation_deg",
                 "observable_rank", "observability", "elapsed_ms", "n_points",
                 "verify_score", "verify_median_m", "yaw_offset_deg", "monotonic_timestamp")

    def __init__(self, ok=False, reason="", T_slam_to_map=None, fitness=0.0,
                 rmse_m=0.0, n_inliers=0, translation_m=0.0, rotation_deg=0.0,
                 observable_rank=0, observability=None, elapsed_ms=0.0,
                 n_points=0, verify_score=0.0, verify_median_m=0.0,
                 yaw_offset_deg=0.0):
        self.monotonic_timestamp = None
        self.ok = ok
        self.reason = reason
        self.T_slam_to_map = T_slam_to_map
        self.fitness = fitness
        self.rmse_m = rmse_m
        self.n_inliers = n_inliers
        self.translation_m = translation_m
        self.rotation_deg = rotation_deg
        self.observable_rank = observable_rank
        self.observability = (np.zeros(3) if observability is None
                              else np.asarray(observability, dtype=float))
        self.elapsed_ms = elapsed_ms
        self.n_points = n_points
        #: fraction of depth points within verify_threshold_m of the scan once
        #: this solution is applied -- the honest "does the camera view match
        #: the LiDAR" number, unlike ICP fitness.
        self.verify_score = verify_score
        self.verify_median_m = verify_median_m
        self.yaw_offset_deg = yaw_offset_deg

    def __repr__(self):
        if not self.ok:
            return f"MapMeasurement(rejected: {self.reason})"
        return (f"MapMeasurement(fit={self.fitness:.2f} "
                f"verify={self.verify_score:.2f} "
                f"rmse={self.rmse_m * 1000:.0f}mm "
                f"d={self.translation_m * 1000:.0f}mm "
                f"r={self.rotation_deg:.2f}deg rank={self.observable_rank})")


class LidarMapMatcher:
    """Localises the live D455 depth stream inside a pre-scanned LiDAR map.

    Call `update_depth()` on depth frames; read the fused result with
    `map_transform()`. `is_aligned` gates everything -- seed it first with
    `align_from_seed()` (the locator workflow) or `set_initial_alignment()`
    (the ArUco workflow).
    """

    def __init__(self, config):
        self.is_aligned = False
        self.has_correction = False
        self.config = config or {}
        cfg = self.config

        self.map_file = cfg.get("map_file", "")
        self.voxel_size = float(cfg.get("voxel_size", 0.05))
        self.icp_rate_hz = float(cfg.get("icp_rate_hz", 5.0))

        # ── measurement gates ────────────────────────────────────────────────
        self.min_fitness = float(cfg.get("min_fitness", 0.3))
        self.min_inlier_points = int(cfg.get("min_inlier_points", 50))
        self.max_rmse_m = float(cfg.get("max_rmse_m", 0.08))
        # A measurement is rejected if it claims the map moved further than
        # this. Distinct from the ICP correspondence radius below, which the
        # old code conflated with it.
        self.max_correction_m = float(cfg.get("max_correction_m", 0.30))
        self.max_correction_deg = float(cfg.get("max_correction_deg", 5.0))

        # ── ICP search radii (coarse-to-fine) ────────────────────────────────
        self.coarsest_radius_m = float(cfg.get("coarsest_radius_m", 1.00))
        self.coarse_radius_m = float(cfg.get("coarse_radius_m", 0.50))
        self.fine_radius_m = float(cfg.get("fine_radius_m", 0.10))
        self.initial_iterations = int(cfg.get("initial_icp_iterations", 200))
        self.continuous_iterations = int(cfg.get("continuous_icp_iterations", 30))

        # ── fusion ───────────────────────────────────────────────────────────
        # Fraction of the measured residual absorbed per accepted ICP fix, at
        # full confidence. The slew limits below then cap how fast that can
        # actually move the camera, in metres and degrees per second.
        self.blend_alpha = float(cfg.get("blend_alpha", 0.1))
        self.max_slew_m_s = float(cfg.get("max_slew_m_s", 0.02))
        self.max_slew_deg_s = float(cfg.get("max_slew_deg_s", 1.0))
        self.freeze_when_stationary = bool(
            cfg.get("freeze_when_stationary", False))

        # ── depth conditioning (D455) ────────────────────────────────────────
        self.min_range_m = float(cfg.get("min_range_m", 0.3))
        self.max_range_m = float(cfg.get("max_range_m", 6.0))
        self.baseline_m = float(cfg.get("baseline_m", D455_BASELINE_M))
        self.disparity_sigma_px = float(
            cfg.get("disparity_sigma_px", D455_DISPARITY_SIGMA_PX))
        # Range at which a depth sample is kept with probability 1. Nearer
        # points are already at p=1 (clipped); further ones thin out as
        # 1/sigma_z^2.
        self.weight_ref_range_m = float(cfg.get("weight_ref_range_m", 2.0))
        self.min_keep_fraction = float(cfg.get("min_keep_fraction", 0.05))
        self.target_points = int(cfg.get("target_points", 8000))

        # Directions whose normal-covariance eigenvalue falls below this are
        # treated as unobservable and the correction is not applied along them.
        self.observability_eps = float(cfg.get("observability_eps", 0.04))

        # ── verifying a lock-on ──────────────────────────────────────────────
        # ICP's own `fitness` is the fraction of points with a correspondence
        # inside the search radius, so at a wide radius it reads ~1.0 almost
        # wherever you put the cloud -- measured on a real room scan, a
        # solution 530 mm out of place still scored 0.75. It cannot be used to
        # decide whether a lock is real. This is the independent check: the
        # fraction of depth points landing within `verify_threshold_m` of the
        # scan once the solution is applied.
        self.verify_threshold_m = float(cfg.get("verify_threshold_m", 0.10))
        self.min_verify_score = float(cfg.get("min_verify_score", 0.55))

        # Yaw is the DOF an operator gets most wrong when placing a hint by
        # hand (the locator's heading ring), and it is also the one ICP is
        # worst at recovering, so the lock-on tries several and keeps whichever
        # verifies best.
        # How far from the hint the answer is allowed to be. The locator says
        # "the camera is around here, search the scan there" -- so a solution
        # outside this is not a refinement of the hint, it is a different place
        # that happened to look similar. Measured on real hardware: gated at
        # 2 m / 45 deg, a close-up of a table corner locked onto geometry
        # 1.1 m and 42 deg away, reported success, and left continuous fusion
        # permanently stuck asking for a 409 mm correction it was not allowed
        # to make.
        self.hint_radius_m = float(cfg.get("hint_radius_m", 0.50))
        self.hint_radius_deg = float(cfg.get("hint_radius_deg", 20.0))

        self.seed_yaw_search_deg = float(cfg.get("seed_yaw_search_deg", 25.0))
        self.seed_yaw_steps = int(cfg.get("seed_yaw_steps", 5))
        #: how many screened candidates get a full-resolution refinement
        self.refine_candidates = int(cfg.get("refine_candidates", 3))
        #: a screened candidate this close to the scan ends the yaw search --
        #: an operator whose hint was already good should not wait for a
        #: search that cannot improve on it
        self.early_exit_median_m = float(
            cfg.get("early_exit_median_m", self.voxel_size))
        #: only refine candidates within this factor of the leader's median;
        #: refining an obviously worse one cannot change the outcome
        self.shortlist_ratio = float(cfg.get("shortlist_ratio", 1.5))
        #: match-quality penalty charged per metre of distance from the hint,
        #: when choosing between candidate solutions. The hint is information,
        #: not just a starting point: in a cluttered room several placements
        #: match the view about equally well, and the one nearest where the
        #: operator said the camera is should win. In units of the verify
        #: score, so 0.1 means "half a metre away must match 5 % better".
        self.hint_weight = float(cfg.get("hint_weight", 0.10))

        self.mesh_sample_points = int(cfg.get("mesh_sample_points", 400000))
        # Metres per map-file unit. A scan exported from Unreal is in
        # centimetres, so it needs 0.01 here; everything downstream then works
        # in metres and every transform stays rigid. The dashboard keeps this
        # in step with the viewport's mesh-alignment scale slider.
        self.map_scale = float(cfg.get("map_scale", 1.0))

        # ── state ────────────────────────────────────────────────────────────
        self.map_cloud = None
        self._map_native = None         # points in the file's own units
        self._map_pyramid = {}          # voxel multiplier -> downsampled cloud
        self._map_tree = None           # cached KD-tree for validate_depth
        self._pyramid_dirty = False
        #: SLAM world -> map-native frame. The fused estimate; the whole point.
        self.T_slam_to_map = np.eye(4)
        self.confidence = 0.0
        self.alignment_source = None

        self.last_measurement = None
        #: bumped every time a fix is fused, so a front-end can poll cheaply
        #: instead of being called back from the worker thread.
        self.fix_counter = 0
        self._generation = 0
        self._worker = None
        self._worker_stop = threading.Event()
        self._pending = None
        self._pending_lock = threading.Condition()
        self.last_icp_time = 0.0
        self._last_fuse_time = None
        self._rng = np.random.default_rng(0)

        # Camera-image lineup against the scan (see image_alignment.py). Used
        # as a read-only check by default: measured on this scan its objective
        # sits 30-60 mm off the true pose, so it must not drive a depth
        # solution that is already good to millimetres.
        self.image_refine = bool(cfg.get("image_align_refine", False))
        self.edge_aligner = (EdgeAligner(cfg) if _HAS_EDGE_ALIGN else None)
        self._mesh_vertices = None
        self._mesh_triangles = None
        self._frame = None

        self.diagnostics = {
            "lidar_aligned": False,
            "alignment_source": None,
            "icp_fitness": 0.0,
            "icp_rmse_mm": 0.0,
            "residual_mm": 0.0,
            "residual_deg": 0.0,
            "applied_mm": 0.0,
            "applied_deg": 0.0,
            "observable_rank": 0,
            "observability": [0.0, 0.0, 0.0],
            "image_score": 0.0,
            "image_median_px": 0.0,
            "image_edges": 0,
            "verify_score": 0.0,          # latest accepted measurement
            "verify_median_mm": 0.0,
            "lock_verify_score": 0.0,     # the lock-on itself; never overwritten
            "lock_verify_median_mm": 0.0,
            "lock_ok": False,
            "hint_offset_mm": 0.0,
            "hint_offset_deg": 0.0,
            "search_region_m": 0.0,
            "search_region_deg": 0.0,
            "inliers": 0,
            "cloud_points": 0,
            "last_icp_ms": 0.0,
            "confidence": 0.0,
            "rejects": 0,
            "last_reject": "",
            "fixes": 0,
        }

        if not _HAS_O3D:
            print("[LiDAR] open3d not installed. Map matching disabled.")
            return
        if not _HAS_SCIPY:
            print("[LiDAR] scipy not installed — lock-on verification and "
                  "depth validation are unavailable.")

        self._load_map()

    # ── map loading ──────────────────────────────────────────────────────────
    def _load_map(self):
        if not self.map_file:
            # Configured empty on purpose: the scan arrives by import instead.
            print("[LiDAR] No map file configured — waiting for an import.")
            return
        if not os.path.exists(self.map_file):
            print(f"[LiDAR] Map file not found: {self.map_file}")
            return

        print(f"[LiDAR] Loading map: {self.map_file}")
        ext = os.path.splitext(self.map_file)[1].lower()
        try:
            if ext in (".obj", ".stl", ".fbx", ".gltf", ".glb"):
                mesh = o3d.io.read_triangle_mesh(self.map_file)
                if not mesh.has_triangles():
                    print(f"[LiDAR] Mesh has no triangles: {self.map_file}")
                    return
                n_points = max(self.mesh_sample_points, len(mesh.triangles) * 10)
                self._mesh_vertices = np.asarray(mesh.vertices).copy()
                self._mesh_triangles = np.asarray(mesh.triangles).copy()
                cloud = o3d.geometry.PointCloud()
                cloud.points = o3d.utility.Vector3dVector(sample_mesh_surface(
                    self._mesh_vertices, self._mesh_triangles, n_points))
                print(f"[LiDAR] Sampled {n_points} points from mesh "
                      f"({len(mesh.triangles)} triangles)")
            else:
                cloud = o3d.io.read_point_cloud(self.map_file)

            if not cloud.has_points():
                print(f"[LiDAR] Map file empty or invalid: {self.map_file}")
                return
            self.set_map_cloud(cloud)
        except Exception as exc:
            print(f"[LiDAR] Failed to load map: {exc}")

    def set_map_cloud(self, cloud):
        """Install a map cloud in the file's own units and build the pyramid."""
        self._map_native = np.asarray(cloud.points).copy()
        self._check_units(self._map_native)
        self._build_pyramid()

    def clear_map(self):
        """Drop the loaded scan and every cache derived from it.

        Deleting a scan has to reach further than the viewport. The pyramid,
        the KD-tree and the raycast surface are all caches of a map that no
        longer exists, and `is_aligned` still claims a fix against it -- left
        behind, the worker goes on matching depth against a scan the operator
        removed, and the dashboard goes on reporting agreement with it.

        The generation bump is what makes this safe to call while the worker
        is mid-ICP: a result computed against the old map is discarded on
        arrival rather than fused.
        """
        self.map_file = ""
        self.map_cloud = None
        self._map_native = None
        self._map_pyramid = {}
        self._map_tree = None
        self._pyramid_dirty = False
        self._mesh_vertices = None
        self._mesh_triangles = None
        self.is_aligned = False
        self.alignment_source = None
        self.confidence = 0.0
        self.T_slam_to_map = np.eye(4)
        self.last_measurement = None
        self._generation += 1
        if self.edge_aligner is not None:
            self.edge_aligner.clear_mesh()
        self.diagnostics["lidar_aligned"] = False
        self.diagnostics["alignment_source"] = None

    def set_map_mesh(self, vertices, triangles):
        """Keep the scan's surface too, for the camera-image lineup.

        ICP works on the sampled point cloud, but a silhouette is a property of
        the surface -- a point cloud has none to find -- so the mesh is kept
        alongside it.
        """
        self._mesh_vertices = np.asarray(vertices, dtype=np.float64).copy()
        self._mesh_triangles = np.asarray(triangles, dtype=np.int32).copy()
        self._install_mesh()

    def _install_mesh(self):
        if (self.edge_aligner is None or self._mesh_vertices is None
                or self._mesh_triangles is None):
            return False
        return self.edge_aligner.set_mesh(self._mesh_vertices,
                                          self._mesh_triangles,
                                          scale=self.map_scale)

    def set_image(self, gray, K, stamp=None):
        """Latest camera frame, for the lineup check and overlay.

        The stamp travels with the frame so both can be drawn from the pose
        that belongs to *this* picture instead of the newest one to arrive.
        Kept as one tuple on purpose: the GUI thread reads it while the ROS
        callback replaces it, and a half-updated pair would put a fresh pose
        on a stale frame -- the very error the stamp is here to prevent.
        """
        self._frame = (gray, None if K is None else np.asarray(K, dtype=float),
                       None if stamp is None else float(stamp))

    def _paired_frame(self, pose):
        """(gray, K, T_slam_cam) for one frame, or None if it cannot be paired.

        `pose` is either a pose or a callable taking the frame's stamp and
        returning the pose recorded at that instant. The callable form is what
        holds the overlay still: the stamp is chosen and used inside a single
        atomic read of the frame, so no newer picture can arrive between
        picking the pose and drawing on the image it belongs to.
        """
        frame = self._frame
        if (self.edge_aligner is None or not self.edge_aligner.available
                or frame is None or not self.is_aligned):
            return None
        gray, K, stamp = frame
        if gray is None or K is None:
            return None
        T = self._as_transform(pose(stamp) if callable(pose) else pose)
        return None if T is None else (gray, K, T)

    def image_lineup(self, pose):
        """How well the camera picture lines up with the scan right now.

        Read-only: returns the score or None, and never moves the pose.
        """
        paired = self._paired_frame(pose)
        if paired is None:
            return None
        gray, K, T = paired
        return self.edge_aligner.score(gray, K, self.T_slam_to_map @ T)

    def image_overlay(self, pose):
        """The camera frame with the scan's edges drawn on it, or None."""
        paired = self._paired_frame(pose)
        if paired is None:
            return None
        gray, K, T = paired
        return self.edge_aligner.overlay(gray, K, self.T_slam_to_map @ T)

    def set_map_scale(self, metres_per_unit):
        """Set the map file's units, in metres per file unit.

        Rebuilding the pyramid over a 400k-point scan takes about a second, so
        this only marks it dirty: the rebuild happens on the next ICP tick, on
        the depth thread, rather than stalling the GUI on every tick of the
        scene alignment scale.
        """
        metres_per_unit = float(metres_per_unit)
        if metres_per_unit <= 0.0 or abs(metres_per_unit - self.map_scale) < 1e-12:
            return False
        self._generation += 1
        self.map_scale = metres_per_unit
        self._pyramid_dirty = True
        return True

    def _ensure_pyramid(self):
        if self._pyramid_dirty and self._map_native is not None:
            self._pyramid_dirty = False
            self._build_pyramid()

    def _check_units(self, points):
        """Warn when the scan's extent says it is not in metres.

        A room-scale scan a hundred metres across is almost always a
        centimetre export. Left unnoticed, ICP is matching a metre-scale depth
        cloud against a centimetre-scale map and can never converge, which
        looks exactly like "the fusion does not work".
        """
        if len(points) == 0:
            return
        extent = float(np.ptp(points, axis=0).max()) * self.map_scale
        if extent > 50.0:
            print(f"[LiDAR] ⚠️  scan is {extent:.0f} m across at the current "
                  f"scale ({self.map_scale}). If this is a centimetre export, "
                  f"set the mesh-alignment S slider to 0.01 (or lidar.map_scale "
                  f"in tracking.yaml) — otherwise map matching cannot converge.")
        elif extent < 0.5:
            print(f"[LiDAR] ⚠️  scan is only {extent:.2f} m across at the "
                  f"current scale ({self.map_scale}); check the units.")

    def _build_pyramid(self):
        """Build the metre-scale cloud, its resolution pyramid and KD-tree.

        The pyramid is built here rather than per ICP call: coarse-to-fine
        matching needs the map at three resolutions every tick, and normal
        estimation over a 400k-point room scan is far too slow to repeat at
        5 Hz.
        """
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(
            self._map_native * self.map_scale)
        base = cloud.voxel_down_sample(voxel_size=self.voxel_size)
        base.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=self.voxel_size * 2, max_nn=30))

        self.map_cloud = base
        self._map_pyramid = {1: base}
        for mult in (2, 4, 8):
            level = base.voxel_down_sample(voxel_size=self.voxel_size * mult)
            level.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=self.voxel_size * mult * 2, max_nn=30))
            self._map_pyramid[mult] = level
        pts = np.asarray(base.points)
        self._map_tree = cKDTree(pts) if _HAS_SCIPY and len(pts) else None
        self._install_mesh()      # the aligner holds its own scaled copy
        extent = np.ptp(np.asarray(base.points), axis=0)
        print(f"[LiDAR] Map ready: {len(base.points)} points, "
              f"{extent[0]:.1f} x {extent[1]:.1f} x {extent[2]:.1f} m "
              f"(scale {self.map_scale}, voxel {self.voxel_size} m, pyramid "
              f"{[len(self._map_pyramid[m].points) for m in (1, 2, 4, 8)]})")

    # ── alignment seeding ────────────────────────────────────────────────────
    def set_initial_alignment(self, T_slam_to_map, source="marker"):
        """Seed the SLAM->map transform without running ICP.

        Used by the ArUco workflow, where the marker solve already gives a good
        absolute fix. Continuous ICP refines it from here.
        """
        self._generation += 1
        self.last_measurement = None
        self.T_slam_to_map = np.asarray(T_slam_to_map, dtype=float).copy()
        self.is_aligned = True
        self.alignment_source = source
        self.confidence = 0.0
        self._last_fuse_time = None
        self.diagnostics["lidar_aligned"] = True
        self.diagnostics["alignment_source"] = source
        print(f"[LiDAR] alignment seeded from {source}: "
              f"origin at {self.T_slam_to_map[:3, 3].round(3)} m")
        return self.T_slam_to_map

    #: Back-compat alias -- older callers say "marker alignment".
    set_marker_alignment = set_initial_alignment

    def rebase_alignment(self, T_slam_to_map):
        """Replace the transform without touching alignment/confidence state.

        The dashboard calls this when the operator drags the mesh-alignment
        gizmo: the camera must stay put on screen while the mesh moves under
        it, so the SLAM->map transform is re-derived rather than re-measured.
        """
        if not self.is_aligned:
            return None
        self._generation += 1
        self.T_slam_to_map = np.asarray(T_slam_to_map, dtype=float).copy()
        return self.T_slam_to_map

    def solve_from_hint(self, depth_mm, K, T_slam_cam, T_hint, source="locator",
                        up_map=None):
        """Work out where the camera really is, given a rough hint.

        The hint -- the operator's locator, or a marker sighting -- says only
        *roughly* where the camera is. It is a starting guess and nothing more:
        the answer comes from registering the live depth cloud against the
        scan, so what the camera sees lines up with the scanned geometry rather
        than with wherever the hint was dropped.

        This is the fix for the fusion snapping the camera onto the locator.
        The ArUco path used to install the marker-to-locator mapping verbatim,
        which pinned the camera exactly on the hint and left continuous fusion
        to creep toward the truth at the slew limit -- 20 mm/s, so tens of
        seconds. Both entry points now solve immediately instead.

        Yaw is tried at several offsets around the hint, because heading is
        what an operator placing a gizmo by hand gets most wrong and what ICP
        is least able to recover on its own. Each candidate is scored by how
        much of the camera's view actually lands on the scan -- not by ICP
        fitness, which at a wide search radius is near 1.0 almost anywhere --
        and the best is kept.

        `up_map` is the vertical, expressed in the map's frame. The caller
        knows it and the matcher does not: the operator's heading error comes
        from dragging the locator's heading ring, which turns about the
        *displayed* vertical, so the axis is whatever the mesh alignment makes
        it -- the dashboard passes `R_align^T @ [0,0,1]`. Left None it is
        inferred from the hint, which assumes the SLAM frame is gravity-aligned
        (true of cuVSLAM with IMU fusion, but not worth relying on silently).

        Returns a MapMeasurement whose `translation_m` / `rotation_deg` are how
        far the solution moved *off the hint*, so the caller can show that the
        fusion did real work.
        """
        if not self._ready():
            return MapMeasurement(reason="no map loaded")
        self._ensure_pyramid()

        T_hint = np.asarray(T_hint, dtype=float)
        t0 = time.time()

        # A hand-placed hint is expected to be off by more than runtime drift,
        # but it is still a statement about *where to look*. The search region
        # is centred on it, so a candidate that wanders outside is rejected
        # rather than believed.
        wide_m = self.hint_radius_m
        wide_deg = self.hint_radius_deg

        # Phase 1 -- screen the yaw candidates on the coarse pyramid levels
        # only. Running every candidate all the way to full resolution costs
        # seconds for a shortlist that is decided at 10-40 cm resolution; this
        # is the same hierarchical trick the ICP stages themselves use.
        screen_iters = max(20, self.initial_iterations // 5)
        screen_stages = ((8, self.coarsest_radius_m, screen_iters),
                         (4, self.coarse_radius_m, screen_iters),
                         (2, self.coarse_radius_m * 0.5, screen_iters))
        screened = []
        for yaw_deg in self._yaw_candidates():
            seed = self._yaw_about_up(T_hint, T_slam_cam, yaw_deg, up_map)
            meas = self._measure(
                depth_mm, K, T_slam_cam, seed, stages=screen_stages,
                max_translation_m=wide_m, max_rotation_deg=wide_deg,
                constrain=False, verify=True, verify_min=0.0,
                # Judged at 10-40 cm resolution, so the fine-stage RMSE limit
                # would reject every candidate including the right one.
                max_rmse_m=self.coarsest_radius_m, gate_context="hint")
            meas.yaw_offset_deg = yaw_deg
            screened.append(meas)
            # Candidates are ordered nearest-first, so a hint that was already
            # good enough stops the search here instead of paying for offsets
            # that cannot beat it.
            if meas.ok and meas.verify_median_m <= self.early_exit_median_m:
                break

        # Phase 2 -- refine a shortlist, not just the leader. A coarse screen
        # ranks candidates only approximately, and in a cluttered room a pose
        # that screens second can refine to the better answer; picking the
        # single leader here was measurably losing good solutions.
        def cost(candidate):
            """How good a candidate is: match quality, with the hint as a prior.

            Match quality is the fraction of the view landing within
            `verify_threshold_m` of the scan -- deliberately NOT the median
            distance. Measured on this scan from an ambiguous viewpoint, a
            solution 233 mm out of place had the *better* median (29 mm vs
            38 mm) because it aligned some structure very tightly; what gave it
            away was the tail, where it left far more outliers (86 % within
            100 mm against 94 %). Ranking on the median picked the wrong pose;
            ranking on the fraction picks the right one.

            The hint then breaks genuine ties, which a cluttered room produces
            often: several placements match about equally well and the operator
            has told us which part of the scan to believe.
            """
            if not candidate.ok:
                return float("inf")
            offset, _ = self._camera_delta(
                T_hint, candidate.T_slam_to_map, T_slam_cam)
            return ((1.0 - candidate.verify_score)
                    + self.hint_weight * offset
                    + 0.01 * candidate.verify_median_m)   # tiebreak only

        ranked = sorted(screened,
                        key=lambda c: (not c.ok, cost(c) if c.ok
                                       else -c.verify_score))
        best = ranked[0]
        cutoff = (cost(ranked[0]) * self.shortlist_ratio
                  if ranked[0].ok else float("inf"))
        for candidate in ranked[:max(1, self.refine_candidates)]:
            if not candidate.ok:
                continue
            if candidate is not ranked[0] and cost(candidate) > cutoff:
                break
            refined = self._measure(
                depth_mm, K, T_slam_cam, candidate.T_slam_to_map,
                stages=((2, self.coarse_radius_m * 0.5, self.initial_iterations),
                        (1, self.fine_radius_m, self.initial_iterations)),
                max_translation_m=wide_m, max_rotation_deg=wide_deg,
                constrain=False, verify=True, gate_context="hint")
            refined.yaw_offset_deg = candidate.yaw_offset_deg
            if refined.ok and (not best.ok or cost(refined) < cost(best)):
                best = refined
            elif best is ranked[0] and not best.ok and not refined.ok:
                best = refined          # carry the specific rejection reason

        # A shortlist entry that only ever passed the loose screening gate is
        # not a lock: re-judge the winner against the real acceptance test.
        if best.ok and best.verify_score < self.min_verify_score:
            best = MapMeasurement(
                reason=(f"only {best.verify_score:.0%} of the view lands on "
                        f"the scan (need {self.min_verify_score:.0%}); "
                        f"median miss {best.verify_median_m * 1000:.0f}mm"),
                fitness=best.fitness, rmse_m=best.rmse_m,
                n_inliers=best.n_inliers, verify_score=best.verify_score,
                verify_median_m=best.verify_median_m,
                n_points=best.n_points)

        if best.ok:
            # Report movement against the hint the operator actually placed,
            # not against the yaw-rotated seed the winning candidate started
            # from -- this is what shows the fusion did real work.
            best.translation_m, best.rotation_deg = self._camera_delta(
                T_hint, best.T_slam_to_map, T_slam_cam)

        best.elapsed_ms = (time.time() - t0) * 1000.0
        self.last_measurement = best
        self._update_diagnostics(best)

        if not best.ok:
            self.diagnostics["lock_ok"] = False
            self.diagnostics["search_region_m"] = self.hint_radius_m
            self.diagnostics["search_region_deg"] = self.hint_radius_deg
            print(f"[LiDAR] ❌ could not place the camera on the scan: "
                  f"{best.reason}")
            if "outside the search region" in best.reason:
                print(f"[LiDAR]    the scan does not match the camera view "
                      f"anywhere within {self.hint_radius_m:.2f} m / "
                      f"{self.hint_radius_deg:.0f}° of the hint. Move the "
                      f"locator closer to where the camera really is, show it "
                      f"more of the room, or widen lidar.hint_radius_m.")
            return best

        return self.accept_hint_solution(best, source)

    def accept_hint_solution(self, best, source, captured_at=None):
        """Install a verified result after the caller validates its scene context."""
        if not best.ok or best.T_slam_to_map is None:
            raise ValueError("Only a successful scan match can be installed")
        self._update_diagnostics(best)
        # Accept the solve outright: there is no prior estimate worth blending
        # toward, and slew-limiting a crawl from a hand-placed hint would take
        # minutes. Continuous fusion holds it from here.
        self._generation += 1
        best.monotonic_timestamp = time.monotonic() if captured_at is None else captured_at
        self.last_measurement = best
        self.T_slam_to_map = best.T_slam_to_map.copy()
        self.is_aligned = True
        self.has_correction = True
        self.alignment_source = source
        self.confidence = best.verify_score
        self._last_fuse_time = None
        self.diagnostics["lidar_aligned"] = True
        self.diagnostics["alignment_source"] = source
        self.diagnostics["hint_offset_mm"] = best.translation_m * 1000.0
        self.diagnostics["hint_offset_deg"] = best.rotation_deg
        self.diagnostics["lock_verify_score"] = float(best.verify_score)
        self.diagnostics["lock_verify_median_mm"] = float(
            best.verify_median_m * 1000.0)
        self.diagnostics["lock_ok"] = True
        self.diagnostics["search_region_m"] = self.hint_radius_m
        self.diagnostics["search_region_deg"] = self.hint_radius_deg
        print(f"[LiDAR] ✅ camera placed on the scan from the {source} hint: "
              f"solved {best.translation_m * 1000:.0f} mm / "
              f"{best.rotation_deg:.2f}° away from the hint "
              f"(yaw start {best.yaw_offset_deg:+.0f}°, "
              f"{best.verify_score:.0%} of the view on the scan, "
              f"median miss {best.verify_median_m * 1000:.0f} mm, "
              f"{best.elapsed_ms:.0f} ms)")
        return best

    #: Older name; the hint semantics are the point, so the new name is
    #: preferred, but bridge_headless.py and existing callers still work.
    align_from_seed = solve_from_hint

    def _yaw_candidates(self):
        """Yaw offsets to try around the hint, nearest first.

        Capped by the search region: starting a candidate outside the region
        the answer must end up in only wastes an ICP run.
        """
        steps = max(1, self.seed_yaw_steps)
        span = min(self.seed_yaw_search_deg, self.hint_radius_deg)
        if steps == 1 or span <= 0.0:
            return [0.0]
        offsets = list(np.linspace(-span, span, steps))
        return sorted(offsets, key=abs)

    @staticmethod
    def _yaw_about_up(T_hint, T_slam_cam, yaw_deg, up_map=None):
        """Swing the hinted camera around the vertical through its own position.

        About the *world* vertical, not the camera's own axis. The operator's
        heading error comes from dragging the locator's heading ring, which
        turns about the world up; those two axes only coincide for a level
        camera, and a camera tilted down at a table (the case this was found
        on) recovers barely half its heading error if the search swings about
        the wrong one.

        Up in the map frame is recovered from the hint itself: cuVSLAM's frame
        is gravity-aligned by the IMU, so SLAM +Z is up, and `T_hint` carries
        that into the map. No assumption about how the scan was exported --
        this room's own vertical is +Z, the previous export's was +Y.
        """
        if abs(yaw_deg) < 1e-9:
            return np.asarray(T_hint, dtype=float).copy()
        T_hint = np.asarray(T_hint, dtype=float)
        if up_map is None:
            up = T_hint[:3, :3] @ np.array([0.0, 0.0, 1.0])
        else:
            up = np.asarray(up_map, dtype=float).reshape(3)
        norm = np.linalg.norm(up)
        if norm < 1e-9:
            return T_hint.copy()
        up = up / norm

        R = so3_exp(up * math.radians(yaw_deg))
        # Rotate about the line through the camera's position, so the hint's
        # location is preserved and only its heading changes.
        p = (T_hint @ np.asarray(T_slam_cam, dtype=float))[:3, 3]
        M = np.eye(4)
        M[:3, :3] = R
        M[:3, 3] = p - R @ p
        return M @ T_hint

    @staticmethod
    def _better(candidate, incumbent):
        """Rank two measurements: a verified solve beats anything unverified.

        Lock-on candidates are ranked by `cost()` inside solve_from_hint, which
        also weighs the hint; this is the plain comparison for everything else.
        """
        if candidate.ok != incumbent.ok:
            return candidate.ok
        return candidate.verify_score > incumbent.verify_score

    # ── running the continuous loop off the caller's thread ─────────────────
    #
    # ICP against a room scan costs 100-200 ms per fix. Called straight from
    # the ROS depth callback that is where it runs, so the single-threaded
    # executor cannot deliver odometry while it works -- measured on hardware
    # as 89 Hz of SLAM arriving but only 16 Hz of FreeD going out. The sensor
    # thread now only hands over the newest frame and returns.

    def start_worker(self):
        """Run continuous matching on our own thread. Idempotent."""
        if self._worker is not None:
            return False
        self._worker_stop.clear()
        self._worker = threading.Thread(target=self._worker_loop,
                                        name="lidar-fusion", daemon=True)
        self._worker.start()
        return True

    def stop_worker(self, timeout=2.0):
        if self._worker is None:
            return
        self._worker_stop.set()
        with self._pending_lock:
            self._pending_lock.notify_all()
        self._worker.join(timeout=timeout)
        self._worker = None

    def submit(self, depth_mm, K, current_pose, timestamp=None,
               is_stationary=False):
        """Hand the worker the newest depth frame and return immediately.

        Single-slot: a frame that arrives while ICP is busy replaces the one
        waiting rather than queueing behind it. Matching a stale frame would
        produce a correction for where the camera *was*, which is worse than
        skipping it.
        """
        if not self._ready() or not self.is_aligned:
            return False
        if self._worker is None:
            self.update_depth(depth_mm, K, current_pose, timestamp,
                              is_stationary)
            return True
        with self._pending_lock:
            self._pending = (np.array(depth_mm, copy=True),
                             np.asarray(K, dtype=float).copy(), current_pose,
                             time.monotonic() if timestamp is None else timestamp,
                             is_stationary, self._generation)
            self._pending_lock.notify()
        return True

    def _worker_loop(self):
        while not self._worker_stop.is_set():
            with self._pending_lock:
                while self._pending is None and not self._worker_stop.is_set():
                    self._pending_lock.wait(timeout=0.5)
                job, self._pending = self._pending, None
            if job is None or self._worker_stop.is_set():
                continue
            try:
                if time.monotonic() - job[3] > self.config.get("max_measurement_age_s", 0.6):
                    continue
                self.update_depth(*job)
            except Exception as exc:                   # keep the thread alive
                print(f"[LiDAR] fusion worker error: {exc}")

    # ── the continuous loop ──────────────────────────────────────────────────
    def update_depth(self, depth_mm, K, current_pose, timestamp=None,
                     is_stationary=False, generation=None):
        """Run one ICP fix against the map and fuse it. Returns MapMeasurement.

        Rate-limited to `icp_rate_hz`; off-tick calls return None cheaply so
        this is safe to call on every depth frame.
        """
        if not self._ready() or not self.is_aligned:
            return None

        version = self._generation
        if generation is not None and generation != version:
            return None
        now = time.monotonic() if timestamp is None else float(timestamp)
        if now - self.last_icp_time < (1.0 / max(self.icp_rate_hz, 1e-6)):
            return None
        self.last_icp_time = now

        if self.freeze_when_stationary and is_stationary:
            return None

        meas = self._measure(
            depth_mm, K, current_pose, self.T_slam_to_map,
            stages=((2, self.coarse_radius_m, self.continuous_iterations),
                    (1, self.fine_radius_m, self.continuous_iterations)),
            max_translation_m=self.max_correction_m,
            max_rotation_deg=self.max_correction_deg,
            constrain=True, verify=True)

        if version != self._generation:
            return None
        meas.monotonic_timestamp = now
        if generation is not None and time.monotonic() - now > self.config.get("max_measurement_age_s", 0.6):
            meas.ok, meas.reason = False, "Scan solve too old for live tracking"
        self.last_measurement = meas
        if meas.ok:
            self._fuse(meas, current_pose, now)
        else:
            self.diagnostics["rejects"] += 1
            self.diagnostics["last_reject"] = meas.reason
            # Confidence decays while the map is not being seen, so downstream
            # consumers can tell "locked on" from "coasting on VIO alone".
            self.confidence *= 0.8
        self._update_diagnostics(meas)
        return meas

    def _measure(self, depth_mm, K, current_pose, T_prior, stages,
                 max_translation_m, max_rotation_deg, constrain, verify=False,
                 verify_min=None, max_rmse_m=None, gate_context="drift"):
        """One absolute measurement of T_slam_to_map from a depth frame."""
        t0 = time.time()
        self._ensure_pyramid()
        T_slam_cam = self._as_transform(current_pose)
        if T_slam_cam is None:
            return MapMeasurement(reason="no SLAM pose")

        pts_optical = self._deproject(depth_mm, K)
        if len(pts_optical) < self.min_inlier_points:
            return MapMeasurement(
                reason=(f"only {len(pts_optical)} depth points in the "
                        f"{self.min_range_m:.1f}-{self.max_range_m:.0f} m band "
                        f"— point the camera at more of the room"),
                n_points=len(pts_optical))

        # Place the cloud in the map frame using the prior, so ICP starts from
        # the current best guess and only has to find the residual.
        T_map_optical = T_prior @ T_slam_cam @ T_BODY_OPTICAL
        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(pts_optical)
        source.transform(T_map_optical)
        source = source.voxel_down_sample(voxel_size=self.voxel_size)
        n_points = len(source.points)
        if n_points < self.min_inlier_points:
            # Almost always the same situation: the camera is close to a flat
            # surface, so the whole view collapses into a handful of voxels and
            # there is no structure to register against. Saying so is far more
            # use to an operator than "cloud too sparse".
            span = float(np.ptp(np.asarray(source.points), axis=0).max()
                         ) if n_points else 0.0
            return MapMeasurement(
                reason=(f"view spans only {span:.2f} m ({n_points} points) — "
                        f"too flat or too close to match; back away so more of "
                        f"the room is in frame"),
                n_points=n_points)

        T_fix, result = self._run_icp(source, stages)
        elapsed = (time.time() - t0) * 1000.0

        if result is None:
            return MapMeasurement(reason="ICP produced no result",
                                  elapsed_ms=elapsed, n_points=n_points)

        n_inliers = len(np.asarray(result.correspondence_set))
        if result.fitness < self.min_fitness:
            return MapMeasurement(
                reason=f"fitness {result.fitness:.2f} < {self.min_fitness:.2f}",
                fitness=result.fitness, rmse_m=result.inlier_rmse,
                n_inliers=n_inliers, elapsed_ms=elapsed, n_points=n_points)
        if n_inliers < self.min_inlier_points:
            return MapMeasurement(
                reason=f"{n_inliers} inliers < {self.min_inlier_points}",
                fitness=result.fitness, rmse_m=result.inlier_rmse,
                n_inliers=n_inliers, elapsed_ms=elapsed, n_points=n_points)
        # Tuned for the finest stage; a coarse screening pass is *expected* to
        # sit further off, so callers judging one pass a limit that matches its
        # resolution instead.
        rmse_limit = self.max_rmse_m if max_rmse_m is None else float(max_rmse_m)
        if result.inlier_rmse > rmse_limit:
            return MapMeasurement(
                reason=(f"rmse {result.inlier_rmse * 1000:.0f}mm > "
                        f"{rmse_limit * 1000:.0f}mm"),
                fitness=result.fitness, rmse_m=result.inlier_rmse,
                n_inliers=n_inliers, elapsed_ms=elapsed, n_points=n_points)

        observability, rank = self._observability(result)
        if constrain:
            T_fix = self._constrain(T_fix, observability)
        if rank == 0:
            return MapMeasurement(
                reason="geometry degenerate in all 3 axes (featureless view)",
                fitness=result.fitness, rmse_m=result.inlier_rmse,
                n_inliers=n_inliers, observability=observability,
                elapsed_ms=elapsed, n_points=n_points)

        # T_fix corrects the cloud inside the map frame, so it left-composes
        # onto the prior. This is the step the old code got wrong by reading
        # the translation column instead.
        T_measured = T_fix @ T_prior

        translation, rotation_deg = self._camera_delta(
            T_prior, T_measured, T_slam_cam)
        if translation > max_translation_m:
            reason = (
                f"best match is {translation * 1000:.0f}mm from the hint, "
                f"outside the search region of {max_translation_m * 1000:.0f}mm"
                if gate_context == "hint" else
                f"correction {translation * 1000:.0f}mm > "
                f"{max_translation_m * 1000:.0f}mm")
            return MapMeasurement(
                reason=reason,
                fitness=result.fitness, rmse_m=result.inlier_rmse,
                n_inliers=n_inliers, translation_m=translation,
                rotation_deg=rotation_deg, observability=observability,
                observable_rank=rank, elapsed_ms=elapsed, n_points=n_points)
        if rotation_deg > max_rotation_deg:
            reason = (
                f"best match is {rotation_deg:.1f}° from the hint, outside "
                f"the search region of {max_rotation_deg:.0f}°"
                if gate_context == "hint" else
                f"rotation {rotation_deg:.2f}° > {max_rotation_deg:.2f}°")
            return MapMeasurement(
                reason=reason,
                fitness=result.fitness, rmse_m=result.inlier_rmse,
                n_inliers=n_inliers, translation_m=translation,
                rotation_deg=rotation_deg, observability=observability,
                observable_rank=rank, elapsed_ms=elapsed, n_points=n_points)

        verify_score, verify_median = 0.0, 0.0
        if verify:
            verify_score, verify_median = self._verify(source, T_fix)
            if gate_context == "drift":
                prior_score, prior_median = self._verify(source, np.eye(4))
                if (prior_score >= self.min_verify_score
                        and (verify_score + 0.005 < prior_score
                             or (verify_score <= prior_score + 0.005
                                 and verify_median > prior_median + 0.001))):
                    # The current pose matches at least as well. Keep its
                    # alignment and report the successful validation; an ICP
                    # local minimum must not pull a surveyed camera away.
                    T_measured = np.asarray(T_prior, dtype=float).copy()
                    translation, rotation_deg = 0.0, 0.0
                    verify_score, verify_median = prior_score, prior_median
            gate = (self.min_verify_score if verify_min is None
                    else float(verify_min))
            if verify_score < gate:
                return MapMeasurement(
                    reason=(f"only {verify_score:.0%} of the view lands on the "
                            f"scan (need {gate:.0%}); "
                            f"median miss {verify_median * 1000:.0f}mm"),
                    fitness=result.fitness, rmse_m=result.inlier_rmse,
                    n_inliers=n_inliers, translation_m=translation,
                    rotation_deg=rotation_deg, observability=observability,
                    observable_rank=rank, elapsed_ms=(time.time() - t0) * 1000,
                    n_points=n_points, verify_score=verify_score,
                    verify_median_m=verify_median)

        return MapMeasurement(
            ok=True, T_slam_to_map=T_measured, fitness=result.fitness,
            rmse_m=result.inlier_rmse, n_inliers=n_inliers,
            translation_m=translation, rotation_deg=rotation_deg,
            observable_rank=rank, observability=observability,
            elapsed_ms=(time.time() - t0) * 1000, n_points=n_points,
            verify_score=verify_score, verify_median_m=verify_median)

    def verify_pose(self, depth_mm, K, T_slam_cam, T_slam_to_map):
        """Validate a surveyed camera pose against depth without optimizing it.

        A low-error multi-point PnP pose is already a measurement, not an
        arbitrary locator hint. Depth may veto it, but must not silently
        substitute a different ICP local minimum.
        """
        if not self._ready():
            return MapMeasurement(reason="No scan loaded")
        self._ensure_pyramid()
        if self._map_tree is None:
            return MapMeasurement(reason="Scan verification requires scipy")
        optical = self._deproject(depth_mm, K)
        if len(optical) < self.min_inlier_points:
            return MapMeasurement(reason="Too few depth points to verify the survey solve")
        T = np.asarray(T_slam_to_map) @ self._as_transform(T_slam_cam) @ T_BODY_OPTICAL
        points = optical @ T[:3, :3].T + T[:3, 3]
        # Evenly spaced subset keeps the verification deterministic.
        points = points[np.linspace(0, len(points)-1, min(len(points), 1200), dtype=int)]
        distances, indices = self._map_tree.query(points, k=1, workers=1)
        good = distances < self.verify_threshold_m
        score, median = float(np.mean(good)), float(np.median(distances))
        n_inliers = int(np.sum(good))
        min_score = self.config.get("survey_min_verify_score", 0.75)
        max_median = self.config.get("survey_max_median_m", 0.05)
        if score < min_score or median > max_median or n_inliers < self.min_inlier_points:
            return MapMeasurement(reason=f"Survey pose has {score:.0%} depth agreement, median miss {median*1000:.0f} mm",
                                  verify_score=score, verify_median_m=median, n_inliers=n_inliers)
        normals = np.asarray(self.map_cloud.normals)[indices[good]]
        norms = np.linalg.norm(normals, axis=1)
        normals = normals[norms > 1e-6] / norms[norms > 1e-6, None]
        if not len(normals):
            return MapMeasurement(reason="Scan normals cannot verify pose geometry")
        eigenvalues = np.linalg.eigvalsh(normals.T @ normals / len(normals))
        rank = int(np.sum(eigenvalues >= self.observability_eps))
        result = MapMeasurement(ok=True, T_slam_to_map=np.array(T_slam_to_map, copy=True),
                                fitness=score, rmse_m=float(np.sqrt(np.mean(distances[good]**2))),
                                verify_score=score, verify_median_m=median, observable_rank=rank,
                                observability=eigenvalues, n_inliers=n_inliers, n_points=len(points))
        return result

    def _verify(self, source, T_fix):
        """Does the camera's view actually land on the scan? (score, median m).

        Independent of ICP's own fitness, which at a wide search radius reads
        near 1.0 almost wherever the cloud is put and so cannot tell a real
        lock from a confident wrong one. This measures the thing the operator
        actually cares about: with this solution applied, what fraction of the
        depth image falls within `verify_threshold_m` of the scanned surface.
        """
        if self._map_tree is None:
            return 0.0, 0.0
        pts = np.asarray(source.points)
        if len(pts) == 0:
            return 0.0, 0.0
        moved = (T_fix @ np.hstack([pts, np.ones((len(pts), 1))]).T).T[:, :3]
        if len(moved) > 600:                       # a score, not a solver
            idx = np.linspace(0, len(moved)-1, 600, dtype=int)
            moved = moved[idx]
        d, _ = self._map_tree.query(moved, k=1, workers=-1)
        return float(np.mean(d < self.verify_threshold_m)), float(np.median(d))

    def _run_icp(self, source, stages):
        """Coarse-to-fine point-to-plane ICP. Returns (4x4 fix, final result).

        Multi-scale because a single radius cannot do both jobs: wide enough to
        find a 30 cm error means the fine solution is pulled about by distant
        wrong correspondences, and narrow enough to be precise means a 30 cm
        error is invisible.
        """
        T = np.eye(4)
        result = None
        estimator = o3d.pipelines.registration.TransformationEstimationPointToPlane()
        for mult, radius, iterations in stages:
            target = self._map_pyramid.get(mult, self.map_cloud)
            src = (source if mult == 1
                   else source.voxel_down_sample(self.voxel_size * mult))
            if len(src.points) < 3:
                continue
            result = o3d.pipelines.registration.registration_icp(
                src, target, radius, T, estimator,
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=iterations))
            T = np.asarray(result.transformation, dtype=float)
        return T, result

    # ── depth -> points ──────────────────────────────────────────────────────
    def _deproject(self, depth_mm, K):
        """Depth image (16UC1 mm) -> Nx3 points in the camera OPTICAL frame.

        Range-gated to the D455's trustworthy band and thinned by inverse
        variance, so ICP sees a cloud whose density reflects how much each
        sample is actually worth.
        """
        depth = np.asarray(depth_mm)
        if depth.ndim != 2:
            return np.empty((0, 3))
        h, w = depth.shape
        K = np.asarray(K, dtype=float).reshape(3, 3)
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        if fx <= 0.0 or fy <= 0.0:
            return np.empty((0, 3))

        # Stride to land near target_points before any filtering, so cost is
        # bounded regardless of the depth profile the operator selects.
        stride = max(1, int(math.sqrt((h * w) / max(self.target_points * 4, 1))))
        rows = np.arange(0, h, stride)
        cols = np.arange(0, w, stride)
        z = depth[np.ix_(rows, cols)].astype(np.float64) / 1000.0
        uu, vv = np.meshgrid(cols.astype(np.float64), rows.astype(np.float64))

        valid = (z >= self.min_range_m) & (z <= self.max_range_m) & np.isfinite(z)
        if not np.any(valid):
            return np.empty((0, 3))
        z = z[valid]
        uu = uu[valid]
        vv = vv[valid]

        # Inverse-variance thinning: keep probability ~ 1/sigma_z^2, normalised
        # so a sample at weight_ref_range_m is always kept.
        sigma = depth_sigma_m(z, fx, self.baseline_m, self.disparity_sigma_px)
        sigma_ref = depth_sigma_m(self.weight_ref_range_m, fx, self.baseline_m,
                                  self.disparity_sigma_px)
        keep_p = np.clip((sigma_ref / np.maximum(sigma, 1e-9)) ** 2,
                         self.min_keep_fraction, 1.0)
        keep = self._rng.random(z.shape) < keep_p
        if not np.any(keep):
            return np.empty((0, 3))
        z = z[keep]
        uu = uu[keep]
        vv = vv[keep]

        x = (uu - cx) / fx * z
        y = (vv - cy) / fy * z
        return np.stack([x, y, z], axis=1)

    # ── observability ────────────────────────────────────────────────────────
    def _observability(self, result):
        """How well the matched geometry pins down each translation axis.

        Returns (eigenvalue-weighted 3x3 basis as a 3-vector of scores paired
        with directions, rank). A translation along direction v is observable
        in proportion to how much matched surface normal points along v: a room
        corner constrains all three axes, one flat wall constrains one.
        """
        corr = np.asarray(result.correspondence_set)
        fine = self._map_pyramid.get(1, self.map_cloud)
        if corr.size == 0 or not fine.has_normals():
            return np.zeros(3), 0
        normals = np.asarray(fine.normals)[corr[:, 1]]
        norms = np.linalg.norm(normals, axis=1)
        good = norms > 1e-6
        if not np.any(good):
            return np.zeros(3), 0
        normals = normals[good] / norms[good, None]

        # Sum of outer products: eigenvalues are the per-direction constraint
        # strength, normalised to sum to 1.
        M = (normals.T @ normals) / len(normals)
        eigvals, eigvecs = np.linalg.eigh(M)
        self._obs_basis = eigvecs
        rank = int(np.sum(eigvals >= self.observability_eps))
        return eigvals, rank

    def _constrain(self, T_fix, eigvals):
        """Zero the translation correction along unobservable directions.

        Rotation is left alone: it is constrained by normal *direction* spread
        rather than by the same rank test, and a wrongly suppressed rotation is
        harder to recover from than a wrongly suppressed slide.
        """
        basis = getattr(self, "_obs_basis", None)
        if basis is None:
            return T_fix
        mask = (np.asarray(eigvals) >= self.observability_eps).astype(float)
        if mask.all():
            return T_fix
        t = T_fix[:3, 3]
        # Project into the eigenbasis, drop the unobservable components, return.
        t_proj = basis @ (mask * (basis.T @ t))
        out = np.array(T_fix, dtype=float, copy=True)
        out[:3, 3] = t_proj
        return out

    # ── fusion ───────────────────────────────────────────────────────────────
    def _fuse(self, meas, current_pose, now):
        """Blend an accepted measurement into T_slam_to_map, rate-limited."""
        T_slam_cam = self._as_transform(current_pose)
        dt = 1.0 / max(self.icp_rate_hz, 1e-6)
        if self._last_fuse_time is not None:
            dt = min(max(now - self._last_fuse_time, 1e-3), 2.0 / max(self.icp_rate_hz, 1e-6))
        self._last_fuse_time = now

        alpha = float(np.clip(self.blend_alpha * meas.fitness, 0.0, 1.0))
        if alpha <= 0.0:
            return

        candidate = se3_interpolate(self.T_slam_to_map, meas.T_slam_to_map, alpha)

        # Rate-limit on what the operator actually sees -- the camera's own
        # displacement -- not on the twist norm, which for a distant camera
        # understates the movement badly.
        moved_m, moved_deg = self._camera_delta(
            self.T_slam_to_map, candidate, T_slam_cam)
        limit_m = self.max_slew_m_s * dt
        limit_deg = self.max_slew_deg_s * dt
        scale = 1.0
        if moved_m > max(limit_m, 0.0):
            scale = min(scale, max(limit_m, 0.0) / moved_m)
        if moved_deg > max(limit_deg, 0.0):
            scale = min(scale, max(limit_deg, 0.0) / moved_deg)
        if scale < 1.0:
            alpha *= scale
            candidate = se3_interpolate(self.T_slam_to_map,
                                        meas.T_slam_to_map, alpha)
            moved_m, moved_deg = self._camera_delta(
                self.T_slam_to_map, candidate, T_slam_cam)

        self.T_slam_to_map = candidate
        self.confidence = meas.fitness
        self.has_correction = True
        self.diagnostics["fixes"] += 1
        self.diagnostics["applied_mm"] = moved_m * 1000.0
        self.diagnostics["applied_deg"] = moved_deg
        self.fix_counter += 1

    @staticmethod
    def _camera_delta(T_a, T_b, T_slam_cam):
        """How far the camera moves when SLAM->map goes from T_a to T_b.

        This is the honest metric for both the rejection gate and the slew
        limit: `(R - I)p + t`, evaluated at the camera's actual position,
        rather than the translation column the previous implementation used.
        """
        if T_slam_cam is None:
            p = np.zeros(3)
        else:
            p = np.asarray(T_slam_cam, dtype=float)[:3, 3]
        p_h = np.append(p, 1.0)
        moved = float(np.linalg.norm((T_b @ p_h)[:3] - (T_a @ p_h)[:3]))
        R_rel = np.asarray(T_a, dtype=float)[:3, :3].T @ np.asarray(
            T_b, dtype=float)[:3, :3]
        angle = math.degrees(float(np.linalg.norm(so3_log(R_rel))))
        return moved, angle

    # ── output ───────────────────────────────────────────────────────────────
    def map_transform(self):
        """(T_slam_to_map 4x4, confidence 0-1). The fused answer."""
        return self.T_slam_to_map.copy(), self.confidence

    def corrected_pose(self, position, quaternion):
        """A raw SLAM pose -> (position, quaternion) in the map frame."""
        from pose_transform import matrix_to_quaternion
        T = self.T_slam_to_map @ make_transform(position, quaternion)
        return T[:3, 3], matrix_to_quaternion(T[:3, :3])

    def get_diagnostics(self):
        return self.diagnostics

    def _update_diagnostics(self, meas):
        d = self.diagnostics
        d["confidence"] = float(self.confidence)
        d["alignment_source"] = self.alignment_source
        d["lidar_aligned"] = self.is_aligned
        if meas is None:
            return
        d["icp_fitness"] = float(meas.fitness)
        d["icp_rmse_mm"] = float(meas.rmse_m * 1000.0)
        d["residual_mm"] = float(meas.translation_m * 1000.0)
        d["residual_deg"] = float(meas.rotation_deg)
        d["observable_rank"] = int(meas.observable_rank)
        # Only an accepted measurement carries a real verification: the
        # translation/rotation gates fire before verification runs, so a
        # rejected one reports 0 and must not clobber the last good figure.
        if meas.ok:
            d["verify_score"] = float(meas.verify_score)
            d["verify_median_mm"] = float(meas.verify_median_m * 1000.0)
        d["observability"] = [round(float(v), 4) for v in meas.observability]
        d["inliers"] = int(meas.n_inliers)
        d["cloud_points"] = int(meas.n_points)
        d["last_icp_ms"] = float(meas.elapsed_ms)

    # ── helpers ──────────────────────────────────────────────────────────────
    def _ready(self):
        return _HAS_O3D and self.map_cloud is not None

    @staticmethod
    def _as_transform(pose):
        """Accept a 4x4, or the (position, quaternion) pair the callers hold."""
        if pose is None:
            return None
        if isinstance(pose, np.ndarray) and pose.shape == (4, 4):
            return pose.astype(float)
        try:
            position, quaternion = pose
        except (TypeError, ValueError):
            return None
        # ROS message objects expose .x/.y/.z rather than being sequences.
        if hasattr(quaternion, "x"):
            quaternion = (quaternion.x, quaternion.y, quaternion.z, quaternion.w)
        if hasattr(position, "x"):
            position = (position.x, position.y, position.z)
        return make_transform(position, quaternion)

    # ── read-only validation (no fusion, no state change) ────────────────────
    def validate_depth(self, depth_mm, K, T_world_camera):
        """Score a depth frame against the map without correcting anything.

        Answers "does what the camera sees match where we think it is?", which
        is the number an operator watches to decide whether to trust the take.
        Uses the cached KD-tree -- the previous implementation rebuilt one over
        the whole map cloud on every call.
        """
        empty = {"valid": False, "fitness": 0.0, "mean_error_mm": 0.0,
                 "median_error_mm": 0.0, "n_rays": 0, "elapsed_ms": 0.0}
        if not self._ready():
            return empty
        self._ensure_pyramid()
        if self._map_tree is None:
            return empty

        t0 = time.time()
        pts_optical = self._deproject(depth_mm, K)
        if len(pts_optical) < 20:
            empty["n_rays"] = len(pts_optical)
            empty["elapsed_ms"] = round((time.time() - t0) * 1000, 1)
            return empty

        # Cap the sample: this is a status readout, not a solver.
        if len(pts_optical) > 500:
            idx = self._rng.choice(len(pts_optical), 500, replace=False)
            pts_optical = pts_optical[idx]

        T = np.asarray(T_world_camera, dtype=float) @ T_BODY_OPTICAL
        if self.is_aligned:
            T = self.T_slam_to_map @ T
        pts_h = np.hstack([pts_optical, np.ones((len(pts_optical), 1))])
        pts_map = (T @ pts_h.T).T[:, :3]

        inlier_threshold = self.voxel_size * 3
        distances, _ = self._map_tree.query(pts_map, k=1, workers=-1)
        fitness = float(np.mean(distances < inlier_threshold))
        elapsed = (time.time() - t0) * 1000.0
        self.diagnostics["validation_fitness"] = fitness
        self.diagnostics["validation_mean_mm"] = float(np.mean(distances)) * 1000
        self.diagnostics["validation_elapsed_ms"] = elapsed
        return {
            "valid": True,
            "fitness": round(fitness, 3),
            "mean_error_mm": round(float(np.mean(distances)) * 1000, 1),
            "median_error_mm": round(float(np.median(distances)) * 1000, 1),
            "n_rays": len(distances),
            "elapsed_ms": round(elapsed, 1),
        }
