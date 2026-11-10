"""Measure the scan's scale against the camera's own metric depth.

A phone or photogrammetry scan can come out a few percent too small or too
large, and a lineup from one view of a room corner never shows it: three
planes meeting at a point look the same at any size, so the fit just slides
the camera nearer or further. Markers then sit where depth puts them, not
where they are on the scan.

The D455's depth is metric (checked against a tape-measured marker to about
1.5%), and so is the tracking. Depth gathered while the camera pans across the
room, each frame placed by its own tracked pose, holds walls metres apart. A
similarity fit of that cloud to the scan -- rotation, translation and one
scale -- then reads the scan's scale off the room itself.

Scale is refused, not guessed, when the swept geometry cannot pin it down:
the fit reports how much of its scale is independent of sliding the camera
(the "lever"). Sweeps across a room measured 1.3-1.5 m; corners 0.02 m in
simulation and 0.16 m on the rig, where clutter adds a little. At 0.16 m a
free fit still shrank the rig's corner to the 0.65x limit with 81% of it
"fitting", so scale is held, and refused, below 0.5 m.
"""
import math
import numpy as np
from scipy.spatial import cKDTree
from pose_transform import so3_exp


def voxel_down(points, voxel):
    """One point (the mean) per occupied voxel."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if not len(points):
        return points
    keys = np.floor(points / voxel).astype(np.int64)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()
    sums = np.zeros((len(counts), 3))
    np.add.at(sums, inverse, points)
    return sums / counts[:, None]


def surface_normals(points, k=12):
    """Unit normals from each point's k nearest neighbours (sign arbitrary)."""
    _, idx = cKDTree(points).query(points, k=min(k, len(points)))
    neighbours = points[idx] - points[idx].mean(axis=1, keepdims=True)
    return np.linalg.svd(np.einsum("nki,nkj->nij", neighbours, neighbours))[0][:, :, 2]


class DepthSweep:
    """Depth frames gathered into the tracking (SLAM) frame as the camera pans."""

    def __init__(self, voxel=.03):
        self.voxel = float(voxel)
        self.frames = 0
        self._chunks = []

    def add(self, points_optical, T_slam_optical):
        points = np.asarray(points_optical, dtype=float).reshape(-1, 3)
        if not len(points):
            return False
        T = np.asarray(T_slam_optical, dtype=float)
        self._chunks.append(voxel_down(points @ T[:3, :3].T + T[:3, 3], self.voxel))
        self.frames += 1
        return True

    def points(self):
        if not self._chunks:
            return np.empty((0, 3))
        return voxel_down(np.vstack(self._chunks), self.voxel)


def fit_locator_scale(centres, targets, *, min_baseline_m=.3, min_shape_ratio=.08,
                      max_error_m=.04, depth_accuracy=.01):
    """Fit scan ~= s * R * metric marker centres + t using every assigned pair.

    Three non-collinear, depth-backed centres establish metric scale even in
    a single room corner. Rotation is free in 3D; neither a tilted scan nor a
    small SLAM gravity error should masquerade as a misplaced Locator.
    Residuals and degeneracy thresholds are evaluated in real metres. This
    fit never edits individual targets or the physical printed marker widths.
    """
    a, b = (np.asarray(p, dtype=float) for p in (centres, targets))
    if (a.ndim != 2 or a.shape[1:] != (3,) or b.shape != a.shape or len(a) < 3
            or not np.isfinite(a).all() or not np.isfinite(b).all()):
        return dict(ok=False, reason="Need at least three finite, paired marker centres with RealSense depth.")
    x, y = a - a.mean(axis=0), b - b.mean(axis=0)
    for points in (x, y):
        shape = np.linalg.svd(points, compute_uv=False)
        if shape[0] <= 1e-12 or shape[1] / shape[0] < min_shape_ratio:
            return dict(ok=False, reason="The three Locators must form a triangle, not a line; use more widely separated markers.")
    distances = np.linalg.norm(a[:, None] - a[None, :], axis=2)
    if distances.max() < min_baseline_m:
        return dict(ok=False, reason="The markers are too close together to measure scale reliably.")
    U, singular, Vt = np.linalg.svd(y.T @ x)
    signs = np.array([1., 1., np.linalg.det(U @ Vt)])
    R = (U * signs) @ Vt
    s = float(singular @ signs / np.sum(x * x))
    if not np.isfinite(s) or s <= 1e-12:
        return dict(ok=False, reason="The marker triangle does not give a positive scan scale.")
    t = b.mean(axis=0) - s * R @ a.mean(axis=0)
    miss = np.linalg.norm(s * a @ R.T + t - b, axis=1) / s
    pairs = np.triu_indices(len(a), 1)
    pair_error = np.abs(np.linalg.norm(b[:, None] - b[None, :], axis=2) / s - distances)
    if (miss.max() > max_error_m or
            np.any(pair_error[pairs] > max_error_m + 2 * depth_accuracy * distances[pairs])):
        return dict(ok=False, reason=("The locator triangle and depth-measured marker triangle do not agree "
                    "under one uniform scale. Check marker IDs and depth, then measure again; "
                    "the scan and Locator placements were kept."))
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = s * R, t
    rms = float(np.sqrt(np.mean(miss ** 2)))
    lever = float(np.sqrt(np.mean(np.sum(x * x, axis=1))))
    return dict(ok=True, reason="", scale=s, T=T, marker_count=len(a),
                marker_rmse_m=rms, lever_m=lever,
                sigma=math.hypot(depth_accuracy, rms / lever))


def verify_locator_scale(slam_points, scan_points, fit, *, voxel=.03,
                         min_points=500, min_inliers=.6, threshold_m=.08,
                         max_points=80000):
    """Check a locator similarity against independent room depth in real metres.

    Keep the triangle's scale and pose fixed: a free ICP fit in a corner can
    collapse scale or slide onto another wall. The scan is converted to its
    proposed metric size before voxelisation and correspondence thresholds.
    This also supports unit corrections much larger than a few percent.
    """
    if not fit["ok"]:
        return fit
    s, T = fit["scale"], fit["T"]

    def cloud(points):
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        points = voxel_down(points[np.isfinite(points).all(axis=1)], voxel)
        if len(points) > max_points:
            points = points[np.linspace(0, len(points) - 1, max_points, dtype=int)]
        return points

    src, scan = cloud(slam_points), cloud(np.asarray(scan_points) / s)
    if min(len(src), len(scan)) < min_points:
        return dict(fit, ok=False, reason="Too few depth or scan points to verify scale; show more scanned surfaces.")
    placed = src @ (T[:3, :3] / s).T + T[:3, 3] / s
    distances, _ = cKDTree(scan).query(placed, distance_upper_bound=threshold_m, workers=-1)
    matched = np.isfinite(distances)
    inliers = float(matched.mean())
    rmse = float(np.sqrt(np.mean(distances[matched] ** 2))) if matched.any() else math.inf
    result = dict(fit, inliers=inliers, rmse_m=rmse, points=len(src))
    if inliers < min_inliers:
        return dict(result, ok=False, reason=(
            f"The three-marker scale has only {inliers:.0%} RealSense depth agreement with the scan "
            f"(need {min_inliers:.0%}). Show more scanned surfaces and check marker IDs. "
            "The scan and Locator placements were kept."))
    return result


def measure_scale(slam_points, scan_points, T_slam_to_map, *, voxel=.03,
                  stages=((.25, 15), (.12, 15), (.06, 20), (.04, 20)),
                  min_points=2000, min_inliers=.5, min_lever_m=.5, max_change=.3,
                  huber_m=.02, max_sweep_points=80000, depth_accuracy=.01):
    """Fit scan ~= s * R * slam + t, starting from the lineup's alignment.

    `scan_points` are the scan in map metres at its current scale; the result's
    `scale` is how big the scan is relative to the room the camera measured,
    so the corrected map scale is the current one divided by it.

    `T_slam_to_map` is a lineup (rigid), or a similarity such as the markers'
    fit (`marker_similarity`), whose own scale is then the first start.

    Point-to-plane, with a Huber weight: depth the scan does not have (clutter,
    wall above where the scan stops) falls outside the correspondence radius
    or is down-weighted, and does not drag the scale. The scale is taken about
    the matched points' centroid, so it is not tangled up with translation.

    Where the sweep cannot constrain scale, a free fit shrinks the cloud
    towards a point, so scale is held at 1 whenever its lever is too short and
    the result is refused; otherwise each step changes it by at most 2%.

    `sigma` includes `depth_accuracy`: the measurement is only as metric as the
    camera's depth, and a scan within about two sigma of 1 needs no change.

    Returns a dict: ok, scale, sigma (1-sigma, as a fraction), lever_m,
    inliers (fraction within the last stage's radius), rmse_m, T (the similarity), points, reason.
    """
    src = voxel_down(slam_points, voxel)
    if len(src) > max_sweep_points:
        src = src[np.linspace(0, len(src) - 1, max_sweep_points, dtype=int)]
    if len(src) < min_points:
        return dict(ok=False, reason=f"Only {len(src)} depth points were gathered; pan across more of the room.")
    scan = voxel_down(scan_points, voxel)
    normals = surface_normals(scan)
    tree = cKDTree(scan)
    T_seed = np.asarray(T_slam_to_map, dtype=float)
    seed_scale = float(np.cbrt(np.linalg.det(T_seed[:3, :3])))
    R_seed = T_seed[:3, :3] / seed_scale
    # Scale about where the sweep sits, which the lineup (or the markers) put right.
    sweep_on_map = T_seed[:3, :3] @ src.mean(axis=0) + T_seed[:3, 3]

    def fit_from(s0):
        s, R = s0, R_seed.copy()
        t = sweep_on_map - s0 * R @ src.mean(axis=0)

        def matched(radius):
            y = s * src @ R.T + t
            d, i = tree.query(y, distance_upper_bound=radius, workers=-1)
            ok = np.isfinite(d)
            return y[ok], scan[i[ok]], normals[i[ok]]

        def lever_of(H, weight):
            """RMS lever (m) the scale column keeps once rotation and translation take their share."""
            schur = H[6, 6] - H[6, :6] @ np.linalg.solve(H[:6, :6] + 1e-9 * np.eye(6), H[:6, 6])
            return math.sqrt(max(schur, 0.) / weight), schur

        for radius, iterations in stages:
            for _ in range(iterations):
                y, q, n = matched(radius)
                if len(y) < min_points:
                    return dict(ok=False, reason=(
                        f"Only {len(y)} depth points land near the scan. Line up first, "
                        "or check that this is the right scan."))
                r = np.sum(n * (y - q), axis=1)
                c = y.mean(axis=0)
                J = np.column_stack((np.cross(y, n), n, np.sum(n * (y - c), axis=1)))
                w = np.minimum(1., huber_m / np.maximum(np.abs(r), 1e-12))
                H, g = (J * w[:, None]).T @ J, -(J * w[:, None]).T @ r
                step = np.zeros(7)
                if lever_of(H, w.sum())[0] < min_lever_m:
                    # Scale is not observable from this geometry: left free, it
                    # wanders (a corner shrinks to its apex) and drags the rest off
                    # the scan. Hold it and fit the rigid part only.
                    step[:6] = np.linalg.solve(H[:6, :6] + 1e-9 * np.eye(6), g[:6])
                else:
                    step = np.linalg.solve(H + 1e-9 * np.eye(7), g)
                    step[6] = np.clip(step[6], -.02, .02)
                    step[6] = np.clip(s * (1 + step[6]), 1 - max_change - .05, 1 + max_change + .05) / s - 1
                turn, grow = so3_exp(step[:3]), 1. + step[6]
                s, R, t = s * grow, turn @ R, grow * turn @ t - step[6] * turn @ c + step[3:6]
                if np.linalg.norm(step) < 1e-6:
                    break

        y, q, n = matched(stages[-1][0])
        r = np.sum(n * (y - q), axis=1)
        c = y.mean(axis=0)
        J = np.column_stack((np.cross(y, n), n, np.sum(n * (y - c), axis=1)))
        # What is left of the scale column once rotation and translation have taken
        # what they can: zero for a lone corner, which looks the same at any size.
        lever, schur = lever_of(J.T @ J, len(y))
        # Neighbouring residuals are not independent (depth noise and scan error
        # both come in patches); 5x is a conservative allowance for that.
        fit = 5 * float(np.sqrt(np.mean(r ** 2))) / math.sqrt(max(schur, 1e-12))
        sigma = math.hypot(fit, depth_accuracy)
        inliers = len(y) / len(src)
        result = dict(scale=s, sigma=sigma, lever_m=lever, inliers=inliers,
                      rmse_m=float(np.sqrt(np.mean(r ** 2))), points=len(src))
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = s * R, t
        result["T"] = T
        if inliers < min_inliers:
            return dict(result, ok=False, reason=(
                f"Only {inliers:.0%} of the swept depth fits the scan after scaling. "
                "Check that this is the right scan and that the lineup was close."))
        if lever < min_lever_m:
            return dict(result, ok=False, reason=(
                "This sweep can't measure scale: what it saw looks the same at any size "
                "(a corner, or one wall and the floor). Pan across the room so the camera "
                "sees walls on opposite sides, or furniture, then measure again."))
        if abs(s - 1) > max_change:
            return dict(result, ok=False, reason=(
                f"The scan would need scaling by {1 / s:.2f}x, more than a scan's units can be "
                "off by mistake. Check its units (lidar.map_scale) and the lineup."))
        return dict(result, ok=True, reason="")

    # A scan 8% off leaves far walls ~20 cm out at scale 1, beyond what the
    # fit can match, so their lever never shows and scale stays held. Starting
    # either side of 1 as well reaches them; the start that fits most wins.
    starts = [fit_from(s0) for s0 in [seed_scale] + [s0 for s0 in (1., .92, 1.08)
                                                     if abs(s0 - seed_scale) > 1e-6]]
    good = [result for result in starts if result["ok"]]
    return max(good, key=lambda r: (r["inliers"], -r["rmse_m"])) if good else starts[0]
