"""Find where the camera is inside the scan, with no hint from the operator.

The marker lineup needs a locator: where the printed marker sits in the scan.
Placing that by hand is the one step nobody can do accurately -- the marker is
not in the scan, so there is nothing to snap to, and a locator rotated by 90
degrees swings the camera estimate three quarters of a metre (measured on a
10 cm marker at half a metre).

This removes the guess. A feature-based global registration puts the live depth
cloud into the scan without any prior, which gives the camera pose; the marker's
pose follows from the camera, and the locator follows from the marker.

It is deliberately separate from LidarMapMatcher: that class refines a pose it
already roughly knows, a few times a second. This runs once, takes seconds, and
answers a different question -- "where am I at all?".
"""
import numpy as np

try:
    import open3d as o3d
    _HAS_O3D = True
except ImportError:                                   # pragma: no cover
    _HAS_O3D = False

#: Below this, the answer is not trustworthy enough to anchor a shoot to.
#: Measured on a D455 against a room scan: a correct fix scores 0.92-0.96 and a
#: wrong one 0.54-0.65, so the gap is wide and the threshold is not delicate.
MIN_FITNESS = 0.90

#: Points closer than this to the camera are mostly the operator's own hands
#: and the desk edge; further than this the D455's stereo noise dominates.
MIN_RANGE_M, MAX_RANGE_M = 0.25, 5.0


def depth_to_cloud(depth_mm, K, stride=1):
    """Depth image (16UC1 mm) -> Nx3 points in the camera optical frame."""
    depth = np.asarray(depth_mm)
    if depth.ndim != 2:
        return np.empty((0, 3))
    K = np.asarray(K, dtype=float).reshape(3, 3)
    rows = np.arange(0, depth.shape[0], stride)
    cols = np.arange(0, depth.shape[1], stride)
    z = depth[np.ix_(rows, cols)].astype(float) / 1000.0
    uu, vv = np.meshgrid(cols.astype(float), rows.astype(float))
    ok = np.isfinite(z) & (z > MIN_RANGE_M) & (z < MAX_RANGE_M)
    if not np.any(ok):
        return np.empty((0, 3))
    return np.stack([(uu[ok] - K[0, 2]) * z[ok] / K[0, 0],
                     (vv[ok] - K[1, 2]) * z[ok] / K[1, 1], z[ok]], axis=1)


def _prepare(points, voxel):
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(points)))
    cloud = cloud.voxel_down_sample(voxel)
    cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 3, max_nn=30))
    feature = o3d.pipelines.registration.compute_fpfh_feature(
        cloud, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5, max_nn=100))
    return cloud, feature


def locate_in_scan(points_optical, scan_points, voxel=0.03,
                   min_fitness=MIN_FITNESS, attempts=12):
    """Camera optical pose inside the scan, or a reason it could not be found.

    RANSAC is restarted rather than run once with a huge iteration budget: a
    room has repeated structure, so the search is bimodal and a restart that
    lands in the right basin converges immediately. Stopping as soon as a fix
    clears the threshold keeps the common case near a second.
    """
    if not _HAS_O3D:
        return dict(ok=False, reason="open3d is not installed", fitness=0.0)
    points_optical = np.asarray(points_optical, dtype=float)
    if len(points_optical) < 500:
        return dict(ok=False, fitness=0.0,
                    reason="too little depth in view to localise against the scan")

    live, live_f = _prepare(points_optical, voxel)
    scan, scan_f = _prepare(scan_points, voxel)
    checkers = [
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel * 2)]
    best = None
    for _ in range(max(1, int(attempts))):
        coarse = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            live, scan, live_f, scan_f, True, voxel * 2,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3, checkers,
            o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 0.999))
        fine = o3d.pipelines.registration.registration_icp(
            live, scan, voxel * 1.7, coarse.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
        if best is None or fine.fitness > best.fitness:
            best = fine
        if best.fitness >= min_fitness:
            break

    fitness = float(best.fitness)
    if fitness < min_fitness:
        return dict(ok=False, fitness=fitness,
                    T_map_optical=np.asarray(best.transformation).copy(),
                    reason=(f"only {fitness:.0%} of the view matches the scan "
                            f"(need {min_fitness:.0%}) — aim the camera at more "
                            f"scanned geometry"))
    return dict(ok=True, fitness=fitness, rmse_m=float(best.inlier_rmse),
                T_map_optical=np.asarray(best.transformation).copy(), reason="")
