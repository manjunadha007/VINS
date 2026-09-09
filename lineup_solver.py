"""Single-frame surveyed camera lineup; no Qt, ROS, or live output mutations."""
from dataclasses import dataclass
import math
import numpy as np
from pose_transform import T_BODY_OPTICAL, invert_transform


@dataclass
class LineupResult:
    ok: bool = False
    reason: str = ""
    T_map_cam: object = None
    projected: object = None
    residuals_px: object = None
    inliers: object = None
    rms_px: float = 0.0


def pick_mesh_surface(vertices, faces, eye, direction, near=0.05, far=10000):
    """Nearest visible triangle hit (two-sided), not a vertex behind a wall."""
    triangles = np.asarray(vertices)[np.asarray(faces, dtype=int)]
    direction = np.asarray(direction, dtype=float)
    direction /= np.linalg.norm(direction)
    a = triangles[:, 0]
    e1, e2 = triangles[:, 1] - a, triangles[:, 2] - a
    h = np.cross(direction, e2)
    det = np.einsum("ij,ij->i", e1, h)
    valid = np.abs(det) > 1e-10
    inv = np.divide(1.0, det, out=np.zeros_like(det), where=valid)
    s = np.asarray(eye) - a
    u = inv * np.einsum("ij,ij->i", s, h)
    q = np.cross(s, e1)
    v = inv * (q @ direction)
    t = inv * np.einsum("ij,ij->i", e2, q)
    valid &= (u >= 0) & (v >= 0) & (u + v <= 1) & (t >= near) & (t <= far)
    if not np.any(valid):
        return None
    distance = np.min(t[valid])
    return np.asarray(eye) + distance * direction


def solve_lineup(points_3d, points_2d, K, image_size=None, max_error_px=3.0):
    """Solve a rectified D455 image against metric scan points.

    At least six well-spread non-coplanar inliers are required. Fixed calibrated
    intrinsics avoid the focal-length/translation ambiguity of free lens solves.
    """
    import cv2
    world = np.asarray(points_3d, dtype=np.float64).reshape(-1, 3)
    image = np.asarray(points_2d, dtype=np.float64).reshape(-1, 2)
    K = np.asarray(K, dtype=float).reshape(3, 3)
    if len(world) != len(image) or len(world) < 6:
        return LineupResult(reason="Add at least 6 matching point pairs (8–12 recommended)")
    if not all(np.isfinite(a).all() for a in (world, image, K)) or K[0,0] <= 0 or K[1,1] <= 0:
        return LineupResult(reason="Invalid points or camera intrinsics")
    if np.linalg.matrix_rank(world - world.mean(axis=0), tol=0.01) < 3:
        return LineupResult(reason="Points lie on one flat surface. Use points at different depths, across two walls or objects; more points on the same wall or floor will not help.")
    if np.linalg.matrix_rank(image - image.mean(axis=0), tol=1.0) < 2:
        return LineupResult(reason="Spread image points horizontally and vertically")
    if len(np.unique(world.round(5), axis=0)) != len(world) or len(np.unique(image.round(2), axis=0)) != len(image):
        return LineupResult(reason="Remove duplicate point pairs")
    if image_size is not None:
        w,h = image_size
        if np.any(image < 0) or np.any(image >= (w,h)):
            return LineupResult(reason="A selected point is outside the frozen image")
        if np.any(np.ptp(image, axis=0) < np.array((w,h)) * 0.1):
            return LineupResult(reason="Spread points over more of the image")
    try:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            world, image, K, None, iterationsCount=250,
            reprojectionError=max_error_px, confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
        required = max(6, math.ceil(len(world)*0.7))
        consistent = 0 if not ok or inliers is None else len(inliers)
        if consistent < required:
            return LineupResult(reason=f"Too few consistent pairs: {consistent}/{len(world)} (need {required}). Check that each pair marks the same feature; add accurate pairs across different surfaces (8–12 recommended). With only 6 pairs, all 6 must agree.")
        ids = inliers.ravel()
        if np.linalg.matrix_rank(world[ids] - world[ids].mean(axis=0), tol=0.01) < 3:
            return LineupResult(reason="Inlier points are planar; add points at another depth")
        if hasattr(cv2, "solvePnPRefineLM"):
            rvec, tvec = cv2.solvePnPRefineLM(world[ids], image[ids], K, None, rvec, tvec)
        else:
            ok, rvec, tvec = cv2.solvePnP(world[ids], image[ids], K, None,
                                        rvec, tvec, True, flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                return LineupResult(reason="Pose refinement failed")
        R = cv2.Rodrigues(rvec)[0]
        camera_points = world @ R.T + tvec.reshape(1,3)
        if not np.isfinite(camera_points).all() or np.any(camera_points[ids,2] <= 0):
            return LineupResult(reason="Solved points are behind the camera")
        projected = cv2.projectPoints(world, rvec, tvec, K, None)[0].reshape(-1,2)
        residuals = np.linalg.norm(projected-image, axis=1)
        rms = float(np.sqrt(np.mean(residuals[ids] ** 2)))
        if rms > max_error_px:
            return LineupResult(reason=f"Reprojection error {rms:.2f} px exceeds {max_error_px:.1f} px")
        T_optical_map = np.eye(4)
        T_optical_map[:3,:3], T_optical_map[:3,3] = R, tvec.ravel()
        T_map_cam = invert_transform(T_optical_map) @ invert_transform(T_BODY_OPTICAL)
        return LineupResult(True, "", T_map_cam, projected, residuals, ids, rms)
    except cv2.error as exc:
        return LineupResult(reason=f"Camera solve failed: {exc}")
