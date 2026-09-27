"""Marker-assisted scan lineup from frozen, timestamp-paired observations."""
from dataclasses import dataclass
import copy
import time
import numpy as np
from pose_transform import (T_BODY_OPTICAL, invert_transform, make_transform,
                            rpy_to_matrix, so3_exp)
from lidar_map_matcher import MapMeasurement


@dataclass
class MarkerObservation:
    family: str
    marker_id: int
    corners: np.ndarray
    K: np.ndarray
    marker_size_m: float
    T_slam_image: np.ndarray
    T_slam_depth: np.ndarray
    T_map_locator: np.ndarray
    depth: np.ndarray
    depth_K: np.ndarray
    up_map: np.ndarray
    captured_at: float


def capture_marker_observation(engine, detections, image_stamp, image_received,
                               image_size, family, marker_id, marker_size_m,
                               T_display_locator, T_display_map, now=None):
    """Return (observation, reason), without anchoring or changing fusion."""
    now = time.monotonic() if now is None else now
    if image_received is None or not 0 <= now - image_received <= 0.5:
        return None, "Waiting for a fresh marker image."
    if not engine.intrinsics_from_camera_info:
        return None, "Waiting for calibrated camera intrinsics."
    matches = [(fid, quad) for name, fid, quad in detections
               if name == family and (marker_id < 0 or fid == marker_id)]
    if not matches:
        expected = "a marker" if marker_id < 0 else f"marker {marker_id}"
        return None, f"Waiting for {family} / {expected}."
    if len(matches) != 1:
        return None, "More than one matching marker is visible. Select its ID or show only one."
    image_pose = engine.pose_at(image_stamp)
    depth_sample = engine.capture_depth()
    if image_pose is None or depth_sample is None:
        return None, "Waiting for fresh depth and timestamp-paired image/depth camera poses."
    if engine._pose_rejected:
        return None, "Waiting for accepted camera tracking."
    marker_size_m = float(marker_size_m)
    if not np.isfinite(marker_size_m) or marker_size_m <= 0:
        return None, "Enter the measured marker square width in metres."
    fid, quad = matches[0]
    corners = np.asarray(quad, dtype=float).reshape(4, 2)
    if (not np.isfinite(corners).all() or np.any(corners < 0)
            or np.any(corners >= image_size)):
        return None, "The marker corners are outside the camera image."
    depth, depth_K, depth_pose, received = depth_sample
    T_map_display = invert_transform(T_display_map)
    return MarkerObservation(
        family, int(fid), corners.copy(), engine.aruco_K.copy(), marker_size_m,
        make_transform(*image_pose), make_transform(*depth_pose),
        T_map_display @ np.asarray(T_display_locator, dtype=float),
        np.array(depth, copy=True), depth_K.copy(),
        T_map_display[:3, :3] @ np.array([0., 0., 1.]),
        min(image_received, received)), ""


def optical_marker_poses(observation):
    """Keep both valid planar PnP branches for independent scan verification."""
    import cv2
    s = observation.marker_size_m / 2
    obj = np.array([[-s, s, 0.], [s, s, 0.],
                    [s, -s, 0.], [-s, -s, 0.]], dtype=np.float64)
    image = np.ascontiguousarray(observation.corners, dtype=np.float64)
    if abs(cv2.contourArea(image.astype(np.float32))) < 25:
        raise ValueError("Marker is too small in the image; move closer.")
    result = cv2.solvePnPGeneric(obj, image, observation.K, None,
                                flags=cv2.SOLVEPNP_IPPE_SQUARE)
    poses = []
    for rvec, tvec in zip(result[1], result[2]):
        R = cv2.Rodrigues(rvec)[0]
        camera_points = obj @ R.T + tvec.reshape(1, 3)
        if not np.isfinite(camera_points).all() or np.any(camera_points[:, 2] <= 0):
            continue
        projected = cv2.projectPoints(obj, rvec, tvec, observation.K, None)[0].reshape(4, 2)
        rms = float(np.sqrt(np.mean(np.sum((projected - image) ** 2, axis=1))))
        if not np.isfinite(rms) or rms > 3:
            continue
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, tvec.ravel()
        poses.append((rms, T))
    if not poses:
        raise ValueError("Marker pose is unreliable; move closer and show a sharper, unobstructed square.")
    return [T for _, T in sorted(poses, key=lambda item: item[0])]


def floor_marker_hint(observation):
    """Fixed-heading helper for callers that already know the locator rotation."""
    T_optical_marker = optical_marker_poses(observation)[0]
    T_locator_marker = np.eye(4)
    T_locator_marker[:3, :3] = rpy_to_matrix(0, 0, -np.pi / 2)
    T_map_marker = observation.T_map_locator @ T_locator_marker
    T_slam_marker = observation.T_slam_image @ T_BODY_OPTICAL @ T_optical_marker
    return T_map_marker @ invert_transform(T_slam_marker)


def flat_locator(position, up, top=None):
    """A level marker target in map coordinates, even when the displayed scan tilts."""
    up = np.asarray(up, dtype=float)
    up = up / np.linalg.norm(up)
    if top is None:
        top = np.eye(3)[np.argmin(np.abs(up))]
    top = np.asarray(top, dtype=float) - up * np.dot(top, up)
    if np.linalg.norm(top) < 1e-8:
        raise ValueError("Marker top direction is perpendicular to the floor.")
    top /= np.linalg.norm(top)
    T = np.eye(4)
    T[:3, :3] = np.column_stack((top, np.cross(up, top), up))
    T[:3, 3] = position
    return T


def marker_heading_seeds(observation):
    """Sweep camera AND heading around the fixed marker, never around the camera."""
    target = flat_locator(observation.T_map_locator[:3, 3], observation.up_map)
    marker_to_locator = np.eye(4)
    marker_to_locator[:3, :3] = rpy_to_matrix(0, 0, np.pi / 2)
    for optical_marker in optical_marker_poses(observation):
        slam_locator = observation.T_slam_image @ T_BODY_OPTICAL @ optical_marker @ marker_to_locator
        # 10 deg, not 15. Half a step of yaw about the marker leaves the seed
        # camera 2*d*sin(step/2) out, and that grows with the horizontal marker
        # distance d: 15 deg is 89 mm at 0.7 m, which the coarse screen still
        # recovers from, but 121 mm at 0.9 m, which it does not -- the level-8
        # stage wanders 33 deg off, trips the 20 deg hint bound, and the solve
        # is rejected with "only 76% of the view lands on the scan", pointing
        # the operator at scan scale and marker width when the only problem is
        # standing 20 cm too far back. 10 deg keeps the offset under ~80 mm out
        # to ~1.2 m for 24 more measures, ~0.2 s against a 12 s search budget.
        for degrees in range(0, 360, 10):
            R = so3_exp(target[:3, 2] * np.radians(degrees))
            rotated = target.copy()
            rotated[:3, :3] = R @ target[:3, :3]
            yield rotated @ invert_transform(slam_locator), slam_locator, degrees


class MarkerMeasurement(MapMeasurement):
    """Verified camera alignment plus the marker heading to show in the locator."""
    __slots__ = ("T_map_locator", "marker_error_m")

    def __init__(self, measurement, locator, marker_error):
        for field in MapMeasurement.__slots__:
            setattr(self, field, getattr(measurement, field))
        self.T_map_locator = locator
        self.marker_error_m = marker_error


def locator_from_scan(observation, T_map_optical):
    """The locator the operator would have placed, derived from the scan.

    Inverts the lineup: instead of the operator saying where the marker is and
    the fusion solving for the camera, the scan says where the camera is and
    this solves for the marker. The result is exactly what `floor_marker_hint`
    expects to be handed, so nothing downstream changes.

    The +90 deg is the same convention as everywhere else -- the locator's +X
    points at the marker's printed top, which is 90 deg from the marker's own
    OpenCV frame. Getting that backwards is worth three quarters of a metre of
    camera error on a 10 cm marker at half a metre, and it looks plausible on
    screen, so it is derived here rather than typed by hand.
    """
    import cv2
    s = observation.marker_size_m / 2
    obj = np.array([[-s, s, 0.], [s, s, 0.],
                    [s, -s, 0.], [-s, -s, 0.]], dtype=np.float64)
    image = np.ascontiguousarray(observation.corners, dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, image, observation.K, None,
                                  flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        raise ValueError("Marker pose could not be solved.")
    T_optical_marker = np.eye(4)
    T_optical_marker[:3, :3] = cv2.Rodrigues(rvec)[0]
    T_optical_marker[:3, 3] = tvec.ravel()
    T_map_marker = np.asarray(T_map_optical, dtype=float) @ T_optical_marker
    T_marker_locator = np.eye(4)
    T_marker_locator[:3, :3] = rpy_to_matrix(0, 0, np.pi / 2)
    return T_map_marker @ T_marker_locator


def matcher_snapshot(matcher):
    """A private solver sharing immutable scan geometry, never live alignment.

    Map imports/scaling replace their arrays and clouds. Keep these references
    so an in-flight job can finish, then let the GUI discard an obsolete job.
    Rebuilt pyramids and diagnostics belong only to this copy.
    """
    result = copy.copy(matcher)
    result.T_slam_to_map = matcher.T_slam_to_map.copy()
    result.diagnostics = dict(matcher.diagnostics)
    result._map_pyramid = dict(matcher._map_pyramid)
    result._rng = np.random.default_rng(0)
    result.edge_aligner = None
    result._worker = None
    result._pending = None
    return result


def solve_marker_observation(observation, matcher, *, estimate_heading=True):
    """Position-only marker lineup; a failed search never installs an alignment.

    The entire heading circle is checked, including both planar PnP branches.
    Final matches must pass fine ICP, the original marker-position bound, the
    floor-normal check and an ambiguity test against distinct camera poses.
    """
    if not estimate_heading:
        hint = floor_marker_hint(observation)
        measurement = matcher.solve_from_hint(
            observation.depth, observation.depth_K, observation.T_slam_depth,
            hint, source="aruco", up_map=observation.up_map)
        measurement.monotonic_timestamp = observation.captured_at
        return measurement
    started = time.monotonic()
    if not matcher._ready():
        return MapMeasurement(reason="Load a scan before marker lineup.")
    matcher._ensure_pyramid()
    if matcher._map_tree is None:
        return MapMeasurement(reason="Scan verification requires scipy.")
    points = matcher._deproject(observation.depth, observation.depth_K)
    if len(points) < matcher.min_inlier_points:
        return MapMeasurement(reason=f"Only {len(points)} usable depth points; show the marker plus more scanned room geometry.")
    # Score every candidate on exactly the same rays, without voxel-order bias.
    points = points[np.linspace(0, len(points)-1, min(len(points), 1600), dtype=int)]
    T_slam_optical = observation.T_slam_depth @ T_BODY_OPTICAL
    centre = observation.T_map_locator[:3, 3]
    up = observation.up_map / np.linalg.norm(observation.up_map)

    def score(measurement):
        T = measurement.T_slam_to_map @ T_slam_optical
        moved = points @ T[:3, :3].T + T[:3, 3]
        distances, _ = matcher._map_tree.query(moved, k=1, workers=1)
        measurement.verify_score = float(np.mean(distances < matcher.verify_threshold_m))
        measurement.verify_median_m = float(np.median(distances))
        return 1.0 - measurement.verify_score + .01 * measurement.verify_median_m

    def distinct(a, b):
        distance, angle = matcher._camera_delta(a.T_slam_to_map, b.T_slam_to_map,
                                                observation.T_slam_depth)
        return distance > .15 or angle > 8.0

    def marker_error(measurement, slam_locator):
        found = measurement.T_slam_to_map @ slam_locator
        distance = float(np.linalg.norm(found[:3, 3] - centre))
        tilt = float(np.degrees(np.arccos(np.clip(found[:3, 2] @ up, -1, 1))))
        return distance, tilt, found

    coarse = []
    last_reason = "No heading matched the scan. Check scan scale, locator height and marker width."
    screen = ((8, matcher.coarsest_radius_m, 30), (4, matcher.coarse_radius_m, 30),
              (2, matcher.coarse_radius_m * .5, 30))
    for seed, slam_locator, degrees in marker_heading_seeds(observation):
        if time.monotonic() - started > 12:
            return MapMeasurement(reason="Heading search timed out. Hold still and try again with more room geometry visible.")
        measurement = matcher._measure(
            observation.depth, observation.depth_K, observation.T_slam_depth, seed,
            stages=screen, max_translation_m=matcher.hint_radius_m,
            max_rotation_deg=matcher.hint_radius_deg, constrain=False,
            verify=True, verify_min=0, max_rmse_m=matcher.coarsest_radius_m,
            gate_context="hint")
        if not measurement.ok:
            last_reason = measurement.reason
            continue
        distance, tilt, _ = marker_error(measurement, slam_locator)
        if distance > matcher.hint_radius_m or tilt > 20:
            continue
        cost = score(measurement)
        measurement.yaw_offset_deg = degrees
        coarse.append((cost, measurement, slam_locator))

    # Refine distinct basins; many adjacent yaw seeds converge to the same pose.
    shortlist = []
    for candidate in sorted(coarse, key=lambda item: item[0]):
        if all(distinct(candidate[1], old[1]) for old in shortlist):
            shortlist.append(candidate)
    accepted = []
    for _, candidate, slam_locator in shortlist[:8]:
        if time.monotonic() - started > 12:
            return MapMeasurement(reason="Heading search timed out. Hold still and try again.")
        measurement = matcher._measure(
            observation.depth, observation.depth_K, observation.T_slam_depth,
            candidate.T_slam_to_map,
            stages=((2, matcher.coarse_radius_m * .5, matcher.initial_iterations),
                    (1, matcher.fine_radius_m, matcher.initial_iterations)),
            max_translation_m=matcher.hint_radius_m,
            max_rotation_deg=matcher.hint_radius_deg, constrain=False,
            verify=True, gate_context="hint")
        if not measurement.ok:
            last_reason = measurement.reason
            continue
        distance, tilt, found = marker_error(measurement, slam_locator)
        if distance > matcher.hint_radius_m:
            last_reason = f"Solved marker is {distance*1000:.0f} mm from the locator (limit {matcher.hint_radius_m*1000:.0f} mm). Check its position, height and the printed width."
            continue
        if tilt > 20:
            last_reason = "Marker pose disagrees with a flat floor/table. Show a clearer marker and check scan orientation."
            continue
        if measurement.observable_rank < 3:
            last_reason = f"Only {measurement.observable_rank}/3 position axes are constrained. Include walls, corners or fixed furniture around the marker."
            continue
        cost = score(measurement)
        if measurement.verify_score < matcher.min_verify_score:
            last_reason = f"Only {measurement.verify_score:.0%} depth agreement. Show more scanned geometry and check scan scale."
            continue
        locator = flat_locator(centre, up, found[:3, 0])
        measurement.yaw_offset_deg = candidate.yaw_offset_deg
        result = MarkerMeasurement(measurement, locator, distance)
        accepted.append((cost, result))
    accepted.sort(key=lambda item: item[0])
    if not accepted:
        return MapMeasurement(reason="No verified marker heading. " + last_reason)
    best = accepted[0][1]
    for cost, other in accepted[1:]:
        if distinct(best, other) and cost - accepted[0][0] < .05:
            return MapMeasurement(reason="Marker heading is ambiguous: different camera positions match the scan almost equally well. Show a distinctive corner or furniture while keeping the marker visible.")
    best.elapsed_ms = (time.monotonic() - started) * 1000
    best.monotonic_timestamp = observation.captured_at
    return best
