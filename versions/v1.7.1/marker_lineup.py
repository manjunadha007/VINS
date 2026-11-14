"""Marker-assisted scan lineup from frozen, timestamp-paired observations."""
from dataclasses import dataclass
import copy
import math
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
                               T_display_locator, T_display_map, now=None, *, check_marker_width=True):
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
        ids = ", ".join(str(fid) for fid, _ in sorted(matches, key=lambda m: m[0]))
        return None, (f"More than one marker is visible (IDs {ids}). Select the ID of the "
                      "marker under the Locator in Setup, or assign it to the Locator.")
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
    measured = (marker_width_from_depth(corners, engine.aruco_K, marker_size_m, depth, depth_K)
                if check_marker_width else None)
    if measured is not None and abs(measured / marker_size_m - 1) > .10:
        return None, (f"Marker {fid} measures about {measured*100:.1f} cm across from depth, but Square "
                      f"width is {marker_size_m*100:.1f} cm. Measure the printed square (black border, "
                      "not the paper) and set Square width in Setup.")
    T_map_display = invert_transform(T_display_map)
    return MarkerObservation(
        family, int(fid), corners.copy(), engine.aruco_K.copy(), marker_size_m,
        make_transform(*image_pose), make_transform(*depth_pose),
        T_map_display @ np.asarray(T_display_locator, dtype=float),
        np.array(depth, copy=True), depth_K.copy(),
        T_map_display[:3, :3] @ np.array([0., 0., 1.]),
        min(image_received, received)), ""


def marker_width_from_depth(corners, K, marker_size_m, depth_mm, depth_K, min_pixels=30):
    """The printed square's width implied by depth, or None if depth can't say.

    PnP range scales with the width typed in; depth measures range directly.
    On the rig, squares entered as 10 cm but printed at 16.7 cm (a printer's
    fit-to-page) put every marker 40% too close, and the lineup failed with a
    search error that said nothing about size. Depth is read over the square
    and its white margin, since black print often returns no depth. The depth
    stream shares the IR camera's optical centre on the D4xx.
    """
    import cv2
    corners = np.asarray(corners, dtype=float).reshape(4, 2)
    K, depth_K = np.asarray(K, dtype=float), np.asarray(depth_K, dtype=float)
    s = marker_size_m / 2
    obj = np.array([[-s, s, 0.], [s, s, 0.], [s, -s, 0.], [-s, -s, 0.]])
    ok, _, tvec = cv2.solvePnP(obj, corners, K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None
    rays = (corners - K[:2, 2]) / np.diag(K)[:2]
    pixels = rays * np.diag(depth_K)[:2] + depth_K[:2, 2]
    depth = np.asarray(depth_mm)
    mask = np.zeros(depth.shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, np.round(pixels).astype(np.int32), 1)
    side = np.mean(np.linalg.norm(pixels - np.roll(pixels, 1, axis=0), axis=1))
    grow = max(3, int(side * .25)) | 1
    mask = cv2.dilate(mask, np.ones((grow, grow), np.uint8))
    z = depth[(mask > 0) & (depth > 0)].astype(float) / 1000.
    if z.size < min_pixels:
        return None
    centre = np.append(rays.mean(axis=0), 1.)
    return float(marker_size_m * np.median(z) * np.linalg.norm(centre) / np.linalg.norm(tvec))


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


def level_marker(R, up):
    """R snapped to the nearest exact mount: lying flat, or upright on a wall.

    A marker lies on a floor or table, or is fixed to a wall; a measured tilt
    in between is noise. Returns (rotation, on_wall). The printed top (+X)
    keeps its direction as far as the new face allows.
    """
    up = np.asarray(up, dtype=float) / np.linalg.norm(up)
    R = np.asarray(R, dtype=float)
    lean = R[:, 2] @ up
    on_wall = abs(lean) < math.cos(math.radians(45))
    face = R[:, 2] - up * lean if on_wall else up * np.sign(lean)
    face /= np.linalg.norm(face)
    top = R[:, 0] - face * (R[:, 0] @ face)
    top /= np.linalg.norm(top)
    return np.column_stack((top, np.cross(face, top), face)), on_wall


def _turn_onto(a, b):
    """The smallest rotation taking unit vector a onto unit vector b."""
    axis = np.cross(a, b)
    angle = math.atan2(np.linalg.norm(axis), float(a @ b))
    if np.linalg.norm(axis) < 1e-12:
        return np.eye(3) if angle < 1 else so3_exp(np.eye(3)[np.argmin(np.abs(a))] * np.pi)
    return so3_exp(axis / np.linalg.norm(axis) * angle)


def marker_heading_seeds(observation, offset=None):
    """Sweep camera AND heading around the fixed marker, never around the camera.

    `offset` (scan metres) moves the marker centre the sweep turns about, for
    a Locator that was not placed exactly on the marker.

    A floor or table marker is assumed flat in the scan, whatever the Locator's
    rotation. A wall marker cannot be: which way it faces is unknown and what
    is up on it is not. It is told apart by its tilt against gravity -- the
    SLAM frame is gravity-aligned with IMU fusion on, and wall markers on this
    rig measured 87-88 deg from vertical -- and keeps that measured attitude,
    stood exactly upright, while the sweep turns it about the vertical.
    """
    centre = observation.T_map_locator[:3, 3]
    if offset is not None:
        centre = centre + np.asarray(offset, dtype=float)
    up = observation.up_map / np.linalg.norm(observation.up_map)
    flat = flat_locator(centre, up)
    slam_to_map_up = _turn_onto(np.array([0., 0., 1.]), up)
    marker_to_locator = np.eye(4)
    marker_to_locator[:3, :3] = rpy_to_matrix(0, 0, np.pi / 2)
    branches = []
    for optical_marker in optical_marker_poses(observation):
        slam_locator = observation.T_slam_image @ T_BODY_OPTICAL @ optical_marker @ marker_to_locator
        upright, on_wall = level_marker(slam_locator[:3, :3], [0., 0., 1.])
        lean = np.degrees(np.arccos(np.clip(slam_locator[:3, 2] @ upright[:, 2], -1, 1)))
        branches.append((lean, slam_locator, upright, on_wall))
    # The other planar PnP branch often leans well off both mounts (57 deg off
    # flat in the floor fixture); a marker cannot, so that branch is dropped.
    # If no branch fits, SLAM is not gravity-aligned and every marker is taken
    # as flat, as before.
    plausible = [b for b in branches if b[0] <= 25]
    # Seen from above, a floor marker's other branch can stand upright and
    # pass as a wall marker, adding a competitor: Room01 floor views lost 8 of
    # 128 lineups to "ambiguous" that way. A wall marker's other branch only
    # lies flat from a steep view, where it reprojects too badly to survive.
    if any(not on_wall for _, _, _, on_wall in plausible):
        plausible = [b for b in plausible if not b[3]]
    if not plausible:
        plausible = [(lean, slam_locator, None, False) for lean, slam_locator, _, _ in branches]
    for _, slam_locator, upright, on_wall in plausible:
        target = flat.copy()
        if on_wall:
            target[:3, :3] = slam_to_map_up @ upright
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
            R = so3_exp(up * np.radians(degrees))
            rotated = target.copy()
            rotated[:3, :3] = R @ target[:3, :3]
            yield rotated @ invert_transform(slam_locator), slam_locator, degrees


def marker_search_offsets(radius_m, step_m):
    """In-plane Locator offsets on a `step_m` grid, nearest first.

    Every point within `radius_m` is within half a grid diagonal of a start,
    including at the edge. The centre itself is left out: its heading seeds
    are always searched.
    """
    if not (radius_m > 0 and step_m > 0):
        return []
    reach = radius_m + step_m / math.sqrt(2)
    n = int(reach / step_m)
    offsets = [(i * step_m, j * step_m) for i in range(-n, n + 1) for j in range(-n, n + 1)
               if (i or j) and math.hypot(i, j) * step_m <= reach + 1e-9]
    return sorted(offsets, key=lambda o: (math.hypot(*o), o))


def offset_heading_seeds(observation, matcher, points, T_slam_optical,
                         radius_m, step_m, keep, deadline):
    """The best heading seeds around marker positions away from the Locator.

    ICP only pulls a start in from about a decimetre, so a Locator placed
    further off than that needs starts near the real marker. ICP-screening a
    whole grid of them is far too slow, so each is first ranked by how close a
    few hundred depth points land to the scan, and only `keep` go on to ICP.
    """
    up = observation.up_map / np.linalg.norm(observation.up_map)
    across = np.eye(3)[np.argmin(np.abs(up))]
    across = across - up * (across @ up)
    across /= np.linalg.norm(across)
    along = np.cross(up, across)
    probe = points[np.linspace(0, len(points) - 1, min(len(points), 400), dtype=int)]
    seeds, costs = [], []

    def rank(batch):
        # One threaded query per ~1M points: per-offset queries were mostly
        # thread start-up and took 6x longer.
        T = np.array([seed @ T_slam_optical for seed, _, _ in batch])
        moved = np.einsum("nij,pj->npi", T[:, :3, :3], probe) + T[:, None, :3, 3]
        distances, _ = matcher._map_tree.query(moved.reshape(-1, 3), k=1, workers=-1,
                                               distance_upper_bound=.25)
        costs.extend(np.minimum(distances, .25).reshape(len(batch), -1).mean(axis=1))
        seeds.extend(batch)

    batch = []
    for dx, dy in marker_search_offsets(radius_m, step_m):
        if time.monotonic() > deadline:
            break
        batch.extend(marker_heading_seeds(observation, dx * across + dy * along))
        if len(batch) * len(probe) >= 1_000_000:
            rank(batch)
            batch = []
    if batch:
        rank(batch)
    order = np.argsort(costs, kind="stable")[:keep]
    return [seeds[i] for i in order]


class MarkerMeasurement(MapMeasurement):
    """Verified camera alignment plus the marker heading to show in the locator.

    `T_map_marker` is where the scan put the marker itself, which can be away
    from the Locator by up to the search radius. None keeps the Locator's
    position and updates only its heading.
    """
    __slots__ = ("T_map_locator", "marker_error_m", "T_map_marker")

    def __init__(self, measurement, locator, marker_error, marker=None):
        for field in MapMeasurement.__slots__:
            setattr(self, field, getattr(measurement, field))
        self.T_map_locator = locator
        self.marker_error_m = marker_error
        self.T_map_marker = marker


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
    result._pending_marker = None
    return result


def solve_marker_observation(observation, matcher, *, estimate_heading=True,
                             search_radius_m=None, search_step_m=.15, search_starts=24):
    """Position-only marker lineup; a failed search never installs an alignment.

    The entire heading circle is checked, including both planar PnP branches,
    around the Locator and around grid points up to `search_radius_m` from it
    (default: lidar.hint_radius_m). Final matches must pass fine ICP, the
    marker-position bound, the floor-normal check and an ambiguity test
    against distinct camera poses.
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
    radius = matcher.hint_radius_m if search_radius_m is None else float(search_radius_m)

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
        # How far the solved marker leans off lying flat or standing upright.
        mount, _ = level_marker(found[:3, :3], up)
        tilt = float(np.degrees(np.arccos(np.clip(found[:3, 2] @ mount[:, 2], -1, 1))))
        return distance, tilt, found

    coarse = []
    last_reason = "No heading matched the scan. Check scan scale, locator height and marker width."
    screen = ((8, matcher.coarsest_radius_m, 30), (4, matcher.coarse_radius_m, 30),
              (2, matcher.coarse_radius_m * .5, 30))
    seeds = list(marker_heading_seeds(observation))
    seeds += offset_heading_seeds(observation, matcher, points, T_slam_optical,
                                  radius, search_step_m, search_starts, started + 4)
    for seed, slam_locator, degrees in seeds:
        if time.monotonic() - started > 12:
            return MapMeasurement(reason="Heading search timed out. Hold still and try again with more room geometry visible.")
        measurement = matcher._measure(
            observation.depth, observation.depth_K, observation.T_slam_depth, seed,
            stages=screen, max_translation_m=matcher.hint_radius_m,
            max_rotation_deg=matcher.hint_radius_deg, constrain=False,
            verify=True, verify_min=0, max_rmse_m=matcher.coarsest_radius_m,
            gate_context="hint")
        # The 0.4 m level can drag a start that already fits far off: from the
        # exact pose, a close view of a table top on Room01 moved 239 mm / 24
        # deg, so the true pose never reached refinement and the solve locked
        # onto a wrong one. A start that fits better than its ICP result goes
        # on to refinement as it is; the fine stages kept that one within 4 mm.
        fit = score(measurement) if measurement.ok else float("inf")
        start = MapMeasurement(ok=True, T_slam_to_map=seed)
        start_fit = score(start)
        if start.verify_score >= matcher.min_verify_score and start_fit <= fit:
            measurement, fit = start, start_fit
        if not measurement.ok:
            last_reason = measurement.reason
            continue
        distance, tilt, _ = marker_error(measurement, slam_locator)
        if distance > radius or tilt > 20:
            continue
        # Like solve_from_hint: when placements match about equally well, the
        # one nearer the Locator wins.
        cost = fit + matcher.hint_weight * distance
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
        if distance > radius:
            last_reason = f"Solved marker is {distance*1000:.0f} mm from the locator (limit {radius*1000:.0f} mm). Check its position, height and the printed width."
            continue
        if tilt > 20:
            last_reason = "Marker pose is neither flat on a floor/table nor upright on a wall. Show a clearer marker and check scan orientation."
            continue
        if measurement.observable_rank < 3:
            last_reason = f"Only {measurement.observable_rank}/3 position axes are constrained. Include walls, corners or fixed furniture around the marker."
            continue
        cost = score(measurement) + matcher.hint_weight * distance
        if measurement.verify_score < matcher.min_verify_score:
            last_reason = f"Only {measurement.verify_score:.0%} depth agreement. Show more scanned geometry and check scan scale."
            continue
        locator = np.eye(4)
        locator[:3, :3], _ = level_marker(found[:3, :3], up)
        locator[:3, 3] = centre
        marker = locator.copy()
        marker[:3, 3] = found[:3, 3]
        measurement.yaw_offset_deg = candidate.yaw_offset_deg
        result = MarkerMeasurement(measurement, locator, distance, marker)
        accepted.append((cost, result))
    accepted.sort(key=lambda item: item[0])
    if not accepted:
        return MapMeasurement(reason="No verified marker heading. " + last_reason)
    best = accepted[0][1]
    for cost, other in accepted[1:]:
        if distinct(best, other) and cost - accepted[0][0] < .05:
            return MapMeasurement(reason="Marker heading is ambiguous: different camera positions match the scan almost equally well. Show a distinctive corner or furniture while keeping the marker visible.")
    # Results within `distinct` of the winner are the same place found twice.
    # The Locator prior chooses between places, not within one: on Room01 a
    # wall view's true pose (19 mm median miss) lost to one 127 mm / 5.5 deg
    # off (30 mm) for sitting 4 cm nearer the Locator, both 100% on the scan.
    best = min((other for _, other in accepted if not distinct(best, other)),
               key=lambda other: other.verify_median_m)
    best.elapsed_ms = (time.monotonic() - started) * 1000
    best.monotonic_timestamp = observation.captured_at
    return best
