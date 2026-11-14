"""Surveyed ArUco layouts and joint, scan-verified camera pose estimation.

Layout poses use the Locator convention: +X towards the printed top, +Z out
of the printed face. Coordinates are scan metres, before its display transform.
"""
from dataclasses import dataclass, replace
import itertools
import json
import math
from pathlib import Path
import numpy as np

from lidar_map_matcher import MapMeasurement
from marker_lineup import _turn_onto, capture_marker_observation, level_marker, optical_marker_poses
from pose_transform import T_BODY_OPTICAL, invert_transform, matrix_to_rpy, rpy_to_matrix, so3_exp

#: Every ID of DICT_4X4_250 can have its own Locator.
MAX_MARKERS = 250


def validate_layout(records):
    """Return independent JSON-safe entries; never invent marker locations."""
    import cv2
    if not isinstance(records, list) or len(records) > MAX_MARKERS:
        raise ValueError(f"A marker layout must contain a list of at most {MAX_MARKERS} markers.")
    result, seen = [], set()
    for entry in records:
        if not isinstance(entry, dict):
            raise ValueError("Each marker must have an ID, width and surveyed pose.")
        family, fid = entry.get("dictionary", "DICT_4X4_50"), entry.get("id")
        if family not in ("DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250", "DICT_4X4_1000"):
            raise ValueError("Use a supported DICT_4X4 dictionary.")
        count = len(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, family)).bytesList)
        if isinstance(fid, bool) or not isinstance(fid, int) or not 0 <= fid < count:
            raise ValueError(f"{family} needs an integer ID from 0 to {count-1}.")
        key = (family, fid)
        if key in seen:
            raise ValueError(f"Duplicate marker {family}/{fid}; each physical marker needs a unique ID.")
        seen.add(key)
        size = float(entry.get("size_m", 0))
        pos = np.asarray(entry.get("position_m", []), dtype=float)
        rot = np.asarray(entry.get("rotation_deg", []), dtype=float)
        if not math.isfinite(size) or not .005 <= size <= 5:
            raise ValueError(f"Marker {fid}: measure the printed square width in metres.")
        if pos.shape != (3,) or rot.shape != (3,) or not np.isfinite(np.r_[pos, rot]).all():
            raise ValueError(f"Marker {fid}: enter all three position and rotation coordinates.")
        result.append(dict(dictionary=family, id=fid, size_m=size,
                           position_m=pos.tolist(), rotation_deg=rot.tolist()))
    return result


def locator_transform(entry):
    T = np.eye(4)
    T[:3, :3] = rpy_to_matrix(*np.radians(entry["rotation_deg"]))
    T[:3, 3] = entry["position_m"]
    return T


def layout_entry(family, fid, size_m, T_map_locator):
    """A layout record for a marker at T_map_locator (scan metres)."""
    return validate_layout([dict(dictionary=family, id=int(fid), size_m=float(size_m),
        position_m=np.asarray(T_map_locator)[:3, 3].tolist(),
        rotation_deg=np.degrees(matrix_to_rpy(np.asarray(T_map_locator)[:3, :3])).tolist())])[0]


class MarkerSurvey:
    """Places markers the camera sees but nobody surveyed, from the locked camera.

    Each sighting gives a marker pose in scan metres for each planar PnP
    branch. The wrong branch swings with the viewpoint while the right one
    stays put, so a marker is placed only when one pose is backed by
    `min_sightings` recent sightings and no clearly different pose is backed
    by half as many. Holding the camera still therefore never places a marker
    whose two branches both look steady; a little camera movement settles it.

    A placed marker carries the camera's error at the time, so it holds later
    tracking to this lineup; it adds no accuracy the lineup did not have.
    """

    def __init__(self, min_sightings=6, tolerance_m=.03, tolerance_deg=6.,
                 min_marker_px=24., max_age_s=10.):
        self.min_sightings = max(2, int(min_sightings))
        self.tolerance_m = float(tolerance_m)
        self.tolerance_deg = float(tolerance_deg)
        self.min_marker_px = float(min_marker_px)
        self.max_age_s = float(max_age_s)
        self._tracks = {}

    def clear(self):
        self._tracks.clear()

    def observe(self, key, corners, K, size_m, T_map_optical, now):
        """Record one sighting of marker `key`; False if it was too small or unclear."""
        import cv2
        corners = np.asarray(corners, dtype=float).reshape(4, 2)
        if math.sqrt(abs(cv2.contourArea(corners.astype(np.float32)))) < self.min_marker_px:
            return False
        sighting = _Sighting(marker_size_m=float(size_m), corners=corners, K=np.asarray(K, dtype=float))
        try:
            poses = optical_marker_poses(sighting)
        except (ValueError, cv2.error):
            return False
        candidates = [np.asarray(T_map_optical) @ T @ _MARKER_TO_LOCATOR for T in poses]
        track = [s for s in self._tracks.get(key, ())
                 if 0 <= now - s[0] <= self.max_age_s and s[1] == float(size_m)]
        track.append((now, float(size_m), candidates))
        self._tracks[key] = track[-3 * self.min_sightings:]
        return True

    def ready(self):
        """Remove and return every settled marker as (key, size_m, T_map_locator)."""
        placed = []
        for key, track in list(self._tracks.items()):
            pose = self._settled(track)
            if pose is not None:
                placed.append((key, track[-1][1], pose))
                del self._tracks[key]
        return placed

    def _settled(self, track):
        if len(track) < self.min_sightings:
            return None
        poses = np.array([T for _, _, candidates in track for T in candidates])
        owner = np.array([i for i, (_, _, candidates) in enumerate(track) for _ in candidates])
        distance = np.linalg.norm(poses[:, None, :3, 3] - poses[None, :, :3, 3], axis=2)
        cosine = (np.einsum("aij,bij->ab", poses[:, :3, :3], poses[:, :3, :3]) - 1) / 2
        angle = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
        close = (distance <= self.tolerance_m) & (angle <= self.tolerance_deg)
        sightings = np.zeros((len(poses), len(track)), dtype=bool)
        for i in range(len(track)):
            sightings[:, i] = close[:, owner == i].any(axis=1)
        support = sightings.sum(axis=1)
        best = int(np.argmax(support))
        if support[best] < self.min_sightings:
            return None
        rivals = (distance[best] > 2 * self.tolerance_m) | (angle[best] > 2 * self.tolerance_deg)
        if rivals.any() and 2 * support[rivals].max() >= self.min_sightings:
            return None
        members = poses[close[best]]
        U, _, Vt = np.linalg.svd(members[:, :3, :3].sum(axis=0))
        T = np.eye(4)
        T[:3, :3] = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
        T[:3, 3] = members[:, :3, 3].mean(axis=0)
        return T


@dataclass
class _Sighting:
    marker_size_m: float
    corners: np.ndarray
    K: np.ndarray


#: Marker (OpenCV) frame to Locator frame: +X toward the printed top.
_MARKER_TO_LOCATOR = np.eye(4)
_MARKER_TO_LOCATOR[:3, :3] = rpy_to_matrix(0, 0, np.pi / 2)


def layout_signature(records):
    # Compare physical orientations, including equivalent Euler angles at
    # +/-180 degrees and gimbal lock, so selecting a Locator cannot cancel it.
    return tuple(sorted((e["dictionary"], e["id"], round(e["size_m"], 6),
        *np.round(e["position_m"], 6), *np.round(locator_transform(e)[:3,:3].ravel(), 8)) for e in records))


def save_layout(path, records):
    records = validate_layout(records)
    Path(path).write_text(json.dumps(dict(format="hyperfusion.markers", version=1,
        coordinates="scan_metres", markers=records), indent=2) + "\n")


def load_layout(path):
    data = json.loads(Path(path).read_text())
    if (not isinstance(data, dict) or data.get("format") != "hyperfusion.markers"
            or data.get("version") != 1 or data.get("coordinates") != "scan_metres"):
        raise ValueError("Choose a HyperFusion marker layout in scan metres.")
    return validate_layout(data.get("markers"))


@dataclass
class MappedObservation:
    markers: tuple

    @property
    def captured_at(self):
        return self.markers[0].captured_at

    @property
    def T_slam_depth(self):
        return self.markers[0].T_slam_depth

    @property
    def marker_id(self):
        return ", ".join(str(m.marker_id) for m in self.markers)

    def with_layout(self, records):
        """The same sighting, against Locator poses that have since changed."""
        poses = {(e["dictionary"], e["id"]): locator_transform(e) for e in records}
        return MappedObservation(tuple(replace(m, T_map_locator=poses[m.family, m.marker_id])
                                       for m in self.markers))


def capture_mapped_observation(engine, detections, image_stamp, image_received,
                               image_size, records, T_display_map, now=None, *, check_marker_width=True):
    """Use only configured IDs from ONE image, paired with its own SLAM pose."""
    entries = {(e["dictionary"], e["id"]): e for e in records}
    known = [(family, fid, corners) for family, fid, corners in detections
             if (family, fid) in entries]
    if not known:
        return None, "Waiting for a registered marker. Unknown IDs are ignored."
    if len({(family, fid) for family, fid, _ in known}) != len(known):
        return None, "A registered ID appears twice; give each printed marker a unique ID."
    first_family, first_id, _ = known[0]
    first = entries[first_family, first_id]
    base, reason = capture_marker_observation(engine, known, image_stamp, image_received,
        image_size, first_family, first_id, first["size_m"],
        T_display_map @ locator_transform(first), T_display_map, now=now,
        check_marker_width=check_marker_width)
    if base is None:
        return None, reason
    observations = []
    for family, fid, quad in known:
        corners = np.asarray(quad, dtype=float)
        if (corners.shape != (4, 2) or not np.isfinite(corners).all()
                or np.any(corners < 0) or np.any(corners >= image_size)):
            continue
        entry = entries[family, fid]
        observations.append(replace(base, family=family, marker_id=fid,
            marker_size_m=entry["size_m"], corners=corners.copy(),
            T_map_locator=locator_transform(entry)))
    if not observations:
        return None, "No complete registered marker is inside the image."
    return MappedObservation(tuple(observations)), ""


def _degrees(a, b):
    """Angle between two vectors, or between two rotation matrices."""
    if np.ndim(a) == 2:
        cosine = (np.trace(np.asarray(a).T @ b) - 1) / 2
    else:
        cosine = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(np.clip(cosine, -1, 1))))


def mean_rotation(rotations):
    """The rotation closest to the average of `rotations` (chordal L2 mean)."""
    U, _, Vt = np.linalg.svd(np.sum(np.asarray(rotations, dtype=float), axis=0))
    return U @ np.diag([1., 1., np.sign(np.linalg.det(U @ Vt))]) @ Vt


def steady_rotation(rotations, count, agree_deg):
    """Mean of the last `count` rotations if every one is within `agree_deg` of it.

    One frame's measurement of a marker's rotation can be several degrees out
    (small or oblique squares, depth noise on the patch around them); turning a
    Locator from it then makes the joint solve reject the layout. None until
    `count` successive frames agree.
    """
    recent = list(rotations)[-count:]
    if len(recent) < count:
        return None
    mean = mean_rotation(recent)
    return mean if all(_degrees(R, mean) <= agree_deg for R in recent) else None


def _surface(marker, max_points=4000):
    """The surface a marker is stuck to, from depth: (unit normal, point) in SLAM.

    Black print often returns no depth, so the plane is fitted over the square
    and about half its width around it. A patch across two surfaces, as at a
    room corner, is not one plane and gives None. The normal faces the camera,
    as the printed face does.

    Inverse depth is linear in the normalised pixel coordinates on a plane, so
    the fit is ordinary least squares with the error along each ray, as stereo
    depth errs. A 3D plane fit (SVD) is pulled towards that ray noise: 1.5 deg
    off at D455 noise levels on a 35 cm patch.
    """
    import cv2
    K, depth_K = np.asarray(marker.K, dtype=float), np.asarray(marker.depth_K, dtype=float)
    rays = (np.asarray(marker.corners, dtype=float) - K[:2, 2]) / np.diag(K)[:2]
    pixels = rays * np.diag(depth_K)[:2] + depth_K[:2, 2]
    depth = np.asarray(marker.depth)
    mask = np.zeros(depth.shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, np.round(pixels).astype(np.int32), 1)
    side = np.mean(np.linalg.norm(pixels - np.roll(pixels, 1, axis=0), axis=1))
    mask = cv2.dilate(mask, np.ones((max(3, int(side)) | 1,) * 2, np.uint8))
    v, u = np.nonzero((mask > 0) & (depth > 0))
    if len(u) < 50:
        return None
    pick = np.linspace(0, len(u) - 1, min(len(u), max_points), dtype=int)
    v, u = v[pick], u[pick]
    z = depth[v, u].astype(float) / 1000.
    rays = np.column_stack(((u - depth_K[0, 2]) / depth_K[0, 0],
                            (v - depth_K[1, 2]) / depth_K[1, 1], np.ones(len(u))))
    # D4xx depth noise grows with range: about 1.5% of it.
    tolerance = max(.03, .015 * float(np.median(z)))
    near = np.ones(len(z), dtype=bool)
    for _ in range(3):
        if near.sum() < 50:
            return None
        # The plane is plane @ X = 1 for every point X = z * ray on it.
        plane = np.linalg.lstsq(rays[near], 1 / z[near], rcond=None)[0]
        expected = rays @ plane
        near = (expected > 0) & (np.abs(z - 1 / np.maximum(expected, 1e-9)) < tolerance)
    if near.mean() < .7:
        return None
    T = marker.T_slam_depth @ T_BODY_OPTICAL
    return -T[:3, :3] @ plane / np.linalg.norm(plane), T[:3, :3] @ plane / (plane @ plane) + T[:3, 3]


def seen_marker(marker):
    """Where the camera sees a marker: (PnP branches as Locator poses, surface, centre), in SLAM.

    Both PnP branches put the centre on the same ray. Its range comes from the
    surface when depth shows one, not from the square's size: 0.4 px of corner
    noise on a 20 px square is 6 cm at 3 m. `surface` is None when depth shows
    none facing the camera. None for a square too small or unclear to solve.
    """
    import cv2
    try:
        poses = optical_marker_poses(marker)
    except (ValueError, cv2.error):
        return None
    T_slam_optical = marker.T_slam_image @ T_BODY_OPTICAL
    branches = [T_slam_optical @ T @ _MARKER_TO_LOCATOR for T in poses]
    surface, camera = _surface(marker), T_slam_optical[:3, 3]
    centre = branches[0][:3, 3]
    if surface is not None:
        ray = centre - camera
        facing = surface[0] @ ray
        if facing < -.2 * np.linalg.norm(ray):
            centre = camera + ray * (surface[0] @ (surface[1] - camera)) / facing
        else:
            surface = None
    return branches, surface, centre


def _floor_frame(up):
    """The turn of SLAM's +Z onto `up`, and two unit columns spanning the floor."""
    lift = _turn_onto(np.array([0., 0., 1.]), up)
    across = np.eye(3)[np.argmin(np.abs(up))]
    across = across - up * (across @ up)
    across /= np.linalg.norm(across)
    return lift, np.column_stack((across, np.cross(up, across)))


def marker_similarity(centres, targets, up, min_spread_m=.3):
    """Scale, rotation and translation taking SLAM marker centres onto their Locators.

    SLAM is gravity-aligned, so besides scale only the heading is free: SLAM's
    +Z turns onto the scan's `up`, the heading is solved across the floor, then
    one scale. Two markers fix it; with more, a misplaced Locator shows in its
    miss. Returns (scale, R, t, miss) with targets ~= scale * R @ centre + t.
    """
    centres, targets = (np.asarray(a, dtype=float).reshape(-1, 3) for a in (centres, targets))
    up = np.asarray(up, dtype=float) / np.linalg.norm(up)
    lift, floor = _floor_frame(up)
    a = (centres - centres.mean(axis=0)) @ lift.T @ floor
    b = (targets - targets.mean(axis=0)) @ floor
    if min(np.linalg.norm(f[:, None] - f[None], axis=2).max() for f in (a, b)) < min_spread_m:
        raise ValueError("The markers need to be further apart across the floor; markers "
                         "above one another show neither heading nor scale.")
    yaw = math.atan2(np.sum(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]), np.sum(a * b))
    R = so3_exp(up * yaw) @ lift
    turned = (centres - centres.mean(axis=0)) @ R.T
    scale = float(np.sum(turned * (targets - targets.mean(axis=0))) / np.sum(turned * turned))
    t = targets.mean(axis=0) - scale * R @ centres.mean(axis=0)
    return scale, R, t, np.linalg.norm(scale * centres @ R.T + t - targets, axis=1)


class MarkerSightings:
    """Where the camera saw each marker's centre, in SLAM metres, over a sweep.

    A marker counts once it has `min_sightings` that agree to `max_spread_m`
    (median distance from their median). Sightings with depth under the
    marker are used when there are enough of them: range from the square's
    size alone is centimetres out at a few metres.
    """

    def __init__(self, min_sightings=3, max_spread_m=.05):
        self.min_sightings = int(min_sightings)
        self.max_spread_m = float(max_spread_m)
        self._seen = {}

    def add(self, marker):
        """Record one MarkerObservation; False if its square could not be solved."""
        sighting = seen_marker(marker)
        if sighting is None:
            return False
        _, surface, centre = sighting
        self._seen.setdefault((marker.family, marker.marker_id), []).append((centre, surface is not None))
        return True

    def counts(self):
        return {key: len(seen) for key, seen in self._seen.items()}

    def centres(self, require_depth=False):
        """Median centre of each marker seen steadily enough, by (family, id)."""
        result = {}
        for key, seen in self._seen.items():
            backed = [centre for centre, on_surface in seen if on_surface]
            points = np.array(backed if require_depth or len(backed) >= self.min_sightings
                              else [c for c, _ in seen])
            if len(points) < self.min_sightings:
                continue
            centre = np.median(points, axis=0)
            if np.median(np.linalg.norm(points - centre, axis=1)) <= self.max_spread_m:
                result[key] = centre
        return result


def level_locators_from_view(observation, max_miss_m=.10, min_error_deg=10., min_spread_m=.3,
                             T_display_map=None, log=None, snap_mounts=True,
                             min_marker_px=None, max_view_deg=None):
    """Assigned Locators turned the way the camera sees their markers.

    A rotation is easy to get wrong by hand: a Locator duplicated from a floor
    one stays flat when moved onto a wall, and Rotation X = 90 stands it up
    with its printed top sideways. Either puts every corner about a side's
    width from where the camera sees it, and the joint solve can only say the
    markers disagree.

    Only the Locators' positions are trusted. Three depth-backed centres fix
    the full camera rotation; with two, gravity supplies the missing axis.
    Each marker's face and printed top come from the camera and its depth.
    `min_error_deg=0` returns every measured orientation, even if it happens
    to match the entered rotation. `snap_mounts=False` preserves the measured
    surface tilt instead of assuming perfectly level walls and floors.
    `min_marker_px` / `max_view_deg` leave alone a marker whose square is
    smaller than that across, or seen more than that off face-on: its
    rotation cannot be measured to a few degrees from there.

    Returns (corrections, note). Each correction is (key, T_map_locator,
    degrees turned, on_wall) for a Locator more than `min_error_deg` off what
    the camera sees; its position is kept. A marker standing neither flat nor
    upright, or whose face cannot be told, is left alone. The note says why
    nothing could be checked, naming a misplaced Locator if there is one;
    positions in it are display coordinates (`T_display_map`), as the Channel
    Box shows them. `log` gets each marker's Locator and camera position.
    """
    display = np.eye(4) if T_display_map is None else np.asarray(T_display_map, dtype=float)
    seen = []
    for marker in observation.markers:
        sighting = seen_marker(marker)
        if sighting is not None:
            seen.append((marker, *sighting))
    if len(seen) < 2:
        return [], "Locator rotations are checked with two or more assigned markers in view."
    up = observation.markers[0].up_map / np.linalg.norm(observation.markers[0].up_map)
    lift, floor = _floor_frame(up)
    centres = np.array([centre for *_, centre in seen])
    placed = np.array([marker.T_map_locator[:3, 3] for marker, *_ in seen])
    ids = [marker.marker_id for marker, *_ in seen]
    everyone = np.arange(len(seen))

    def spread(index):
        b = placed[index] @ floor
        return np.linalg.norm(b[:, None] - b[None], axis=2).max()

    def fit(index, targets):
        """SLAM-to-map rotation and translation through markers `index`; every marker's miss."""
        a = (centres[index] - centres[index].mean(axis=0)) @ lift.T @ floor
        b = (targets[index] - targets[index].mean(axis=0)) @ floor
        yaw = math.atan2(np.sum(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]), np.sum(a * b))
        R = so3_exp(up * yaw) @ lift
        t = targets[index].mean(axis=0) - R @ centres[index].mean(axis=0)
        return R, t, np.linalg.norm(centres @ R.T + t - targets, axis=1)

    def shown(position):
        return "X {:.3f}, Y {:.3f}, Z {:.3f} m".format(*(display @ np.append(position, 1.))[:3])

    def misplaced():
        """Which Locator does not fit what the camera sees, as specifically as it can be told."""
        if len(seen) >= 3:
            for i, j in itertools.combinations(everyone, 2):
                swapped = placed.copy()
                swapped[[i, j]] = placed[[j, i]]
                if fit(everyone, swapped)[2].max() <= max_miss_m:
                    return (f"The Locators of markers {ids[i]} and {ids[j]} look swapped: each marker is "
                            "where the other's Locator is. Swap their IDs.")
            lone = []
            for i in everyone:
                others = everyone[everyone != i]
                if spread(others) < min_spread_m:
                    continue
                R_others, t_others, miss_others = fit(others, placed)
                if miss_others[others].max() <= max_miss_m:
                    lone.append((i, others, R_others @ centres[i] + t_others, miss_others[i]))
            if len(lone) == 1:
                i, others, expected, gap = lone[0]
                return (f"Marker {ids[i]}'s Locator is {gap*100:.0f} cm from where markers "
                        f"{' and '.join(str(ids[k]) for k in others)} put it: {shown(expected)}. "
                        "Move it there if their Locators are right.")
        if len(seen) == 2:
            (seen_apart, seen_rise), (placed_apart, placed_rise) = (
                (np.linalg.norm(d @ floor), d @ up) for d in (R @ (centres[1] - centres[0]), placed[1] - placed[0]))
            return (f"Markers {ids[0]} and {ids[1]} are {seen_apart:.2f} m apart across the floor and "
                    f"{seen_rise:+.2f} m in height, but their Locators are {placed_apart:.2f} m and "
                    f"{placed_rise:+.2f} m. Check the Locator positions.")
        worst = int(np.argmax(miss))
        return (f"Marker {ids[worst]}'s Locator is {miss[worst]*100:.0f} cm from where the other "
                "markers in view put it. Check the Locator positions.")

    if spread(everyone) < min_spread_m:
        return [], ("Locator rotations need markers further apart across the floor; "
                    "markers above one another do not show the camera's heading.")
    R, t, miss = fit(everyone, placed)
    scaled = None
    if len(seen) >= 3 and all(surface is not None for _, _, surface, _ in seen):
        from scan_scale import fit_locator_scale
        scaled = fit_locator_scale(centres, placed)
        if scaled["ok"] and abs(scaled["scale"] - 1) <= 2 * scaled["sigma"]:
            # The depth triangle also fixes SLAM's tilt relative to the scan.
            # Holding gravity exact here would undo the 3D calibration and
            # blame correctly scaled Locators for a tilted tracking frame.
            R = scaled["T"][:3, :3] / scaled["scale"]
            t = placed.mean(axis=0) - R @ centres.mean(axis=0)
            miss = np.linalg.norm(centres @ R.T + t - placed, axis=1)
    if log is not None:
        for i in everyone:
            log(f"Marker {ids[i]}: Locator at {shown(placed[i])}; the camera puts it at "
                f"{shown(R @ centres[i] + t)} ({miss[i]*100:.0f} cm apart)")
    if miss.max() > max_miss_m:
        # A rigid fit makes a uniformly mis-scaled scan look like misplaced
        # Locators. Diagnose the common scale first, without moving anything.
        if scaled is not None and scaled["ok"] and abs(scaled["scale"] - 1) > 2 * scaled["sigma"]:
            return [], (f"The locator triangle indicates a scan scale mismatch "
                        f"({scaled['scale']:.3f}x the RealSense measurements). "
                        "Run Auto lineup with three assigned Locators to calibrate scale; "
                        "keep their placements on the scan.")
        return [], misplaced()
    corrections, unclear, distant = [], [], []
    for marker, branches, surface, centre in seen:
        if min_marker_px is not None or max_view_deg is not None:
            corners = np.asarray(marker.corners, dtype=float)
            across = float(np.linalg.norm(corners - np.roll(corners, 1, axis=0), axis=1).max())
            face = surface[0] if surface is not None else branches[0][:3, 2]
            camera = (marker.T_slam_image @ T_BODY_OPTICAL)[:3, 3]
            if ((min_marker_px is not None and across < min_marker_px)
                    or (max_view_deg is not None and _degrees(face, camera - centre) > max_view_deg)):
                distant.append(marker.marker_id)
                continue
        rotations = [R @ T[:3, :3] for T in branches]
        measured = None
        if surface is not None:
            face = R @ surface[0]
            off = [_degrees(rotation[:, 2], face) for rotation in rotations]
            if min(off) <= 30:
                # The face from depth, fitted over far more pixels than the
                # square has; the printed top from its corners.
                top = rotations[int(np.argmin(off))][:, 0]
                top = top - face * (top @ face)
                top /= np.linalg.norm(top)
                measured = np.column_stack((top, np.cross(face, top), face))
        else:
            # No surface to go by: the same choice as marker_heading_seeds. A
            # floor marker's other branch can stand upright; a wall marker's
            # stands upright too when seen from about its own height, facing
            # tens of degrees off, so that one stays unclear.
            mounts = [(rotation, *level_marker(rotation, up)) for rotation in rotations]
            mounts = [m for m in mounts if _degrees(m[0][:, 2], m[1][:, 2]) <= 25]
            if any(not on_wall for *_, on_wall in mounts):
                mounts = [m for m in mounts if not m[2]]
            ambiguity_deg = min_error_deg if min_error_deg > 0 else 5.
            if mounts and all(_degrees(m[1], mounts[0][1]) <= ambiguity_deg for m in mounts[1:]):
                measured = mounts[0][0]
        if measured is None:
            unclear.append(marker.marker_id)
            continue
        level, on_wall = level_marker(measured, up)
        if snap_mounts and _degrees(measured[:, 2], level[:, 2]) > 25:
            unclear.append(marker.marker_id)
            continue
        rotation = level if snap_mounts else measured
        if min_error_deg > 0 and _degrees(marker.T_map_locator[:3, :3], measured) <= min_error_deg:
            continue
        T = marker.T_map_locator.copy()
        T[:3, :3] = rotation
        corrections.append(((marker.family, marker.marker_id), T,
                            _degrees(marker.T_map_locator[:3, :3], rotation), on_wall))
    notes = []
    if distant:
        notes.append(f"Marker(s) {', '.join(map(str, distant))} are too small or too edge-on to measure "
                     "their rotation: move closer and face them.")
    if unclear:
        notes.append(f"Could not check the rotation of marker(s) {', '.join(map(str, unclear))}: "
                     "keep the wall or floor around them in the depth view.")
    return corrections, " ".join(notes)


class MappedMeasurement(MapMeasurement):
    __slots__ = ("marker_keys", "rejected_marker_keys", "marker_rms_px")

    def __init__(self, scan, keys, rejected, rms):
        for field in MapMeasurement.__slots__:
            setattr(self, field, getattr(scan, field))
        self.marker_keys = tuple(keys)
        self.rejected_marker_keys = tuple(rejected)
        self.marker_rms_px = float(rms)


def solve_mapped_observation(observation, matcher, max_error_px=3.0):
    """Robust marker-level consensus, joint corner refinement, then scan check.

    Two visible markers must agree. With three or more, a strict majority is
    required, allowing one bad ID/placement to be excluded from a three-tag
    solve. Occluded/unknown markers contribute nothing. A single visible known
    marker is allowed only with full scan geometry and an unambiguous pose.
    """
    import cv2
    markers, world, image, seeds = [], [], [], []
    local_rotation = np.eye(4)
    local_rotation[:3, :3] = rpy_to_matrix(0, 0, -np.pi / 2)
    for marker in observation.markers:
        try:
            poses = optical_marker_poses(marker)
        except (ValueError, cv2.error):
            continue
        s = marker.marker_size_m / 2
        corners = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]])
        T_map_marker = marker.T_map_locator @ local_rotation
        world.append(corners @ T_map_marker[:3, :3].T + T_map_marker[:3, 3])
        image.append(marker.corners)
        markers.append(marker)
        seeds.extend(T_optical_marker @ invert_transform(T_map_marker) for T_optical_marker in poses)
    if not markers:
        return MapMeasurement(reason="Registered markers are too small or unclear; move closer.")
    world, image = np.asarray(world), np.asarray(image)
    K = markers[0].K
    # A visible but unusable marker must not silently turn a two-tag conflict
    # into a trusted single-tag solve.
    required = 1 if len(observation.markers) == 1 else max(2, len(observation.markers)//2 + 1)

    def residual(T):
        camera = world @ T[:3, :3].T + T[:3, 3]
        projected = camera @ K.T
        uv = projected[:, :, :2] / np.maximum(projected[:, :, 2:], 1e-9)
        errors = np.sqrt(np.mean(np.sum((uv - image)**2, axis=2), axis=1))
        errors[np.any(camera[:, :, 2] <= 0, axis=1)] = np.inf
        return errors

    # Also solve all corners together: distant markers may each have noisy
    # depth/heading, while the baseline between them pins the joint camera.
    if len(markers) > 1:
        ok, rvec, tvec, _ = cv2.solvePnPRansac(world.reshape(-1, 3), image.reshape(-1, 2),
            K, None, iterationsCount=200, reprojectionError=max_error_px,
            confidence=.999, flags=cv2.SOLVEPNP_EPNP)
        if ok:
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = cv2.Rodrigues(rvec)[0], tvec.ravel()
            seeds.append(T)

    candidates = []
    for T in seeds:
        errors = residual(T)
        ids = np.flatnonzero(errors <= max_error_px)
        if len(ids) < required:
            continue
        rvec, tvec = cv2.Rodrigues(T[:3, :3])[0], T[:3, 3].copy()
        rvec, tvec = cv2.solvePnPRefineLM(world[ids].reshape(-1, 3),
            image[ids].reshape(-1, 2), K, None, rvec, tvec)
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = cv2.Rodrigues(rvec)[0], tvec.ravel()
        errors = residual(T)
        ids = np.flatnonzero(errors <= max_error_px)
        if len(ids) < required:
            continue
        T_map_camera = invert_transform(T) @ invert_transform(T_BODY_OPTICAL)
        alignment = T_map_camera @ invert_transform(markers[0].T_slam_image)
        rms = float(np.sqrt(np.mean(errors[ids]**2)))
        candidates.append((len(ids), rms, alignment, ids))
    if not candidates:
        return MapMeasurement(reason=f"Registered markers disagree; need {required} consistent markers. Check IDs, sizes and surveyed poses.")
    candidates.sort(key=lambda c: (-c[0], c[1]))
    best_count = candidates[0][0]
    verified, checked = [], []
    last_reason = "No marker pose agrees with the scan."
    sample = markers[0]
    for count, rms, alignment, ids in candidates:
        if count != best_count:
            continue
        if any(matcher._camera_delta(alignment, old, sample.T_slam_image)[0] < .005
               and matcher._camera_delta(alignment, old, sample.T_slam_image)[1] < .2 for old in checked):
            continue
        if len(checked) >= 6:
            break
        checked.append(alignment)
        scan = matcher.verify_pose(sample.depth, sample.depth_K, sample.T_slam_depth, alignment)
        if not scan.ok:
            last_reason = scan.reason
            continue
        if count == 1 and scan.observable_rank < 3:
            last_reason = "One marker needs more scanned geometry; show another registered marker or a room corner."
            continue
        keys = [(markers[i].family, markers[i].marker_id) for i in ids]
        rejected = [(m.family, m.marker_id) for m in observation.markers
                    if (m.family, m.marker_id) not in keys]
        result = MappedMeasurement(scan, keys, rejected, rms)
        result.monotonic_timestamp = observation.captured_at
        verified.append(result)
    if not verified:
        return MapMeasurement(reason="Marker pose failed scan verification: " + last_reason)
    verified.sort(key=lambda r: (r.marker_rms_px, -r.verify_score))
    best = verified[0]
    for other in verified[1:]:
        distance, angle = matcher._camera_delta(best.T_slam_to_map, other.T_slam_to_map, sample.T_slam_image)
        if ((distance > .10 or angle > 5) and abs(best.marker_rms_px-other.marker_rms_px) < .5
                and abs(best.verify_score-other.verify_score) < .05):
            return MapMeasurement(reason="Marker pose is ambiguous; show another registered marker or more room geometry.")
    return best
