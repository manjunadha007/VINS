"""Testing arena: turn a recorded take into what 3DEqualizer needs.

For a take in arena/inputs/<take>/ this writes, beside the bag:

    plate/<take>.00001.png ...  the left IR camera at 30 fps -- the footage to
                                track. Same exposure cuVSLAM tracked, already
                                rectified: no lens distortion to solve.
    frames.csv                  plate frame -> exact sensor timestamp
    plate.json                  frame rate, size, and the lens for 3DE
    survey_points.txt           every mapped marker's 4 corners and centre,
                                metres, in the dashboard's world
    scan_3de.obj                the scan exactly as the dashboard uses it:
                                scaled (Setup > Scan scale) and placed
    gnss.csv                    GNSS fixes, if any were recorded
    README_3DE.txt              how to set the 3DE project up

3DE files are Y-up: a dashboard (ROS, Z-up) point (x, y, z) is written as
(x, z, -y). That is the same axis change Blender undoes when it reads the
.abc back (arena_abc.py), so the solved camera returns in dashboard axes.

Needs the Isaac container (rosbag2_py, OpenCV, open3d).
"""
import csv
import json
import math
import os
import struct

import numpy as np

from pose_transform import rpy_to_matrix

PLATE_TOPIC = "/camera/infra1/image_rect_raw"
PLATE_INFO_TOPIC = "/camera/infra1/camera_info"
GNSS_FIX_TOPIC = "/hyperfusion/gnss/fix"
PLATE_FPS = 30.0
FIRST_FRAME = 1
# 3DE wants a filmback and focal length in physical units; only their ratio
# to the image size matters, so a virtual 10 um pixel is used throughout.
VIRTUAL_PIXEL_CM = 0.001

# Marker (OpenCV) frame from a Locator: the solver's own convention
# (marker_map.solve_mapped_observation) -- Locator +X is the printed top.
_LOCATOR_TO_MARKER = np.eye(4)
_LOCATOR_TO_MARKER[:3, :3] = rpy_to_matrix(0, 0, -np.pi / 2)
# ArUco corner order: top-left, top-right, bottom-right, bottom-left.
_CORNERS = np.array([[-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0]], dtype=float)


# ── Pure helpers (unit-tested) ───────────────────────────────────────────────
def ros_to_3de(points):
    """Dashboard / ROS world (Z-up) -> 3DE / Alembic (Y-up): (x, z, -y)."""
    p = np.asarray(points, dtype=float)
    return np.stack([p[..., 0], p[..., 2], -p[..., 1]], axis=-1)


def alignment_transform(alignment):
    """The scene's scan alignment as one 4x4 similarity: native scan -> world.

    Exactly MapViewerWidget._update_transform: rotation scaled by S, then the
    translation. S is the scan's metres-per-file-unit (Setup > Scan scale).
    """
    T = np.eye(4)
    R = rpy_to_matrix(*(math.radians(alignment.get(k, 0.0)) for k in ("roll", "pitch", "yaw")))
    T[:3, :3] = R * float(alignment.get("scale", 1.0))
    T[:3, 3] = [float(alignment.get(k, 0.0)) for k in ("tx", "ty", "tz")]
    return T


def rigid_part(T_align):
    """Rigid part of the alignment (what poses compose with)."""
    scale = np.cbrt(np.linalg.det(T_align[:3, :3]))
    R = np.eye(4)
    R[:3, :3] = T_align[:3, :3] / scale
    R[:3, 3] = T_align[:3, 3]
    return R


def survey_points(scene):
    """[(name, xyz_world)] for each mapped marker's corners and centre.

    Mapped markers are stored in scan metres; the alignment's rigid part puts
    them in the world the dashboard's camera path is in.
    """
    from marker_map import locator_transform
    rigid = rigid_part(alignment_transform(scene.get("alignment") or {}))
    points = []
    for entry in (scene.get("marker_lineup") or {}).get("mapped_markers") or []:
        T_world_marker = rigid @ locator_transform(entry) @ _LOCATOR_TO_MARKER
        s = float(entry["size_m"]) / 2.0
        corners = (_CORNERS * s) @ T_world_marker[:3, :3].T + T_world_marker[:3, 3]
        base = f"{entry.get('dictionary', 'DICT')}_id{entry['id']}"
        for k, corner in enumerate(corners):
            points.append((f"{base}_c{k}", corner))
        points.append((f"{base}_centre", T_world_marker[:3, 3].copy()))
    return points


def pick_plate_frames(stamps, fps=PLATE_FPS):
    """Choose the source frames for a fixed-rate plate, by timestamp.

    stamps -- sorted sensor stamps (s) of every recorded frame
    Returns [(source_index, grid_time, delta_s, duplicate)]: for each plate
    frame, the recorded frame nearest its slot on a uniform `fps` grid. Where
    the camera dropped frames the nearest one is used and `delta_s` says how
    far off it is; `duplicate` marks a source frame used for two slots.
    """
    stamps = np.asarray(stamps, dtype=float)
    if len(stamps) == 0:
        return []
    grid = stamps[0] + np.arange(int((stamps[-1] - stamps[0]) * fps) + 1) / fps
    right = np.clip(np.searchsorted(stamps, grid), 1, len(stamps) - 1)
    left = right - 1
    pick = np.where(np.abs(stamps[left] - grid) <= np.abs(stamps[right] - grid), left, right)
    if len(stamps) == 1:
        pick = np.zeros(len(grid), dtype=int)
    frames, previous = [], None
    for g, i in zip(grid, pick):
        frames.append((int(i), float(g), float(stamps[i] - g), previous == int(i)))
        previous = int(i)
    return frames


def lens_for_3de(info):
    """3DE camera / lens values from a camera_info dict (K, width, height)."""
    fx, fy, cx, cy = info["K"][0], info["K"][4], info["K"][2], info["K"][5]
    w, h = int(info["width"]), int(info["height"])
    return dict(
        width_px=w, height_px=h, fx_px=fx, fy_px=fy, cx_px=cx, cy_px=cy,
        pixel_aspect=round(fy / fx, 6),
        distortion="none (rectified image)",
        filmback_width_cm=round(w * VIRTUAL_PIXEL_CM, 6),
        filmback_height_cm=round(h * VIRTUAL_PIXEL_CM, 6),
        focal_length_cm=round(fx * VIRTUAL_PIXEL_CM, 6),
        lens_center_offset_x_cm=round((cx - w / 2.0) * VIRTUAL_PIXEL_CM, 6),
        lens_center_offset_y_cm=round(-(cy - h / 2.0) * VIRTUAL_PIXEL_CM, 6),
        horizontal_fov_deg=round(math.degrees(2 * math.atan(w / 2.0 / fx)), 3),
    )


# ── Bag access ───────────────────────────────────────────────────────────────
def _reader(bag_dir, topics):
    import rosbag2_py
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag_dir, storage_id="mcap"),
                rosbag2_py.ConverterOptions("", ""))
    reader.set_filter(rosbag2_py.StorageFilter(topics=list(topics)))
    return reader


def _stamp(raw):
    order = "<" if raw[1] & 1 else ">"
    sec, nsec = struct.unpack_from(order + "iI", raw, 4)
    return sec + nsec * 1e-9


def _camera_info(bag_dir):
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import CameraInfo
    reader = _reader(bag_dir, [PLATE_INFO_TOPIC])
    if not reader.has_next():
        raise ValueError(f"the take has no {PLATE_INFO_TOPIC}")
    msg = deserialize_message(reader.read_next()[1], CameraInfo)
    return dict(K=list(msg.k), D=list(msg.d), width=msg.width, height=msg.height,
                distortion_model=msg.distortion_model, frame_id=msg.header.frame_id)


def _write_obj(path, vertices, faces, header):
    with open(path, "w") as fh:
        fh.write(header)
        np.savetxt(fh, vertices, fmt="v %.6f %.6f %.6f")
        if faces is not None and len(faces):
            np.savetxt(fh, np.asarray(faces) + 1, fmt="f %d %d %d")


# ── Export ───────────────────────────────────────────────────────────────────
def export_take(take_dir, log=print):
    """Write the 3DE plate and its companions for one take. Returns plate.json."""
    import cv2
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import Image

    name = os.path.basename(os.path.normpath(take_dir))
    bag_dir = os.path.join(take_dir, "bag")
    with open(os.path.join(take_dir, "take.json")) as fh:
        take = json.load(fh)

    # Pass 1: every frame's stamp (header only, no image decode).
    stamps = []
    reader = _reader(bag_dir, [PLATE_TOPIC])
    while reader.has_next():
        stamps.append(_stamp(reader.read_next()[1]))
    if len(stamps) < 2:
        raise ValueError(f"the take has no {PLATE_TOPIC} frames to export")
    order = np.argsort(stamps)
    sorted_stamps = np.asarray(stamps)[order]
    frames = pick_plate_frames(sorted_stamps)
    wanted = {}                       # bag message index -> [plate frame numbers]
    for n, (i, _, _, _) in enumerate(frames):
        wanted.setdefault(int(order[i]), []).append(FIRST_FRAME + n)

    # Pass 2: decode and write only the chosen frames.
    plate_dir = os.path.join(take_dir, "plate")
    os.makedirs(plate_dir, exist_ok=True)
    pattern = f"{name}.%05d.png"
    log(f"Exporting {len(frames)} plate frames at {PLATE_FPS:.0f} fps -> {plate_dir}")
    reader = _reader(bag_dir, [PLATE_TOPIC])
    index = written = 0
    while reader.has_next():
        raw = reader.read_next()[1]
        if index in wanted:
            msg = deserialize_message(raw, Image)
            if msg.encoding not in ("mono8", "8UC1"):
                raise ValueError(f"plate encoding {msg.encoding!r} is not mono8")
            img = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(msg.height, msg.step)[:, :msg.width]
            for number in wanted[index]:
                cv2.imwrite(os.path.join(plate_dir, pattern % number), img,
                            [cv2.IMWRITE_PNG_COMPRESSION, 1])
                written += 1
                if written % 500 == 0:
                    log(f"   {written}/{len(frames)}")
        index += 1

    start = float(take.get("start_time", sorted_stamps[0]))
    with open(os.path.join(take_dir, "frames.csv"), "w", newline="") as fh:
        out = csv.writer(fh)
        out.writerow(["frame", "file", "stamp_s", "take_offset_s", "off_grid_ms", "duplicate"])
        for n, (i, grid, delta, dup) in enumerate(frames):
            number = FIRST_FRAME + n
            out.writerow([number, pattern % number, f"{sorted_stamps[i]:.6f}",
                          f"{sorted_stamps[i] - start:.4f}", f"{delta * 1000:.2f}", int(dup)])
    late = [f for f in frames if abs(f[2]) > 0.5 / 90.0]

    info = _camera_info(bag_dir)
    plate = dict(format="hyperfusion.plate", version=1, take=name,
                 fps=PLATE_FPS, first_frame=FIRST_FRAME, frame_count=len(frames),
                 file_pattern=os.path.join("plate", pattern), source_topic=PLATE_TOPIC,
                 frames_off_grid=len(late), duplicate_frames=sum(f[3] for f in frames),
                 camera_frame=info["frame_id"], lens=lens_for_3de(info),
                 coordinates="metres, Y-up: dashboard (x, y, z) is written (x, z, -y)")

    # Survey and scan, from the scene the take ran with.
    scene_path = os.path.join(take_dir, "scene.json")
    if os.path.isfile(scene_path):
        with open(scene_path) as fh:
            scene = json.load(fh)
        points = survey_points(scene)
        with open(os.path.join(take_dir, "survey_points.txt"), "w") as fh:
            fh.write("# name x y z -- metres, 3DE Y-up; marker corners c0..c3 = "
                     "top-left, top-right, bottom-right, bottom-left\n")
            for pname, xyz in points:
                x, y, z = ros_to_3de(xyz)
                fh.write(f"{pname} {x:.6f} {y:.6f} {z:.6f}\n")
        plate["survey_points"] = len(points)
        plate["scan_alignment"] = scene.get("alignment")
        map_file = scene.get("map_file")
        if map_file and os.path.isfile(map_file):
            import open3d as o3d
            mesh = o3d.io.read_triangle_mesh(map_file)
            native = np.asarray(mesh.vertices)
            T = alignment_transform(scene.get("alignment") or {})
            world = native @ T[:3, :3].T + T[:3, 3]
            _write_obj(os.path.join(take_dir, "scan_3de.obj"), ros_to_3de(world),
                       np.asarray(mesh.triangles),
                       f"# {map_file} as the dashboard used it: scale "
                       f"{(scene.get('alignment') or {}).get('scale', 1.0)}, metres, Y-up\n")
            plate["scan_file"] = map_file
        else:
            log(f"   ⚠️  scan {map_file!r} not found; scan_3de.obj not written")
    else:
        log("   ⚠️  no scene.json in the take: no survey points or scan for 3DE")

    gnss = _export_gnss(bag_dir, os.path.join(take_dir, "gnss.csv"))
    plate["gnss_fixes"] = gnss
    with open(os.path.join(take_dir, "plate.json"), "w") as fh:
        json.dump(plate, fh, indent=2)
    _write_3de_readme(take_dir, plate)
    log(f"   plate: {len(frames)} frames ({plate['frames_off_grid']} nearest-frame substitutes "
        f"where the camera dropped frames), survey points: {plate.get('survey_points', 0)}, "
        f"GNSS fixes: {gnss}")
    return plate


def _export_gnss(bag_dir, path):
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import NavSatFix
    reader = _reader(bag_dir, [GNSS_FIX_TOPIC])
    rows = []
    while reader.has_next():
        m = deserialize_message(reader.read_next()[1], NavSatFix)
        cov = list(m.position_covariance)
        rows.append([f"{m.header.stamp.sec + m.header.stamp.nanosec * 1e-9:.6f}",
                     f"{m.latitude:.9f}", f"{m.longitude:.9f}", f"{m.altitude:.4f}",
                     m.status.status,
                     f"{math.sqrt(2 * cov[0]):.4f}" if m.position_covariance_type else "",
                     f"{math.sqrt(cov[8]):.4f}" if m.position_covariance_type else ""])
    if rows:
        with open(path, "w", newline="") as fh:
            out = csv.writer(fh)
            out.writerow(["stamp_s", "lat", "lon", "alt_m", "navsat_status", "hacc_m", "vacc_m"])
            out.writerows(rows)
    return len(rows)


def _write_3de_readme(take_dir, plate):
    lens = plate["lens"]
    name = plate["take"]
    text = f"""3DEqualizer setup for {name}
{'=' * (21 + len(name))}

Footage   plate/{name}.#####.png, frames {plate['first_frame']}-{plate['first_frame'] + plate['frame_count'] - 1}
          Project frame rate: {plate['fps']:.0f} fps, start frame {plate['first_frame']}.
          Grayscale left IR camera of the D455 -- the camera the dashboard tracks.

Camera    Filmback {lens['filmback_width_cm']} x {lens['filmback_height_cm']} cm,
          focal length {lens['focal_length_cm']} cm (fixed),
          pixel aspect {lens['pixel_aspect']},
          lens center offset {lens['lens_center_offset_x_cm']} / {lens['lens_center_offset_y_cm']} cm.
          Distortion: none -- the image is already rectified. Do not solve lens distortion.
          (Pixels: {lens['width_px']}x{lens['height_px']}, fx {lens['fx_px']:.3f}, fy {lens['fy_px']:.3f},
          cx {lens['cx_px']:.3f}, cy {lens['cy_px']:.3f}; horizontal FOV {lens['horizontal_fov_deg']} deg.)

World     survey_points.txt: marker corners in METRES, Y-up -- the dashboard's world.
          Track the marker corners and give them these survey positions ("exactly
          surveyed"), so the solve lands in the dashboard's world and scale.
          scan_3de.obj: the scan exactly as the dashboard used it, same coordinates,
          to check the solve against.

Export    Alembic (.abc), units METRES (no unit scaling), {plate['fps']:.0f} fps,
          frames {plate['first_frame']}-{plate['first_frame'] + plate['frame_count'] - 1}, one camera.
          Save as:  arena/reference/{name}.abc

          The frame rate must match. A camera read at the wrong rate is silently
          shifted in time (tested: a 30 fps file read at 24 fps came back 0.125 m off).
"""
    with open(os.path.join(take_dir, "README_3DE.txt"), "w") as fh:
        fh.write(text)
