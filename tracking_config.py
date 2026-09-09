#!/usr/bin/env python3
"""
Configuration for the virtual production camera tracking bridge.

Everything an operator needs to change on set lives in tracking.yaml — most
importantly the IP of the machine running Unreal Engine, which used to be
hardcoded to 127.0.0.1 (so the bridge could only ever talk to Unreal on the same
box, not "over your local network").

Loading order, first hit wins:
  1. $VP_TRACKING_CONFIG
  2. ./tracking.yaml  (current working directory)
  3. tracking.yaml next to this module

A partial file is fine: it is deep-merged over DEFAULTS, so you only write the
keys you actually want to change. Missing file is fine too — defaults are a
working localhost setup.
"""
import copy
import os

try:
    import yaml
except ImportError:                                   # pragma: no cover
    yaml = None

CONFIG_FILENAME = "tracking.yaml"

# Keep these ordered from the smallest codebook to the largest. OpenCV's
# predefined 4x4 dictionaries can share marker patterns; when that happens the
# first matching family is reported consistently.
ARUCO_4X4_FAMILIES = (
    "DICT_4X4_50",
    "DICT_4X4_100",
    "DICT_4X4_250",
    "DICT_4X4_1000",
)


def aruco_dictionary_names(aruco_config):
    """Return configured marker families, accepting the legacy singular key.

    ``dictionaries`` is the new list form. ``dictionary`` remains supported so
    existing on-set tracking.yaml files continue to work unchanged.
    """
    names = aruco_config.get("dictionaries")
    if names is None:
        names = aruco_config.get("dictionary", "DICT_4X4_1000")
    if isinstance(names, str):
        names = [names]

    # Preserve order while removing accidental duplicate YAML entries.
    return tuple(dict.fromkeys(str(name) for name in names if str(name).strip()))

DEFAULTS = {
    # ── Where Unreal Engine is ────────────────────────────────────────────────
    "network": {
        "unreal_ip": "127.0.0.1",   # set to the Unreal machine's LAN address
        "unreal_port": 11111,
        "camera_id": 1,             # must match the LiveLinkFreeD subject in UE
        "send_rate_hz": 60,         # 0 = send on every pose, lowest latency
        # Inbound JSON commands from the Unreal plugin (anchor_lidar, reset_pose,
        # set_offset, ping). Set to 0 to disable listening.
        "command_port": 11114,
    },

    # ── Isaac ROS topics ──────────────────────────────────────────────────────
    "topics": {
        "odometry": "/visual_slam/tracking/odometry",
        "vo_pose": "/visual_slam/tracking/vo_pose_covariance",
        "status": "/visual_slam/status",
        "image": "/camera/infra1/image_rect_raw",
        "camera_info": "/camera/infra1/camera_info",
        "imu": "/camera/imu",
        "depth": "/camera/depth/image_rect_raw",
        "depth_camera_info": "/camera/depth/camera_info",
    },

    # ── ROS -> Unreal frame conversion (see pose_transform.py) ────────────────
    # Defaults are the mathematically derived ROS-canonical -> Unreal mapping.
    # If an axis moves the wrong way in Unreal, flip its sign here; use
    # calibrate_axes.py to work out which one.
    "transform": {
        "position_scale": 1.0,            # 1.0 = send metres, per FreeD spec
        "position_sign": [1.0, -1.0, 1.0],
        "pan_sign": -1.0,
        "tilt_sign": -1.0,
        "roll_sign": 1.0,
        "pan_offset_deg": 0.0,
        "tilt_offset_deg": 0.0,
        "roll_offset_deg": 0.0,
        # Where the film camera's nodal point sits relative to the RealSense,
        # in ROS body axes: x forward, y left, z up (metres / degrees).
        "rig_offset": {
            "translation_m": [0.0, 0.0, 0.0],
            "rotation_deg": [0.0, 0.0, 0.0],
        },
    },

    # ── Timing / stability ────────────────────────────────────────────────────
    "tracking": {
        # Predict this far ahead to offset capture-to-render latency. Measure it
        # before raising this: over-prediction shows up as overshoot on whip pans.
        "lead_time_s": 0.0,
        "max_extrapolation_s": 0.05,   # freeze rather than fly on tracking loss
        "stale_pose_timeout_s": 0.25,  # after this, hold pose and flag INVALID
        "smoothing_alpha": 0.0,        # 0 = off; adds latency, use sparingly

        # Outlier gate. cuVSLAM relocalisations and loop closures can snap the
        # pose by over a metre between two frames; that must never reach Unreal.
        # Raise the limits if you genuinely move the rig faster than this.
        "gate_enabled": True,
        "gate_max_speed_m_s": 3.0,
        "gate_max_angular_deg_s": 900.0,
        "gate_max_jump_m": 0.05,
        "gate_max_consecutive_rejects": 15,

        # Stationary hold (ZUPT). With the rig physically still, pure visual
        # odometry still creeps ~8.5 mm per 12 s on this camera. The IMU can tell
        # that nothing is moving, so the output is pinned and the creep absorbed.
        # Entering needs a quiet window; leaving takes one lively sample, so the
        # start of a real move is never delayed.
        "hold_enabled": True,
        "hold_gyro_still_rad_s": 0.05,      # measured stationary gyro max 0.023
        "hold_gyro_release_rad_s": 0.08,
        "hold_accel_still_std": 0.15,
        "hold_window_samples": 60,          # 0.3 s at 200 Hz
        # Vision's veto over the IMU. An accelerometer cannot tell constant
        # velocity from rest and a dolly move rotates nothing, so the IMU alone
        # reads a straight tracking shot as "still". Release if the raw VO
        # estimate travels this far over the window. 3 mm in ~0.13 s is 22 mm/s,
        # well above the 0.7-1.4 mm/s creep and well below any real move.
        "hold_vo_release_m": 0.003,
        "hold_vo_window_samples": 12,       # ~0.13 s at 90 Hz
    },

    # ── ArUco ground-zero anchoring ───────────────────────────────────────────
    "aruco": {
        "enabled": True,
        "dictionary": "DICT_4X4_1000",
        "marker_size_m": 0.10,
        "lineup_marker_id": -1,    # -1 requires exactly one visible marker
        # Depth cross-check on the marker solve. Four coplanar corners are
        # pose-ambiguous, so a mono PnP can silently settle on the mirrored
        # solution — and a flipped anchor takes the whole virtual set with it.
        # Depth measures distance independently, so a disagreement means refuse
        # to anchor rather than anchor wrongly.
        "depth_check_enabled": True,
        "depth_tolerance_m": 0.10,
        "depth_refine": True,       # snap PnP distance to the measured depth
        # Only used until camera_info arrives; real intrinsics come from the topic.
        # Only used if camera_info never arrives; the console says which was
        # used. Read off this D455 at the 848x480 profile isaac_vp.launch.py
        # now requests -- the previous values (fx 389.4, cx 319.5) were for a
        # 640x480 stream and would have put a marker solve badly out of scale.
        "fallback_intrinsics": {
            "fx": 429.948, "fy": 429.948, "cx": 423.497, "cy": 241.663,
        },
    },

    # ── LiDAR map matching ────────────────────────────────────────────────────
    "lidar": {
        "enabled": False,
        "map_file": "",                  # .ply/.pcd point cloud or .obj mesh
        "voxel_size": 0.05,              # downsampling size (metres)
        "icp_rate_hz": 5.0,              # how often to run continuous ICP
        "max_measurement_age_s": 0.6,    # host monotonic capture-to-finish bound

        # ── measurement gates ────────────────────────────────────────────────
        "min_fitness": 0.3,              # minimum ICP fitness score to trust
        "min_inlier_points": 50,         # minimum matched points
        "max_rmse_m": 0.08,              # reject a sloppy fit
        "max_correction_m": 0.30,        # reject a fix that moves the camera further
        "max_correction_deg": 5.0,       # ...or rotates it further
        "observability_eps": 0.04,       # below this an axis counts as unconstrained

        # ── verifying a lock-on (the locator/marker is only a hint) ──────────
        "verify_threshold_m": 0.10,      # a depth point this close counts as "on the scan"
        "min_verify_score": 0.55,        # fraction of the view that must land on it
        # The hint defines WHERE IN THE SCAN to search. A match outside this
        # region is a different place that happened to look similar, not a
        # refinement of the hint, and is rejected.
        "hint_radius_m": 0.50,
        "hint_radius_deg": 20.0,
        "seed_yaw_search_deg": 25.0,     # yaw offsets tried around the hint
        "seed_yaw_steps": 5,
        "refine_candidates": 3,          # shortlist refined at full resolution
        "shortlist_ratio": 1.5,          # only refine within this of the leader
        "hint_weight": 0.10,             # how much the hint counts when candidates tie
        "early_exit_median_m": 0.05,     # a hint this good ends the search

        # ── ICP correspondence radii (coarse -> fine) ────────────────────────
        # Kept separate from max_correction_m, which the old code reused as the
        # search radius: an error larger than the radius is invisible to ICP,
        # so tying the two together capped the recoverable error at the
        # rejection threshold.
        "coarsest_radius_m": 1.00,
        "coarse_radius_m": 0.50,
        "fine_radius_m": 0.10,
        "initial_icp_iterations": 200,   # more iterations for first alignment
        "continuous_icp_iterations": 30, # fewer for runtime speed

        # ── fusion ───────────────────────────────────────────────────────────
        "blend_alpha": 0.1,              # 0=off, 1=instant, 0.1=gradual
        "max_slew_m_s": 0.02,            # hard cap on how fast a fix may move the camera
        "max_slew_deg_s": 1.0,
        "freeze_when_stationary": False,

        # ── D455 depth conditioning ──────────────────────────────────────────
        "min_range_m": 0.3,              # inside this the D455 sees nothing usable
        "max_range_m": 6.0,              # past this stereo noise swamps the signal
        "weight_ref_range_m": 2.0,       # range kept at full density
        "min_keep_fraction": 0.05,
        "target_points": 8000,
        "baseline_m": 0.095,             # D455 stereo baseline
        "disparity_sigma_px": 0.1,
        "mesh_sample_points": 400000,    # points sampled off an .obj/.stl map
        "map_scale": 1.0,                # metres per map-file unit (0.01 for a cm scan)

        # ── Camera-image lineup against the scan (image_alignment.py) ────────
        # Read-only by default: it reports whether the video lines up with the
        # LiDAR and draws the scan's edges over the picture. Letting it move
        # the pose is opt-in and off, because its objective was measured
        # 30-60 mm off true on this scan.
        "image_align_enabled": True,
        "image_align_refine": False,
        "image_align_width": 320,
        "image_align_iterations": 20,
        "image_align_inlier_px": 3.0,
        "image_align_search_px": 24.0,
        "image_align_min_edges": 150,
        "image_align_min_score": 0.35,
        "image_align_max_edges": 4000,
        "image_align_use_creases": False,
        "image_align_max_range_m": 15.0,
        "image_align_max_move_m": 0.25,
        "image_align_max_move_deg": 10.0,
    },

    # Legacy configs retain their pose-only workflow unless explicitly enabled.
    "fusion": {
        "enabled": False,
        "require_lineup": True,
        "scan_timeout_s": 0.8,
        "gnss_timeout_s": 0.8,
        "single_source_grace_s": 1.0,
        "recovery_samples": 3,
        "min_scan_rank": 3,
        "depth_pose_tolerance_s": 0.035,
        "depth_timeout_s": 0.5,
        "recovery_max_distance_m": 0.10,
        "recovery_max_angle_deg": 5.0,
        "recovery_slew_m_s": 0.10,
        "recovery_slew_deg_s": 5.0,
    },

    # ── Optional extras ───────────────────────────────────────────────────────
    # GNSS is off by default: indoors there is no fix, and substituting a
    # GNSS-derived position into a SLAM-origin FreeD stream mid-take teleports
    # the virtual camera. Useful only for outdoor, large-volume work.
    "gnss": {
        "enabled": False,
        "port": "/dev/ttyACM0",
        "baud": 115200,
        "calibrated": False,
        "origin_lla": None,
        "origin_world_m": [0.0, 0.0, 0.0],
        "enu_yaw_deg": 0.0,
        "antenna_offset_body_m": [0.0, 0.0, 0.0],
        "latency_s": 0.0,
        "pose_tolerance_s": 0.05,
        "stale_timeout_s": 0.8,
        "min_satellites": 8,
        "max_hdop": 2.0,
        "max_correction_age_s": 2.0,
        "max_hacc_m": 0.05,
        "max_vacc_m": 0.10,
        "max_innovation_m": 0.20,
    },
    "timecode": {
        "enabled": True,
        "samplerate": 48000,
    },
    "ui": {
        "refresh_hz": 60,
        "trajectory_points": 3000,
    },
}


def _deep_merge(base, override):
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def candidate_paths():
    paths = []
    env = os.environ.get("VP_TRACKING_CONFIG")
    if env:
        paths.append(env)
    paths.append(os.path.join(os.getcwd(), CONFIG_FILENAME))
    paths.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              CONFIG_FILENAME))
    return paths


def load_config(path=None, verbose=True):
    """Load configuration, deep-merged over DEFAULTS. Never raises on a missing
    file; a malformed file is reported and ignored so a typo on set cannot take
    the pipeline down mid-shoot."""
    search = [path] if path else candidate_paths()

    for candidate in search:
        if not candidate or not os.path.isfile(candidate):
            continue
        if yaml is None:                              # pragma: no cover
            if verbose:
                print("[config] PyYAML not installed — using defaults")
            break
        try:
            with open(candidate, "r") as fh:
                user = yaml.safe_load(fh) or {}
            if not isinstance(user, dict):
                raise ValueError("top level of the config must be a mapping")
            cfg = _deep_merge(DEFAULTS, user)
            cfg["_source"] = candidate
            map_file = cfg.get("lidar", {}).get("map_file", "")
            if map_file and not os.path.isabs(map_file):
                cfg["lidar"]["map_file"] = os.path.abspath(
                    os.path.join(os.path.dirname(candidate), map_file))
            if verbose:
                print(f"[config] loaded {candidate}")
            return cfg
        except Exception as exc:
            print(f"[config] ERROR reading {candidate}: {exc}")
            print("[config] falling back to defaults")
            break

    if verbose:
        print("[config] no tracking.yaml found — using defaults "
              "(Unreal target 127.0.0.1:11111)")
    cfg = copy.deepcopy(DEFAULTS)
    cfg["_source"] = None
    return cfg


def describe(cfg):
    """One-screen summary for the console at startup."""
    n, t, tr = cfg["network"], cfg["tracking"], cfg["transform"]
    rate = n["send_rate_hz"]
    lines = [
        f"Unreal target : {n['unreal_ip']}:{n['unreal_port']}  (camera {n['camera_id']})",
        f"Send rate     : {'on every pose' if not rate else str(rate) + ' Hz'}",
        f"Prediction    : lead {t['lead_time_s'] * 1000:.0f} ms, "
        f"cap {t['max_extrapolation_s'] * 1000:.0f} ms",
        f"Smoothing     : {'off' if not t['smoothing_alpha'] else t['smoothing_alpha']}",
        f"Signs         : pos {tr['position_sign']} pan {tr['pan_sign']:+.0f} "
        f"tilt {tr['tilt_sign']:+.0f} roll {tr['roll_sign']:+.0f}",
    ]
    f = cfg.get("fusion", {})
    lines.append(f"Fusion policy : {'supervised; lineup required' if f.get('enabled') else 'legacy pose only'}")
    lines.append(f"GNSS          : {'enabled' if cfg['gnss']['enabled'] else 'off'}; "
                 f"{'calibration configured' if cfg['gnss'].get('calibrated') else 'reference mode'}")
    rig = tr["rig_offset"]["translation_m"]
    if any(rig):
        lines.append(f"Rig offset    : {rig} m (ROS body axes)")
    return "\n".join("  " + line for line in lines)
