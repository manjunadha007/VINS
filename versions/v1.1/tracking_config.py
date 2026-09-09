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

DEFAULTS = {
    # ── Where Unreal Engine is ────────────────────────────────────────────────
    "network": {
        "unreal_ip": "127.0.0.1",   # set to the Unreal machine's LAN address
        "unreal_port": 11111,
        "camera_id": 1,             # must match the LiveLinkFreeD subject in UE
        "send_rate_hz": 60,         # 0 = send on every pose, lowest latency
    },

    # ── Isaac ROS topics ──────────────────────────────────────────────────────
    "topics": {
        "odometry": "/visual_slam/tracking/odometry",
        "vo_pose": "/visual_slam/tracking/vo_pose_covariance",
        "status": "/visual_slam/status",
        "image": "/camera/infra1/image_rect_raw",
        "camera_info": "/camera/infra1/camera_info",
        "imu": "/camera/imu",
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
    },

    # ── ArUco ground-zero anchoring ───────────────────────────────────────────
    "aruco": {
        "enabled": True,
        "dictionary": "DICT_4X4_1000",
        "marker_size_m": 0.10,
        # Only used until camera_info arrives; real intrinsics come from the topic.
        "fallback_intrinsics": {
            "fx": 389.386, "fy": 389.386, "cx": 319.544, "cy": 241.506,
        },
    },

    # ── Optional extras ───────────────────────────────────────────────────────
    # GNSS is off by default: indoors there is no fix, and substituting a
    # GNSS-derived position into a SLAM-origin FreeD stream mid-take teleports
    # the virtual camera. Useful only for outdoor, large-volume work.
    "gnss": {
        "enabled": False,
        "port": "/dev/ttyACM0",
        "baud": 115200,
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
    rig = tr["rig_offset"]["translation_m"]
    if any(rig):
        lines.append(f"Rig offset    : {rig} m (ROS body axes)")
    return "\n".join("  " + line for line in lines)
