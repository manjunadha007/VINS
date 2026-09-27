#!/usr/bin/env python3
"""
Axis calibration for the Unreal Engine side of the bridge.

Drives the virtual camera through one axis at a time with synthetic FreeD data —
no camera, no SLAM, no ROS needed — so you can confirm that each axis moves the
way you expect in Unreal before you trust the tracking. If an axis moves the
wrong way, flip the matching sign in tracking.yaml.

    python3 calibrate_axes.py                     # full guided sequence
    python3 calibrate_axes.py --axis pan          # just one axis, looping
    python3 calibrate_axes.py --hold x=1,pan=45   # park at a fixed pose
    python3 calibrate_axes.py --ip 192.168.1.50   # override the target
    python3 calibrate_axes.py --show-config       # print effective settings

Before running, in Unreal:
  1. Enable the LiveLink FreeD plugin and restart the editor.
  2. Window > Virtual Production > Live Link, Add Source > FreeD,
     set the UDP port to match (default 11111).
  3. Confirm a subject appears and its Camera ID matches network.camera_id.
  4. Attach the subject to a CineCameraActor via a LiveLinkComponentController.
"""
import argparse
import math
import socket
import sys
import time

import freed
import tracking_config

# label -> (which tracking.yaml key to flip, what you should see in Unreal)
AXES = {
    "x":    ("transform.position_sign[0]", "move FORWARD (away from you)"),
    "y":    ("transform.position_sign[1]", "move RIGHT"),
    "z":    ("transform.position_sign[2]", "move UP"),
    "pan":  ("transform.pan_sign",         "pan/rotate to the RIGHT"),
    "tilt": ("transform.tilt_sign",        "tilt UP"),
    "roll": ("transform.roll_sign",        "roll CLOCKWISE (right side down)"),
}

TRANSLATION_AMPLITUDE_M = 1.0
ROTATION_AMPLITUDE_DEG = 45.0
SEND_HZ = 60


def build_pose(axis, value):
    """Return (x, y, z, pan, tilt, roll) with a single axis displaced."""
    pose = {"x": 0.0, "y": 0.0, "z": 0.0, "pan": 0.0, "tilt": 0.0, "roll": 0.0}
    if axis in pose:
        pose[axis] = value
    return (pose["x"], pose["y"], pose["z"],
            pose["pan"], pose["tilt"], pose["roll"])


def send(sock, target, cam_id, pose):
    x, y, z, pan, tilt, roll = pose
    # get_freed_packet takes (pitch, yaw, roll) = (tilt, pan, roll)
    sock.sendto(freed.get_freed_packet(cam_id, tilt, pan, roll, x, y, z), target)


def sweep(sock, target, cam_id, axis, duration=6.0, cycles=1.5):
    """Smoothly oscillate one axis so the motion is unmistakable in Unreal."""
    amplitude = (ROTATION_AMPLITUDE_DEG if axis in ("pan", "tilt", "roll")
                 else TRANSLATION_AMPLITUDE_M)
    unit = "deg" if axis in ("pan", "tilt", "roll") else "m"
    key, expectation = AXES[axis]

    print()
    print(f"── {axis.upper()} " + "─" * 56)
    print(f"   Sending {axis} = 0 -> +{amplitude:g} {unit}")
    print(f"   Unreal should: {expectation}")
    print(f"   If it does the OPPOSITE, negate {key} in tracking.yaml")

    period = 1.0 / SEND_HZ
    steps = int(duration * SEND_HZ)
    for i in range(steps):
        # 0 -> 1 -> 0 raised cosine: always starts and ends at rest.
        phase = (i / steps) * cycles * 2.0 * math.pi
        value = amplitude * 0.5 * (1.0 - math.cos(phase))
        send(sock, target, cam_id, build_pose(axis, value))
        time.sleep(period)

    send(sock, target, cam_id, build_pose(axis, 0.0))


def hold(sock, target, cam_id, spec, duration=None):
    """Park at a fixed pose, e.g. 'x=1.5,pan=90'."""
    pose = {"x": 0.0, "y": 0.0, "z": 0.0, "pan": 0.0, "tilt": 0.0, "roll": 0.0}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise SystemExit(f"bad --hold term {part!r}; expected name=value")
        name, _, value = part.partition("=")
        name = name.strip().lower()
        if name not in pose:
            raise SystemExit(f"unknown axis {name!r}; choose from {list(pose)}")
        pose[name] = float(value)

    tup = (pose["x"], pose["y"], pose["z"],
           pose["pan"], pose["tilt"], pose["roll"])
    decoded = freed.decode_freed_packet(
        freed.get_freed_packet(cam_id, pose["tilt"], pose["pan"], pose["roll"],
                              pose["x"], pose["y"], pose["z"]))
    print(f"Holding  X {decoded['x_m']:+.3f}  Y {decoded['y_m']:+.3f}  "
          f"Z {decoded['z_m']:+.3f}  pan {decoded['yaw_deg']:+.2f}  "
          f"tilt {decoded['pitch_deg']:+.2f}  roll {decoded['roll_deg']:+.2f}")
    print("Ctrl+C to stop.")

    deadline = None if duration is None else time.time() + duration
    while deadline is None or time.time() < deadline:
        send(sock, target, cam_id, tup)
        time.sleep(1.0 / SEND_HZ)


def main():
    ap = argparse.ArgumentParser(
        description="Verify Unreal Engine axis conventions with synthetic FreeD.")
    ap.add_argument("--ip", help="override the Unreal Engine IP")
    ap.add_argument("--port", type=int, help="override the UDP port")
    ap.add_argument("--camera-id", type=int, help="override the FreeD camera ID")
    ap.add_argument("--axis", choices=sorted(AXES), help="sweep one axis, looping")
    ap.add_argument("--hold", metavar="SPEC",
                    help="hold a fixed pose, e.g. 'x=1,pan=45'")
    ap.add_argument("--duration", type=float, default=6.0,
                    help="seconds per axis sweep (default 6)")
    ap.add_argument("--show-config", action="store_true",
                    help="print the effective configuration and exit")
    args = ap.parse_args()

    cfg = tracking_config.load_config()
    if args.show_config:
        print(tracking_config.describe(cfg))
        print(f"  Config file   : {cfg.get('_source') or 'defaults (no file found)'}")
        return 0

    net = cfg["network"]
    ip = args.ip or net["unreal_ip"]
    port = args.port or net["unreal_port"]
    cam_id = args.camera_id if args.camera_id is not None else net["camera_id"]
    target = (ip, port)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    print("═" * 64)
    print("  FreeD AXIS CALIBRATION")
    print("═" * 64)
    print(f"  Target      : {ip}:{port}  (camera {cam_id})")
    print(f"  Config      : {cfg.get('_source') or 'defaults'}")
    if ip in ("127.0.0.1", "localhost"):
        print("  NOTE: targeting localhost. If Unreal runs on another machine,")
        print("        set network.unreal_ip in tracking.yaml or pass --ip.")
    print()
    print("  Watch the virtual camera in Unreal and compare each move to the")
    print("  expected direction printed below.")

    try:
        if args.hold:
            hold(sock, target, cam_id, args.hold)
            return 0

        if args.axis:
            print(f"\nLooping {args.axis}. Ctrl+C to stop.")
            while True:
                sweep(sock, target, cam_id, args.axis, duration=args.duration)
            return 0

        print("\n  Parking at origin for 3s — the virtual camera should sit")
        print("  at the world origin, perfectly level.")
        deadline = time.time() + 3.0
        while time.time() < deadline:
            send(sock, target, cam_id, build_pose("none", 0.0))
            time.sleep(1.0 / SEND_HZ)

        for axis in ("x", "y", "z", "pan", "tilt", "roll"):
            sweep(sock, target, cam_id, axis, duration=args.duration)

        print()
        print("═" * 64)
        print("  DONE. For every axis that moved the wrong way, negate the")
        print("  matching sign in tracking.yaml, then re-run to confirm:")
        print()
        for axis, (key, _) in AXES.items():
            print(f"    {axis:<5} -> {key}")
        print("═" * 64)
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
