#!/usr/bin/env python3
"""
Configure the tracking anchor from the command line.

Speaks the same JSON-over-UDP protocol the Unreal plugin uses, so you can set up
and test the LiDAR-anchor workflow before the plugin side exists — and diagnose
it afterwards. Works against either front-end (dashboard.py or
bridge_headless.py); whichever is running owns the port.

    # is anything listening?
    python3 anchor_cli.py ping

    # marker becomes the world origin (standalone shoot)
    python3 anchor_cli.py origin

    # marker becomes the HFLidarAnchor Actor's spot in the LiDAR scan.
    # Position is Unreal CENTIMETRES, rotation is pan/tilt/roll DEGREES,
    # exactly as the Actor reports them.
    python3 anchor_cli.py lidar --pos 250 -400 120 --rot 30 0 0

    # forget the anchor and go back to the raw SLAM frame
    python3 anchor_cli.py reset

    # what does the bridge think is going on?
    python3 anchor_cli.py status

Arming is not the same as anchoring. These commands *arm* the anchor; it fires on
the next frame in which the marker is actually detected, and it is refused if the
depth cross-check disagrees with the PnP distance. `status` shows both the armed
flag and the last marker result, so you can tell the difference between "no
marker in view" and "marker seen but rejected".
"""
import argparse
import json
import socket
import sys

import tracking_config

STATUS_PORT = 11113   # bridge_headless.py broadcasts JSON status here


def send_command(payload, host, port, timeout=2.0):
    """Send one command and return the reply dict, or None on no reply."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(timeout)
    try:
        sock.sendto(json.dumps(payload).encode("utf-8"), (host, port))
        data, _ = sock.recvfrom(65535)
        return json.loads(data.decode("utf-8"))
    except socket.timeout:
        return None
    finally:
        sock.close()


def listen_status(port, timeout=3.0):
    """Grab one status datagram from the headless bridge's broadcast."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.settimeout(timeout)
    try:
        sock.bind(("0.0.0.0", port))
        data, _ = sock.recvfrom(65535)
        return json.loads(data.decode("utf-8"))
    except (socket.timeout, OSError):
        return None
    finally:
        sock.close()


def main():
    cfg = tracking_config.load_config(verbose=False)
    default_port = int(cfg["network"].get("command_port", 11114))

    ap = argparse.ArgumentParser(
        description="Configure the camera-tracking anchor over UDP.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Configure the tracking anchor")[-1])
    ap.add_argument("action",
                    choices=["ping", "origin", "lidar", "reset", "status",
                             "offset"],
                    help="ping | origin | lidar | reset | status | offset")
    ap.add_argument("--host", default="127.0.0.1",
                    help="where the bridge is listening (default 127.0.0.1)")
    ap.add_argument("--port", type=int, default=default_port,
                    help=f"command port (default {default_port})")
    ap.add_argument("--pos", nargs=3, type=float, metavar=("X", "Y", "Z"),
                    help="anchor position in Unreal CENTIMETRES")
    ap.add_argument("--rot", nargs=3, type=float, default=[0.0, 0.0, 0.0],
                    metavar=("PAN", "TILT", "ROLL"),
                    help="anchor rotation in degrees (default 0 0 0)")
    args = ap.parse_args()

    if args.action == "status":
        status = listen_status(STATUS_PORT)
        if status is None:
            print(f"No status on UDP {STATUS_PORT} within 3 s.")
            print("Only bridge_headless.py broadcasts status; if you are running")
            print("the GUI dashboard, read the status panel in the window.")
            return 1
        print(json.dumps(status, indent=2))
        return 0

    if args.action == "ping":
        payload = {"cmd": "ping"}
    elif args.action == "origin":
        payload = {"cmd": "anchor_aruco"}
    elif args.action == "reset":
        payload = {"cmd": "reset_pose"}
    elif args.action == "offset":
        if not args.pos:
            ap.error("offset needs --pos X Y Z (metres, ROS frame)")
        payload = {"cmd": "set_offset",
                   "tx": args.pos[0], "ty": args.pos[1], "tz": args.pos[2],
                   "rx": args.rot[0], "ry": args.rot[1], "rz": args.rot[2]}
    else:   # lidar
        if not args.pos:
            ap.error("lidar needs --pos X Y Z in Unreal centimetres")
        payload = {"cmd": "anchor_lidar",
                   "marker_map_position": args.pos,
                   "marker_map_rotation": args.rot}

    print(f"-> {args.host}:{args.port}  {json.dumps(payload)}")
    reply = send_command(payload, args.host, args.port)
    if reply is None:
        print("\nNo reply. Either nothing is listening on that port, or the")
        print("bridge is running but its command channel failed to bind.")
        print("Check:")
        print("  * is dashboard.py or bridge_headless.py actually running?")
        print("  * inside a container, is the port reachable from here?")
        print("  * network.command_port in tracking.yaml matches --port")
        return 1

    print(f"<- {json.dumps(reply, indent=2)}")
    if reply.get("ok") and args.action in ("lidar", "origin"):
        print("\nArmed. Point the camera at the marker — it anchors on the next")
        print("detection. Run 'status' (headless) or watch the dashboard panel to")
        print("confirm; a depth/PnP disagreement refuses the anchor and stays armed.")
    return 0 if reply.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
