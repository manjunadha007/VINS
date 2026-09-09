#!/usr/bin/env python3
"""
Bridge simulator / protocol checker for the HyperFusionVP Unreal plugin.

This exists so the Unreal side can be developed and tested with no RealSense, no
ROS and no Docker container running. It speaks exactly the same wire protocol as
bridge_headless.py:

    Port 11112 (out) -> JPEG frames, 4-byte big-endian length header + payload
    Port 11113 (out) -> JSON status, one object per datagram
    Port 11114 (in)  <- JSON commands from Unreal

Modes
-----
  serve     Pretend to be the bridge: stream synthetic frames + status, and print
            any command Unreal sends back. Use this to bring up the plugin.

  sniff     Bind a port and validate what arrives. Point it at 11112 to confirm
            the real bridge is emitting well-formed frames, or at 11113 to watch
            the status JSON.

  send-cmd  Fire a single command at a running bridge on 11114, to check that its
            listener reacts without needing Unreal in the loop.

Examples
--------
    python3 test_video_stream.py serve
    python3 test_video_stream.py serve --fps 30 --size 640x480
    python3 test_video_stream.py sniff --port 11112
    python3 test_video_stream.py sniff --port 11113
    python3 test_video_stream.py send-cmd store_pose
    python3 test_video_stream.py send-cmd set_offset --tx 0.1 --rz 15
"""

import argparse
import json
import socket
import struct
import sys
import threading
import time

try:
    import cv2
    import numpy as np
except ImportError:
    sys.exit("[sim] needs opencv-python and numpy: pip install opencv-python numpy")


# Keep these in lockstep with bridge_headless.py
VIDEO_PORT = 11112
STATUS_PORT = 11113
COMMAND_PORT = 11114
JPEG_QUALITY = 70
MAX_UDP_PAYLOAD = 65000


# ──────────────────────────────────────────────────
# Synthetic frame generation
# ──────────────────────────────────────────────────

def make_frame(width, height, n):
    """A frame whose content changes every tick.

    The moving bar and the counter both matter: a static test pattern looks
    identical whether the texture is updating at 30 fps or frozen on frame 1.
    """
    img = np.zeros((height, width), dtype=np.uint8)

    # Diagonal gradient so JPEG has something to actually compress
    xs = np.linspace(0, 120, width, dtype=np.uint8)
    img[:, :] = xs[None, :]

    # Sweeping vertical bar
    bar_x = int((n * 6) % max(width - 40, 1))
    img[:, bar_x:bar_x + 40] = 255

    # Horizon line + frame counter
    cv2.line(img, (0, height // 2), (width, height // 2), 90, 1)
    cv2.putText(img, f"HyperFusion SIM  frame {n}", (12, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2, cv2.LINE_AA)
    cv2.putText(img, time.strftime("%H:%M:%S"), (12, height - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, 200, 1, cv2.LINE_AA)

    # Corner ticks make it obvious in Unreal if the overlay is cropped or flipped
    cv2.rectangle(img, (2, 2), (width - 3, height - 3), 255, 2)
    cv2.circle(img, (width - 30, 30), 12, 255, -1)
    return img


def encode_jpeg(img):
    ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        return None
    return buf.tobytes()


# ──────────────────────────────────────────────────
# serve — act as the bridge
# ──────────────────────────────────────────────────

def cmd_serve(args):
    try:
        width, height = (int(v) for v in args.size.lower().split('x'))
    except ValueError:
        sys.exit(f"[sim] --size must look like 640x480, got {args.size!r}")

    video_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    status_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    state = {"running": True, "sent": 0, "dropped": 0, "commands": 0}

    def command_listener():
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(('0.0.0.0', COMMAND_PORT))
        except OSError as e:
            print(f"[sim] cannot bind command port {COMMAND_PORT}: {e}")
            print("[sim] (is the real bridge already running?)")
            return
        sock.settimeout(0.5)
        print(f"[sim] command listener on {COMMAND_PORT}")
        while state["running"]:
            try:
                data, addr = sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            state["commands"] += 1
            try:
                msg = json.loads(data.decode('utf-8'))
                print(f"[sim] <- command from {addr[0]}: {msg}")
            except (UnicodeDecodeError, json.JSONDecodeError):
                print(f"[sim] <- malformed command from {addr[0]}: {data[:80]!r}")
        sock.close()

    def status_loop():
        while state["running"]:
            status = {
                "status": "TRACKING",
                "fps_in": 89.5,
                "fps_out": float(args.fps),
                "gate_rejects": state["dropped"],
                "gate_resyncs": 0,
                "anchored": False,
                "pose_stored": False,
            }
            if args.marker:
                status["marker"] = {"id": 42, "tvec": [0.10, -0.05, 1.25]}
            try:
                status_sock.sendto(json.dumps(status).encode('utf-8'),
                                   (args.host, STATUS_PORT))
            except OSError:
                pass
            time.sleep(0.5)

    threading.Thread(target=command_listener, daemon=True).start()
    threading.Thread(target=status_loop, daemon=True).start()

    print(f"[sim] streaming {width}x{height} @ {args.fps} fps -> {args.host}:{VIDEO_PORT}")
    print(f"[sim] status -> {args.host}:{STATUS_PORT} every 500 ms")
    print("[sim] Ctrl+C to stop\n")

    interval = 1.0 / args.fps
    n = 0
    last_report = time.time()
    try:
        while True:
            tick = time.time()
            jpeg = encode_jpeg(make_frame(width, height, n))
            if jpeg is None:
                state["dropped"] += 1
            elif len(jpeg) > MAX_UDP_PAYLOAD:
                # The real bridge downscales here; for the sim just note it.
                state["dropped"] += 1
                print(f"[sim] frame {n} too large ({len(jpeg)} B), dropped")
            else:
                video_sock.sendto(struct.pack('>I', len(jpeg)) + jpeg,
                                  (args.host, VIDEO_PORT))
                state["sent"] += 1

            n += 1
            if tick - last_report >= 2.0:
                print(f"[sim] frame {n}  sent={state['sent']}  "
                      f"dropped={state['dropped']}  cmds={state['commands']}  "
                      f"last={len(jpeg) if jpeg else 0} B")
                last_report = tick

            slack = interval - (time.time() - tick)
            if slack > 0:
                time.sleep(slack)
    except KeyboardInterrupt:
        print("\n[sim] stopping")
    finally:
        state["running"] = False
        video_sock.close()
        status_sock.close()
        print(f"[sim] sent {state['sent']} frames, {state['commands']} commands received")


# ──────────────────────────────────────────────────
# sniff — validate what a real bridge emits
# ──────────────────────────────────────────────────

def cmd_sniff(args):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2 * 1024 * 1024)
    try:
        sock.bind(('0.0.0.0', args.port))
    except OSError as e:
        sys.exit(f"[sniff] cannot bind {args.port}: {e}\n"
                 f"[sniff] Unreal may already hold it — stop PIE and retry.")
    sock.settimeout(2.0)
    print(f"[sniff] listening on {args.port}, Ctrl+C to stop\n")

    count = bad = 0
    window_start = time.time()
    window_count = 0
    window_bytes = 0
    try:
        while True:
            try:
                data, addr = sock.recvfrom(65536)
            except socket.timeout:
                print("[sniff] ...nothing for 2s")
                continue

            count += 1
            window_count += 1
            window_bytes += len(data)

            if args.port == STATUS_PORT:
                try:
                    print(f"[sniff] {json.loads(data.decode('utf-8'))}")
                except (UnicodeDecodeError, json.JSONDecodeError):
                    bad += 1
                    print(f"[sniff] BAD status payload: {data[:80]!r}")
                continue

            # Video framing check
            problem = None
            if len(data) < 5:
                problem = f"runt datagram ({len(data)} B)"
            else:
                declared = struct.unpack('>I', data[:4])[0]
                payload = data[4:]
                if declared != len(payload):
                    problem = f"header says {declared} B, carries {len(payload)} B"
                elif payload[:2] != b'\xff\xd8':
                    problem = f"payload is not JPEG (starts {payload[:2].hex()})"
                elif payload[-2:] != b'\xff\xd9':
                    problem = "JPEG missing EOI marker"

            if problem:
                bad += 1
                print(f"[sniff] BAD frame #{count}: {problem}")
            elif args.decode:
                arr = np.frombuffer(data[4:], dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    bad += 1
                    print(f"[sniff] BAD frame #{count}: JPEG failed to decode")

            now = time.time()
            if now - window_start >= 1.0:
                fps = window_count / (now - window_start)
                kbps = window_bytes / (now - window_start) / 1024
                shape = ""
                if args.decode and not problem:
                    shape = f"  {img.shape[1]}x{img.shape[0]}"
                print(f"[sniff] {fps:5.1f} fps  {kbps:7.1f} KiB/s  "
                      f"total={count} bad={bad}{shape}  from {addr[0]}")
                window_start, window_count, window_bytes = now, 0, 0
    except KeyboardInterrupt:
        print(f"\n[sniff] {count} datagrams, {bad} malformed")
        sys.exit(1 if bad else 0)


# ──────────────────────────────────────────────────
# send-cmd — poke a running bridge
# ──────────────────────────────────────────────────

def cmd_send(args):
    if args.action == "set_offset":
        msg = {"cmd": "set_offset", "tx": args.tx, "ty": args.ty, "tz": args.tz,
               "rx": args.rx, "ry": args.ry, "rz": args.rz, "scale": args.scale}
    else:
        msg = {"cmd": args.action}

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(json.dumps(msg).encode('utf-8'), (args.host, COMMAND_PORT))
    sock.close()
    print(f"[cmd] -> {args.host}:{COMMAND_PORT}  {msg}")


# ──────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="HyperFusion bridge simulator and protocol checker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Examples")[-1])
    sub = ap.add_subparsers(dest="mode", required=True)

    p = sub.add_parser("serve", help="stream synthetic frames + status like the bridge")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--size", default="640x480")
    p.add_argument("--marker", action="store_true",
                   help="include a fake ArUco detection in the status JSON")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("sniff", help="bind a port and validate arriving datagrams")
    p.add_argument("--port", type=int, default=VIDEO_PORT)
    p.add_argument("--decode", action="store_true",
                   help="fully decode each JPEG (slower, catches corrupt payloads)")
    p.set_defaults(func=cmd_sniff)

    p = sub.add_parser("send-cmd", help="send one command to a running bridge")
    p.add_argument("action", choices=["store_pose", "reset_pose", "anchor_aruco",
                                      "set_offset", "ping"])
    p.add_argument("--host", default="127.0.0.1")
    for axis in ("tx", "ty", "tz", "rx", "ry", "rz"):
        p.add_argument(f"--{axis}", type=float, default=0.0)
    p.add_argument("--scale", type=float, default=1.0)
    p.set_defaults(func=cmd_send)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
