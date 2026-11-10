#!/usr/bin/env python3
"""
Manual FreeD injector — no camera, no SLAM, no ROS required.

Streams a slowly panning camera at a fixed position so you can confirm the
Unreal Engine side (LiveLinkFreeD subject, port, camera ID) is wired up before
blaming the tracking pipeline.

    python3 test_freed.py            # camera ID 1 (matches the rest of the repo)
    python3 test_freed.py 0          # override camera ID

This used to default to camera ID 0 while every real bridge sent ID 1, so a
"working" test showed nothing in Unreal.
"""
import socket
import sys
import time

from freed import get_freed_packet, FREED_DEFAULT_IP, FREED_DEFAULT_PORT

UDP_IP = FREED_DEFAULT_IP
UDP_PORT = FREED_DEFAULT_PORT
CAMERA_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 1

print(f"Sending test FreeD packets to {UDP_IP}:{UDP_PORT} as camera {CAMERA_ID}...")
print("Press Ctrl+C to stop.")

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
angle = 0
try:
    while True:
        pkt = get_freed_packet(CAMERA_ID, 0, angle, 0, 1.0, 2.0, 3.0)
        sock.sendto(pkt, (UDP_IP, UDP_PORT))
        angle = (angle + 1) % 360
        time.sleep(1.0 / 30.0)
except KeyboardInterrupt:
    print("\nStopped.")
