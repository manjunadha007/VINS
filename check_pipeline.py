#!/usr/bin/env python3
"""Require live stereo, IMU and odometry before declaring startup ready.

Run inside the Isaac container after starting SLAM. This subscribes only; it
does not open the camera, change parameters, or send FreeD packets.
"""
import argparse
from dataclasses import dataclass
import math
import time


@dataclass
class StreamHealth:
    count: int = 0
    first: float = None
    last: float = None
    stamp: float = None

    def observe(self, stamp, now, valid=True):
        if not valid or not math.isfinite(stamp) or stamp <= 0:
            return
        if self.stamp is not None and stamp <= self.stamp:
            return  # Repeated/stale frames cannot satisfy readiness.
        if self.last is None or now - self.last > 0.5:
            self.first, self.count = now, 0
        self.last, self.stamp = now, stamp
        self.count += 1

    def ready(self, now):
        return (self.count >= 5 and self.last - self.first >= 1.0
                and now - self.last <= 0.5)

    @property
    def hz(self):
        span = 0.0 if self.last is None else self.last - self.first
        return (self.count - 1) / span if span > 0 else 0.0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--require-depth", action="store_true")
    parser.add_argument("--require-active-depth", action="store_true")
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be positive and finite")

    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, Image, Imu
    from nav_msgs.msg import Odometry
    from tracking_config import load_config

    cfg = load_config(args.config, verbose=False)
    topics = cfg["topics"]
    required = {
        "left IR": (Image, topics["image"]),
        "right IR": (Image, topics.get("image_right", "/camera/infra2/image_rect_raw")),
        "IR calibration": (CameraInfo, topics["camera_info"]),
        "IMU": (Imu, topics["imu"]),
        "SLAM odometry": (Odometry, topics["odometry"]),
    }
    if args.require_depth or args.require_active_depth or cfg["lidar"]["enabled"]:
        required.update({
            "depth": (Image, topics["depth"]),
            "depth calibration": (CameraInfo, topics["depth_camera_info"]),
        })
    if args.require_active_depth:
        required["active-depth preview"] = (Image, topics["depth_preview"])
    health = {name: StreamHealth() for name in required}
    rclpy.init(args=[])
    node = rclpy.create_node("hyperfusion_startup_check")

    def receive(msg, name):
        valid = True
        if isinstance(msg, Image):
            valid = msg.width > 0 and msg.height > 0 and len(msg.data) > 0
        elif isinstance(msg, CameraInfo):
            valid = (all(math.isfinite(v) for v in msg.k)
                     and msg.k[0] > 0 and msg.k[4] > 0)
        elif isinstance(msg, Imu):
            a, g = msg.linear_acceleration, msg.angular_velocity
            valid = (all(math.isfinite(v) for v in (a.x, a.y, a.z, g.x, g.y, g.z))
                     and a.x*a.x + a.y*a.y + a.z*a.z > 0)
        elif isinstance(msg, Odometry):
            p, q = msg.pose.pose.position, msg.pose.pose.orientation
            valid = (all(math.isfinite(v) for v in (p.x, p.y, p.z, q.x, q.y, q.z, q.w))
                     and abs(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w - 1) < 0.1)
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        health[name].observe(stamp, time.monotonic(), valid)

    subscriptions = [
        node.create_subscription(kind, topic, lambda msg, name=name: receive(msg, name),
                                 qos_profile_sensor_data)
        for name, (kind, topic) in required.items()
    ]
    print("    Waiting for live stereo, IMU, calibration and SLAM data...", flush=True)
    deadline = time.monotonic() + args.timeout
    ready = False
    try:
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            now = time.monotonic()
            if all(stream.ready(now) for stream in health.values()):
                ready = True
                break
        now = time.monotonic()
        for name, stream in health.items():
            status = "OK" if stream.ready(now) else "MISSING / STALE"
            print(f"    {name}: {status}, {stream.hz:.1f} Hz ({required[name][1]})")
        if not ready:
            print("    Pipeline is not ready. Keep IMU enabled; check camera/SLAM logs.")
        return 0 if ready else 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
