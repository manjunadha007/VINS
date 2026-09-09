#!/usr/bin/env python3
"""
Real-time consumer for the D455 depth stream — measures whether depth can be
used live, and reports metrics worth using.

Run it inside the Isaac container alongside the running pipeline:

    docker exec -it -u admin -w /workspaces/isaac_ros-dev \
        isaac_ros_dev-x86_64-container bash -lc \
        "source /opt/ros/humble/setup.bash && source install/setup.bash && \
         python3 src/depth_probe.py"

Needs the camera launched with VP_ENABLE_DEPTH=1, otherwise the topic never
appears and this says so rather than hanging silently.

What it answers
---------------
Timing, because "real time" is the actual question:
    rate        frames actually arriving per second
    latency     capture timestamp to this process receiving it
    proc        wall time to decode and analyse one frame
    budget      proc as a share of the frame period -- over 100% means the
                consumer cannot keep up and the stream will back up

Content, because raw depth is not directly useful:
    valid       share of pixels with a depth return at all. Low values mean
                featureless or out-of-range scene.
    median      median scene distance
    in-band     share of returns inside the range where a 95 mm baseline
                actually resolves disparity. Beyond that, depth is present but
                its precision degrades with the square of distance, and so does
                the translation component of stereo odometry.

Depth is decoded straight out of the message buffer with numpy rather than
through cv_bridge: it is a plain 16-bit array, and the copy cv_bridge adds is
a real cost at 90 Hz.
"""

import argparse
import sys
import time

import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from sensor_msgs.msg import Image, CameraInfo
except ImportError:
    sys.exit("[probe] no rclpy — run this inside the Isaac ROS container "
             "with the ROS environment sourced (see module docstring)")


DEPTH_TOPIC = "/camera/depth/image_rect_raw"
INFO_TOPIC = "/camera/depth/camera_info"

# A 95 mm baseline stops resolving useful disparity well before the sensor
# stops returning a number. Depth outside this band is reported, not trusted.
GOOD_MIN_M = 0.35
GOOD_MAX_M = 5.0


class DepthProbe(Node):

    def __init__(self, args):
        super().__init__("hf_depth_probe")
        self.args = args
        self.stride = max(1, args.stride)

        # Sensor data QoS: best-effort, matching how the RealSense driver
        # publishes. A reliable subscriber silently receives nothing here.
        qos = QoSProfile(depth=1,
                         reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST)

        self.create_subscription(Image, args.topic, self.on_depth, qos)
        self.create_subscription(CameraInfo, args.info_topic, self.on_info, qos)

        self.fx = self.fy = self.cx = self.cy = None

        self.started = time.time()
        self.first_frame_at = None
        self.count = 0
        self.window_start = time.time()
        self.window_frames = 0
        self.proc_times = []
        self.latencies = []
        self.last_stamp = None
        self.gaps = 0

        self.create_timer(1.0, self.report)
        print("[probe] listening on {}".format(args.topic))
        print("[probe] waiting for first frame — if nothing arrives, relaunch "
              "with VP_ENABLE_DEPTH=1")

    def on_info(self, msg):
        if self.fx is None:
            k = msg.k
            self.fx, self.fy, self.cx, self.cy = k[0], k[4], k[2], k[5]

    def on_depth(self, msg):
        t0 = time.perf_counter()

        if self.first_frame_at is None:
            self.first_frame_at = time.time()
            print("[probe] first depth frame after {:.2f} s — {}x{} {}".format(
                self.first_frame_at - self.started,
                msg.width, msg.height, msg.encoding))

        # 16UC1 in millimetres is what the RealSense driver publishes. Read the
        # buffer in place; no copy, no cv_bridge.
        if msg.encoding not in ("16UC1", "mono16"):
            self.get_logger().warn("unexpected encoding {}".format(msg.encoding),
                                   throttle_duration_sec=5.0)
            return

        depth = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
        if self.stride > 1:
            depth = depth[::self.stride, ::self.stride]

        metres = depth.astype(np.float32) * 0.001
        valid = metres > 0.0
        n_valid = int(valid.sum())

        if n_valid:
            vals = metres[valid]
            median = float(np.median(vals))
            in_band = float(np.count_nonzero(
                (vals >= GOOD_MIN_M) & (vals <= GOOD_MAX_M))) / n_valid
            nearest = float(vals.min())
        else:
            median = in_band = nearest = 0.0

        proc_ms = (time.perf_counter() - t0) * 1000.0

        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        now = self.get_clock().now().nanoseconds * 1e-9
        self.latencies.append((now - stamp) * 1000.0)

        # A gap much larger than the nominal period means frames were dropped
        # between the driver and here.
        if self.last_stamp is not None and self.args.expect_hz > 0:
            period = 1.0 / self.args.expect_hz
            if (stamp - self.last_stamp) > period * 1.8:
                self.gaps += 1
        self.last_stamp = stamp

        self.proc_times.append(proc_ms)
        self.count += 1
        self.window_frames += 1
        self.latest = (n_valid, valid.size, median, in_band, nearest)

    def report(self):
        if not self.window_frames:
            if self.first_frame_at is None:
                print("[probe] ...no depth frames yet ({:.0f} s)".format(
                    time.time() - self.started))
            return

        elapsed = time.time() - self.window_start
        rate = self.window_frames / elapsed
        proc = np.asarray(self.proc_times)
        lat = np.asarray(self.latencies)
        n_valid, n_total, median, in_band, nearest = self.latest

        budget = 0.0
        if self.args.expect_hz > 0:
            budget = float(proc.mean()) / (1000.0 / self.args.expect_hz) * 100.0

        print("[probe] {:5.1f} fps | proc {:5.2f} ms (p95 {:5.2f}) = {:4.0f}% of "
              "budget | latency {:5.1f} ms | valid {:4.1f}% | median {:4.2f} m | "
              "in-band {:4.1f}% | near {:4.2f} m | gaps {}".format(
                  rate, float(proc.mean()), float(np.percentile(proc, 95)),
                  budget, float(lat.mean()),
                  100.0 * n_valid / max(n_total, 1), median,
                  100.0 * in_band, nearest, self.gaps))

        self.window_start = time.time()
        self.window_frames = 0
        self.proc_times.clear()
        self.latencies.clear()


def main():
    ap = argparse.ArgumentParser(description="Real-time D455 depth probe")
    ap.add_argument("--topic", default=DEPTH_TOPIC)
    ap.add_argument("--info-topic", default=INFO_TOPIC)
    ap.add_argument("--stride", type=int, default=1,
                    help="subsample by this factor before analysis (1 = full res)")
    ap.add_argument("--expect-hz", type=float, default=90.0,
                    help="nominal stream rate, used for budget and gap detection")
    args = ap.parse_args()

    rclpy.init()
    node = DepthProbe(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.count:
            print("\n[probe] {} frames, {} gaps".format(node.count, node.gaps))
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
