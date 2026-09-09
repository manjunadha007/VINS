#!/usr/bin/env python3
"""
Headless end-to-end verification of the camera tracking pipeline.

Answers, with numbers rather than impressions:
  1. Is stereo + IMU actually reaching Isaac ROS?
  2. Is the 6DoF pose sane, and how much does it drift when the rig is still?
  3. Does the ROS -> Unreal conversion and FreeD encoding round-trip exactly?
  4. Does the reported trajectory match a real movement of the camera?

Run inside the Isaac ROS container, with the SLAM pipeline already up:

    source /opt/ros/humble/setup.bash
    python3 src/verify_tracking.py --seconds 10            # stationary check
    python3 src/verify_tracking.py --seconds 20 --motion   # while you move the rig

Uses the real pose_transform and freed modules, so what it validates is exactly
what dashboard.py streams to Unreal Engine.
"""
import argparse
import math
import statistics
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu

import freed
import tracking_config
from pose_transform import PoseTransformer, quaternion_to_rpy


class Verifier(Node):
    def __init__(self, cfg):
        super().__init__("verify_tracking")
        self.transformer = PoseTransformer(cfg["transform"])
        self.cam_id = cfg["network"]["camera_id"]

        self.poses = []      # (t, position, quaternion)
        self.imu = []        # (t, accel_mag, gyro_mag)
        self.encode_errors = []
        self.max_encode_err = 0.0

        sensor_qos = QoSProfile(depth=200,
                                reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(
            Odometry, cfg["topics"]["odometry"], self._odom, 100)
        self.create_subscription(
            Imu, cfg["topics"]["imu"], self._imu, sensor_qos)

    def _odom(self, msg):
        p, o = msg.pose.pose.position, msg.pose.pose.orientation
        position = (p.x, p.y, p.z)
        quat = (o.x, o.y, o.z, o.w)
        self.poses.append((time.time(), position, quat))

        # Full output path: convert -> encode -> decode, and confirm the wire
        # values match the converted pose to within one FreeD quantisation step.
        pose = self.transformer.ros_to_unreal(position, quat)
        pkt = freed.get_freed_packet(self.cam_id, pose.tilt, pose.pan, pose.roll,
                                     pose.x, pose.y, pose.z)
        try:
            d = freed.decode_freed_packet(pkt)
        except ValueError as exc:
            self.encode_errors.append(str(exc))
            return
        err = max(
            abs(d["x_m"] - pose.x), abs(d["y_m"] - pose.y),
            abs(d["z_m"] - pose.z),
            abs(freed.wrap_deg(d["yaw_deg"] - pose.pan)),
            abs(freed.wrap_deg(d["pitch_deg"] - pose.tilt)),
            abs(freed.wrap_deg(d["roll_deg"] - pose.roll)),
        )
        self.max_encode_err = max(self.max_encode_err, err)

    def _imu(self, msg):
        a, g = msg.linear_acceleration, msg.angular_velocity
        self.imu.append((
            time.time(),
            math.sqrt(a.x ** 2 + a.y ** 2 + a.z ** 2),
            math.sqrt(g.x ** 2 + g.y ** 2 + g.z ** 2),
        ))


def rate_of(stamps):
    if len(stamps) < 2:
        return 0.0
    span = stamps[-1] - stamps[0]
    return (len(stamps) - 1) / span if span > 0 else 0.0


def report(v, duration, motion):
    ok = True
    print()
    print("=" * 68)
    print("  PIPELINE VERIFICATION REPORT")
    print("=" * 68)

    # ── 1. inputs ────────────────────────────────────────────────────────────
    imu_t = [s[0] for s in v.imu]
    pose_t = [s[0] for s in v.poses]
    imu_hz, pose_hz = rate_of(imu_t), rate_of(pose_t)

    print("\n1. SENSOR INPUT")
    print(f"   IMU messages        : {len(v.imu):5d}  ({imu_hz:6.1f} Hz)")
    print(f"   SLAM pose messages  : {len(v.poses):5d}  ({pose_hz:6.1f} Hz)")
    if v.imu:
        acc = [s[1] for s in v.imu]
        gyr = [s[2] for s in v.imu]
        mean_acc = statistics.mean(acc)
        print(f"   Accel magnitude     : {mean_acc:6.3f} m/s²  "
              f"(gravity = 9.807, error {abs(mean_acc - 9.807):.3f})")
        print(f"   Gyro magnitude      : mean {statistics.mean(gyr):.4f}  "
              f"max {max(gyr):.4f} rad/s")
        if abs(mean_acc - 9.807) > 1.0:
            print("   ⚠️  accelerometer magnitude is off — IMU may be miscalibrated")
            ok = False
    else:
        print("   ❌ NO IMU DATA — visual-inertial fusion is not being fed")
        ok = False
    if not v.poses:
        print("   ❌ NO SLAM POSES — tracking is not running")
        return False

    # ── 2. 6DoF sanity ───────────────────────────────────────────────────────
    positions = np.array([p[1] for p in v.poses])
    rpy = np.array([quaternion_to_rpy(p[2]) for p in v.poses])
    rpy_deg = np.degrees(rpy)

    print("\n2. 6DoF POSE (ROS frame: X fwd, Y left, Z up)")
    for i, axis in enumerate("XYZ"):
        col = positions[:, i]
        print(f"   {axis}: min {col.min():+8.4f}  max {col.max():+8.4f}  "
              f"range {col.max() - col.min():7.4f} m")
    for i, name in enumerate(("roll", "pitch", "yaw")):
        col = rpy_deg[:, i]
        print(f"   {name:<5}: min {col.min():+8.2f}  max {col.max():+8.2f}  "
              f"range {col.max() - col.min():7.2f}°")

    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(rpy)):
        print("   ❌ non-finite values in the pose stream")
        ok = False

    # ── 3. drift or motion ───────────────────────────────────────────────────
    steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    path_len = float(steps.sum())
    displacement = float(np.linalg.norm(positions[-1] - positions[0]))
    excursion = float(np.max(np.linalg.norm(positions - positions[0], axis=1)))

    if motion:
        print("\n3. MOTION TRACKING")
        print(f"   Path length         : {path_len:7.4f} m")
        print(f"   Net displacement    : {displacement:7.4f} m")
        print(f"   Max excursion       : {excursion:7.4f} m")
        print(f"   Peak speed          : {(steps.max() * pose_hz):7.4f} m/s")
        print(f"   Yaw swept           : {rpy_deg[:, 2].max() - rpy_deg[:, 2].min():7.2f}°")
        if path_len < 0.02:
            print("   ⚠️  almost no movement detected — was the rig actually moved?")
            ok = False
        else:
            print("   ✅ movement was tracked")
    else:
        print("\n3. STATIONARY DRIFT  (rig should be still)")
        print(f"   Max excursion       : {excursion * 1000:7.2f} mm")
        print(f"   Net displacement    : {displacement * 1000:7.2f} mm")
        print(f"   Accumulated path    : {path_len * 1000:7.2f} mm "
              f"(integrated jitter)")
        print(f"   Per-sample jitter   : mean {steps.mean() * 1e6:6.1f} µm  "
              f"max {steps.max() * 1e6:6.1f} µm")
        print(f"   Angular wander      : "
              f"{max(rpy_deg[:, i].max() - rpy_deg[:, i].min() for i in range(3)):.4f}°")
        if excursion > 0.02:
            print(f"   ⚠️  drifted {excursion * 1000:.1f} mm while stationary "
                  f"(>20 mm) — check lighting/texture and that the IR emitter is off")
            ok = False
        else:
            print("   ✅ drift within 20 mm")

    # ── 4. output path ───────────────────────────────────────────────────────
    print("\n4. ROS -> UNREAL -> FreeD OUTPUT PATH")
    tf = v.transformer
    first = tf.ros_to_unreal(v.poses[0][1], v.poses[0][2])
    last = tf.ros_to_unreal(v.poses[-1][1], v.poses[-1][2])
    print(f"   First pose -> {first}")
    print(f"   Last  pose -> {last}")
    print(f"   Encode/decode round-trip max error : {v.max_encode_err:.6f}")
    print(f"   Malformed packets                  : {len(v.encode_errors)}")

    # One FreeD quantisation step: 1/64000 m and 1/32768 deg.
    tolerance = max(1.0 / freed.TRANS_SCALE, 1.0 / freed.ANGLE_SCALE) * 1.5
    if v.encode_errors or v.max_encode_err > tolerance:
        print(f"   ❌ output path error exceeds one quantisation step "
              f"({tolerance:.6f})")
        ok = False
    else:
        print(f"   ✅ every pose encoded losslessly to within one LSB "
              f"({tolerance:.6f})")

    # Cross-check the sign convention on the axis that used to be broken.
    yaw_span = rpy_deg[:, 2].max() - rpy_deg[:, 2].min()
    if yaw_span > 2.0:
        idx_lo = int(np.argmin(rpy[:, 2]))
        idx_hi = int(np.argmax(rpy[:, 2]))
        pan_lo = tf.ros_to_unreal(v.poses[idx_lo][1], v.poses[idx_lo][2]).pan
        pan_hi = tf.ros_to_unreal(v.poses[idx_hi][1], v.poses[idx_hi][2]).pan
        print(f"\n   Pan response check (the old gimbal-lock bug):")
        print(f"     ROS yaw {rpy_deg[idx_lo, 2]:+7.2f}° -> FreeD pan {pan_lo:+7.2f}°")
        print(f"     ROS yaw {rpy_deg[idx_hi, 2]:+7.2f}° -> FreeD pan {pan_hi:+7.2f}°")
        if abs(pan_hi - pan_lo) < 0.5 * yaw_span:
            print("     ❌ pan is not tracking yaw — conversion is broken")
            ok = False
        else:
            print(f"     ✅ pan tracks yaw over {abs(pan_hi - pan_lo):.2f}°")

    print()
    print("=" * 68)
    print("  RESULT: " + ("✅ PASS" if ok else "⚠️  ISSUES FOUND — see above"))
    print("=" * 68)
    return ok


def run_benchmark(node, distance_m, report_path):
    """Guided there-and-back test: the real accuracy number.

    Phases are timed and announced so one person can run it alone. Move the rig
    a measured distance, bring it back to exactly the same spot, and the two
    numbers that matter fall out:

      scale error  -- reported outbound travel vs the distance you actually moved
      return error -- how far from the origin the estimate thinks you are, after
                      you physically returned to it. This is accumulated drift,
                      and it is the number to compare against any tracker's spec.
    """
    phases = [
        (6.0, "HOLD STILL — establishing origin"),
        (12.0, f"MOVE the camera {distance_m:.2f} m to the RIGHT, smoothly"),
        (4.0, "HOLD at the far point"),
        (12.0, "RETURN to EXACTLY the starting spot"),
        (6.0, "HOLD STILL — measuring return error"),
    ]
    marks = {}
    t0 = time.time()
    for duration, label in phases:
        print(f"\n>>> {label}  ({duration:.0f}s)", flush=True)
        end = time.time() + duration
        while time.time() < end:
            rclpy.spin_once(node, timeout_sec=0.02)
        marks[label] = len(node.poses)

    if len(node.poses) < 50:
        print("\n❌ not enough pose data — is the pipeline running?")
        return False

    P = np.array([p[1] for p in node.poses])
    idx = list(marks.values())
    origin = P[max(1, idx[0] // 2):idx[0]].mean(axis=0)          # settled origin
    far = P[idx[1]:idx[2]].mean(axis=0)                          # far point
    ret = P[idx[3]:].mean(axis=0)                                # returned point

    outbound = float(np.linalg.norm(far - origin))
    return_err = float(np.linalg.norm(ret - origin))
    scale_err = (outbound - distance_m) / distance_m * 100.0
    path = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())

    lines = [
        "=" * 68,
        "  TRAJECTORY FIDELITY BENCHMARK",
        "=" * 68,
        f"  commanded move      : {distance_m * 1000:8.1f} mm",
        f"  reported move       : {outbound * 1000:8.1f} mm",
        f"  SCALE ERROR         : {scale_err:+8.2f} %",
        "",
        f"  RETURN ERROR        : {return_err * 1000:8.1f} mm "
        f"({return_err / max(path, 1e-9) * 100:.2f} % of {path:.2f} m travelled)",
        "",
        f"  poses               : {len(P)}",
        f"  path length         : {path:8.3f} m",
    ]
    verdict = []
    if abs(scale_err) <= 5.0:
        verdict.append("  ✅ scale within 5 % — the virtual camera moves as far as the real one")
    else:
        verdict.append(f"  ⚠️  scale off by {scale_err:+.1f} % — check the measured distance, "
                       "and that transform.position_scale is 1.0")
    if return_err <= 0.02:
        verdict.append("  ✅ return error within 20 mm — good enough for virtual production")
    elif return_err <= 0.05:
        verdict.append("  ⚠️  return error 20-50 mm — usable for wide shots, visible on close work")
    else:
        verdict.append(f"  ❌ return error {return_err * 1000:.0f} mm — too much drift; "
                       "more scene texture, slower motion, or re-anchor on ArUco more often")
    out = "\n".join(lines + [""] + verdict + ["=" * 68])
    print("\n" + out)
    try:
        with open(report_path, "w") as fh:
            fh.write(out + "\n")
        print(f"\n(report written to {report_path})")
    except OSError as exc:
        print(f"(could not write report: {exc})")
    return return_err <= 0.05 and abs(scale_err) <= 5.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--motion", action="store_true",
                    help="expect the rig to be moved during the capture")
    ap.add_argument("--benchmark", action="store_true",
                    help="guided there-and-back accuracy test")
    ap.add_argument("--distance", type=float, default=0.5,
                    help="how far you will move the rig, in metres (default 0.5)")
    ap.add_argument("--report",
                    default="/workspaces/isaac_ros-dev/fidelity_report.txt")
    args = ap.parse_args()

    cfg = tracking_config.load_config(verbose=False)

    rclpy.init()
    node = Verifier(cfg)

    if args.benchmark:
        print("=" * 68)
        print("  GUIDED ACCURACY BENCHMARK")
        print("=" * 68)
        print(f"  Mark two spots {args.distance:.2f} m apart on the bench or floor.")
        print("  Follow the prompts. Move SMOOTHLY and keep textured objects in")
        print("  view. Total time about 40 s.")
        ok = run_benchmark(node, args.distance, args.report)
        node.destroy_node()
        rclpy.shutdown()
        return 0 if ok else 1

    print(f"Capturing {args.seconds:.0f}s from "
          f"{cfg['topics']['odometry']} and {cfg['topics']['imu']} ...")
    if args.motion:
        print(">>> MOVE THE CAMERA NOW — smooth pans and translations. <<<")
    else:
        print(">>> Keep the camera COMPLETELY STILL. <<<")

    deadline = time.time() + args.seconds
    tty = sys.stdout.isatty()
    next_tick = time.time() + 1.0
    while time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
        now = time.time()
        if now >= next_tick:          # once a second, and never \r into a pipe
            next_tick = now + 1.0
            end, pre = ("\r", "  ") if tty else ("\n", "  ")
            print(f"{pre}{deadline - now:4.1f}s left   "
                  f"poses={len(node.poses):5d} imu={len(node.imu):5d}",
                  end=end, flush=True)
    print()

    ok = report(node, args.seconds, args.motion)
    node.destroy_node()
    rclpy.shutdown()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
