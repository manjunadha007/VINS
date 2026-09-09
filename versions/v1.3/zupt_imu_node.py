#!/usr/bin/env python3
"""
ZUPT IMU + Image Gate for VINS-Fusion (v3 — Dual Gate)
=======================================================
Gates BOTH the IMU AND stereo images until still-calibration
is complete. This ensures VINS-Fusion always receives IMU data
BEFORE the first image, guaranteeing correct VIO initialization.

Why dual-gate?
  VINS-Fusion's estimator.inputImage() requires IMU data in its
  internal buffer with timestamps BEFORE the image timestamp.
  If images arrive before any IMU (because the ZUPT gate held
  back IMU but images flowed freely), the visual-inertial
  alignment fails and VINS falls back to pure IMU integration.
  Gating both topics ensures they open simultaneously.

Pipeline:
  /camera/imu            → [ZUPT Gate] → /camera/imu_corrected
  /camera/infra1/image_rect_raw → [Gate] → /camera/infra1/gated
  /camera/infra2/image_rect_raw → [Gate] → /camera/infra2/gated

VINS config must use:
  imu_topic:    /camera/imu_corrected
  image0_topic: /camera/infra1/gated
  image1_topic: /camera/infra2/gated
"""
import rospy
import numpy as np
from sensor_msgs.msg import Imu, Image
from nav_msgs.msg import Odometry

# --- Calibration Parameters ---
STILL_GYRO_STD_THRESHOLD = 0.01    # rad/s  — max gyro std dev to be considered "still"
STILL_ACCEL_MAG_RANGE    = 0.5     # m/s²   — tolerance around gravity
CALIBRATION_SAMPLES      = 200     # still samples (~2s at 100Hz)
GRAVITY_MAGNITUDE        = 9.806   # m/s²

# --- Watchdog (read-only — LOG only, never restart VINS) ---
WATCHDOG_MAX_POSITION_M  = 3.0
WATCHDOG_MAX_SPEED_M_S   = 1.5
WATCHDOG_MIN_SAMPLES     = 50
WATCHDOG_SETTLE_SECS     = 5.0

class ZUPTNode:
    def __init__(self):
        rospy.init_node('zupt_imu_preprocessor', anonymous=False)

        # Gate state
        self._calibrated   = False
        self._calib_buffer = []
        self._calib_time   = None

        # Bias estimates
        self._gyro_bias  = np.zeros(3)
        self._accel_bias = np.zeros(3)

        # Watchdog
        self._last_pos      = None
        self._last_pos_time = None
        self._vio_samples   = 0
        self._drift_warned  = False

        # --- IMU publisher ---
        self._pub_imu   = rospy.Publisher('/camera/imu_corrected',  Imu,   queue_size=200)

        # --- Image gate publishers ---
        self._pub_img0  = rospy.Publisher('/camera/infra1/gated',   Image, queue_size=10)
        self._pub_img1  = rospy.Publisher('/camera/infra2/gated',   Image, queue_size=10)

        # --- Subscribers ---
        rospy.Subscriber('/camera/imu',                   Imu,      self._imu_callback,   queue_size=300)
        rospy.Subscriber('/camera/infra1/image_rect_raw', Image,    self._img0_callback,  queue_size=30)
        rospy.Subscriber('/camera/infra2/image_rect_raw', Image,    self._img1_callback,  queue_size=30)
        rospy.Subscriber('/vins_estimator/imu_propagate', Odometry, self._watchdog_cb,    queue_size=10)

        rospy.loginfo("=" * 60)
        rospy.loginfo("  ZUPT DUAL GATE — IMU + Image gate CLOSED")
        rospy.loginfo("  Place RealSense FLAT and STILL on a surface.")
        rospy.loginfo("  Both IMU and images are held until bias is calibrated.")
        rospy.loginfo("=" * 60)

    # ── Stillness check ──────────────────────────────────────────────────────
    def _is_still(self, samples):
        arr      = np.array(samples)
        accel_mag = np.linalg.norm(np.mean(arr[:, :3], axis=0))
        gyro_std  = np.std(arr[:, 3:], axis=0).max()
        return (abs(accel_mag - GRAVITY_MAGNITUDE) < STILL_ACCEL_MAG_RANGE
                and gyro_std < STILL_GYRO_STD_THRESHOLD)

    # ── Calibration ──────────────────────────────────────────────────────────
    def _calibrate(self, samples):
        arr        = np.array(samples)
        mean_accel = np.mean(arr[:, :3], axis=0)
        mean_gyro  = np.mean(arr[:, 3:], axis=0)

        g_axis = int(np.argmax(np.abs(mean_accel)))
        g_sign = float(np.sign(mean_accel[g_axis]))
        g_vec  = np.zeros(3)
        g_vec[g_axis] = g_sign * GRAVITY_MAGNITUDE

        self._accel_bias = mean_accel - g_vec
        self._gyro_bias  = mean_gyro
        self._calib_time = rospy.get_time()

        rospy.loginfo("=" * 60)
        rospy.loginfo("  ✅ ZUPT GATE OPEN — IMU + Images now flowing to VINS!")
        rospy.loginfo(f"  Gravity axis : {'XYZ'[g_axis]} "
                      f"({'+'  if g_sign > 0 else '-'}{GRAVITY_MAGNITUDE} m/s²)")
        rospy.loginfo(f"  Gyro  bias   : {self._gyro_bias.round(5)}")
        rospy.loginfo(f"  Accel bias   : {self._accel_bias.round(5)}")
        rospy.loginfo("  Slowly pick up and move the camera to init VINS.")
        rospy.loginfo("=" * 60)

    # ── IMU callback ──────────────────────────────────────────────────────────
    def _imu_callback(self, msg):
        ax, ay, az = (msg.linear_acceleration.x,
                      msg.linear_acceleration.y,
                      msg.linear_acceleration.z)
        gx, gy, gz = (msg.angular_velocity.x,
                      msg.angular_velocity.y,
                      msg.angular_velocity.z)

        if not self._calibrated:
            self._calib_buffer.append([ax, ay, az, gx, gy, gz])
            if len(self._calib_buffer) > CALIBRATION_SAMPLES:
                self._calib_buffer.pop(0)

            if len(self._calib_buffer) == CALIBRATION_SAMPLES:
                if self._is_still(self._calib_buffer):
                    self._calibrate(self._calib_buffer)
                    self._calibrated = True
                else:
                    self._calib_buffer = []
                    rospy.logwarn_throttle(
                        2.0, "[ZUPT] Camera moving — keep still to calibrate...")
            else:
                rem = CALIBRATION_SAMPLES - len(self._calib_buffer)
                if rem % 50 == 0 and rem > 0:
                    rospy.loginfo(f"[ZUPT] Collecting... {rem} samples left, keep still.")
            return  # Gate closed

        # Gate open — debias and publish
        out = Imu()
        out.header = msg.header
        out.linear_acceleration.x  = float(ax - self._accel_bias[0])
        out.linear_acceleration.y  = float(ay - self._accel_bias[1])
        out.linear_acceleration.z  = float(az - self._accel_bias[2])
        out.angular_velocity.x     = float(gx - self._gyro_bias[0])
        out.angular_velocity.y     = float(gy - self._gyro_bias[1])
        out.angular_velocity.z     = float(gz - self._gyro_bias[2])
        out.orientation                    = msg.orientation
        out.orientation_covariance         = msg.orientation_covariance
        out.linear_acceleration_covariance = msg.linear_acceleration_covariance
        out.angular_velocity_covariance    = msg.angular_velocity_covariance
        self._pub_imu.publish(out)

    # ── Image gate callbacks ──────────────────────────────────────────────────
    def _img0_callback(self, msg):
        if self._calibrated:
            self._pub_img0.publish(msg)

    def _img1_callback(self, msg):
        if self._calibrated:
            self._pub_img1.publish(msg)

    # ── Read-only watchdog ────────────────────────────────────────────────────
    def _watchdog_cb(self, msg):
        if not self._calibrated or self._calib_time is None:
            return
        if rospy.get_time() - self._calib_time < WATCHDOG_SETTLE_SECS:
            return

        px, py, pz = (msg.pose.pose.position.x,
                      msg.pose.pose.position.y,
                      msg.pose.pose.position.z)
        pos     = np.array([px, py, pz])
        pos_mag = float(np.linalg.norm(pos))

        self._vio_samples += 1
        if self._vio_samples < WATCHDOG_MIN_SAMPLES:
            return

        if pos_mag > WATCHDOG_MAX_POSITION_M and not self._drift_warned:
            rospy.logwarn("=" * 60)
            rospy.logwarn(f"  ⚠️  WATCHDOG: Position {pos_mag:.1f}m — drift detected!")
            rospy.logwarn("  Stop pipeline (Ctrl+C), replug RealSense, restart.")
            rospy.logwarn("=" * 60)
            self._drift_warned = True
            return

        if self._last_pos is not None and self._last_pos_time is not None:
            dt = rospy.get_time() - self._last_pos_time
            if dt > 0:
                speed = float(np.linalg.norm(pos - self._last_pos)) / dt
                if speed > WATCHDOG_MAX_SPEED_M_S and not self._drift_warned:
                    rospy.logwarn(f"[ZUPT] Watchdog: impossible speed {speed:.1f} m/s — bad init.")
                    self._drift_warned = True

        self._last_pos      = pos
        self._last_pos_time = rospy.get_time()


def main():
    node = ZUPTNode()
    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
