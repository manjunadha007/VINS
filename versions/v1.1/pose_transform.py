#!/usr/bin/env python3
"""
ROS -> Unreal Engine pose transform for virtual production camera tracking.

This is the mathematical core of the bridge: it turns an Isaac ROS Visual SLAM
pose into the pan/tilt/roll + XYZ that Unreal Engine's LiveLinkFreeD plugin
expects. Deliberately free of ROS, Qt and OpenCV imports so it can be unit
tested anywhere and reasoned about in isolation.

Frames
------
Isaac ROS Visual SLAM publishes the pose of `base_frame` (camera_link in the
shipped RealSense launch) in the `odom` frame. Per NVIDIA's own header,
isaac_ros_visual_slam/impl/cuvslam_ros_conversion.hpp:

    Canonical ROS Frame (x-forward, y-left, z-up)     right-handed

Unreal Engine uses:

    x-forward, y-right, z-up                          LEFT-handed

So the conversion is a single mirror of the Y axis, plus the rotation-sign
consequences of swapping handedness:

    position:  X_ue = +X_ros,  Y_ue = -Y_ros,  Z_ue = +Z_ros
    rotation:  roll = +roll_ros,  tilt = -pitch_ros,  pan = -yaw_ros

Sanity check on each rotation axis:
  * ROS +yaw   = counter-clockwise seen from above = camera pans LEFT.
    Unreal +yaw = clockwise from above = pans RIGHT.        -> sign flip
  * ROS +pitch = right-hand rule about Y(left) = nose DOWN.
    Unreal +pitch = nose UP.                                -> sign flip
  * ROS +roll  = right-hand rule about X(forward) = left side up.
    Unreal +roll = right side down = left side up.          -> same sign

Why not the old matrix pair
---------------------------
The previous implementation did `R_ue = P_w . R_vins . P_b` where P_b mapped a
camera *optical* frame (z-forward, x-right, y-down) into the body frame. Isaac
publishes camera_link, which is already x-forward, so that extra 90-degree
double rotation was spurious. Worse, it landed the result exactly on the Euler
singularity (pitch == 90 for a level camera), so `euler_from_matrix` zeroed the
third angle: FreeD pan was pinned at 0.0 forever and real pan motion came out on
the roll channel. A level camera is the *normal* operating state, so the bug was
always active.

Extracting roll/pitch/yaw directly in ROS RPY order (equivalently intrinsic
Z-Y-X) is singular only when the camera points straight up or down, which is the
standard, well-understood pan/tilt limitation.
"""
import math

import numpy as np

# ── Basic rotation helpers (no scipy / tf_transformations dependency) ─────────


def quaternion_to_matrix(q):
    """Quaternion (x, y, z, w) -> 3x3 rotation matrix.

    Matches tf_transformations.quaternion_matrix()[:3, :3].
    """
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def matrix_to_rpy(R):
    """3x3 rotation matrix -> (roll, pitch, yaw) radians, ROS RPY convention.

    ROS RPY is extrinsic X-Y-Z, which is identical to intrinsic Z-Y-X, i.e.
    R = Rz(yaw) @ Ry(pitch) @ Rx(roll). Equivalent to
    tf_transformations.euler_from_matrix(R, 'sxyz').
    """
    R = np.asarray(R, dtype=float)
    sin_pitch = -R[2, 0]
    sin_pitch = max(-1.0, min(1.0, sin_pitch))
    pitch = math.asin(sin_pitch)

    # cos(pitch) ~ 0 means the camera points straight up or down: roll and yaw
    # become degenerate (only their sum is observable). Attribute it all to yaw
    # and hold roll at zero, which keeps pan continuous through the pole.
    if abs(math.cos(pitch)) < 1e-8:
        roll = 0.0
        yaw = math.atan2(-R[0, 1], R[1, 1])
    else:
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw


def quaternion_to_rpy(q):
    """Quaternion (x, y, z, w) -> (roll, pitch, yaw) radians, ROS convention."""
    return matrix_to_rpy(quaternion_to_matrix(q))


def rpy_to_matrix(roll, pitch, yaw):
    """(roll, pitch, yaw) radians -> 3x3 matrix, R = Rz(yaw) Ry(pitch) Rx(roll)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ])


def make_transform(position, quaternion):
    """(position, quaternion) -> 4x4 homogeneous transform."""
    T = np.eye(4)
    T[:3, :3] = quaternion_to_matrix(quaternion)
    T[:3, 3] = np.asarray(position, dtype=float)
    return T


def invert_transform(T):
    """Inverse of a rigid 4x4 transform (transpose + rotated translation)."""
    R = np.asarray(T, dtype=float)[:3, :3]
    t = np.asarray(T, dtype=float)[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


# ── The transform itself ─────────────────────────────────────────────────────

class UnrealPose:
    """A pose in Unreal/FreeD terms: metres and degrees, ready to encode."""

    __slots__ = ("x", "y", "z", "pan", "tilt", "roll")

    def __init__(self, x, y, z, pan, tilt, roll):
        self.x, self.y, self.z = x, y, z
        self.pan, self.tilt, self.roll = pan, tilt, roll

    def as_tuple(self):
        return (self.x, self.y, self.z, self.pan, self.tilt, self.roll)

    def __repr__(self):
        return (f"UnrealPose(x={self.x:+.4f} y={self.y:+.4f} z={self.z:+.4f} "
                f"pan={self.pan:+.2f} tilt={self.tilt:+.2f} roll={self.roll:+.2f})")

    def __eq__(self, other):
        return (isinstance(other, UnrealPose)
                and self.as_tuple() == other.as_tuple())


#: Correct ROS-canonical -> Unreal mapping, derived in the module docstring.
#: Overridable per-axis because the final sign depends on how the Unreal project
#: orients its LiveLink subject; see calibrate_axes.py and README.md.
DEFAULT_SIGNS = {
    "position": (1.0, -1.0, 1.0),   # X, Y, Z
    "pan": -1.0,
    "tilt": -1.0,
    "roll": 1.0,
}


class PoseTransformer:
    """Converts Isaac ROS poses into Unreal/FreeD poses.

    Handles, in order:
      1. rig offset   -- camera_link to the film camera's nodal point
      2. ground zero  -- make some chosen pose the Unreal origin
      3. frame change -- ROS canonical (RH) to Unreal (LH)
      4. trim         -- per-axis sign, angle offset, position scale
    """

    def __init__(self, config=None):
        cfg = config or {}
        self.position_scale = float(cfg.get("position_scale", 1.0))
        self.position_sign = tuple(
            float(s) for s in cfg.get("position_sign", DEFAULT_SIGNS["position"]))
        self.pan_sign = float(cfg.get("pan_sign", DEFAULT_SIGNS["pan"]))
        self.tilt_sign = float(cfg.get("tilt_sign", DEFAULT_SIGNS["tilt"]))
        self.roll_sign = float(cfg.get("roll_sign", DEFAULT_SIGNS["roll"]))

        # Trim, applied after the frame change, in Unreal degrees.
        self.pan_offset = float(cfg.get("pan_offset_deg", 0.0))
        self.tilt_offset = float(cfg.get("tilt_offset_deg", 0.0))
        self.roll_offset = float(cfg.get("roll_offset_deg", 0.0))

        # Rig offset: where the real camera's nodal point sits relative to
        # camera_link, expressed in ROS body axes (x fwd, y left, z up).
        self._rig = self._build_rig_offset(cfg.get("rig_offset") or {})

        self._zero = None          # 4x4 inverse of the ground-zero pose

    @staticmethod
    def _build_rig_offset(rig_cfg):
        xyz = rig_cfg.get("translation_m", (0.0, 0.0, 0.0))
        rpy = rig_cfg.get("rotation_deg", (0.0, 0.0, 0.0))
        T = np.eye(4)
        T[:3, :3] = rpy_to_matrix(*(math.radians(float(a)) for a in rpy))
        T[:3, 3] = np.asarray([float(v) for v in xyz])
        return T

    @property
    def has_rig_offset(self):
        return not np.allclose(self._rig, np.eye(4))

    @property
    def is_zeroed(self):
        return self._zero is not None

    # ── ground zero ──────────────────────────────────────────────────────────
    def set_ground_zero(self, position, quaternion):
        """Make this pose the Unreal origin (position and heading)."""
        T = make_transform(position, quaternion) @ self._rig
        self._zero = invert_transform(T)

    def set_ground_zero_from_transform(self, T_world_marker):
        """Ground zero from an externally computed 4x4 pose (e.g. ArUco)."""
        self._zero = invert_transform(np.asarray(T_world_marker, dtype=float))

    def clear_ground_zero(self):
        self._zero = None

    # ── the conversion ───────────────────────────────────────────────────────
    def ros_to_unreal(self, position, quaternion):
        """Isaac ROS pose -> UnrealPose (metres, degrees)."""
        T = make_transform(position, quaternion) @ self._rig
        if self._zero is not None:
            T = self._zero @ T

        roll, pitch, yaw = matrix_to_rpy(T[:3, :3])
        tx, ty, tz = T[:3, 3]

        sx, sy, sz = self.position_sign
        k = self.position_scale

        return UnrealPose(
            x=tx * sx * k,
            y=ty * sy * k,
            z=tz * sz * k,
            pan=math.degrees(yaw) * self.pan_sign + self.pan_offset,
            tilt=math.degrees(pitch) * self.tilt_sign + self.tilt_offset,
            roll=math.degrees(roll) * self.roll_sign + self.roll_offset,
        )


# ── Latency compensation ─────────────────────────────────────────────────────

class PosePredictor:
    """Extrapolates the last pose forward to hide pipeline latency.

    Isaac ROS publishes odometry at image rate (30 fps in the shipped RealSense
    launch) but Unreal wants a smooth 60 Hz+ stream, and there is always some
    capture-to-render delay. Isaac fills Odometry.twist with a real cuVSLAM
    velocity estimate, so prediction is a straight constant-velocity step rather
    than the hand-rolled gyro integration this replaces -- that competed with
    Isaac's own visual-inertial fusion and raced it for `latest_pose`.

    Extrapolation is capped: if tracking stalls, holding the last good pose is
    far better than flying the virtual camera off into space.
    """

    def __init__(self, lead_time_s=0.0, max_extrapolation_s=0.05):
        self.lead_time_s = float(lead_time_s)
        self.max_extrapolation_s = float(max_extrapolation_s)
        self._pos = None
        self._quat = None
        self._lin = np.zeros(3)
        self._ang = np.zeros(3)
        self._stamp = None

    def update(self, position, quaternion, linear_velocity, angular_velocity, now):
        self._pos = np.asarray([float(v) for v in position])
        self._quat = tuple(float(v) for v in quaternion)
        self._lin = np.asarray([float(v) for v in linear_velocity])
        self._ang = np.asarray([float(v) for v in angular_velocity])
        self._stamp = float(now)

    @property
    def has_pose(self):
        return self._pos is not None

    def age(self, now):
        if self._stamp is None:
            return float("inf")
        return float(now) - self._stamp

    def predict(self, now):
        """(position, quaternion) extrapolated to `now` + lead_time."""
        if self._pos is None:
            return None
        dt = (float(now) - self._stamp) + self.lead_time_s
        dt = max(0.0, min(dt, self.max_extrapolation_s))
        if dt == 0.0:
            return self._pos.copy(), self._quat

        position = self._pos + self._lin * dt

        # Angular velocity is body-rate; integrate as a small-angle rotation.
        omega = self._ang * dt
        theta = float(np.linalg.norm(omega))
        R = quaternion_to_matrix(self._quat)
        if theta > 1e-12:
            axis = omega / theta
            K = np.array([[0.0, -axis[2], axis[1]],
                          [axis[2], 0.0, -axis[0]],
                          [-axis[1], axis[0], 0.0]])
            dR = (np.eye(3) + math.sin(theta) * K
                  + (1.0 - math.cos(theta)) * (K @ K))
            R = R @ dR
        return position, matrix_to_quaternion(R)


#: Below this dt the per-second rate tests are meaningless (see PoseGate.check).
_MIN_RATE_DT_S = 0.001


class PoseGate:
    """Rejects physically impossible pose updates before they reach Unreal.

    Measured on this rig: cuVSLAM can emit a 1254 mm position change between two
    consecutive 90 Hz frames — an implied 113 m/s. That is not camera motion, it
    is the tracker relocalising or a loop closure snapping the estimate. Passed
    through untouched it teleports the virtual camera mid-shot, which is
    unusable; a professional tracker holds the last good pose instead.

    After `max_consecutive_rejects` rejections in a row the new pose is accepted
    anyway and the gate re-anchors. Without that, a genuine relocalisation (say
    the operator carried the rig to a new position while it was occluded) would
    freeze the output forever.
    """

    def __init__(self, max_speed_m_s=5.0, max_angular_deg_s=900.0,
                 max_jump_m=0.15, max_consecutive_rejects=15, enabled=True):
        self.enabled = bool(enabled)
        self.max_speed_m_s = float(max_speed_m_s)
        self.max_angular_deg_s = float(max_angular_deg_s)
        self.max_jump_m = float(max_jump_m)
        self.max_consecutive_rejects = int(max_consecutive_rejects)

        self._last_pos = None
        self._last_quat = None
        self._last_t = None
        self.consecutive_rejects = 0
        self.total_rejects = 0
        self.total_accepts = 0
        self.resyncs = 0
        self.last_reason = None

    @staticmethod
    def _angle_between(q1, q2):
        """Smallest rotation angle between two quaternions, in degrees."""
        d = abs(sum(a * b for a, b in zip(q1, q2)))
        d = max(-1.0, min(1.0, d))
        return math.degrees(2.0 * math.acos(d))

    def check(self, position, quaternion, now):
        """Return (accept, reason). reason is None when accepted normally."""
        position = np.asarray([float(v) for v in position])
        quaternion = tuple(float(v) for v in quaternion)
        now = float(now)

        if not self.enabled or self._last_pos is None or self._last_t is None:
            self._commit(position, quaternion, now)
            return True, None

        dt = now - self._last_t
        if dt <= 0.0:
            # Same or out-of-order timestamp: nothing to test against.
            self._commit(position, quaternion, now)
            return True, None

        step = float(np.linalg.norm(position - self._last_pos))

        # The absolute step test always applies. The rate tests only apply when
        # dt is large enough to be meaningful: `now` should be the message
        # timestamp, but if poses ever arrive batched (or a caller passes arrival
        # time) dt collapses towards zero and every real move looks like it
        # happened at infinite speed. Rejecting genuine camera motion is worse
        # than missing an outlier, and max_jump_m still bounds the damage.
        rate_tests_valid = dt >= _MIN_RATE_DT_S

        reason = None
        if step > self.max_jump_m:
            reason = (f"jump {step * 1000:.0f}mm > {self.max_jump_m * 1000:.0f}mm")
        elif rate_tests_valid:
            speed = step / dt
            ang_speed = self._angle_between(self._last_quat, quaternion) / dt
            if speed > self.max_speed_m_s:
                reason = f"speed {speed:.1f} > {self.max_speed_m_s:.1f} m/s"
            elif ang_speed > self.max_angular_deg_s:
                reason = f"spin {ang_speed:.0f} > {self.max_angular_deg_s:.0f} deg/s"

        if reason is None:
            self.consecutive_rejects = 0
            self._commit(position, quaternion, now)
            return True, None

        self.consecutive_rejects += 1
        self.total_rejects += 1
        self.last_reason = reason

        if self.consecutive_rejects >= self.max_consecutive_rejects:
            # Persistent disagreement means the tracker really has moved on.
            self.resyncs += 1
            self.consecutive_rejects = 0
            self._commit(position, quaternion, now)
            return True, f"resync after {reason}"

        return False, reason

    def _commit(self, position, quaternion, now):
        self._last_pos = position
        self._last_quat = quaternion
        self._last_t = now
        self.total_accepts += 1

    def reset(self):
        self._last_pos = None
        self._last_quat = None
        self._last_t = None
        self.consecutive_rejects = 0
        self._offset_pos = np.zeros(3)
        self._offset_rot = np.eye(3)

    # ── continuity filtering ─────────────────────────────────────────────────
    # Rejecting a snap only defers it: the tracker keeps reporting the new
    # position, so after `max_consecutive_rejects` the gate resyncs and the jump
    # lands anyway. Replaying the recorded 90 s divergence through reject-only
    # logic cut peak excursion from 138.9 m to just 135.3 m -- useless.
    #
    # Absorbing is what actually works. When the estimate snaps by delta, fold
    # -delta into a standing offset so the OUTPUT stays continuous. Real camera
    # motion passes through untouched; relocalisations and loop closures become
    # invisible. Absolute accuracy is not recovered -- that is what the ArUco
    # ground-zero re-anchor is for -- but the virtual camera never teleports.

    def filter(self, position, quaternion, now):
        """Return (position, quaternion, snapped, reason) with snaps absorbed."""
        if not hasattr(self, "_offset_pos"):
            self._offset_pos = np.zeros(3)
            self._offset_rot = np.eye(3)

        raw_pos = np.asarray([float(v) for v in position])
        raw_quat = tuple(float(v) for v in quaternion)

        prev_pos = None if self._last_pos is None else self._last_pos.copy()
        prev_quat = self._last_quat
        accepted, reason = self.check(raw_pos, raw_quat, now)

        if not accepted:
            # Hold: report the previous corrected pose unchanged.
            out_pos = self._offset_pos + (prev_pos if prev_pos is not None
                                          else raw_pos)
            out_rot = self._offset_rot @ quaternion_to_matrix(
                prev_quat if prev_quat is not None else raw_quat)
            return out_pos, matrix_to_quaternion(out_rot), True, reason

        if reason and prev_pos is not None and prev_quat is not None:
            # Resync: absorb the discontinuity instead of passing it on.
            target_pos = self._offset_pos + prev_pos
            self._offset_pos = target_pos - raw_pos
            target_rot = self._offset_rot @ quaternion_to_matrix(prev_quat)
            self._offset_rot = target_rot @ quaternion_to_matrix(raw_quat).T
            self.absorbed = getattr(self, "absorbed", 0) + 1

        out_pos = self._offset_pos + raw_pos
        out_rot = self._offset_rot @ quaternion_to_matrix(raw_quat)
        return out_pos, matrix_to_quaternion(out_rot), bool(reason), reason


def matrix_to_quaternion(R):
    """3x3 rotation matrix -> quaternion (x, y, z, w). Shepperd's method."""
    R = np.asarray(R, dtype=float)
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return (x, y, z, w)


# ── Optional output smoothing ────────────────────────────────────────────────

class PoseSmoother:
    """One-pole low-pass on the Unreal-side pose, with an angle-aware path.

    Off by default: filtering trades latency for steadiness, and on a tracked
    camera latency is usually the more expensive of the two. Useful when the
    plate is locked off or the rig is on a slider and micro-jitter shows up in
    the composite.
    """

    def __init__(self, alpha=0.0):
        self.alpha = float(alpha)      # 0 = no smoothing, ->1 = heavy
        self._prev = None

    @staticmethod
    def _blend_angle(prev, new, a):
        """Interpolate along the shortest arc so 179 -> -179 does not spin."""
        delta = (new - prev + 180.0) % 360.0 - 180.0
        return prev + (1.0 - a) * delta

    def apply(self, pose):
        if self.alpha <= 0.0:
            self._prev = pose
            return pose
        a = min(self.alpha, 0.99)
        if self._prev is None:
            self._prev = pose
            return pose
        p = self._prev
        out = UnrealPose(
            x=p.x + (1 - a) * (pose.x - p.x),
            y=p.y + (1 - a) * (pose.y - p.y),
            z=p.z + (1 - a) * (pose.z - p.z),
            pan=self._blend_angle(p.pan, pose.pan, a),
            tilt=self._blend_angle(p.tilt, pose.tilt, a),
            roll=self._blend_angle(p.roll, pose.roll, a),
        )
        self._prev = out
        return out

    def reset(self):
        self._prev = None
