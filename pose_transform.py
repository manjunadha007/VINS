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
import collections
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


def decompose_similarity(T):
    """Split a similarity transform into (uniform scale, rigid 4x4).

    The 3D viewport's mesh-alignment gizmo builds `T_align` as ``R * scale``
    with a translation, because the operator needs a scale slider: a scan
    exported from Unreal is in centimetres while everything else here is in
    metres, and that slider is how the two are reconciled.

    That makes `T_align` a *similarity*, not a rigid transform, and composing
    it with poses as if it were rigid is silently wrong -- `invert_transform`
    assumes ``R^-1 == R^T`` and comes back off by the square of the scale, and
    `matrix_to_rpy` reads angles off a matrix whose entries are no longer
    sines and cosines. So callers that need to compose it with a pose must
    take the rigid part from here and handle the scale explicitly.

    For ``M = s * R``, every column has norm s, which is what this measures.
    """
    T = np.asarray(T, dtype=float)
    M = T[:3, :3]
    scale = float(np.mean(np.linalg.norm(M, axis=0)))
    if scale < 1e-12:
        return 1.0, np.eye(4)
    rigid = np.eye(4)
    rigid[:3, :3] = M / scale
    rigid[:3, 3] = T[:3, 3]
    return scale, rigid


def invert_similarity(T):
    """Inverse of a similarity transform, scale included.

    `invert_transform` is only correct for a rigid transform. Use this when
    the matrix may carry the mesh-alignment scale.
    """
    scale, rigid = decompose_similarity(T)
    R = rigid[:3, :3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T / scale
    Ti[:3, 3] = -(R.T @ np.asarray(T, dtype=float)[:3, 3]) / scale
    return Ti


#: camera OPTICAL frame (x right, y down, z forward) -> camera_link BODY frame
#: (x forward, y left, z up). Depth images, point clouds and solvePnP all answer
#: in the optical frame; the SLAM pose is in the body frame. Composing the two
#: without this mixes frames -- it puts an ArUco anchor in the wrong place and
#: sprays the depth point cloud into the wrong part of the LiDAR map.
#:
#: Lives here, in the frame-maths module, because three separate copies of it had
#: grown across tracking_engine.py and lidar_map_matcher.py.
T_BODY_OPTICAL = np.array([
    [0.0,  0.0, 1.0, 0.0],
    [-1.0, 0.0, 0.0, 0.0],
    [0.0, -1.0, 0.0, 0.0],
    [0.0,  0.0, 0.0, 1.0],
])


# ── SE(3) log / exp, for filtering a transform rather than a sample ───────────
#
# The map-localisation fusion (lidar_map_matcher.py) filters a *transform* --
# the slowly-varying SLAM-to-map drift correction -- not a position. Averaging
# 4x4 matrices elementwise does not produce a rigid transform, so the blend has
# to happen in the Lie algebra: convert the difference to a 6-vector twist,
# scale it, and map it back. Kept here, numpy-only, so it is testable without
# open3d or a camera.

def so3_log(R):
    """3x3 rotation matrix -> rotation vector (axis * angle, radians)."""
    R = np.asarray(R, dtype=float)
    cos_theta = (np.trace(R) - 1.0) / 2.0
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    theta = math.acos(cos_theta)

    if theta < 1e-8:
        # Near identity: the first-order term is exact to machine precision here.
        return np.array([R[2, 1] - R[1, 2],
                         R[0, 2] - R[2, 0],
                         R[1, 0] - R[0, 1]]) / 2.0

    if theta > math.pi - 1e-5:
        # Near pi the antisymmetric part vanishes; recover the axis from the
        # symmetric part instead, where it is still well conditioned.
        A = (R + np.eye(3)) / 2.0
        axis = np.sqrt(np.clip(np.diag(A), 0.0, None))
        # Signs are lost by the squaring; fix them against the largest component.
        k = int(np.argmax(axis))
        if axis[k] > 1e-8:
            axis = A[:, k] / axis[k]
        n = np.linalg.norm(axis)
        if n < 1e-8:
            return np.zeros(3)
        return axis / n * theta

    return theta / (2.0 * math.sin(theta)) * np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1]])


def so3_exp(omega):
    """Rotation vector (axis * angle, radians) -> 3x3 rotation matrix."""
    omega = np.asarray(omega, dtype=float).reshape(3)
    theta = float(np.linalg.norm(omega))
    K = np.array([[0.0, -omega[2], omega[1]],
                  [omega[2], 0.0, -omega[0]],
                  [-omega[1], omega[0], 0.0]])
    if theta < 1e-8:
        return np.eye(3) + K          # second order term is below float64 noise
    return (np.eye(3)
            + math.sin(theta) / theta * K
            + (1.0 - math.cos(theta)) / (theta * theta) * (K @ K))


def se3_log(T):
    """Rigid 4x4 -> twist (rho[3], omega[3]). Inverse of se3_exp()."""
    T = np.asarray(T, dtype=float)
    omega = so3_log(T[:3, :3])
    t = T[:3, 3]
    theta = float(np.linalg.norm(omega))

    if theta < 1e-8:
        return np.concatenate([t, omega])

    K = np.array([[0.0, -omega[2], omega[1]],
                  [omega[2], 0.0, -omega[0]],
                  [-omega[1], omega[0], 0.0]]) / theta
    # V^-1 for the SE(3) exponential, from the closed form of the left Jacobian.
    half = theta / 2.0
    V_inv = (np.eye(3)
             - theta / 2.0 * K
             + (1.0 - half / math.tan(half)) * (K @ K))
    return np.concatenate([V_inv @ t, omega])


def se3_exp(xi):
    """Twist (rho[3], omega[3]) -> rigid 4x4. Inverse of se3_log()."""
    xi = np.asarray(xi, dtype=float).reshape(6)
    rho, omega = xi[:3], xi[3:]
    theta = float(np.linalg.norm(omega))

    T = np.eye(4)
    T[:3, :3] = so3_exp(omega)
    if theta < 1e-8:
        T[:3, 3] = rho
        return T

    K = np.array([[0.0, -omega[2], omega[1]],
                  [omega[2], 0.0, -omega[0]],
                  [-omega[1], omega[0], 0.0]]) / theta
    V = (np.eye(3)
         + (1.0 - math.cos(theta)) / theta * K
         + (1.0 - math.sin(theta) / theta) * (K @ K))
    T[:3, 3] = V @ rho
    return T


def se3_interpolate(T_from, T_to, alpha):
    """Geodesic blend: alpha=0 gives T_from, alpha=1 gives T_to.

    Stays on the manifold at every alpha, unlike an elementwise matrix lerp,
    so the result is always a valid rigid transform.
    """
    if alpha <= 0.0:
        return np.asarray(T_from, dtype=float).copy()
    if alpha >= 1.0:
        return np.asarray(T_to, dtype=float).copy()
    delta = invert_transform(T_from) @ np.asarray(T_to, dtype=float)
    return np.asarray(T_from, dtype=float) @ se3_exp(alpha * se3_log(delta))


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


#: Unreal world units per metre. Unreal is centimetre-based, so an Actor's
#: GetActorLocation() is in centimetres; anchors arriving from the Unreal side
#: are divided by this to reach metres.
UNREAL_UNITS_PER_METRE = 100.0

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

        self._zero = None          # 4x4 mapping VIO world -> output frame
        self._target = None        # what the marker was anchored onto, if any

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
    def set_ground_zero(self, position, quaternion, T_target=None):
        """Map this pose onto T_target (the Unreal origin when target is None)."""
        T = make_transform(position, quaternion) @ self._rig
        self._set_zero(T, T_target)

    def set_ground_zero_from_transform(self, T_world_marker, T_target=None):
        """Map an externally computed pose (e.g. an ArUco marker) onto T_target.

        With `T_target` left as None the marker becomes the world origin, which
        is the right behaviour for a standalone shoot: press anchor, the marker is
        (0, 0, 0).

        Pass `T_target` to put the marker somewhere specific instead. That is the
        LiDAR-scan workflow: the marker is taped to a known spot in the scanned
        set, an anchor Actor sits at that same spot in Unreal, and the tracking
        output has to come out in the *scan's* coordinate frame rather than in a
        frame invented by wherever the operator happened to stand.

        Without this, anchoring pinned the marker to the origin and the virtual
        camera came out offset by exactly the anchor Actor's position -- which
        looks perfect whenever that Actor happens to sit near the origin, and is
        silently wrong everywhere else.
        """
        self._set_zero(np.asarray(T_world_marker, dtype=float), T_target)

    def _set_zero(self, T_world_marker, T_target):
        inv_marker = invert_transform(T_world_marker)
        if T_target is None:
            self._zero = inv_marker
            self._target = None
        else:
            T_target = np.asarray(T_target, dtype=float)
            self._zero = T_target @ inv_marker
            self._target = T_target

    @property
    def anchor_target(self):
        """The 4x4 the marker was mapped onto, or None if it went to the origin."""
        return None if self._target is None else self._target.copy()

    def clear_ground_zero(self):
        self._zero = None
        self._target = None

    def set_map_alignment(self, T_zero):
        """Drive ground zero straight from a LiDAR-map localisation fix.

        `_zero` is the SLAM-world -> output-frame transform, and map fusion
        estimates precisely that (see lidar_map_matcher.py). Writing it here
        rather than nudging each incoming sample is what makes the correction
        apply to position *and* rotation, in one place, with no per-sample
        offset bookkeeping to fall out of step with the measurement.

        `anchor_target` is deliberately left alone: it records what the
        operator anchored onto, and refining the fit against the scan does not
        change that intent.
        """
        self._zero = np.asarray(T_zero, dtype=float).copy()

    # ── Unreal -> ROS, for anchors that arrive from the engine ───────────────
    def unreal_to_ros_transform(self, position_unreal, rotation_deg,
                                unreal_units_per_metre=UNREAL_UNITS_PER_METRE):
        """An Unreal Actor transform -> 4x4 in ROS canonical metres.

        Exactly inverts this instance's own ros_to_unreal() mapping, signs, trim
        and scale included, so a flipped axis in tracking.yaml cannot put the
        anchor and the live pose into different frames.

        `position_unreal` is in Unreal world units (centimetres by default, which
        is what GetActorLocation returns) and `rotation_deg` is (pan, tilt, roll)
        to match the order this class emits.
        """
        pan, tilt, roll = (float(a) for a in rotation_deg)
        sx, sy, sz = self.position_sign

        # Undo the trim, then the per-axis sign, to recover ROS RPY.
        yaw_ros = math.radians((pan - self.pan_offset) / self.pan_sign)
        pitch_ros = math.radians((tilt - self.tilt_offset) / self.tilt_sign)
        roll_ros = math.radians((roll - self.roll_offset) / self.roll_sign)

        # Unreal units -> metres, then undo the axis mirror. position_scale is an
        # output gain on an already-metric value, so it plays no part here: the
        # incoming number is in Unreal world units whatever that gain is set to.
        p = [float(v) / float(unreal_units_per_metre) for v in position_unreal]

        T = np.eye(4)
        T[:3, :3] = rpy_to_matrix(roll_ros, pitch_ros, yaw_ros)
        T[:3, 3] = [p[0] / sx, p[1] / sy, p[2] / sz]
        return T

    # ── the conversion ───────────────────────────────────────────────────────
    def ros_to_unreal(self, position, quaternion):
        """Isaac ROS pose -> UnrealPose (metres, degrees)."""
        T = make_transform(position, quaternion) @ self._rig
        if self._zero is not None:
            T = self._zero @ T

        return self.world_transform_to_unreal(T)

    def world_transform_to_unreal(self, T):
        """Convert an already aligned nodal-camera transform to FreeD units."""
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

    def pose_with_zero_ros(self, position, quaternion):
        """(position, quaternion) with ground zero applied, in ROS canonical
        metres -- no Unreal axis-flip or sign trim, unlike ros_to_unreal().

        For displaying the camera in a viewport that shares the marker/anchor
        frame (e.g. the dashboard's own 3D viewport, which draws the locator
        in this same frame) rather than for FreeD output.

        Unanchored (no ground zero set) returns the input completely
        unmodified, so callers relying on this only change behaviour once the
        operator actually anchors -- the unanchored case stays byte-identical
        to reading the raw pose directly.
        """
        if self._zero is None:
            return (np.asarray(position, dtype=float),
                    tuple(float(v) for v in quaternion))
        T = self._zero @ (make_transform(position, quaternion) @ self._rig)
        return T[:3, 3], matrix_to_quaternion(T[:3, :3])


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


class StationaryHold:
    """Freeze the output pose while the rig is physically still (a ZUPT).

    Measured on this rig with the camera untouched on a bench: the pure visual
    odometry estimate creeps 8.5 mm per 12 s, with 8.5 um of jitter per sample.
    Visual odometry has no absolute position reference, so sub-pixel feature
    match errors integrate into a slow slide. On a locked-off plate that slide is
    exactly the artefact you notice -- the virtual camera drifting while the real
    one sits on sticks.

    The IMU settles it. A quiet gyro plus an accelerometer reading a constant 1 g
    means the camera is not moving, whatever the vision thinks. So while that
    holds, pin the output and absorb the creep into an offset -- the same trick
    PoseGate uses for snaps, so releasing produces no jump.

    Asymmetric on purpose: entering the hold needs a full quiet window, leaving
    it takes a single sample above the release threshold. Holding a fraction of a
    second too long at the start of a move would be a far worse artefact than the
    drift being removed.
    """

    def __init__(self, enabled=True, gyro_still_rad_s=0.05,
                 gyro_release_rad_s=0.08, accel_still_std=0.15,
                 window_samples=60, vo_release_m=0.003,
                 vo_window_samples=12):
        self.enabled = bool(enabled)
        self.gyro_still_rad_s = float(gyro_still_rad_s)
        self.gyro_release_rad_s = float(gyro_release_rad_s)
        self.accel_still_std = float(accel_still_std)
        self.window_samples = int(window_samples)
        self.vo_release_m = float(vo_release_m)
        self.vo_window_samples = int(vo_window_samples)

        self._gyro = collections.deque(maxlen=self.window_samples)
        self._accel = collections.deque(maxlen=self.window_samples)
        self._vo_pos = collections.deque(maxlen=max(2, self.vo_window_samples))
        self._stationary = False
        self._vo_moving = False        # vision's latched verdict
        self.vo_releases = 0           # times vision overruled the IMU

        self._offset_pos = np.zeros(3)
        self._offset_rot = np.eye(3)
        self._freeze_pos = None
        self._freeze_rot = None

        self.holds = 0                 # times the hold engaged
        self.absorbed_m = 0.0          # total creep suppressed

    @property
    def is_stationary(self):
        return self._stationary

    # ── IMU side, called at 200 Hz ───────────────────────────────────────────
    def update_imu(self, gyro, accel):
        if not self.enabled:
            return
        gm = float(np.linalg.norm(np.asarray(gyro, dtype=float)))
        am = float(np.linalg.norm(np.asarray(accel, dtype=float)))
        self._gyro.append(gm)
        self._accel.append(am)

        # Release on a single lively sample — never lag the start of a move.
        if gm > self.gyro_release_rad_s:
            self._stationary = False
            return

        if len(self._gyro) < self.window_samples:
            return

        # Vision's latch also gates engagement, not just release. This runs at
        # 200 Hz against a ~90 Hz pose stream, so without the gate it re-arms the
        # hold between poses and the output chatters pinned/live every frame.
        accel_std = float(np.std(np.asarray(self._accel)))
        if (not self._vo_moving
                and max(self._gyro) <= self.gyro_still_rad_s
                and accel_std <= self.accel_still_std):
            if not self._stationary:
                self._stationary = True
                self._freeze_pos = None      # captured on the next apply()
                self._freeze_rot = None
                self.holds += 1

    def _update_vo_moving(self):
        """Latch vision's verdict on whether the rig is actually travelling.

        Hysteresis mirrors the gyro's still/release split: crossing vo_release_m
        says "moving", and it takes a drop to half that to say "stopped" again.
        A bare threshold would chatter for any move hovering near the limit.
        """
        travel = self._vo_travel()
        if self._vo_moving:
            if travel < self.vo_release_m * 0.5:
                self._vo_moving = False
        elif travel > self.vo_release_m:
            self._vo_moving = True

    def _vo_travel(self):
        """Straight-line distance the raw VO estimate has covered over the window.

        Measured end-to-end rather than as path length: drift is a random walk
        that largely cancels, while real motion is directional and accumulates.
        """
        if len(self._vo_pos) < 2:
            return 0.0
        return float(np.linalg.norm(self._vo_pos[-1] - self._vo_pos[0]))

    # ── pose side ────────────────────────────────────────────────────────────
    def apply(self, position, quaternion):
        """Return (position, quaternion, held)."""
        if not self.enabled:
            return position, quaternion, False

        raw_pos = np.asarray([float(v) for v in position])
        raw_rot = quaternion_to_matrix(quaternion)

        # Keep vision's verdict current even while released, so update_imu knows
        # not to re-arm mid-move.
        self._vo_pos.append(raw_pos.copy())
        self._update_vo_moving()

        out_pos = self._offset_pos + raw_pos
        out_rot = self._offset_rot @ raw_rot

        if not self._stationary:
            return out_pos, matrix_to_quaternion(out_rot), False

        # Vision gets a veto. An accelerometer cannot tell constant velocity from
        # rest, and a dolly or truck move rotates nothing, so the IMU alone reads
        # a straight tracking shot as "still" and the hold silently eats it.
        # Vision sees that motion perfectly well, so believe it and release.
        #
        # The margin is wide: the creep this hold exists to remove runs about
        # 0.7 mm/s, while even a slow dolly is two orders of magnitude faster.
        if self._vo_moving:
            self._stationary = False
            self.vo_releases += 1
            return out_pos, matrix_to_quaternion(out_rot), False

        if self._freeze_pos is None:
            # First pose of this hold defines where we pin.
            self._freeze_pos = out_pos.copy()
            self._freeze_rot = out_rot.copy()

        # Re-derive the offset every frame so the output stays pinned and the
        # offset is already correct the instant we release.
        self.absorbed_m += float(np.linalg.norm(
            (self._offset_pos + raw_pos) - self._freeze_pos))
        self._offset_pos = self._freeze_pos - raw_pos
        self._offset_rot = self._freeze_rot @ raw_rot.T

        return (self._freeze_pos.copy(),
                matrix_to_quaternion(self._freeze_rot), True)

    def reset(self):
        self._gyro.clear()
        self._accel.clear()
        self._vo_pos.clear()
        self._stationary = False
        self._vo_moving = False
        self._offset_pos = np.zeros(3)
        self._offset_rot = np.eye(3)
        self._freeze_pos = None
        self._freeze_rot = None


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
                 max_jump_m=0.15, max_consecutive_rejects=15, enabled=True,
                 max_gap_s=0.2):
        self.enabled = bool(enabled)
        self.max_speed_m_s = float(max_speed_m_s)
        self.max_angular_deg_s = float(max_angular_deg_s)
        self.max_jump_m = float(max_jump_m)
        self.max_consecutive_rejects = int(max_consecutive_rejects)
        self.max_gap_s = max(0.0, float(max_gap_s))
        self._rejection_step_limit = None

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

        # Allow physically plausible travel over a short delivery gap. Keep
        # the allowance fixed throughout a rejection streak: repeated bad
        # poses must not buy themselves a larger jump allowance by waiting.
        step_limit = self._rejection_step_limit
        if step_limit is None:
            step_limit = max(self.max_jump_m,
                             self.max_speed_m_s * min(dt, self.max_gap_s))

        # The bounded step test always applies. The rate tests only apply when
        # dt is large enough to be meaningful: `now` should be the message
        # timestamp, but if poses ever arrive batched (or a caller passes arrival
        # time) dt collapses towards zero and every real move looks like it
        # happened at infinite speed. Rejecting genuine camera motion is worse
        # than missing an outlier, and max_jump_m still bounds the damage.
        rate_tests_valid = dt >= _MIN_RATE_DT_S

        reason = None
        if step > step_limit:
            reason = (f"jump {step * 1000:.0f}mm > {step_limit * 1000:.0f}mm")
        elif rate_tests_valid:
            speed = step / dt
            ang_speed = self._angle_between(self._last_quat, quaternion) / dt
            if speed > self.max_speed_m_s:
                reason = f"speed {speed:.1f} > {self.max_speed_m_s:.1f} m/s"
            elif ang_speed > self.max_angular_deg_s:
                reason = f"spin {ang_speed:.0f} > {self.max_angular_deg_s:.0f} deg/s"

        if reason is None:
            self.consecutive_rejects = 0
            self._rejection_step_limit = None
            self._commit(position, quaternion, now)
            return True, None

        self.consecutive_rejects += 1
        self._rejection_step_limit = step_limit
        self.total_rejects += 1
        self.last_reason = reason

        if self.consecutive_rejects >= self.max_consecutive_rejects:
            # Persistent disagreement means the tracker really has moved on.
            self.resyncs += 1
            self.consecutive_rejects = 0
            self._rejection_step_limit = None
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
        self._rejection_step_limit = None
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
            out_pos = self._offset_pos + self._offset_rot @ (
                prev_pos if prev_pos is not None else raw_pos)
            out_rot = self._offset_rot @ quaternion_to_matrix(
                prev_quat if prev_quat is not None else raw_quat)
            return out_pos, matrix_to_quaternion(out_rot), True, reason

        if reason and prev_pos is not None and prev_quat is not None:
            # Resync: absorb the discontinuity instead of passing it on.
            target_pos = self._offset_pos + self._offset_rot @ prev_pos
            target_rot = self._offset_rot @ quaternion_to_matrix(prev_quat)
            self._offset_rot = target_rot @ quaternion_to_matrix(raw_quat).T
            self._offset_pos = target_pos - self._offset_rot @ raw_pos
            self.absorbed = getattr(self, "absorbed", 0) + 1

        out_pos = self._offset_pos + self._offset_rot @ raw_pos
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
