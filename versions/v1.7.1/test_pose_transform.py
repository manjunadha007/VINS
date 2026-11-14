#!/usr/bin/env python3
"""
Tests for pose_transform.py — the ROS -> Unreal Engine conversion.

Pure numpy, no ROS/Qt/OpenCV, so this runs on the host as well as in the Isaac
container:

    python3 -m unittest test_pose_transform -v

Where scipy is installed the rotation maths is checked against
scipy.spatial.transform as an independent oracle; those tests skip without it.
"""
import math
import unittest

import numpy as np

import pose_transform as pt
from pose_transform import (PosePredictor, PoseSmoother, PoseTransformer,
                            matrix_to_quaternion, matrix_to_rpy,
                            quaternion_to_matrix, quaternion_to_rpy,
                            rpy_to_matrix)

try:
    from scipy.spatial.transform import Rotation as _R
    HAVE_SCIPY = True
except ImportError:                                  # pragma: no cover
    HAVE_SCIPY = False


def rand_quats(n, seed=0):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n, 4))
    return q / np.linalg.norm(q, axis=1, keepdims=True)


# ── Rotation primitives vs scipy ─────────────────────────────────────────────

@unittest.skipUnless(HAVE_SCIPY, "scipy not installed")
class TestRotationMathsAgainstScipy(unittest.TestCase):
    def test_quaternion_to_matrix(self):
        for q in rand_quats(200, seed=1):
            np.testing.assert_allclose(
                quaternion_to_matrix(q), _R.from_quat(q).as_matrix(), atol=1e-12)

    def test_matrix_to_rpy_matches_extrinsic_xyz(self):
        """ROS RPY == extrinsic xyz == tf_transformations 'sxyz' == scipy 'xyz'."""
        for q in rand_quats(500, seed=2):
            R = _R.from_quat(q)
            expected = R.as_euler("xyz")
            got = matrix_to_rpy(R.as_matrix())
            # Compare as rotations to sidestep equivalent-angle representations.
            np.testing.assert_allclose(
                rpy_to_matrix(*got), rpy_to_matrix(*expected), atol=1e-9)

    def test_matrix_to_quaternion(self):
        for q in rand_quats(200, seed=3):
            R = quaternion_to_matrix(q)
            got = np.array(matrix_to_quaternion(R))
            # q and -q are the same rotation.
            if np.dot(got, q) < 0:
                got = -got
            np.testing.assert_allclose(got, q, atol=1e-8)

    def test_rpy_to_matrix(self):
        rng = np.random.default_rng(4)
        for _ in range(200):
            rpy = rng.uniform(-math.pi, math.pi, 3)
            np.testing.assert_allclose(
                rpy_to_matrix(*rpy), _R.from_euler("xyz", rpy).as_matrix(),
                atol=1e-12)


class TestRotationRoundTrips(unittest.TestCase):
    def test_rpy_matrix_rpy(self):
        for roll in (-2.0, -0.5, 0.0, 0.3, 1.2):
            for pitch in (-1.2, -0.4, 0.0, 0.7, 1.4):
                for yaw in (-3.0, -1.0, 0.0, 2.0):
                    R = rpy_to_matrix(roll, pitch, yaw)
                    r2, p2, y2 = matrix_to_rpy(R)
                    np.testing.assert_allclose(
                        rpy_to_matrix(r2, p2, y2), R, atol=1e-12)

    def test_identity(self):
        r, p, y = quaternion_to_rpy((0.0, 0.0, 0.0, 1.0))
        self.assertAlmostEqual(r, 0.0)
        self.assertAlmostEqual(p, 0.0)
        self.assertAlmostEqual(y, 0.0)

    def test_unnormalised_quaternion_is_handled(self):
        r, p, y = quaternion_to_rpy((0.0, 0.0, 0.0, 5.0))
        self.assertAlmostEqual(r, 0.0)
        self.assertAlmostEqual(p, 0.0)
        self.assertAlmostEqual(y, 0.0)

    def test_zero_quaternion_does_not_blow_up(self):
        np.testing.assert_allclose(quaternion_to_matrix((0, 0, 0, 0)), np.eye(3))

    def test_straight_down_is_continuous_not_nan(self):
        """Camera pointing at the floor sits on the Euler pole."""
        R = rpy_to_matrix(0.0, math.pi / 2, 0.7)
        roll, pitch, yaw = matrix_to_rpy(R)
        self.assertFalse(any(math.isnan(v) for v in (roll, pitch, yaw)))
        self.assertAlmostEqual(abs(pitch), math.pi / 2, places=6)
        np.testing.assert_allclose(rpy_to_matrix(roll, pitch, yaw), R, atol=1e-9)


# ── The conversion that matters ──────────────────────────────────────────────

class TestRosToUnreal(unittest.TestCase):
    def setUp(self):
        self.tf = PoseTransformer()

    @staticmethod
    def _q_from_rpy(roll_deg=0.0, pitch_deg=0.0, yaw_deg=0.0):
        R = rpy_to_matrix(math.radians(roll_deg), math.radians(pitch_deg),
                          math.radians(yaw_deg))
        return matrix_to_quaternion(R)

    def test_level_camera_at_origin_is_all_zero(self):
        """THE regression test. The old transform reported pan/tilt/roll =
        0/90/90 for a level camera; now a level camera reads as level."""
        p = self.tf.ros_to_unreal((0, 0, 0), (0, 0, 0, 1))
        self.assertAlmostEqual(p.pan, 0.0, places=9)
        self.assertAlmostEqual(p.tilt, 0.0, places=9)
        self.assertAlmostEqual(p.roll, 0.0, places=9)

    def test_pan_actually_pans(self):
        """The old transform pinned pan at 0.0 for every yaw input."""
        for yaw_deg in (5.0, 15.0, 45.0, 90.0, 135.0, -30.0, -90.0):
            p = self.tf.ros_to_unreal((0, 0, 0), self._q_from_rpy(yaw_deg=yaw_deg))
            self.assertAlmostEqual(p.pan, -yaw_deg, places=6,
                                   msg=f"pan wrong for ROS yaw {yaw_deg}")
            self.assertAlmostEqual(p.tilt, 0.0, places=6,
                                   msg="pure pan must not produce tilt")
            self.assertAlmostEqual(p.roll, 0.0, places=6,
                                   msg="pure pan must not leak into roll")

    def test_tilt_only_affects_tilt(self):
        for pitch_deg in (10.0, 30.0, -25.0, 60.0):
            p = self.tf.ros_to_unreal((0, 0, 0), self._q_from_rpy(pitch_deg=pitch_deg))
            self.assertAlmostEqual(p.tilt, -pitch_deg, places=6)
            self.assertAlmostEqual(p.pan, 0.0, places=6)
            self.assertAlmostEqual(p.roll, 0.0, places=6)

    def test_roll_only_affects_roll(self):
        for roll_deg in (10.0, -45.0, 90.0):
            p = self.tf.ros_to_unreal((0, 0, 0), self._q_from_rpy(roll_deg=roll_deg))
            self.assertAlmostEqual(p.roll, roll_deg, places=6)
            self.assertAlmostEqual(p.pan, 0.0, places=6)
            self.assertAlmostEqual(p.tilt, 0.0, places=6)

    def test_position_mirrors_y_only(self):
        p = self.tf.ros_to_unreal((1.5, 2.0, -3.25), (0, 0, 0, 1))
        self.assertAlmostEqual(p.x, 1.5)
        self.assertAlmostEqual(p.y, -2.0, msg="Y must mirror for left-handed UE")
        self.assertAlmostEqual(p.z, -3.25)

    def test_pan_direction_is_physically_correct(self):
        """ROS +yaw pans LEFT; Unreal pans left on NEGATIVE yaw."""
        p = self.tf.ros_to_unreal((0, 0, 0), self._q_from_rpy(yaw_deg=30.0))
        self.assertLess(p.pan, 0.0)

    def test_tilt_direction_is_physically_correct(self):
        """ROS +pitch is nose-DOWN; Unreal nose-down is NEGATIVE pitch."""
        p = self.tf.ros_to_unreal((0, 0, 0), self._q_from_rpy(pitch_deg=20.0))
        self.assertLess(p.tilt, 0.0)

    def test_combined_motion_is_reversible(self):
        """A full 6DOF pose survives the trip with all axes distinguishable."""
        q = self._q_from_rpy(roll_deg=8.0, pitch_deg=-12.0, yaw_deg=57.0)
        p = self.tf.ros_to_unreal((1.0, -2.0, 0.5), q)
        self.assertAlmostEqual(p.roll, 8.0, places=5)
        self.assertAlmostEqual(p.tilt, 12.0, places=5)
        self.assertAlmostEqual(p.pan, -57.0, places=5)
        self.assertAlmostEqual(p.x, 1.0, places=9)
        self.assertAlmostEqual(p.y, 2.0, places=9)
        self.assertAlmostEqual(p.z, 0.5, places=9)

    def test_monotonic_pan_sweep_has_no_discontinuity(self):
        """Sweeping the camera must not make the virtual camera jump."""
        prev = None
        for yaw_deg in np.arange(-170.0, 170.0, 2.0):
            p = self.tf.ros_to_unreal((0, 0, 0), self._q_from_rpy(yaw_deg=float(yaw_deg)))
            if prev is not None:
                self.assertLess(abs(p.pan - prev), 5.0,
                                f"pan jumped at yaw {yaw_deg}")
            prev = p.pan


class TestOldTransformWasBroken(unittest.TestCase):
    """Pins the defect this module replaces, so the fix cannot silently regress."""

    @staticmethod
    def legacy_euler_from_matrix(M):
        """Faithful replica of transformations.euler_from_matrix(M, 'sxyz').

        Note the gimbal-lock branch: it zeroes the THIRD angle (yaw). That is
        what pinned FreeD pan at 0.0. pose_transform.matrix_to_rpy deliberately
        zeroes roll instead and keeps yaw live, so a camera tipped straight up or
        down still pans.
        """
        cy = math.sqrt(M[0, 0] * M[0, 0] + M[1, 0] * M[1, 0])
        if cy > 1e-8:
            ax = math.atan2(M[2, 1], M[2, 2])
            ay = math.atan2(-M[2, 0], cy)
            az = math.atan2(M[1, 0], M[0, 0])
        else:
            ax = math.atan2(-M[1, 2], M[1, 1])
            ay = math.atan2(-M[2, 0], cy)
            az = 0.0
        return ax, ay, az

    @classmethod
    def legacy_vins_to_ue_pose(cls, tx, ty, tz, q):
        """Verbatim maths from dashboard._vins_to_ue_pose before the fix."""
        R_vins = quaternion_to_matrix(q)
        P_w = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
        P_b = np.array([[0, 1, 0], [0, 0, -1], [1, 0, 0]])
        R_ue = P_w.dot(R_vins).dot(P_b)
        rh_roll, rh_pitch, rh_yaw = cls.legacy_euler_from_matrix(R_ue)
        return (tx, -ty, tz, -math.degrees(rh_roll),
                -math.degrees(rh_pitch), math.degrees(rh_yaw))

    def test_legacy_was_permanently_gimbal_locked_for_level_pans(self):
        """P_w . P_b puts a level camera exactly on the pole."""
        R_ue = (np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
                .dot(np.array([[0, 1, 0], [0, 0, -1], [1, 0, 0]])))
        cy = math.sqrt(R_ue[0, 0] ** 2 + R_ue[1, 0] ** 2)
        self.assertAlmostEqual(cy, 0.0, places=12,
                               msg="expected the singular configuration")

    def test_legacy_leaked_pan_into_roll(self):
        for yaw_deg in (15.0, 45.0):
            q = matrix_to_quaternion(rpy_to_matrix(0.0, 0.0, math.radians(yaw_deg)))
            _, _, _, legacy_roll, _, _ = self.legacy_vins_to_ue_pose(0, 0, 0, q)
            self.assertAlmostEqual(legacy_roll, 90.0 + yaw_deg, places=5,
                                   msg="pan should have appeared on roll")

    def test_new_transform_keeps_pan_alive_at_the_pole(self):
        """Design choice: pointing straight down must not kill pan."""
        tf = PoseTransformer()
        q = matrix_to_quaternion(
            rpy_to_matrix(0.0, math.radians(90.0), math.radians(40.0)))
        p = tf.ros_to_unreal((0, 0, 0), q)
        self.assertAlmostEqual(abs(p.tilt), 90.0, places=5)
        self.assertGreater(abs(p.pan), 1.0,
                           "pan collapsed to zero at the tilt pole")

    def test_legacy_pinned_pan_at_zero(self):
        for yaw_deg in (15.0, 45.0, 90.0):
            R = rpy_to_matrix(0.0, 0.0, math.radians(yaw_deg))
            q = matrix_to_quaternion(R)
            *_, legacy_yaw = self.legacy_vins_to_ue_pose(0, 0, 0, q)
            self.assertAlmostEqual(legacy_yaw, 0.0, places=6,
                                   msg="legacy pan was not actually stuck?")

    def test_new_transform_does_not_pin_pan(self):
        tf = PoseTransformer()
        R = rpy_to_matrix(0.0, 0.0, math.radians(45.0))
        p = tf.ros_to_unreal((0, 0, 0), matrix_to_quaternion(R))
        self.assertAlmostEqual(abs(p.pan), 45.0, places=5)


# ── Ground zero / rig offset ─────────────────────────────────────────────────

class TestGroundZero(unittest.TestCase):
    def test_zeroing_puts_that_pose_at_the_origin(self):
        tf = PoseTransformer()
        pos, quat = (3.0, -1.0, 2.0), (0.0, 0.0, 0.3827, 0.9239)   # yaw 45
        tf.set_ground_zero(pos, quat)
        p = tf.ros_to_unreal(pos, quat)
        for v in (p.x, p.y, p.z, p.pan, p.tilt, p.roll):
            self.assertAlmostEqual(v, 0.0, places=4)

    def test_motion_after_zeroing_is_relative(self):
        tf = PoseTransformer()
        tf.set_ground_zero((10.0, 0.0, 0.0), (0, 0, 0, 1))
        p = tf.ros_to_unreal((11.0, 0.0, 0.0), (0, 0, 0, 1))
        self.assertAlmostEqual(p.x, 1.0, places=6)

    def test_zeroing_rotates_the_world_into_the_new_heading(self):
        """After zeroing at 90 deg yaw, driving forward is +X in Unreal."""
        tf = PoseTransformer()
        q90 = matrix_to_quaternion(rpy_to_matrix(0, 0, math.radians(90)))
        tf.set_ground_zero((0.0, 0.0, 0.0), q90)
        # In ROS, heading 90 deg means "forward" is +Y.
        p = tf.ros_to_unreal((0.0, 1.0, 0.0), q90)
        self.assertAlmostEqual(p.x, 1.0, places=5)
        self.assertAlmostEqual(p.y, 0.0, places=5)
        self.assertAlmostEqual(p.pan, 0.0, places=5)

    def test_clear_restores_absolute_poses(self):
        tf = PoseTransformer()
        tf.set_ground_zero((5.0, 0.0, 0.0), (0, 0, 0, 1))
        tf.clear_ground_zero()
        self.assertFalse(tf.is_zeroed)
        self.assertAlmostEqual(tf.ros_to_unreal((5.0, 0, 0), (0, 0, 0, 1)).x, 5.0)


class TestPoseWithZeroRos(unittest.TestCase):
    """pose_with_zero_ros(): ground-zero applied, but in ROS canonical metres
    (no Unreal axis-flip/sign-trim) -- for a viewport sharing the marker/
    locator frame, e.g. the dashboard's own 3D viewer."""

    def test_unanchored_returns_the_input_unmodified(self):
        tf = PoseTransformer()
        pos, quat = (1.5, -2.5, 0.75), (0.0, 0.0, 0.3827, 0.9239)
        out_pos, out_quat = tf.pose_with_zero_ros(pos, quat)
        np.testing.assert_allclose(out_pos, pos)
        self.assertEqual(tuple(out_quat), quat)

    def test_unanchored_ignores_a_configured_rig_offset(self):
        """Byte-identical to the raw pose pre-anchor, even with a rig offset
        configured -- callers must see zero behaviour change until they
        actually anchor."""
        tf = PoseTransformer({"rig_offset": {"translation_m": (0.1, 0.0, 0.0)}})
        pos, quat = (1.0, 2.0, 3.0), (0, 0, 0, 1)
        out_pos, out_quat = tf.pose_with_zero_ros(pos, quat)
        np.testing.assert_allclose(out_pos, pos)

    def test_anchored_matches_ros_to_unreal_up_to_the_frame_conversion(self):
        tf = PoseTransformer()
        T_marker = pt.make_transform((3.0, -1.0, 0.5), (0.0, 0.0, 0.0, 1.0))
        T_target = pt.make_transform((5.0, -2.0, 1.0), (0.0, 0.0, 0.0, 1.0))
        tf.set_ground_zero_from_transform(T_marker, T_target=T_target)

        cam_pos, cam_quat = (3.0, -1.0, 0.5), (0, 0, 0, 1)
        ros_pos, ros_quat = tf.pose_with_zero_ros(cam_pos, cam_quat)
        # The camera sits on the marker, which maps onto the target: it
        # should land exactly on the target's position, un-mirrored (ROS Y,
        # not Unreal Y).
        np.testing.assert_allclose(ros_pos, [5.0, -2.0, 1.0], atol=1e-6)

        # And it must agree with ros_to_unreal() once you undo that
        # function's own Y-mirror -- same anchor maths, different frame.
        unreal = tf.ros_to_unreal(cam_pos, cam_quat)
        self.assertAlmostEqual(unreal.x, ros_pos[0], places=6)
        self.assertAlmostEqual(unreal.y, -ros_pos[1], places=6)
        self.assertAlmostEqual(unreal.z, ros_pos[2], places=6)


class TestAnchorToTarget(unittest.TestCase):
    """The LiDAR-scan workflow: the marker must land on a *chosen* pose.

    Anchoring used to pin the marker to the origin unconditionally, so the
    virtual camera came out offset by exactly the Unreal anchor Actor's position.
    That looks correct whenever the Actor happens to sit near the origin, which
    is what makes it dangerous.
    """

    def setUp(self):
        self.tf = PoseTransformer()
        # Marker physically at (3, -1, 0.5) m in the VIO frame, level.
        self.T_marker = pt.make_transform((3.0, -1.0, 0.5), (0.0, 0.0, 0.0, 1.0))

    def test_no_target_still_puts_the_marker_at_the_origin(self):
        self.tf.set_ground_zero_from_transform(self.T_marker)
        p = self.tf.ros_to_unreal((3.0, -1.0, 0.5), (0, 0, 0, 1))
        for v in (p.x, p.y, p.z, p.pan, p.tilt, p.roll):
            self.assertAlmostEqual(v, 0.0, places=6)
        self.assertIsNone(self.tf.anchor_target)

    def test_camera_at_the_marker_reports_the_anchor_pose(self):
        T_target = pt.make_transform((5.0, -2.0, 1.0), (0.0, 0.0, 0.0, 1.0))
        self.tf.set_ground_zero_from_transform(self.T_marker, T_target=T_target)
        p = self.tf.ros_to_unreal((3.0, -1.0, 0.5), (0, 0, 0, 1))
        self.assertAlmostEqual(p.x, 5.0, places=6)
        self.assertAlmostEqual(p.y, 2.0, places=6)   # Y mirrors into Unreal
        self.assertAlmostEqual(p.z, 1.0, places=6)
        self.assertIsNotNone(self.tf.anchor_target)

    def test_movement_away_from_the_marker_is_relative_to_the_anchor(self):
        T_target = pt.make_transform((5.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
        self.tf.set_ground_zero_from_transform(self.T_marker, T_target=T_target)
        # Step 1 m along +X from the marker.
        p = self.tf.ros_to_unreal((4.0, -1.0, 0.5), (0, 0, 0, 1))
        self.assertAlmostEqual(p.x, 6.0, places=6)

    def test_target_heading_rotates_the_world(self):
        """Anchor Actor yawed 90 deg: driving marker-forward must follow it."""
        q90 = matrix_to_quaternion(rpy_to_matrix(0, 0, math.radians(90)))
        T_target = pt.make_transform((0.0, 0.0, 0.0), q90)
        self.tf.set_ground_zero_from_transform(self.T_marker, T_target=T_target)
        p = self.tf.ros_to_unreal((4.0, -1.0, 0.5), (0, 0, 0, 1))
        # +1 m of marker-forward becomes +1 m along the target's forward (ROS +Y),
        # which mirrors to -1 m in Unreal Y.
        self.assertAlmostEqual(p.x, 0.0, places=5)
        self.assertAlmostEqual(p.y, -1.0, places=5)

    def test_clear_removes_the_target_too(self):
        self.tf.set_ground_zero_from_transform(
            self.T_marker, T_target=pt.make_transform((9.0, 0, 0), (0, 0, 0, 1)))
        self.tf.clear_ground_zero()
        self.assertFalse(self.tf.is_zeroed)
        self.assertIsNone(self.tf.anchor_target)


class TestUnrealToRosTransform(unittest.TestCase):
    """Anchors arrive as Unreal Actor transforms: centimetres, left-handed."""

    def test_centimetres_become_metres_and_y_mirrors(self):
        tf = PoseTransformer()
        T = tf.unreal_to_ros_transform((500.0, 200.0, 100.0), (0.0, 0.0, 0.0))
        np.testing.assert_allclose(T[:3, 3], [5.0, -2.0, 1.0], atol=1e-9)

    def test_exactly_inverts_ros_to_unreal(self):
        """Round-trip through the pair must be the identity, signs and all."""
        tf = PoseTransformer()
        for xyz, rpy in (((1.0, -2.0, 0.5), (0.0, 0.0, 0.0)),
                         ((0.0, 0.0, 0.0), (30.0, -10.0, 5.0)),
                         ((-3.25, 4.0, -1.5), (-120.0, 25.0, -8.0))):
            fwd = tf.ros_to_unreal(xyz, matrix_to_quaternion(
                rpy_to_matrix(math.radians(rpy[2]), math.radians(rpy[1]),
                              math.radians(rpy[0]))))
            back = tf.unreal_to_ros_transform(
                (fwd.x * pt.UNREAL_UNITS_PER_METRE,
                 fwd.y * pt.UNREAL_UNITS_PER_METRE,
                 fwd.z * pt.UNREAL_UNITS_PER_METRE),
                (fwd.pan, fwd.tilt, fwd.roll))
            np.testing.assert_allclose(back[:3, 3], xyz, atol=1e-9)

    def test_honours_flipped_axis_signs(self):
        """A sign flipped in tracking.yaml must move the anchor with it.

        Otherwise the anchor and the live pose end up in different frames and the
        error is a constant offset that is very hard to spot on set.
        """
        tf = PoseTransformer({"position_sign": (1.0, 1.0, 1.0), "pan_sign": 1.0})
        T = tf.unreal_to_ros_transform((100.0, 200.0, 0.0), (45.0, 0.0, 0.0))
        np.testing.assert_allclose(T[:3, 3], [1.0, 2.0, 0.0], atol=1e-9)
        _r, _p, yaw = matrix_to_rpy(T[:3, :3])
        self.assertAlmostEqual(math.degrees(yaw), 45.0, places=6)

    def test_honours_angle_trim(self):
        tf = PoseTransformer({"pan_offset_deg": 90.0})
        T = tf.unreal_to_ros_transform((0.0, 0.0, 0.0), (90.0, 0.0, 0.0))
        _r, _p, yaw = matrix_to_rpy(T[:3, :3])
        self.assertAlmostEqual(math.degrees(yaw), 0.0, places=6)

    def test_custom_world_units(self):
        tf = PoseTransformer()
        T = tf.unreal_to_ros_transform((5.0, 0.0, 0.0), (0.0, 0.0, 0.0),
                                       unreal_units_per_metre=1.0)
        self.assertAlmostEqual(T[0, 3], 5.0, places=9)

    def test_end_to_end_anchor_from_unreal(self):
        """The whole workflow in one assertion.

        Anchor Actor at (250, -400, 120) cm, pan 30. Marker seen at some
        arbitrary VIO pose. Camera sitting on the marker must report the Actor's
        transform back, in Unreal units.
        """
        tf = PoseTransformer()
        T_target = tf.unreal_to_ros_transform((250.0, -400.0, 120.0),
                                              (30.0, 0.0, 0.0))
        T_marker = pt.make_transform((7.5, 2.25, -0.75), matrix_to_quaternion(
            rpy_to_matrix(0, 0, math.radians(-63.0))))
        tf.set_ground_zero_from_transform(T_marker, T_target=T_target)

        p = tf.ros_to_unreal((7.5, 2.25, -0.75), matrix_to_quaternion(
            rpy_to_matrix(0, 0, math.radians(-63.0))))
        self.assertAlmostEqual(p.x * 100.0, 250.0, places=4)
        self.assertAlmostEqual(p.y * 100.0, -400.0, places=4)
        self.assertAlmostEqual(p.z * 100.0, 120.0, places=4)
        self.assertAlmostEqual(p.pan, 30.0, places=4)


class TestRigOffset(unittest.TestCase):
    def test_no_offset_by_default(self):
        self.assertFalse(PoseTransformer().has_rig_offset)

    def test_forward_offset_moves_with_heading(self):
        """A nodal point 20 cm ahead of the tracker must orbit as it pans."""
        tf = PoseTransformer({"rig_offset": {"translation_m": (0.2, 0.0, 0.0)}})
        self.assertTrue(tf.has_rig_offset)

        p = tf.ros_to_unreal((0, 0, 0), (0, 0, 0, 1))
        self.assertAlmostEqual(p.x, 0.2, places=9)
        self.assertAlmostEqual(p.y, 0.0, places=9)

        q90 = matrix_to_quaternion(rpy_to_matrix(0, 0, math.radians(90)))
        p = tf.ros_to_unreal((0, 0, 0), q90)
        self.assertAlmostEqual(p.x, 0.0, places=6)
        # ROS +Y is left; Unreal mirrors it to -Y.
        self.assertAlmostEqual(p.y, -0.2, places=6)

    def test_vertical_offset(self):
        tf = PoseTransformer({"rig_offset": {"translation_m": (0.0, 0.0, -0.1)}})
        self.assertAlmostEqual(tf.ros_to_unreal((0, 0, 1.0), (0, 0, 0, 1)).z, 0.9,
                               places=9)


class TestTrimAndScale(unittest.TestCase):
    def test_position_scale(self):
        tf = PoseTransformer({"position_scale": 100.0})     # metres -> cm
        p = tf.ros_to_unreal((1.0, 0.0, 0.0), (0, 0, 0, 1))
        self.assertAlmostEqual(p.x, 100.0)

    def test_sign_overrides(self):
        tf = PoseTransformer({"pan_sign": 1.0, "position_sign": (1.0, 1.0, 1.0)})
        q = matrix_to_quaternion(rpy_to_matrix(0, 0, math.radians(30)))
        p = tf.ros_to_unreal((0, 1.0, 0), q)
        self.assertAlmostEqual(p.pan, 30.0, places=5)
        self.assertAlmostEqual(p.y, 1.0, places=9)

    def test_angle_trim(self):
        tf = PoseTransformer({"pan_offset_deg": 90.0, "tilt_offset_deg": -5.0})
        p = tf.ros_to_unreal((0, 0, 0), (0, 0, 0, 1))
        self.assertAlmostEqual(p.pan, 90.0)
        self.assertAlmostEqual(p.tilt, -5.0)


# ── Prediction & smoothing ───────────────────────────────────────────────────

class TestPosePredictor(unittest.TestCase):
    def test_no_pose_yet(self):
        p = PosePredictor()
        self.assertFalse(p.has_pose)
        self.assertIsNone(p.predict(1.0))
        self.assertEqual(p.age(1.0), float("inf"))

    def test_constant_velocity_extrapolation(self):
        p = PosePredictor(max_extrapolation_s=1.0)
        p.update((0, 0, 0), (0, 0, 0, 1), (2.0, 0, 0), (0, 0, 0), now=10.0)
        pos, _q = p.predict(10.5)
        self.assertAlmostEqual(pos[0], 1.0, places=9)

    def test_extrapolation_is_capped(self):
        """Tracking loss must freeze the camera, not launch it."""
        p = PosePredictor(max_extrapolation_s=0.05)
        p.update((0, 0, 0), (0, 0, 0, 1), (10.0, 0, 0), (0, 0, 0), now=0.0)
        pos, _q = p.predict(100.0)
        self.assertAlmostEqual(pos[0], 0.5, places=9)

    def test_lead_time_adds_prediction(self):
        p = PosePredictor(lead_time_s=0.1, max_extrapolation_s=1.0)
        p.update((0, 0, 0), (0, 0, 0, 1), (1.0, 0, 0), (0, 0, 0), now=0.0)
        pos, _q = p.predict(0.0)
        self.assertAlmostEqual(pos[0], 0.1, places=9)

    def test_angular_extrapolation(self):
        p = PosePredictor(max_extrapolation_s=1.0)
        rate = math.radians(60.0)      # 60 deg/s yaw
        p.update((0, 0, 0), (0, 0, 0, 1), (0, 0, 0), (0, 0, rate), now=0.0)
        _pos, q = p.predict(0.5)
        _r, _pi, yaw = quaternion_to_rpy(q)
        self.assertAlmostEqual(math.degrees(yaw), 30.0, places=4)

    def test_zero_velocity_is_exact(self):
        p = PosePredictor(max_extrapolation_s=1.0)
        p.update((1.0, 2.0, 3.0), (0, 0, 0, 1), (0, 0, 0), (0, 0, 0), now=0.0)
        pos, q = p.predict(0.5)
        np.testing.assert_allclose(pos, [1.0, 2.0, 3.0], atol=1e-12)
        np.testing.assert_allclose(quaternion_to_matrix(q), np.eye(3), atol=1e-12)

    def test_age_tracks_staleness(self):
        p = PosePredictor()
        p.update((0, 0, 0), (0, 0, 0, 1), (0, 0, 0), (0, 0, 0), now=5.0)
        self.assertAlmostEqual(p.age(5.25), 0.25)


class TestPoseGate(unittest.TestCase):
    """The gate is what stops a cuVSLAM relocalisation teleporting Unreal."""

    def setUp(self):
        self.gate = pt.PoseGate(max_speed_m_s=5.0, max_angular_deg_s=900.0,
                                max_jump_m=0.15)
        self.q = (0.0, 0.0, 0.0, 1.0)

    def test_first_pose_is_always_accepted(self):
        ok, reason = self.gate.check((0, 0, 0), self.q, 0.0)
        self.assertTrue(ok)
        self.assertIsNone(reason)

    def test_normal_motion_passes(self):
        self.gate.check((0, 0, 0), self.q, 0.0)
        # 1 cm in 11 ms = 0.9 m/s, an ordinary handheld move at 90 Hz.
        ok, reason = self.gate.check((0.01, 0, 0), self.q, 0.011)
        self.assertTrue(ok, reason)

    def test_real_motion_survives_dropped_frames_without_absorbing_travel(self):
        gate = pt.PoseGate(max_jump_m=.05, max_speed_m_s=3.)
        for t in (0., .033, .066, .199, .232, .410):
            position, _, rejected, reason = gate.filter((.5*t, 0, 0), self.q, t)
            self.assertFalse(rejected, reason)
            np.testing.assert_allclose(position, (.5*t, 0, 0))
        self.assertEqual(gate.resyncs, 0)

    def test_rejected_jump_cannot_expand_its_allowance_by_waiting(self):
        gate = pt.PoseGate(max_jump_m=.05, max_speed_m_s=3., max_consecutive_rejects=100)
        gate.check((0, 0, 0), self.q, 0.)
        for i in range(1, 30):
            accepted, reason = gate.check((.4, 0, 0), self.q, i*.01)
            self.assertFalse(accepted, reason)

    def test_long_outage_still_rejects_large_unverified_displacement(self):
        gate = pt.PoseGate(max_jump_m=.05, max_speed_m_s=3.)
        gate.check((0, 0, 0), self.q, 0.)
        self.assertFalse(gate.check((2., 0, 0), self.q, 10.)[0])

    def test_rotated_tracking_reset_keeps_subsequent_translation_in_the_same_world(self):
        gate = pt.PoseGate(max_jump_m=.05, max_speed_m_s=3., max_consecutive_rejects=2)
        gate.filter((0, 0, 0), self.q, 0.)
        rotated = matrix_to_quaternion(rpy_to_matrix(0, 0, math.pi/2))
        gate.filter((10, 0, 0), rotated, .02)
        p, q, _, _ = gate.filter((10, 0, 0), rotated, .04)
        np.testing.assert_allclose(p, (0, 0, 0), atol=1e-9)
        np.testing.assert_allclose(quaternion_to_matrix(q), np.eye(3), atol=1e-9)
        # Raw +Y is now the original world's +X, following the 90-degree reset.
        p, _, rejected, _ = gate.filter((10, .02, 0), rotated, .06)
        self.assertFalse(rejected)
        np.testing.assert_allclose(p, (.02, 0, 0), atol=1e-9)

    def test_metre_scale_jump_is_rejected(self):
        """The exact failure measured on hardware: 1254 mm in one frame."""
        self.gate.check((0, 0, 0), self.q, 0.0)
        ok, reason = self.gate.check((1.2546, 0, 0), self.q, 0.011)
        self.assertFalse(ok)
        self.assertIn("jump", reason)

    def test_excessive_speed_is_rejected(self):
        gate = pt.PoseGate(max_speed_m_s=2.0, max_jump_m=10.0)
        gate.check((0, 0, 0), self.q, 0.0)
        ok, reason = gate.check((0.1, 0, 0), self.q, 0.01)   # 10 m/s
        self.assertFalse(ok)
        self.assertIn("speed", reason)

    def test_excessive_spin_is_rejected(self):
        gate = pt.PoseGate(max_angular_deg_s=100.0, max_jump_m=10.0,
                           max_speed_m_s=100.0)
        gate.check((0, 0, 0), self.q, 0.0)
        q90 = matrix_to_quaternion(rpy_to_matrix(0, 0, math.radians(90)))
        ok, reason = gate.check((0, 0, 0), q90, 0.05)     # 1800 deg/s
        self.assertFalse(ok)
        self.assertIn("spin", reason)

    def test_rejected_pose_does_not_become_the_reference(self):
        """Otherwise one bad pose would drag the gate along with it."""
        self.gate.check((0, 0, 0), self.q, 0.0)
        self.gate.check((5.0, 0, 0), self.q, 0.011)        # rejected
        ok, _ = self.gate.check((0.01, 0, 0), self.q, 0.022)
        self.assertTrue(ok, "gate should still be anchored at the good pose")

    def test_persistent_disagreement_resyncs(self):
        """A real relocalisation must eventually be accepted, not frozen out."""
        gate = pt.PoseGate(max_jump_m=0.15, max_consecutive_rejects=5)
        gate.check((0, 0, 0), self.q, 0.0)
        accepts = []
        for i in range(1, 8):
            accepts.append(gate.check((3.0, 0, 0), self.q, i * 0.011)[0])
        self.assertFalse(accepts[0])
        self.assertIn(True, accepts, "gate never resynced — output frozen forever")
        self.assertEqual(gate.resyncs, 1)

    def test_counters(self):
        self.gate.check((0, 0, 0), self.q, 0.0)
        self.gate.check((9.0, 0, 0), self.q, 0.011)
        self.assertEqual(self.gate.total_rejects, 1)
        self.assertEqual(self.gate.consecutive_rejects, 1)

    def test_disabled_gate_passes_everything(self):
        gate = pt.PoseGate(enabled=False, max_jump_m=0.001)
        gate.check((0, 0, 0), self.q, 0.0)
        ok, _ = gate.check((500.0, 0, 0), self.q, 0.011)
        self.assertTrue(ok)

    def test_zero_or_backwards_dt_does_not_divide_by_zero(self):
        self.gate.check((0, 0, 0), self.q, 1.0)
        ok, _ = self.gate.check((0.001, 0, 0), self.q, 1.0)
        self.assertTrue(ok)
        ok, _ = self.gate.check((0.002, 0, 0), self.q, 0.5)
        self.assertTrue(ok)

    def test_reset_clears_the_reference(self):
        self.gate.check((0, 0, 0), self.q, 0.0)
        self.gate.reset()
        ok, _ = self.gate.check((99.0, 0, 0), self.q, 0.011)
        self.assertTrue(ok, "after reset the next pose re-anchors")

    def test_replays_recorded_divergence(self):
        """Feed the gate the pathological pattern seen on hardware."""
        gate = pt.PoseGate(max_jump_m=0.15, max_speed_m_s=5.0)
        t, x = 0.0, 0.0
        rejected = 0
        for i in range(200):
            t += 1 / 90.0
            x += 0.002 if i % 20 else 1.25      # a 1.25 m snap every 20 frames
            ok, _ = gate.check((x, 0, 0), self.q, t)
            if not ok:
                rejected += 1
        self.assertGreater(rejected, 5, "gate let the snaps through")


class TestStationaryHold(unittest.TestCase):
    """Kills stationary drift: measured 27.89 mm -> 0.70 mm over 20 s on hardware."""

    QUIET_GYRO = (0.001, 0.001, 0.001)
    GRAVITY = (0.0, 0.0, 9.807)
    Q = (0.0, 0.0, 0.0, 1.0)

    def _settled(self, window=10, **kw):
        h = pt.StationaryHold(window_samples=window, **kw)
        for _ in range(window + 2):
            h.update_imu(self.QUIET_GYRO, self.GRAVITY)
        return h

    def test_needs_a_full_quiet_window_to_engage(self):
        h = pt.StationaryHold(window_samples=10)
        for _ in range(9):
            h.update_imu(self.QUIET_GYRO, self.GRAVITY)
        self.assertFalse(h.is_stationary, "engaged before the window filled")
        h.update_imu(self.QUIET_GYRO, self.GRAVITY)
        self.assertTrue(h.is_stationary)

    def test_pins_the_output_while_still(self):
        h = self._settled()
        first, _q, held = h.apply((1.0, 0.0, 0.0), self.Q)
        self.assertTrue(held)
        # SLAM creeps 50 um -- a few samples' worth at the measured 0.7 mm/s.
        # (Was 5 mm here, which at 90 Hz is 450 mm/s: that is a dolly move, not
        # drift, and the VO cross-check now correctly refuses to pin it.)
        drifted, _q, held = h.apply((1.00005, 0.0, 0.0), self.Q)
        self.assertTrue(held)
        np.testing.assert_allclose(drifted, first, atol=1e-12)

    def test_absorbs_creep_so_release_does_not_jump(self):
        h = self._settled()
        pinned, _q, _ = h.apply((1.0, 0.0, 0.0), self.Q)
        for n in range(1, 11):                      # creep 100 um while held
            h.apply((1.0 + n * 0.00001, 0.0, 0.0), self.Q)
        h.update_imu((5.0, 0.0, 0.0), self.GRAVITY)  # motion starts
        self.assertFalse(h.is_stationary)
        released, _q, held = h.apply((1.0001, 0.0, 0.0), self.Q)
        self.assertFalse(held)
        self.assertLess(float(np.linalg.norm(released - pinned)), 1e-9,
                        "released with a jump — the creep was not absorbed")

    def test_releases_on_a_single_lively_sample(self):
        """Lagging the start of a move is worse than the drift being removed."""
        h = self._settled()
        self.assertTrue(h.is_stationary)
        h.update_imu((0.5, 0.0, 0.0), self.GRAVITY)
        self.assertFalse(h.is_stationary)

    def test_real_motion_passes_through_untouched(self):
        h = pt.StationaryHold(window_samples=10)
        for i in range(30):
            h.update_imu((1.0, 0.0, 0.0), self.GRAVITY)   # clearly moving
        outs = []
        for i in range(10):
            p, _q, held = h.apply((i * 0.01, 0.0, 0.0), self.Q)
            self.assertFalse(held)
            outs.append(p)
        np.testing.assert_allclose(outs[-1], [0.09, 0.0, 0.0], atol=1e-12)

    def test_pure_translation_is_not_swallowed(self):
        """A dolly/truck move rotates nothing and, at constant velocity, produces
        no acceleration either — so an IMU-only stillness test cannot see it.

        The gyro stays quiet the whole way, which means the release condition
        never fires and the hold pins the output. Vision can see the move
        perfectly well, so the hold must not override it."""
        h = self._settled()
        out = None
        for i in range(1, 21):                      # 20 cm of steady travel
            h.update_imu(self.QUIET_GYRO, self.GRAVITY)
            out, _q, _held = h.apply((i * 0.01, 0.0, 0.0), self.Q)
        self.assertAlmostEqual(float(out[0]), 0.20, places=3,
                               msg="pure translation was pinned by the ZUPT")

    def test_sustained_translation_does_not_chatter(self):
        """Vision releasing the hold must not be re-armed by a still-quiet IMU.

        update_imu runs at 200 Hz against a ~90 Hz pose stream, so clearing the
        flag inside apply() is not enough: the IMU re-engages the hold before the
        next pose arrives, the output alternates pinned/live every frame, and the
        virtual camera flickers."""
        h = self._settled()
        outs = []
        for i in range(1, 41):                  # 20 cm at 5 mm per pose
            for _ in range(2):                  # 200 Hz IMU vs 90 Hz pose
                h.update_imu(self.QUIET_GYRO, self.GRAVITY)
            out, _q, _held = h.apply((i * 0.005, 0.0, 0.0), self.Q)
            outs.append(float(out[0]))

        steps = np.diff(np.asarray(outs[3:]))
        self.assertTrue(bool(np.all(steps > 1e-9)),
                        "output froze mid-move — smallest step {:.6f} m"
                        .format(float(steps.min())))
        self.assertEqual(h.holds, 1, "hold re-armed during a continuous move")

    def test_soak_still_move_still_at_production_rates(self):
        """Full-rate simulation with noise: 3 s still, 2 s pure dolly, 3 s still.

        Runs the real 200 Hz IMU / 90 Hz pose interleave and production window
        sizes, with measured noise figures — stationary gyro noise, 0.7 mm/s VO
        creep, 8.5 um per-sample jitter. Unit tests with clean inputs missed both
        the swallowed-translation bug and the chatter it was replaced by, because
        neither shows up without the two rates actually running against each
        other.
        """
        rng = np.random.default_rng(7)
        h = pt.StationaryHold(window_samples=60)      # production default
        POSE_HZ, IMU_PER_POSE = 90.0, 2

        truth = np.zeros(3)
        drift = np.zeros(3)
        outs, held_flags = [], []

        def tick(velocity):
            nonlocal truth, drift
            for _ in range(IMU_PER_POSE):
                # Pure translation: the gyro sees nothing but its own noise.
                h.update_imu(tuple(rng.normal(0.0, 0.004, 3)),
                             (0.0, 0.0, 9.807 + rng.normal(0.0, 0.03)))
            truth = truth + np.asarray(velocity, dtype=float) / POSE_HZ
            drift = drift + np.array([0.7e-3 / POSE_HZ, 0.0, 0.0]) \
                          + rng.normal(0.0, 8.5e-6, 3)
            out, _q, held = h.apply(tuple(truth + drift), self.Q)
            outs.append(out.copy())
            held_flags.append(held)

        for _ in range(270):                 # 3 s locked off
            tick((0.0, 0.0, 0.0))
        still_end = len(outs)

        for _ in range(180):                 # 2 s dolly at 0.15 m/s
            tick((0.15, 0.0, 0.0))
        move_end = len(outs)

        for _ in range(270):                 # 3 s locked off again
            tick((0.0, 0.0, 0.0))

        arr = np.asarray(outs)

        # 1. Locked off: the creep must not reach the output.
        settled = arr[60:still_end]
        excursion = float(np.max(np.linalg.norm(settled - settled[0], axis=1)))
        self.assertLess(excursion, 2e-3,
                        "stationary excursion {:.4f} m — hold not suppressing creep"
                        .format(excursion))

        # 2. The dolly must actually come through.
        travelled = float(arr[move_end - 1][0] - arr[still_end - 1][0])
        self.assertGreater(travelled, 0.28,
                           "only {:.3f} m of a 0.30 m dolly reached the output"
                           .format(travelled))

        # 3. No chatter: the hold must not re-arm repeatedly mid-move.
        self.assertLessEqual(h.holds, 3,
                             "hold engaged {} times — chattering".format(h.holds))
        mid_move_holds = sum(held_flags[still_end + 20:move_end])
        self.assertEqual(mid_move_holds, 0,
                         "output was pinned for {} frames mid-dolly"
                         .format(mid_move_holds))

        # 4. It must settle again afterwards, or drift returns on the next take.
        self.assertTrue(held_flags[-1], "hold never re-engaged after the move")

    def test_hold_still_suppresses_sub_millimetre_creep(self):
        """The VO cross-check must not cost us the drift suppression."""
        h = self._settled()
        first, _q, _ = h.apply((1.0, 0.0, 0.0), self.Q)
        for n in range(1, 41):                      # 4 mm of creep, 0.1 mm/sample
            h.update_imu(self.QUIET_GYRO, self.GRAVITY)
            out, _q, held = h.apply((1.0 + n * 0.0001, 0.0, 0.0), self.Q)
            self.assertTrue(held, "released on creep at sample {}".format(n))
        np.testing.assert_allclose(out, first, atol=1e-12)

    def test_motion_after_a_hold_tracks_relative_movement(self):
        h = self._settled()
        h.apply((1.0, 0.0, 0.0), self.Q)
        h.update_imu((5.0, 0.0, 0.0), self.GRAVITY)
        h.apply((1.0, 0.0, 0.0), self.Q)
        moved, _q, _ = h.apply((1.20, 0.0, 0.0), self.Q)
        self.assertAlmostEqual(float(moved[0]), 1.20, places=9,
                               msg="20 cm of real movement was not reproduced")

    def test_vibration_prevents_the_hold(self):
        """A shaking accelerometer must not read as still."""
        h = pt.StationaryHold(window_samples=10, accel_still_std=0.05)
        for i in range(20):
            h.update_imu(self.QUIET_GYRO,
                         (0.0, 0.0, 9.807 + (0.5 if i % 2 else -0.5)))
        self.assertFalse(h.is_stationary)

    def test_rotation_is_pinned_too(self):
        h = self._settled()
        _p, q0, _ = h.apply((0.0, 0.0, 0.0), self.Q)
        tilted = matrix_to_quaternion(rpy_to_matrix(0, 0, math.radians(0.5)))
        _p, q1, held = h.apply((0.0, 0.0, 0.0), tilted)
        self.assertTrue(held)
        self.assertLess(pt.PoseGate._angle_between(q0, q1), 1e-6,
                        "angular creep leaked through the hold")

    def test_disabled_is_a_passthrough(self):
        h = pt.StationaryHold(enabled=False, window_samples=5)
        for _ in range(20):
            h.update_imu(self.QUIET_GYRO, self.GRAVITY)
        self.assertFalse(h.is_stationary)
        p, _q, held = h.apply((3.0, 0.0, 0.0), self.Q)
        self.assertFalse(held)
        np.testing.assert_allclose(p, [3.0, 0.0, 0.0])

    def test_counters_and_reset(self):
        h = self._settled()
        h.apply((0.0, 0.0, 0.0), self.Q)
        h.apply((0.0001, 0.0, 0.0), self.Q)     # 100 um of creep, not a move
        self.assertEqual(h.holds, 1)
        self.assertGreater(h.absorbed_m, 0.0)
        h.reset()
        self.assertFalse(h.is_stationary)

    def test_simulated_still_then_move_then_still(self):
        """End to end: drift suppressed, movement preserved, no discontinuity."""
        h = pt.StationaryHold(window_samples=20)
        pos, outs = np.zeros(3), []

        def step(gyro, delta):
            nonlocal pos
            h.update_imu(gyro, self.GRAVITY)
            pos = pos + np.asarray(delta)
            outs.append(h.apply(tuple(pos), self.Q)[0])

        for _ in range(120):                       # still, creeping 0.1 mm/frame
            step(self.QUIET_GYRO, (0.0001, 0, 0))
        after_still = outs[-1][0]
        for _ in range(100):                       # real move, 2 mm/frame
            step((1.0, 0, 0), (0.002, 0, 0))
        after_move = outs[-1][0]
        for _ in range(120):                       # still again
            step(self.QUIET_GYRO, (0.0001, 0, 0))

        self.assertLess(after_still, 0.006, "creep leaked during the first hold")
        self.assertAlmostEqual(after_move - after_still, 0.200, delta=0.01,
                               msg="the 200 mm move was not reproduced")
        self.assertAlmostEqual(outs[-1][0], after_move, delta=0.006,
                               msg="creep leaked during the second hold")


class TestPoseSmoother(unittest.TestCase):
    def test_disabled_by_default_is_passthrough(self):
        s = PoseSmoother()
        pose = pt.UnrealPose(1, 2, 3, 4, 5, 6)
        self.assertIs(s.apply(pose), pose)

    def test_smoothing_lags_towards_the_target(self):
        s = PoseSmoother(alpha=0.5)
        s.apply(pt.UnrealPose(0, 0, 0, 0, 0, 0))
        out = s.apply(pt.UnrealPose(10, 0, 0, 0, 0, 0))
        self.assertAlmostEqual(out.x, 5.0, places=9)

    def test_converges(self):
        s = PoseSmoother(alpha=0.5)
        s.apply(pt.UnrealPose(0, 0, 0, 0, 0, 0))
        for _ in range(60):
            out = s.apply(pt.UnrealPose(10, 0, 0, 0, 0, 0))
        self.assertAlmostEqual(out.x, 10.0, places=6)

    def test_angle_wrap_does_not_spin_the_camera(self):
        """179 -> -179 is a 2 degree move, not a 358 degree sweep."""
        s = PoseSmoother(alpha=0.5)
        s.apply(pt.UnrealPose(0, 0, 0, 179.0, 0, 0))
        out = s.apply(pt.UnrealPose(0, 0, 0, -179.0, 0, 0))
        self.assertLess(abs((out.pan - 179.0 + 180) % 360 - 180), 2.0,
                        f"smoother took the long way round: pan={out.pan}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
