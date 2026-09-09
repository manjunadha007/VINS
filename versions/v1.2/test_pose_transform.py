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
        # SLAM creeps 5 mm; output must not move.
        drifted, _q, held = h.apply((1.005, 0.0, 0.0), self.Q)
        self.assertTrue(held)
        np.testing.assert_allclose(drifted, first, atol=1e-12)

    def test_absorbs_creep_so_release_does_not_jump(self):
        h = self._settled()
        pinned, _q, _ = h.apply((1.0, 0.0, 0.0), self.Q)
        for mm in range(1, 11):                     # creep 10 mm while held
            h.apply((1.0 + mm * 0.001, 0.0, 0.0), self.Q)
        h.update_imu((5.0, 0.0, 0.0), self.GRAVITY)  # motion starts
        self.assertFalse(h.is_stationary)
        released, _q, held = h.apply((1.010, 0.0, 0.0), self.Q)
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
        h.apply((0.01, 0.0, 0.0), self.Q)
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
