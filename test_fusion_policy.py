"""Failure/recovery regressions for the supervised camera output."""
import copy
from dataclasses import replace
import unittest
from unittest.mock import patch
import numpy as np

import freed
from fusion_supervisor import FusionSupervisor
from gnss_receiver import GNSSFix, NMEAParser, local_position, quality_error
from lidar_map_matcher import MapMeasurement, LidarMapMatcher
from tracking_config import DEFAULTS
from tracking_engine import TrackingEngine
from pose_transform import make_transform, matrix_to_quaternion, rpy_to_matrix


class Clock:
    t = 100.0
    def mono(self):
        return self.t
    def wall(self):
        return 1000.0 + self.t


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.addCleanup(patch.stopall)
        patch("time.monotonic", self.clock.mono).start()
        patch("time.time", self.clock.wall).start()
        self.cfg = copy.deepcopy(DEFAULTS)
        self.cfg["fusion"]["enabled"] = True
        self.cfg["tracking"].update(gate_enabled=False, hold_enabled=False)
        self.e = TrackingEngine(self.cfg, log=lambda *a: None)

    def pose(self, x=0, q=(0,0,0,1)):
        self.clock.t += 0.02
        return self.e.accept_pose((x,0,0), q, (0,0,0), (0,0,0), stamp=self.clock.t)

    def check(self, ok=True, rank=3):
        m = MapMeasurement(ok=ok, observable_rank=rank, verify_score=0.95,
                           rmse_m=0.01, reason="" if ok else "Occluded")
        m.monotonic_timestamp = self.clock.t
        self.e.observe_scan(m)
        return m

    def ready(self):
        self.e.apply_map_alignment(np.eye(4))
        for _ in range(3):
            self.pose()
            self.check()
        self.assertEqual(self.e.fusion_status()["state"], "VALIDATED")
        return self.e.freed_packet()

    def test_strict_mode_needs_lineup_and_three_checks(self):
        self.cfg["fusion"]["allow_vio_fallback"] = False
        self.pose()
        self.check()
        self.assertIsNone(self.e.freed_packet())
        self.e.apply_map_alignment(np.eye(4))
        self.assertIsNone(self.e.freed_packet())
        for _ in range(2):
            self.pose()
            self.check()
        self.assertIsNotNone(self.e.freed_packet())

    def test_replaying_measurement_does_not_manufacture_consensus(self):
        self.cfg["fusion"]["allow_vio_fallback"] = False
        self.e.apply_map_alignment(np.eye(4))
        self.pose()
        m = self.check()
        for _ in range(20):
            self.e.observe_scan(m)
        self.assertIsNone(self.e.freed_packet())

    def test_depth_occlusion_has_bounded_fallback_then_exact_hold(self):
        self.cfg["fusion"]["allow_vio_fallback"] = False
        self.ready()
        self.pose(0.01)
        self.check(False)
        self.assertEqual(self.e.fusion_status()["state"], "DEGRADED")
        packet = self.e.freed_packet()
        self.clock.t += 1.1
        self.pose(0.03)
        self.assertEqual(self.e.fusion_status()["state"], "HOLD")
        self.assertEqual(self.e.freed_packet(), packet)
        np.testing.assert_allclose(self.e.camera_pose_ros_corrected()[0], (0.01,0,0))

    def test_stale_camera_cannot_be_replaced_by_scan_or_gnss(self):
        packet = self.ready()
        self.clock.t += 0.3
        self.check()
        self.e.fusion.observe("gnss", True, self.clock.t)
        self.assertEqual(self.e.freed_packet(), packet)
        self.assertFalse(self.e.camera_pose_now()[1])

    def test_map_updates_cannot_move_held_output(self):
        packet = self.ready()
        self.clock.t += 1.0
        self.e.freed_packet()
        self.e.apply_map_alignment(make_transform((5,0,0),(0,0,0,1)))
        self.assertEqual(self.e.freed_packet(), packet)

    def test_large_recovery_requires_explicit_lineup(self):
        packet = self.ready()
        self.clock.t += 2.0
        self.e.freed_packet()
        for _ in range(3):
            self.pose(1.0)
            self.check()
        self.assertEqual(self.e.freed_packet(), packet)
        self.assertEqual(self.e.fusion_status()["state"], "RELINEUP")
        self.e.confirm_lineup()
        for _ in range(3):
            self.pose(1.0)
            self.check()
        self.assertNotEqual(self.e.freed_packet(), packet)

    def test_verified_reacquisition_preserves_output_then_slews_without_teleport(self):
        packet = self.ready()
        self.e.begin_recovery()
        self.pose(1.0)
        self.assertEqual(self.e.freed_packet(), packet)
        self.assertFalse(self.e.resume_verified_recovery())
        for _ in range(3):
            self.pose(1.0)
            self.check()
        self.assertEqual(self.e.freed_packet(), packet)
        self.assertTrue(self.e.resume_verified_recovery())
        self.assertEqual(self.e.freed_packet(), packet)
        previous = self.e._output_T.copy()
        self.pose(1.0)
        self.e.freed_packet()
        step = np.linalg.norm(self.e._output_T[:3, 3] - previous[:3, 3])
        self.assertGreater(step, 0.)
        self.assertLessEqual(step, self.cfg['fusion']['recovery_slew_m_s'] * .02 + 1e-9)
        self.assertFalse(self.e._relineup_required)
        # A new outage revokes the exception for the previously verified fix.
        self.clock.t += 1.
        held = self.e.freed_packet()
        self.assertFalse(self.e._verified_recovery)
        self.pose(2.0)
        self.assertEqual(self.e.freed_packet(), held)
        self.assertEqual(self.e.fusion_status()['state'], 'RELINEUP')

    def test_gnss_and_replayed_scan_cannot_release_recovery(self):
        self.ready()
        self.e.begin_recovery()
        self.pose(1.)
        measurement = self.check()
        for _ in range(5):
            self.e.observe_scan(measurement)
            self.pose(1.)
            self.e.fusion.observe('gnss', True, self.clock.t)
        self.assertFalse(self.e.resume_verified_recovery())

    def test_tracking_resync_invalidates_pose_history_for_image_pairing(self):
        self.e.gate.enabled = True
        self.e.gate.max_consecutive_rejects = 2
        self.pose()
        old_stamp = self.clock.t
        self.pose(2.)
        self.pose(2.)
        self.assertEqual(self.e.gate.resyncs, 1)
        self.assertIsNone(self.e.pose_at(old_stamp))

    def test_live_dot_follows_camera_while_output_waits_for_relineup(self):
        packet = self.ready()
        position, _, live = self.e.camera_pose_ros_live()
        np.testing.assert_allclose(position, (0, 0, 0))
        self.assertTrue(live)
        self.clock.t += 2.0
        self.e.freed_packet()
        for i in range(10):
            x = 1.0 + i * .05
            self.pose(x)
            self.check()
            self.assertEqual(self.e.freed_packet(), packet)
            self.assertEqual(self.e.fusion_status()["state"], "RELINEUP")
            # Unreal stays held; the 3D view keeps following the camera.
            np.testing.assert_allclose(self.e.camera_pose_ros_corrected()[0], (0, 0, 0))
            position, _, live = self.e.camera_pose_ros_live()
            np.testing.assert_allclose(position, (x, 0, 0))
            self.assertFalse(live)
        self.e.confirm_lineup()
        for _ in range(3):
            self.pose(2.0)
            self.check()
        self.e.freed_packet()
        position, _, live = self.e.camera_pose_ros_live()
        np.testing.assert_allclose(position, self.e.camera_pose_ros_corrected()[0])
        self.assertTrue(live)

    def test_live_dot_moves_before_lineup_but_is_not_output(self):
        self.pose(0.3)
        self.assertEqual(self.e.fusion_status()["state"], "ALIGNING")
        self.assertIsNone(self.e.camera_pose_ros_corrected())
        position, _, live = self.e.camera_pose_ros_live()
        np.testing.assert_allclose(position, (0.3, 0, 0))
        self.assertFalse(live)

    def test_small_recovery_is_slew_limited(self):
        self.ready()
        self.clock.t += 2.0
        self.e.freed_packet()
        for _ in range(3):
            self.pose(0.05)
            self.check()
        packet = freed.decode_freed_packet(self.e.freed_packet())
        self.assertLessEqual(packet["x_m"], 0.0051)
        self.assertGreater(packet["x_m"], 0)

    def test_single_wall_does_not_claim_full_validation(self):
        self.cfg["fusion"]["allow_vio_fallback"] = False
        self.e.apply_map_alignment(np.eye(4))
        for _ in range(3):
            self.pose()
            self.check(rank=1)
        self.assertIsNone(self.e.freed_packet())

    def test_gaps_reset_recovery_streak(self):
        self.cfg["fusion"]["allow_vio_fallback"] = False
        self.e.apply_map_alignment(np.eye(4))
        for _ in range(4):
            self.clock.t += 2
            self.pose()
            self.check()
        self.assertIsNone(self.e.freed_packet())

    def test_vio_fallback_still_requires_lineup(self):
        self.pose()
        self.assertEqual(self.e.fusion_status()["state"], "ALIGNING")
        self.assertIsNone(self.e.freed_packet())

    def test_camera_follows_alignment_before_runtime_scan_validation(self):
        alignment = make_transform((2, -3, 1),
            matrix_to_quaternion(rpy_to_matrix(0, 0, np.pi/3)))
        rig = make_transform((0.08, 0, 0.12), (0, 0, 0, 1))
        self.e.transformer._rig = rig
        self.e.apply_map_alignment(alignment)
        for i in range(120):  # >1 s grace, >10 cm/5 degree recovery limits
            x = i * 0.01
            q = matrix_to_quaternion(rpy_to_matrix(0, i * 0.002, i * 0.005))
            self.pose(x, q)
            packet = freed.decode_freed_packet(self.e.freed_packet())
            expected = alignment @ make_transform((x, 0, 0), q) @ rig
            actual = make_transform(*self.e.camera_pose_ros_corrected())
            np.testing.assert_allclose(actual, expected, atol=1e-12)
            np.testing.assert_allclose([packet["x_m"], packet["y_m"], packet["z_m"]],
                                       expected[:3, 3] * [1, -1, 1], atol=1e-4)
            self.assertEqual(self.e.fusion_status()["state"], "DEGRADED")
            self.assertFalse(self.e.camera_pose_now()[1])  # motion is not scan validation
            self.assertFalse(self.e._output_held)

    def test_scan_rejection_partial_geometry_and_outage_do_not_freeze_motion(self):
        self.ready()
        for i in range(1, 361):
            self.pose(i * 0.01)
            if i % 10 == 0:
                if i <= 120:
                    self.check(False)
                elif i <= 240:
                    self.check(rank=1)
                # Remaining 2.4 seconds: depth absent; keep fresh camera poses.
            self.e.freed_packet()
            np.testing.assert_allclose(self.e.camera_pose_ros_corrected()[0], (i * .01, 0, 0))
            self.assertFalse(self.e._output_held)
            self.assertFalse(self.e._relineup_required)
        self.assertEqual(self.e.fusion_status()["state"], "DEGRADED")
        # Fresh checks restore green without slewing back from the lineup pose.
        for i in range(3):
            self.pose(3.7 + i * .01)
            self.check()
            self.e.freed_packet()
            self.assertEqual(self.e.fusion_status()["state"], "VALIDATED" if i == 2 else "DEGRADED")
            np.testing.assert_allclose(self.e.camera_pose_ros_corrected()[0], (3.7 + i * .01, 0, 0))

    def test_fallback_holds_when_camera_is_rejected_or_stale(self):
        self.e.apply_map_alignment(np.eye(4))
        self.pose(0.2)
        packet = self.e.freed_packet()
        self.assertIsNotNone(packet)
        self.assertFalse(self.pose(float("nan")))
        self.assertEqual(self.e.freed_packet(), packet)
        self.assertEqual(self.e.fusion_status()["state"], "HOLD")
        self.pose(0.2)
        self.e.freed_packet()
        self.clock.t += 1
        self.assertEqual(self.e.freed_packet(), packet)
        self.assertEqual(self.e.fusion_status()["state"], "HOLD")

    def test_stationary_hold_releases_for_straight_motion_without_scan_checks(self):
        self.cfg["tracking"].update(gate_enabled=True, hold_enabled=True)
        self.e = TrackingEngine(self.cfg, log=lambda *a: None)
        self.ready()
        for _ in range(60):
            self.e.update_imu((0, 0, 0), (0, 0, 9.81))
        self.pose()
        self.e.freed_packet()
        self.assertTrue(self.e.hold.is_stationary)
        for i in range(1, 181):
            # A straight dolly move keeps the gyro quiet; vision must release it.
            self.e.update_imu((0, 0, 0), (0, 0, 9.81))
            self.pose(i * .001)
            self.e.freed_packet()
        self.assertGreater(self.e.hold.vo_releases, 0)
        self.assertFalse(self.e._pose_rejected)
        self.assertEqual(self.e.fusion_status()["state"], "DEGRADED")
        self.assertAlmostEqual(self.e.camera_pose_ros_corrected()[0][0], .18, delta=.004)

    def test_vo_fallback_runs_at_camera_rate_and_yields_to_fresh_odometry(self):
        self.e.apply_map_alignment(np.eye(4))
        def feed(source, x):
            return self.e.accept_pose((x, 0, 0), (0, 0, 0, 1), (0, 0, 0),
                                      (0, 0, 0), stamp=self.clock.t, source=source)
        self.assertTrue(feed("odometry", 0))
        self.e.freed_packet()
        self.clock.t += .02
        self.assertFalse(feed("vo", 1))
        # Switch as soon as primary odometry is stale, then consume every VO frame.
        self.clock.t += self.e.stale_timeout
        for i in range(90):
            self.clock.t += 1/90
            self.assertTrue(feed("vo", i * .01))
            self.e.freed_packet()
            self.assertAlmostEqual(self.e.camera_pose_ros_corrected()[0][0], i * .01)
        self.clock.t += 1/90
        self.assertTrue(feed("odometry", .9))
        self.clock.t += 1/90
        self.assertFalse(feed("vo", 2))
        self.e.freed_packet()
        self.assertAlmostEqual(self.e.camera_pose_ros_corrected()[0][0], .9)

    def test_nan_and_zero_quaternion_are_rejected(self):
        self.ready()
        self.assertFalse(self.pose(float("nan")))
        self.assertFalse(self.pose(q=(0,0,0,0)))
        self.assertFalse(self.e.camera_pose_now()[1])

    def test_duplicate_or_backwards_pose_cannot_refresh_tracking(self):
        self.pose()
        t = self.e._last_pose_mono
        self.clock.t += 1
        self.assertFalse(self.e.accept_pose((0,0,0),(0,0,0,1),(0,0,0),(0,0,0),stamp=t))
        self.assertEqual(self.e._last_pose_mono,t)

    def test_depth_needs_real_intrinsics_and_paired_timestamp(self):
        self.pose(1)
        K = np.diag([400,400,1])
        self.e.set_depth(np.ones((4,4)), stamp=self.clock.t)
        self.assertIsNone(self.e.depth_pose())
        self.e.set_depth_intrinsics(K)
        np.testing.assert_allclose(self.e.depth_pose()[0], (1,0,0))
        self.e.set_depth(np.ones((4,4)), stamp=self.clock.t-0.1)
        self.assertIsNone(self.e.depth_pose())

    def test_depth_pairs_to_capture_pose_not_newest_pose(self):
        self.pose(1)
        old = self.clock.t
        self.pose(2)
        self.e.set_depth_intrinsics(np.diag([400,400,1]))
        self.e.set_depth(np.ones((4,4)), stamp=old)
        np.testing.assert_allclose(self.e.depth_pose()[0], (1,0,0))
        self.clock.t += 1
        self.assertIsNone(self.e.depth_pose())

    def test_uncalibrated_gnss_never_validates(self):
        self.ready()
        fix=good_fix(self.clock.t)
        self.assertFalse(self.e.update_gnss(fix))
        self.assertIn("surveyed", self.e.gnss_reason)

    def test_calibrated_gnss_checks_rotated_antenna_lever_arm(self):
        self.cfg["gnss"].update(calibrated=True, origin_lla=[0,0,0],
                                origin_world_m=[0,1,0], antenna_offset_body_m=[1,0,0])
        self.e.apply_map_alignment(np.eye(4))
        q = matrix_to_quaternion(rpy_to_matrix(0,0,np.pi/2))
        for i in range(3):
            self.pose(q=q)
            fix=replace(good_fix(self.clock.t),epoch=str(i))
            self.assertTrue(self.e.update_gnss(fix), self.e.gnss_reason)
        self.assertEqual(self.e.fusion_status()["state"], "RTK_ASSISTED")
        self.assertIsNotNone(self.e.freed_packet())

    def test_gnss_disagreement_is_rejected_without_moving_pose(self):
        packet=self.ready()
        self.cfg["gnss"].update(calibrated=True, origin_lla=[0,0,0], origin_world_m=[10,0,0])
        self.assertFalse(self.e.update_gnss(good_fix(self.clock.t)))
        self.assertEqual(self.e.freed_packet(),packet)

    def test_reset_invalidates_old_evidence(self):
        self.ready()
        self.e.handle_command({"cmd":"reset_pose"})
        self.assertIsNone(self.e.freed_packet())
        self.assertFalse(self.e.fusion.evidence)


def good_fix(now=100):
    return GNSSFix("120000.000",0,0,0,4,15,0.7,0.5,0.02,0.03,now)


def sentence(body):
    check=0
    for c in body:
        check ^= ord(c)
    return f"${body}*{check:02X}"


class GNSSTests(unittest.TestCase):
    def gga(self, epoch="120000.00", quality="4"):
        return sentence(f"GNGGA,{epoch},1258.0000,N,07735.0000,E,{quality},15,0.7,800.0,M,-80.0,M,0.5,1234")

    def gst(self, epoch="120000.00"):
        return sentence(f"GNGST,{epoch},0.02,0.02,0.01,0,0.01,0.02,0.03")

    def test_checksum_is_required(self):
        p=NMEAParser()
        self.assertIsNone(p.feed(self.gga()[:-2]+"FF"))
        self.assertIsNone(p.feed(self.gga().split("*")[0]))

    def test_same_epoch_accuracy_and_ellipsoid_height(self):
        p=NMEAParser()
        first=p.feed(self.gga(),100)
        self.assertIsNone(first.hacc_m)
        fix=p.feed(self.gst(),100.01)
        self.assertAlmostEqual(fix.hacc_m,np.hypot(0.01,0.02))
        self.assertEqual(fix.altitude_m,720)
        self.assertEqual(fix.received,100)
        self.assertEqual(quality_error(fix,{},100.1),"")

    def test_gst_before_gga(self):
        p=NMEAParser()
        self.assertIsNone(p.feed(self.gst(),100))
        self.assertIsNotNone(p.feed(self.gga(),100.01).hacc_m)

    def test_accuracy_from_another_epoch_cannot_be_reused(self):
        p=NMEAParser()
        p.feed(self.gst(),100)
        self.assertIsNone(p.feed(self.gga("120001.00"),101).hacc_m)

    def test_duplicate_complete_epoch_is_not_emitted(self):
        p=NMEAParser()
        p.feed(self.gga(),100)
        p.feed(self.gst(),100.01)
        self.assertIsNone(p.feed(self.gga(),100.1))

    def test_bad_quality_stale_and_missing_corrections(self):
        for change in (dict(quality=5),dict(quality=1),dict(received=98),
                       dict(correction_age_s=None),dict(correction_age_s=20),
                       dict(hacc_m=1),dict(vacc_m=float("nan")),dict(satellites=3)):
            self.assertTrue(quality_error(replace(good_fix(),**change),{},100))

    def test_world_origin_and_enu_yaw(self):
        f=replace(good_fix(),longitude=0.00001)
        p=local_position(f,dict(calibrated=True,origin_lla=[0,0,0],
                               enu_yaw_deg=90,origin_world_m=[2,3,4]))
        self.assertAlmostEqual(p[0],2,places=4)
        self.assertAlmostEqual(p[1],4.1131949,places=4)


class SlewTests(unittest.TestCase):
    def test_long_outage_does_not_accumulate_correction_budget(self):
        m=LidarMapMatcher(dict(map_file="",blend_alpha=1,max_slew_m_s=0.02,
                              icp_rate_hz=5))
        m._last_fuse_time=1
        measurement=MapMeasurement(ok=True,fitness=1,
                        T_slam_to_map=make_transform((1,0,0),(0,0,0,1)))
        m._fuse(measurement,np.eye(4),100)
        self.assertLessEqual(np.linalg.norm(m.T_slam_to_map[:3,3]),0.00801)

    def test_zero_slew_limit_means_no_translation_correction(self):
        m=LidarMapMatcher(dict(map_file="",blend_alpha=1,max_slew_m_s=0))
        measurement=MapMeasurement(ok=True,fitness=1,
                        T_slam_to_map=make_transform((1,0,0),(0,0,0,1)))
        m._fuse(measurement,np.eye(4),100)
        np.testing.assert_allclose(m.T_slam_to_map,np.eye(4))


if __name__ == "__main__":
    unittest.main()
