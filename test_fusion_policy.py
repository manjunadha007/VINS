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

    def test_no_output_before_lineup_and_three_checks(self):
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
        self.e.apply_map_alignment(np.eye(4))
        self.pose()
        m = self.check()
        for _ in range(20):
            self.e.observe_scan(m)
        self.assertIsNone(self.e.freed_packet())

    def test_depth_occlusion_has_bounded_fallback_then_exact_hold(self):
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
        self.e.apply_map_alignment(np.eye(4))
        for _ in range(3):
            self.pose()
            self.check(rank=1)
        self.assertIsNone(self.e.freed_packet())

    def test_gaps_reset_recovery_streak(self):
        self.e.apply_map_alignment(np.eye(4))
        for _ in range(4):
            self.clock.t += 2
            self.pose()
            self.check()
        self.assertIsNone(self.e.freed_packet())

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
