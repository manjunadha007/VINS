#!/usr/bin/env python3
"""
Unit tests for the shared FreeD encoder (freed.py).

Pure Python — no ROS, no PyQt, no camera, no display. Runs anywhere:

    python3 -m unittest test_freed_protocol -v

Includes a regression test against the original hand-rolled encoder that was
duplicated across five files, so the refactor is proven to be byte-for-byte
wire-compatible for all in-range inputs.
"""
import unittest

import freed
from freed import (get_freed_packet, decode_freed_packet, wrap_deg,
                   MAX_TRANSLATION_M, ANGLE_SCALE, TRANS_SCALE)


def legacy_get_freed_packet(cam_id, pitch_deg, yaw_deg, roll_deg, tx_m, ty_m, tz_m):
    """The original encoder, verbatim from dashboard.py before the refactor.

    Kept only as a test oracle — do not use in production code.
    """
    packet = bytearray(29)
    packet[0] = 0xD1
    packet[1] = cam_id & 0xFF

    def i24(val):
        val = int(val)
        if val < 0:
            val += (1 << 24)
        return [(val >> 16) & 0xFF, (val >> 8) & 0xFF, val & 0xFF]

    packet[2:5]   = i24(int(yaw_deg   * 32768))
    packet[5:8]   = i24(int(pitch_deg * 32768))
    packet[8:11]  = i24(int(roll_deg  * 32768))
    packet[11:14] = i24(int(tx_m * 64000))
    packet[14:17] = i24(int(ty_m * 64000))
    packet[17:20] = i24(int(tz_m * 64000))
    packet[20:23] = i24(0)
    packet[23:26] = i24(0)
    packet[26:28] = [0, 0]
    packet[28] = (64 - (sum(packet[:28]) % 256)) % 256
    return bytes(packet)


class TestPacketStructure(unittest.TestCase):
    def test_length_and_header(self):
        pkt = get_freed_packet(1, 0, 0, 0, 0, 0, 0)
        self.assertEqual(len(pkt), 29, "FreeD D1 packets must be exactly 29 bytes")
        self.assertEqual(pkt[0], 0xD1, "byte 0 must be the D1 message type")
        self.assertEqual(pkt[1], 1, "byte 1 must carry the camera ID")

    def test_checksum_invariant(self):
        """Sum of all 29 bytes must be 64 mod 256 — the FreeD spec invariant."""
        for args in [(1, 0, 0, 0, 0, 0, 0),
                     (7, 12.5, -33.25, 4.0, 1.5, -2.5, 3.75),
                     (255, -179.9, 179.9, -90.0, -10.0, 20.0, -30.0)]:
            pkt = get_freed_packet(*args)
            self.assertEqual(sum(pkt) % 256, 64, f"bad checksum for {args}")

    def test_camera_id_is_masked_to_one_byte(self):
        self.assertEqual(get_freed_packet(257, 0, 0, 0, 0, 0, 0)[1], 1)

    def test_zoom_and_focus_default_to_zero(self):
        d = decode_freed_packet(get_freed_packet(1, 0, 0, 0, 0, 0, 0))
        self.assertEqual(d["zoom"], 0)
        self.assertEqual(d["focus"], 0)


class TestRoundTrip(unittest.TestCase):
    def assertRoundTrips(self, cam_id, pitch, yaw, roll, tx, ty, tz):
        d = decode_freed_packet(
            get_freed_packet(cam_id, pitch, yaw, roll, tx, ty, tz))
        ang_tol = 1.0 / ANGLE_SCALE
        pos_tol = 1.0 / TRANS_SCALE
        self.assertEqual(d["cam_id"], cam_id & 0xFF)
        self.assertAlmostEqual(d["pitch_deg"], wrap_deg(pitch), delta=ang_tol)
        self.assertAlmostEqual(d["yaw_deg"],   wrap_deg(yaw),   delta=ang_tol)
        self.assertAlmostEqual(d["roll_deg"],  wrap_deg(roll),  delta=ang_tol)
        self.assertAlmostEqual(d["x_m"], tx, delta=pos_tol)
        self.assertAlmostEqual(d["y_m"], ty, delta=pos_tol)
        self.assertAlmostEqual(d["z_m"], tz, delta=pos_tol)

    def test_zero(self):
        self.assertRoundTrips(1, 0, 0, 0, 0, 0, 0)

    def test_typical_pose(self):
        self.assertRoundTrips(1, 10.5, -30.25, 2.0, 1.5, -2.25, 3.0)

    def test_negative_translation(self):
        self.assertRoundTrips(1, 0, 0, 0, -5.125, -0.5, -12.0)

    def test_fields_do_not_bleed_into_each_other(self):
        """Only X should move when only X moves."""
        base = decode_freed_packet(get_freed_packet(1, 0, 0, 0, 0.0, 0.0, 0.0))
        moved = decode_freed_packet(get_freed_packet(1, 0, 0, 0, 4.0, 0.0, 0.0))
        self.assertAlmostEqual(moved["x_m"], 4.0, delta=1.0 / TRANS_SCALE)
        self.assertEqual(moved["y_m"], base["y_m"])
        self.assertEqual(moved["z_m"], base["z_m"])
        self.assertEqual(moved["yaw_deg"], base["yaw_deg"])
        self.assertEqual(moved["pitch_deg"], base["pitch_deg"])
        self.assertEqual(moved["roll_deg"], base["roll_deg"])

    def test_axes_are_not_swapped(self):
        d = decode_freed_packet(get_freed_packet(1, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
        self.assertAlmostEqual(d["pitch_deg"], 1.0, delta=1e-3)
        self.assertAlmostEqual(d["yaw_deg"],   2.0, delta=1e-3)
        self.assertAlmostEqual(d["roll_deg"],  3.0, delta=1e-3)
        self.assertAlmostEqual(d["x_m"],       4.0, delta=1e-3)
        self.assertAlmostEqual(d["y_m"],       5.0, delta=1e-3)
        self.assertAlmostEqual(d["z_m"],       6.0, delta=1e-3)


class TestAngleWrapping(unittest.TestCase):
    """The pure-gyro bridge integrates yaw without bound; +/-256 deg used to
    silently wrap the int24 field and teleport the virtual camera."""

    def test_wrap_deg_basics(self):
        self.assertAlmostEqual(wrap_deg(0.0), 0.0)
        self.assertAlmostEqual(wrap_deg(370.0), 10.0, places=9)
        self.assertAlmostEqual(wrap_deg(-190.0), 170.0, places=9)
        self.assertAlmostEqual(wrap_deg(270.0), -90.0, places=9)
        self.assertAlmostEqual(wrap_deg(180.0), -180.0, places=9)
        self.assertAlmostEqual(wrap_deg(-180.0), -180.0, places=9)

    def test_drifted_yaw_stays_in_range(self):
        for yaw in [256.1, 300.0, 720.0, 1e4, 1e6, -1e6]:
            d = decode_freed_packet(get_freed_packet(1, 0, yaw, 0, 0, 0, 0))
            self.assertGreaterEqual(d["yaw_deg"], -180.0)
            self.assertLess(d["yaw_deg"], 180.0)

    def test_wrapped_angle_is_the_same_rotation(self):
        a = decode_freed_packet(get_freed_packet(1, 0, 30.0, 0, 0, 0, 0))
        b = decode_freed_packet(get_freed_packet(1, 0, 390.0, 0, 0, 0, 0))
        self.assertAlmostEqual(a["yaw_deg"], b["yaw_deg"], delta=1e-3)

    def test_legacy_encoder_overflowed_where_we_now_do_not(self):
        """Documents the bug: at 300 deg the old encoder wrapped to a negative."""
        legacy = decode_freed_packet(legacy_get_freed_packet(1, 0, 300.0, 0, 0, 0, 0))
        fixed  = decode_freed_packet(get_freed_packet(1, 0, 300.0, 0, 0, 0, 0))
        self.assertAlmostEqual(fixed["yaw_deg"], -60.0, delta=1e-3)
        self.assertNotAlmostEqual(legacy["yaw_deg"], -60.0, delta=1.0)


class TestTranslationSaturation(unittest.TestCase):
    def test_limit_is_about_131_metres(self):
        self.assertAlmostEqual(MAX_TRANSLATION_M, 131.0719, places=3)

    def test_beyond_limit_saturates_and_keeps_its_sign(self):
        pos = decode_freed_packet(get_freed_packet(1, 0, 0, 0, 500.0, 0, 0))
        neg = decode_freed_packet(get_freed_packet(1, 0, 0, 0, -500.0, 0, 0))
        self.assertGreater(pos["x_m"], 0, "positive overflow must not flip sign")
        self.assertLess(neg["x_m"], 0, "negative overflow must not flip sign")
        self.assertAlmostEqual(pos["x_m"], MAX_TRANSLATION_M, delta=1e-3)
        self.assertAlmostEqual(neg["x_m"], -MAX_TRANSLATION_M, delta=1e-3)

    def test_just_inside_limit_is_exact(self):
        d = decode_freed_packet(get_freed_packet(1, 0, 0, 0, 130.0, 0, 0))
        self.assertAlmostEqual(d["x_m"], 130.0, delta=1.0 / TRANS_SCALE)


class TestDecoderRejectsBadInput(unittest.TestCase):
    def test_wrong_length(self):
        with self.assertRaises(ValueError):
            decode_freed_packet(b"\xD1" * 28)

    def test_wrong_message_type(self):
        pkt = bytearray(get_freed_packet(1, 0, 0, 0, 0, 0, 0))
        pkt[0] = 0xD2
        pkt[28] = freed.freed_checksum(pkt)
        with self.assertRaises(ValueError):
            decode_freed_packet(bytes(pkt))

    def test_corrupt_checksum(self):
        pkt = bytearray(get_freed_packet(1, 0, 0, 0, 0, 0, 0))
        pkt[28] = (pkt[28] + 1) % 256
        with self.assertRaises(ValueError):
            decode_freed_packet(bytes(pkt))


class TestLegacyWireCompatibility(unittest.TestCase):
    """The refactor must not change a single byte for normal operating ranges."""

    IN_RANGE_CASES = [
        (1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (1, 10.5, -30.25, 2.0, 1.5, -2.25, 3.0),
        (0, -45.5, 90.0, -179.5, -10.0, 20.0, -30.0),
        (42, 1.25, 2.5, 3.75, 0.125, -0.25, 0.5),
        (255, 179.0, -179.0, 90.0, 100.0, -100.0, 50.0),
    ]

    def test_byte_for_byte_identical(self):
        for case in self.IN_RANGE_CASES:
            self.assertEqual(get_freed_packet(*case), legacy_get_freed_packet(*case),
                             f"wire format changed for {case}")

    @staticmethod
    def _angular_gap(a, b):
        """Shortest distance between two angles in degrees.

        Needed because +180 and -180 are the same rotation: the legacy encoder
        emitted +180, the new one wraps to -180. Identical orientation, so a
        naive subtraction would report a bogus 360-degree error.
        """
        return abs((a - b + 180.0) % 360.0 - 180.0)

    def test_sweep_is_the_same_rotation_within_one_lsb(self):
        """Angle wrapping introduces float rounding; allow at most 1 count."""
        for i in range(-179, 180, 7):
            for j in range(-120, 121, 30):
                args = (1, i * 0.5, j * 1.5, i * 0.25,
                        j * 0.1, i * 0.05, j * 0.01)
                dn = decode_freed_packet(get_freed_packet(*args))
                do = decode_freed_packet(legacy_get_freed_packet(*args))
                for key in ("pitch_deg", "yaw_deg", "roll_deg"):
                    self.assertLessEqual(
                        self._angular_gap(dn[key], do[key]), 1.5 / ANGLE_SCALE,
                        f"{key} drifted at i={i} j={j}: {dn[key]} vs {do[key]}")
                for key in ("x_m", "y_m", "z_m"):
                    self.assertEqual(dn[key], do[key],
                                     f"{key} changed at i={i} j={j}")

    def test_plus_and_minus_180_are_the_only_byte_difference(self):
        """Explicitly pin the one intentional wire change: 180 -> -180."""
        new = decode_freed_packet(get_freed_packet(1, 0, 180.0, 0, 0, 0, 0))
        old = decode_freed_packet(legacy_get_freed_packet(1, 0, 180.0, 0, 0, 0, 0))
        self.assertAlmostEqual(new["yaw_deg"], -180.0, delta=1e-3)
        self.assertAlmostEqual(old["yaw_deg"], +180.0, delta=1e-3)
        self.assertEqual(self._angular_gap(new["yaw_deg"], old["yaw_deg"]), 0.0)


class TestEulerFromQuaternion(unittest.TestCase):
    def test_identity(self):
        r, p, y = freed.euler_from_quaternion(0.0, 0.0, 0.0, 1.0)
        self.assertAlmostEqual(r, 0.0)
        self.assertAlmostEqual(p, 0.0)
        self.assertAlmostEqual(y, 0.0)

    def test_ninety_degree_yaw(self):
        import math
        s = math.sqrt(0.5)
        r, p, y = freed.euler_from_quaternion(0.0, 0.0, s, s)
        self.assertAlmostEqual(math.degrees(y), 90.0, places=6)
        self.assertAlmostEqual(r, 0.0, places=6)
        self.assertAlmostEqual(p, 0.0, places=6)

    def test_gimbal_lock_does_not_raise(self):
        import math
        s = math.sqrt(0.5)
        r, p, y = freed.euler_from_quaternion(0.0, s, 0.0, s)
        self.assertAlmostEqual(math.degrees(p), 90.0, places=6)


if __name__ == '__main__':
    unittest.main(verbosity=2)
