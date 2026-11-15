#!/usr/bin/env python3
"""Testing arena: scenario file, take bookkeeping, and the dashboard's records.

The scenario/manifest/state tests need nothing beyond PyYAML and numpy. The
dashboard tests need the Isaac image (ROS 2 + PyQt) and skip themselves
elsewhere, like the other dashboard suites.
"""
import json
import os
import struct
import tempfile
import unittest

import numpy as np
import yaml

import arena
import arena_state
from freed import get_freed_packet

try:
    import dashboard
    _DASHBOARD_ERR = None
except Exception as exc:                      # pragma: no cover - env dependent
    dashboard, _DASHBOARD_ERR = None, exc


def _write(directory, name, text):
    path = os.path.join(directory, name)
    with open(path, "w") as fh:
        fh.write(text)
    return path


class TestScenarioFile(unittest.TestCase):
    def setUp(self):
        self.scenarios = arena.load_scenarios()

    def test_the_four_recorded_scenarios_are_defined(self):
        self.assertEqual([s["short"] for s in self.scenarios], ["s1", "s2", "s3", "s4"])
        for sc in self.scenarios:
            self.assertGreaterEqual(len(sc["steps"]), 2, sc["id"])
            self.assertTrue(sc["description"], sc["id"])

    def test_pass_criteria_are_the_agreed_ones(self):
        for sc in self.scenarios:
            self.assertEqual(sc["criteria"], dict(frame_mm=20, frame_deg=0.5, min_frames=0.90))
        self.assertTrue(any("tape" in item for item in self.scenarios[0]["setup"]))

    def test_a_scenario_can_override_one_criterion(self):
        text = ("defaults: {pass: {frame_mm: 20}}\n"
                "scenarios: [{id: s1_a, title: t, steps: [a], pass: {frame_deg: 1.0}}]")
        with tempfile.TemporaryDirectory() as d:
            sc = arena.load_scenarios(_write(d, "s.yaml", text))[0]
        self.assertEqual(sc["criteria"], dict(frame_mm=20, frame_deg=1.0, min_frames=0.90))

    def test_bad_criteria_are_refused(self):
        for bad in ("{frame_mm: -1}", "{min_frames: 90}", "{tolerance: 3}"):
            text = f"scenarios: [{{id: s1_a, title: t, steps: [a], pass: {bad}}}]"
            with self.subTest(bad), tempfile.TemporaryDirectory() as d:
                with self.assertRaises(ValueError):
                    arena.load_scenarios(_write(d, "s.yaml", text))

    def test_resolve_by_short_name_or_full_id(self):
        self.assertEqual(arena.resolve_scenario(self.scenarios, "S2")["id"], "s2_lidar_gap")
        self.assertEqual(arena.resolve_scenario(self.scenarios, "s3_single_marker_start")["short"], "s3")
        with self.assertRaises(KeyError):
            arena.resolve_scenario(self.scenarios, "s9")

    def test_bad_files_are_refused_with_the_scenario_named(self):
        cases = {
            "bad id": "scenarios: [{id: one, title: t, steps: [a]}]",
            "duplicate short name": ("scenarios: [{id: s1_a, title: t, steps: [a]},"
                                     " {id: s1_b, title: t, steps: [a]}]"),
            "no steps": "scenarios: [{id: s1_a, title: t, steps: []}]",
            "no title": "scenarios: [{id: s1_a, steps: [a]}]",
            "empty file": "",
        }
        with tempfile.TemporaryDirectory() as d:
            for what, text in cases.items():
                with self.subTest(what):
                    with self.assertRaises(ValueError):
                        arena.load_scenarios(_write(d, "s.yaml", text))


class TestStreamTiming(unittest.TestCase):
    def test_rate_comes_from_stamps_even_with_frames_missing(self):
        stamps = [i / 90.0 for i in range(900) if i % 10 != 5]   # every 10th lost
        t = arena.stamp_timing(stamps)
        self.assertAlmostEqual(t["rate_hz"], 90.0, places=1)
        self.assertEqual(t["expected"], 900)
        self.assertEqual(t["gaps"], 90)
        self.assertAlmostEqual(t["coverage"], 0.9, places=3)
        self.assertAlmostEqual(t["max_gap_ms"], 22.2, places=1)

    def test_too_few_stamps_give_only_a_count(self):
        self.assertEqual(arena.stamp_timing([1.0, 2.0]), {"count": 2})

    def test_header_stamp_read_from_raw_cdr(self):
        for order, encap in (("<", b"\x00\x01\x00\x00"), (">", b"\x00\x00\x00\x00")):
            raw = encap + struct.pack(order + "iI", 12, 500_000_000) + b"frame_id..."
            self.assertAlmostEqual(arena.header_stamp(raw), 12.5)

    @unittest.skipIf(dashboard is None, f"ROS 2 unavailable: {_DASHBOARD_ERR}")
    def test_header_stamp_matches_real_ros_serialisation(self):
        from rclpy.serialization import serialize_message
        from sensor_msgs.msg import Image, Imu
        for cls in (Image, Imu):
            msg = cls()
            msg.header.stamp.sec, msg.header.stamp.nanosec = 1794734268, 908108000
            self.assertAlmostEqual(arena.header_stamp(serialize_message(msg)),
                                   1794734268.908108, places=6)


class TestTakeBookkeeping(unittest.TestCase):
    INFRA = "/camera/infra1/image_rect_raw"

    def test_complete_streams_have_no_warnings(self):
        meta = {self.INFRA: dict(type="sensor_msgs/msg/Image", count=9000),
                "/tf_static": dict(type="tf2_msgs/msg/TFMessage", count=1)}
        timing = {self.INFRA: arena.stamp_timing([i / 90.0 for i in range(9000)])}
        report, warnings = arena.coverage_report(meta, 100.0, timing, [self.INFRA])
        self.assertEqual(warnings, [])
        self.assertEqual(report[self.INFRA]["coverage"], 1.0)
        self.assertNotIn("coverage", report["/tf_static"])      # latched: no rate

    def test_missing_frames_and_streams_are_named(self):
        meta = {self.INFRA: dict(type="sensor_msgs/msg/Image", count=7200),
                "/visual_slam/tracking/odometry": dict(type="nav_msgs/msg/Odometry", count=10)}
        timing = {self.INFRA: arena.stamp_timing([i / 90.0 for i in range(9000) if i % 5]),
                  # SLAM skipping frames is reported, never warned about
                  "/visual_slam/tracking/odometry": arena.stamp_timing([0, 1, 2, 10])}
        report, warnings = arena.coverage_report(
            meta, 100.0, timing, [self.INFRA, "/camera/imu"])
        self.assertAlmostEqual(report[self.INFRA]["coverage"], 0.8, places=3)
        self.assertEqual(len(warnings), 2, warnings)
        self.assertIn("20.0% of frames missing", warnings[0])
        self.assertIn("/camera/imu: was live before recording but is not in the bag", warnings[1])

    def test_depth_tolerates_more_missing_frames_than_stereo(self):
        depth = "/camera/depth/image_rect_raw"
        ninety = [i / 90.0 for i in range(9000) if i % 10]       # 90% of frames
        meta = {t: dict(type="sensor_msgs/msg/Image", count=8100) for t in (depth, self.INFRA)}
        timing = {t: arena.stamp_timing(ninety) for t in (depth, self.INFRA)}
        _, warnings = arena.coverage_report(meta, 100.0, timing, [])
        self.assertEqual(len(warnings), 1, warnings)
        self.assertIn(self.INFRA, warnings[0])

    def test_reads_rosbag2_metadata(self):
        info = {"rosbag2_bagfile_information": {
            "duration": {"nanoseconds": 12_500_000_000},
            "topics_with_message_count": [
                {"topic_metadata": {"name": "/camera/imu", "type": "sensor_msgs/msg/Imu"},
                 "message_count": 5000}]}}
        with tempfile.TemporaryDirectory() as d:
            _write(d, "metadata.yaml", yaml.safe_dump(info))
            duration, topics = arena.read_bag_metadata(d)
        self.assertEqual(duration, 12.5)
        self.assertEqual(topics, {"/camera/imu": dict(type="sensor_msgs/msg/Imu", count=5000)})

    def test_takes_listed_oldest_first_and_only_finished_ones(self):
        with tempfile.TemporaryDirectory() as root:
            base = arena.inputs_dir(root)
            for take in ("20261115-150000", "20261115-140000", "20261115-160000"):
                os.makedirs(os.path.join(base, arena.take_name("s1_x", take)))
            os.makedirs(os.path.join(base, "s10_y_20261115-150000"))   # another scenario
            for take in ("20261115-150000", "20261115-140000"):
                _write(os.path.join(base, arena.take_name("s1_x", take)), "take.json", "{}")
            _write(os.path.join(base, "s10_y_20261115-150000"), "take.json", "{}")
            names = [os.path.basename(p) for p in arena.list_takes("s1_x", root)]
            self.assertEqual(names, ["s1_x_20261115-140000", "s1_x_20261115-150000"])
            scenarios = [dict(id="s1_x", short="s1")]
            latest = arena.resolve_take("s1", scenarios, root)
            self.assertEqual(os.path.basename(latest), "s1_x_20261115-150000")
            self.assertEqual(arena.resolve_take("s1_x_20261115-140000", scenarios, root),
                             os.path.join(base, "s1_x_20261115-140000"))
            self.assertEqual(arena.reference_abc(latest, root),
                             os.path.join(root, "reference", "s1_x_20261115-150000.abc"))

    def test_free_space_of_a_directory_not_created_yet(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertGreater(arena.free_bytes(os.path.join(d, "a", "b", "c")), 0)

    def test_mmss(self):
        self.assertEqual(arena.mmss(0), "00:00")
        self.assertEqual(arena.mmss(161.9), "02:41")


class TestArenaState(unittest.TestCase):
    STATUS = {
        "state": "VALIDATED", "reason": "Stereo / IMU with depth-to-scan validation",
        "validated": True, "output_allowed": True, "supported_by": ["scan"],
        "sources": {"scan": {"ok": True, "timestamp": np.float64(12.5),
                             "consecutive": np.int64(4), "score": np.float32(0.94),
                             "reason": "", "fresh": True, "ready": True,
                             "matrix": np.eye(4)}},
    }

    def test_heartbeat_record(self):
        rec = arena_state.build_state(100.0, "Waiting", None, None)
        self.assertIsNone(rec["freed"])
        self.assertIsNone(rec["fusion"])
        self.assertEqual(arena_state.decode(arena_state.encode(rec)), rec)

    def test_packet_status_and_pose_survive_json(self):
        pkt = get_freed_packet(1, 2.0, 30.0, -1.0, 1.25, -0.5, 1.6)
        T = np.eye(4)
        T[:3, 3] = [1.25, -0.5, 1.6]
        rec = arena_state.build_state(
            100.0, "Live", (True, "locked", "Auto lineup · Locked"), self.STATUS,
            packet=pkt, packets_sent=np.int64(42), output_T=T)
        back = arena_state.decode(arena_state.encode(rec))
        self.assertAlmostEqual(back["freed"]["x_m"], 1.25, places=3)
        self.assertAlmostEqual(back["freed"]["yaw_deg"], 30.0, places=3)
        self.assertEqual(back["output_T"][3], 1.25)
        self.assertEqual(back["packets_sent"], 42)
        self.assertEqual(back["lineup"], {"enabled": True, "state": "locked",
                                          "title": "Auto lineup · Locked"})
        scan = back["fusion"]["sources"]["scan"]
        self.assertEqual(scan["consecutive"], 4)
        self.assertAlmostEqual(scan["score"], 0.94, places=5)
        self.assertNotIn("matrix", scan)         # non-scalars are left out

    def test_a_bad_packet_is_reported_not_raised(self):
        rec = arena_state.build_state(1.0, "Live", None, None, packet=b"\x00" * 3)
        self.assertIn("error", rec["freed"])

    def test_other_json_is_refused(self):
        with self.assertRaises(ValueError):
            arena_state.decode(json.dumps({"format": "hyperfusion.scene"}))


def _nmea(body):
    checksum = 0
    for c in body:
        checksum ^= ord(c)
    return f"${body}*{checksum:02X}\r\n".encode()


# One RTK-fixed epoch: GGA, then GST with lat/lon/alt sigmas 8/6/15 mm.
GGA = _nmea("GNGGA,123519.00,4807.0380,N,01131.0000,E,4,18,0.6,545.4,M,46.9,M,1.0,0000")
GST = _nmea("GNGST,123519.00,0.5,0.010,0.008,45.0,0.008,0.006,0.015")
RMC = _nmea("GNRMC,123519.00,A,4807.0380,N,01131.0000,E,0.0,,151126,,,R")


def _run_receiver(lines, on_line):
    """Run GNSSReceiver.run() over `lines` from a fake serial port."""
    from unittest import mock
    from gnss_receiver import GNSSReceiver
    fixes, logs = [], []
    receiver = GNSSReceiver({"port": "fake"}, fixes.append, log=logs.append, on_line=on_line)
    pending = list(lines)

    class FakePort:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def readline(self, limit):
            if pending:
                return pending.pop(0)
            receiver.stop_event.set()
            return b""                       # a read timeout

    with mock.patch.dict("sys.modules", {"serial": mock.Mock(Serial=FakePort)}):
        receiver.run()
    return fixes, logs


class TestGnssRecords(unittest.TestCase):
    def test_every_raw_line_reaches_the_hook_and_fixes_still_parse(self):
        lines = []
        fixes, logs = _run_receiver([GGA, RMC, GST], lines.append)
        self.assertEqual(lines, [GGA, RMC, GST])       # RMC too: replay needs it all
        self.assertEqual(len(fixes), 2)                 # GGA, then GGA+GST
        self.assertAlmostEqual(fixes[-1].hacc_m, 0.010, places=6)
        self.assertEqual(logs, [])

    def test_a_failing_hook_never_stops_gnss(self):
        def broken(line):
            raise RuntimeError("publisher gone")
        fixes, logs = _run_receiver([GGA, RMC, GST], broken)
        self.assertEqual(len(fixes), 2)
        self.assertEqual(len(logs), 1)                  # reported once
        self.assertIn("raw NMEA recording stopped", logs[0])

    def _fix(self):
        fixes, _ = _run_receiver([GGA, GST], None)
        return fixes[-1]

    def test_navsat_fields_for_an_rtk_fix(self):
        f = arena_state.navsat_fields(self._fix())
        self.assertEqual(f["status"], 2)                # STATUS_GBAS_FIX
        self.assertAlmostEqual(f["latitude"], 48.1173, places=4)
        self.assertAlmostEqual(f["altitude"], 545.4 + 46.9, places=6)   # ellipsoidal
        self.assertEqual(f["position_covariance_type"], 2)
        self.assertAlmostEqual(f["position_covariance"][0], 0.010 ** 2 / 2, places=9)
        self.assertAlmostEqual(f["position_covariance"][8], 0.015 ** 2, places=9)

    def test_navsat_status_by_gga_quality(self):
        import dataclasses
        fix = self._fix()
        for quality, status in {0: -1, 1: 0, 2: 1, 4: 2, 5: 2, 6: -1}.items():
            with self.subTest(quality=quality):
                f = arena_state.navsat_fields(dataclasses.replace(fix, quality=quality))
                self.assertEqual(f["status"], status)
        f = arena_state.navsat_fields(dataclasses.replace(fix, hacc_m=None))
        self.assertEqual(f["position_covariance_type"], 0)

    def test_gnss_summary_in_state_records(self):
        import dataclasses
        fix = dataclasses.replace(self._fix(), hdop=float("inf"))
        g = arena_state.gnss_summary(fix, fix.received + 0.25)
        self.assertEqual((g["label"], g["satellites"], g["age_s"]), ("RTK fixed", 18, 0.25))
        self.assertIsNone(g["hdop"])                    # no "Infinity" in the JSON
        rec = arena_state.build_state(1.0, "Live", None, None, gnss=g)
        self.assertEqual(arena_state.decode(arena_state.encode(rec))["gnss"], g)
        self.assertIsNone(arena_state.gnss_summary(None, 0.0))
        self.assertEqual(arena.gnss_text(g), "RTK fixed, 18 sats, ±10 mm")
        self.assertEqual(arena.gnss_text(dict(g, age_s=12)), "silent 12s")

    def test_nmea_record(self):
        rec = json.loads(arena_state.encode_nmea(GGA, 1794735000.25))
        self.assertEqual(rec["format"], "hyperfusion.nmea")
        self.assertEqual(rec["t"], 1794735000.25)
        self.assertEqual(rec["line"], GGA.decode().strip())


import arena_export
import arena_replay
import arena_score
from pose_transform import (T_BODY_OPTICAL, invert_transform, make_transform,
                            matrix_to_quaternion, quaternion_to_matrix, rpy_to_matrix)


def _blender_from_abc(p):
    """What Blender's Alembic importer does: Y-up file -> Z-up scene."""
    p = np.asarray(p, float)
    return np.stack([p[..., 0], -p[..., 2], p[..., 1]], axis=-1)


class TestExportFor3DE(unittest.TestCase):
    SCENE = {"alignment": dict(tx=0.4, ty=-0.2, tz=0.1, roll=0.0, pitch=0.0, yaw=30.0,
                               scale=0.021023596065833067),
             "marker_lineup": {"mapped_markers": [
                 dict(dictionary="DICT_4X4_250", id=0, size_m=0.167,
                      position_m=[-1.066, -2.5896, 0.798],
                      rotation_deg=[-22.7, -89.66, -68.78])]}}

    def test_3de_axes_come_back_from_blender_as_dashboard_axes(self):
        p = np.array([[1.0, 2.0, 3.0], [-0.5, 0.25, 1.5]])
        np.testing.assert_allclose(_blender_from_abc(arena_export.ros_to_3de(p)), p)
        self.assertAlmostEqual(arena_export.ros_to_3de([0, 0, 1])[1], 1.0)   # Z-up -> Y-up

    def test_alignment_matches_the_viewer(self):
        T = arena_export.alignment_transform(self.SCENE["alignment"])
        a = self.SCENE["alignment"]
        R = rpy_to_matrix(0.0, 0.0, np.radians(30.0))
        native = np.array([100.0, -50.0, 20.0])
        expected = R @ (native * a["scale"]) + [0.4, -0.2, 0.1]
        np.testing.assert_allclose(T[:3, :3] @ native + T[:3, 3], expected)
        np.testing.assert_allclose(arena_export.rigid_part(T)[:3, :3], R, atol=1e-12)

    def test_survey_corners_match_the_marker_solver(self):
        # marker_map.solve_mapped_observation's own corner maths, by hand.
        from marker_map import locator_transform
        entry = self.SCENE["marker_lineup"]["mapped_markers"][0]
        rigid = arena_export.rigid_part(arena_export.alignment_transform(self.SCENE["alignment"]))
        local = np.eye(4)
        local[:3, :3] = rpy_to_matrix(0, 0, -np.pi / 2)
        T = rigid @ locator_transform(entry) @ local
        s = entry["size_m"] / 2
        corners = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]])
        expected = corners @ T[:3, :3].T + T[:3, 3]
        points = dict(arena_export.survey_points(self.SCENE))
        for k in range(4):
            np.testing.assert_allclose(points[f"DICT_4X4_250_id0_c{k}"], expected[k], atol=1e-12)
        side = np.linalg.norm(points["DICT_4X4_250_id0_c0"] - points["DICT_4X4_250_id0_c1"])
        self.assertAlmostEqual(side, 0.167, places=9)          # true marker size survives

    def test_plate_frames_follow_a_uniform_30_fps_grid(self):
        stamps = [i / 90.0 for i in range(270) if i not in (30, 31, 32)]   # slot 10 lost
        frames = arena_export.pick_plate_frames(stamps)
        self.assertEqual(len(frames), 90)                      # 3 s at 30 fps
        deltas = [abs(d) for _, _, d, _ in frames]
        self.assertLess(max(deltas), 1.5 / 90)                 # nearest frame used at the gap
        self.assertEqual(sum(d > 0.5 / 90 for d in deltas), 1)  # only the dropped slot is off
        idx = [i for i, _, _, _ in frames]
        self.assertEqual(idx, sorted(idx))

    def test_lens_values_for_3de(self):
        lens = arena_export.lens_for_3de(dict(K=[429.948, 0, 423.5, 0, 429.948, 241.7, 0, 0, 1],
                                              width=848, height=480))
        self.assertAlmostEqual(lens["focal_length_cm"] / lens["filmback_width_cm"], 429.948 / 848)
        self.assertAlmostEqual(lens["horizontal_fov_deg"], 89.18, places=1)


class TestReplay(unittest.TestCase):
    def test_what_is_played(self):
        for topic in ("/camera/infra1/image_rect_raw", "/camera/imu", "/camera/depth/camera_info"):
            self.assertTrue(arena_replay.plays(topic), topic)
        for topic in ("/tf", "/tf_static", "/visual_slam/tracking/odometry",
                      "/hyperfusion/state", "/hyperfusion/gnss/nmea", "/rosout"):
            self.assertFalse(arena_replay.plays(topic), topic)

    def test_restamp_shifts_only_the_stamp(self):
        raw = b"\x00\x01\x00\x00" + struct.pack("<iI", 100, 900_000_000) + b"payload"
        out = arena_replay.restamp(raw, 1_250_000_000)          # +1.25 s
        self.assertEqual(struct.unpack_from("<iI", out, 4), (102, 150_000_000))
        self.assertEqual(out[12:], b"payload")
        self.assertEqual(arena.header_stamp(out), 102.15)

    def test_replay_config_cannot_reach_unreal_or_hardware(self):
        with tempfile.TemporaryDirectory() as d:
            base = _write(d, "tracking.yaml", "network: {unreal_ip: 192.168.1.50, unreal_port: 11111, "
                          "send_rate_hz: 60}\ngnss: {enabled: true, port: /dev/ttyACM0}\n"
                          "unreal_video: {enabled: true, port: 11112}\nlidar: {enabled: true}\n")
            cfg = arena_replay.replay_config(base, os.path.join(d, "replay.yaml"))
            with open(os.path.join(d, "replay.yaml")) as fh:
                written = yaml.safe_load(fh)
        self.assertEqual(written, cfg)
        self.assertEqual((cfg["network"]["unreal_ip"], cfg["network"]["unreal_port"]),
                         ("127.0.0.1", arena_replay.DEAD_FREED_PORT))
        self.assertEqual(cfg["network"]["send_rate_hz"], 60)     # the rest untouched
        self.assertFalse(cfg["gnss"]["enabled"])
        self.assertFalse(cfg["unreal_video"]["enabled"])
        self.assertTrue(cfg["lidar"]["enabled"])

    @unittest.skipIf(dashboard is None, f"ROS 2 unavailable: {_DASHBOARD_ERR}")
    def test_header_detection_from_message_definitions(self):
        from sensor_msgs.msg import Image, CameraInfo, Imu
        from std_msgs.msg import String
        from tf2_msgs.msg import TFMessage
        for cls in (Image, CameraInfo, Imu):
            self.assertTrue(arena_replay.header_first(cls), cls)
        for cls in (String, TFMessage):
            self.assertFalse(arena_replay.header_first(cls), cls)


def _trajectory(t):
    """A camera body pose moving 1 m/s along x while panning 10 deg/s."""
    return make_transform([1.0 * t, 0.2, 1.5],
                          matrix_to_quaternion(rpy_to_matrix(0, 0, np.radians(10 * t))))


class TestScoring(unittest.TestCase):
    def _case(self, *, ref_shift=(0, 0, 0), offset=0.0, gap=None, lead=0.0, lag=0.0):
        frames = [dict(frame=1 + k, stamp=100.0 + k / 30.0) for k in range(90)]
        reference = {"frames": []}
        for f in frames:
            T_opt = _trajectory(f["stamp"] - 100.0) @ T_BODY_OPTICAL
            T_opt[:3, 3] += ref_shift
            # As 3DE/Blender would hold it: GL camera (looks -Z, +Y up).
            reference["frames"].append(dict(frame=f["frame"], T=(T_opt @ arena_score.GL_TO_OPTICAL).tolist()))
        records = []
        for k in range(200):                                   # 60 Hz output, replay clock
            t = 100.0 + k / 60.0
            if gap and gap[0] <= t < gap[1]:
                continue
            records.append(dict(t=t + offset - lead, mode="Live",
                                output_T=_trajectory(t - 100.0 - lag).flatten().tolist()))
        marks = [dict(t=100.0, step=1, label="start"), dict(t=101.5, step=2, label="away")]
        return arena_score.score(frames, reference, records, offset_s=offset, lead_s=lead,
                                 marks=marks)

    def test_a_perfect_path_passes(self):
        report, rows = self._case()
        self.assertTrue(report["passed"])
        self.assertEqual(report["pass_ratio"], 1.0)
        self.assertLess(report["pos_mm"]["max"], 0.5)
        self.assertLess(report["rot_deg"]["max"], 0.01)
        self.assertEqual([p["step"] for p in report["phases"]], [1, 2])

    def test_30mm_off_fails_and_the_fit_says_it_is_an_offset(self):
        report, _ = self._case(ref_shift=(0.03, 0, 0))
        self.assertFalse(report["passed"])
        self.assertAlmostEqual(report["pos_mm"]["median"], 30.0, delta=0.5)
        self.assertLess(report["diagnostics"]["best_fit"]["median_mm_after_fit"], 1.0)

    def test_replay_clock_shift_and_lead_time_are_undone(self):
        report, _ = self._case(offset=1_794_735_000.0, lead=0.05)
        self.assertTrue(report["passed"], report)

    def test_a_lagging_output_is_called_out(self):
        report, _ = self._case(lag=0.08)      # output describes 80 ms ago
        self.assertFalse(report["passed"])
        self.assertAlmostEqual(report["diagnostics"]["best_time_shift_ms"], 80, delta=10)
        self.assertTrue(any("lags the 3DE camera by 80 ms" in line
                            for line in arena_score.summary_lines(report, "t")))
        report, _ = self._case()
        self.assertNotIn("best_time_shift_ms", report["diagnostics"])   # no lag, no claim

    def test_frames_with_no_output_fail(self):
        report, rows = self._case(gap=(101.0, 101.5))
        self.assertGreater(report["frames_without_output"], 10)
        self.assertTrue(all(not r["passed"] for r in rows if r["pos_mm"] is None))

    def test_a_centimetre_abc_is_called_out(self):
        frames = [dict(frame=1 + k, stamp=100.0 + k / 30.0) for k in range(90)]
        ref = {"frames": []}
        for f in frames:
            T = _trajectory(f["stamp"] - 100.0) @ T_BODY_OPTICAL @ arena_score.GL_TO_OPTICAL
            T[:3, 3] *= 100.0
            ref["frames"].append(dict(frame=f["frame"], T=T.tolist()))
        records = [dict(t=100.0 + k / 60.0, mode="Live",
                        output_T=_trajectory(k / 60.0).flatten().tolist()) for k in range(200)]
        report, _ = arena_score.score(frames, ref, records)
        self.assertIn("not in metres", report["diagnostics"].get("warning", ""))

    def test_body_to_lens_matches_the_cameras_live_tf(self):
        # ros2 run tf2_ros tf2_echo camera_link camera_infra1_optical_frame (2026-11-15):
        # translation 0, rotation xyzw (-0.5, 0.5, -0.5, 0.5).
        np.testing.assert_allclose(T_BODY_OPTICAL[:3, :3],
                                   quaternion_to_matrix([-0.5, 0.5, -0.5, 0.5]), atol=1e-12)

    def test_rig_offset_is_removed_before_comparing(self):
        cfg = {"transform": {"rig_offset": {"translation_m": [0.1, 0.0, -0.05],
                                            "rotation_deg": [0, 5, 0]}}}
        rig = arena_score.rig_transform(cfg)
        body = _trajectory(0.7)
        output = body @ rig                                     # what the engine sends
        np.testing.assert_allclose(arena_score.dashboard_optical(output, rig),
                                   body @ T_BODY_OPTICAL, atol=1e-12)


class _Pub:
    def __init__(self, fail=False):
        self.sent, self.fail = [], fail

    def publish(self, msg):
        if self.fail:
            raise RuntimeError("publisher gone")
        self.sent.append(msg.data)


class _Engine:
    packets_sent = 7
    _output_T = None

    def fusion_status(self):
        return dict(TestArenaState.STATUS, sources={})


@unittest.skipIf(dashboard is None, f"dashboard unavailable: {_DASHBOARD_ERR}")
class TestDashboardArenaRecords(unittest.TestCase):
    def _app(self, pub):
        app = dashboard.HyperFusionApp.__new__(dashboard.HyperFusionApp)
        app.engine = _Engine()
        app._arena_state_pub = pub
        app._lineup_indicator = (True, "locked", "Auto lineup · Locked")
        app._output_mode = lambda status=None: "Live"
        return app

    def test_each_packet_is_published_with_what_unreal_was_sent(self):
        pub = _Pub()
        app = self._app(pub)
        app._arena_publish(get_freed_packet(1, 0.0, 10.0, 0.0, 1.0, 2.0, 3.0))
        rec = arena_state.decode(pub.sent[0])
        self.assertEqual(rec["mode"], "Live")
        self.assertAlmostEqual(rec["freed"]["z_m"], 3.0, places=3)
        self.assertEqual(rec["lineup"]["state"], "locked")

    def test_a_failing_publisher_never_breaks_the_freed_path(self):
        app = self._app(_Pub(fail=True))
        app._arena_publish(None)                 # must not raise
        self.assertTrue(app._arena_failed)
        app._arena_publish(None)                 # and stays quiet afterwards

    def test_no_publisher_is_a_no_op(self):
        app = self._app(None)
        app._arena_publish(None)
        self.assertFalse(app._arena_failed)

    def _gnss_app(self, pub):
        import time as _time
        fixes, _ = _run_receiver([GGA, GST], None)
        app = self._app(None)
        app.engine = mock_engine = type("E", (), {})()
        mock_engine.seen = []
        mock_engine.update_gnss = lambda fix: mock_engine.seen.append(fix) or True
        app._arena_gnss_fix_pub = pub
        fix = fixes[-1]
        return app, fix, _time

    def test_gnss_fix_reaches_the_engine_and_the_bag(self):
        from sensor_msgs.msg import NavSatFix
        sent = []
        pub = type("P", (), {"publish": lambda self, msg: sent.append(msg)})()
        app, fix, _time = self._gnss_app(pub)
        self.assertTrue(app._on_gnss_fix(fix))
        self.assertEqual(app.engine.seen, [fix])
        msg = sent[0]
        self.assertIsInstance(msg, NavSatFix)
        self.assertEqual(msg.status.status, 2)
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        expected = _time.time() - (_time.monotonic() - fix.received)
        self.assertAlmostEqual(stamp, expected, delta=0.5)

    def test_gnss_publishing_failure_leaves_the_engine_fed(self):
        app, fix, _ = self._gnss_app(_Pub(fail=True))
        app._on_gnss_fix(fix)
        app._on_gnss_fix(fix)
        self.assertEqual(len(app.engine.seen), 2)
        self.assertTrue(app._arena_gnss_failed)

    def test_raw_nmea_line_is_published(self):
        pub = _Pub()
        app = self._app(None)
        app._arena_gnss_nmea_pub = pub
        app._arena_gnss_line(RMC)
        self.assertEqual(json.loads(pub.sent[0])["line"], RMC.decode().strip())


if __name__ == "__main__":
    unittest.main()
