import json
import unittest
from types import SimpleNamespace
from realsense_streams import FrameJoin, REQUIRED, emitter_state, exposure_state, route_exposure


class EmitterTests(unittest.TestCase):
    def test_switch_keeps_passive_depth_and_rejects_projected_tracking(self):
        for state, off, expected in ((0, False, (True, False)),
                                     (1, False, (False, True)),
                                     (0, True, (True, True)),
                                     (1, True, (False, True))):
            parts = {s + "_metadata": SimpleNamespace(
                json_data=json.dumps({"frame_emitter_mode": state}))
                for s in ("infra1", "infra2", "depth")}
            self.assertEqual(route_exposure(parts, off), expected)
            parts["depth_metadata"].json_data = '{}'
            self.assertEqual(route_exposure(parts, off), (False, False))

    def test_only_explicit_current_metadata_is_trusted(self):
        for state in (0, 1):
            self.assertEqual(emitter_state(json.dumps({"frame_emitter_mode": state})), state)
        for payload in ("", "not json", "[]", "null", '{"frame_laser_power_mode": 0}',
                        '{"frame_emitter_mode": true}', '{"frame_emitter_mode": "0"}',
                        '{"frame_emitter_mode": 2}', '{"frame_emitter_mode": -1}'):
            self.assertIsNone(emitter_state(payload), payload)

    def test_all_streams_must_agree(self):
        parts = {s + "_metadata": SimpleNamespace(json_data='{"frame_emitter_mode": 0}')
                 for s in ("infra1", "infra2", "depth")}
        self.assertEqual(exposure_state(parts), 0)
        parts["depth_metadata"].json_data = '{"frame_emitter_mode": 1}'
        self.assertIsNone(exposure_state(parts))
        parts["depth_metadata"].json_data = '{}'
        self.assertIsNone(exposure_state(parts))


class JoinTests(unittest.TestCase):
    def test_interleaved_arrival_preserves_exposure_and_never_uses_parity(self):
        join = FrameJoin(REQUIRED)
        for key in reversed(REQUIRED[:-1]):
            self.assertIsNone(join.add(key, 101, (key, 101), now=0))
        self.assertIsNone(join.add(REQUIRED[-1], 102, ("wrong exposure", 102), now=.01))
        result = join.add(REQUIRED[-1], 101, ("correct", 101), now=.02)
        self.assertEqual(set(result), set(REQUIRED))
        self.assertTrue(all(value[1] == 101 for value in result.values()))
        self.assertIsNone(join.add(REQUIRED[-1], 101, "duplicate", now=.03))

    def test_preview_joins_image_depth_in_either_arrival_order(self):
        for first, second in (("image", "depth"), ("depth", "image")):
            join = FrameJoin(("image", "depth"))
            self.assertIsNone(join.add(first, 100, first, now=0))
            result = join.add(second, 100, second, now=.01)
            self.assertEqual(result, {"image": "image", "depth": "depth"})

    def test_missing_metadata_and_late_messages_do_not_complete_old_exposure(self):
        join = FrameJoin(("image", "metadata"), max_age_s=.25)
        join.add("image", 10, "old image", now=0)
        self.assertIsNone(join.add("metadata", 10, "late metadata", now=.3))
        self.assertEqual(join.dropped, 1)
        join.add("image", 11, "new image", now=.31)
        self.assertEqual(join.add("metadata", 11, "current", now=.32)["image"], "new image")
        self.assertIsNone(join.add("image", 10, "old image", now=.33))

    def test_pending_frames_are_bounded_when_a_stream_is_missing(self):
        join = FrameJoin(("image", "metadata"), max_pending=4)
        for stamp in range(1, 100):
            join.add("image", stamp, stamp, now=0)
        self.assertEqual(len(join.pending), 4)
        self.assertEqual(join.dropped, 95)

    def test_newer_complete_exposure_prevents_backwards_output(self):
        join = FrameJoin(("image", "depth"))
        join.add("image", 10, "older", now=0)
        join.add("depth", 11, "depth", now=.01)
        self.assertIsNotNone(join.add("image", 11, "newer", now=.02))
        self.assertIsNone(join.add("depth", 10, "older", now=.03))


if __name__ == "__main__":
    unittest.main()
