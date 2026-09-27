import unittest

from check_pipeline import StreamHealth


class StreamHealthTests(unittest.TestCase):
    def test_continuous_live_samples_become_ready_then_expire(self):
        stream = StreamHealth()
        for i in range(7):
            stream.observe(100 + i * 0.2, 10 + i * 0.2)
        self.assertTrue(stream.ready(11.3))
        self.assertAlmostEqual(stream.hz, 5.0)
        self.assertFalse(stream.ready(12.0))

    def test_repeated_timestamp_cannot_make_a_frozen_feed_ready(self):
        stream = StreamHealth()
        for i in range(20):
            stream.observe(100.0, 10 + i * 0.1)
        self.assertFalse(stream.ready(12.0))
        self.assertEqual(stream.count, 1)

    def test_startup_burst_does_not_count_as_sustained_streaming(self):
        stream = StreamHealth()
        for i in range(20):
            stream.observe(100 + i * 0.01, 10 + i * 0.001)
        self.assertFalse(stream.ready(10.02))

    def test_gap_requires_new_stable_window(self):
        stream = StreamHealth()
        for i in range(7):
            stream.observe(100 + i * 0.2, 10 + i * 0.2)
        stream.observe(103, 13)
        self.assertFalse(stream.ready(13.1))
        self.assertEqual(stream.count, 1)

    def test_invalid_data_and_out_of_order_stamps_are_ignored(self):
        stream = StreamHealth()
        for stamp in (float("nan"), float("inf"), 0, -1):
            stream.observe(stamp, 10)
        stream.observe(100, 10, valid=False)
        self.assertFalse(stream.ready(10))
        self.assertEqual(stream.count, 0)
        stream.observe(100, 10)
        stream.observe(99, 10.1)
        self.assertEqual(stream.count, 1)

    def test_live_images_do_not_hide_missing_imu(self):
        streams = [StreamHealth(), StreamHealth()]
        for i in range(7):
            streams[0].observe(100 + i * 0.2, 10 + i * 0.2)
        self.assertFalse(all(s.ready(11.3) for s in streams))


if __name__ == "__main__":
    unittest.main()
