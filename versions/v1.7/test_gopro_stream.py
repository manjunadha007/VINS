"""Exercise the GoPro preview's connect/reconnect logic without a camera."""
import io
import json
import threading
import time
import unittest

import numpy as np

from gopro_stream import GoProClient, GoProError, GoProStream, camera_host, local_ipv4_addresses


def wait_for(condition, timeout=2.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if condition():
            return True
        time.sleep(0.005)
    return False


class Client:
    """Records calls; `status` is what /gopro/webcam/status reports."""

    def __init__(self, status=1):
        self.status = status
        self.calls = []
        self.lock = threading.Lock()

    def __call__(self, host):
        self.calls.append(("host", host))
        return self

    def webcam_status(self):
        self.calls.append(("status",))
        return self.status

    def start_webcam(self, resolution, fov):
        self.calls.append(("start", resolution, fov))
        self.status = 2

    def stop_webcam(self):
        self.calls.append(("stop",))
        self.status = 1

    def keep_alive(self):
        self.calls.append(("keep_alive",))

    def names(self):
        return [call[0] for call in self.calls]


class Capture:
    """Serves `frames` frames, then fails like a stream that stopped."""

    def __init__(self, frames=None, opened=True):
        self.frames = frames
        self.opened = opened
        self.reads = 0
        self.released = False

    def isOpened(self):
        return self.opened

    def read(self):
        if self.frames is not None and self.reads >= self.frames:
            return False, None
        self.reads += 1
        time.sleep(0.001)
        return True, np.full((720, 1280, 3), self.reads % 255, np.uint8)

    def release(self):
        self.released = True


class CameraHost(unittest.TestCase):
    def test_camera_is_dot_51_on_the_usb_link(self):
        self.assertEqual(camera_host(["127.0.0.1", "10.141.171.78", "172.28.136.52"]),
                         "172.28.136.51")

    def test_no_usb_link_means_no_camera(self):
        self.assertIsNone(camera_host(["127.0.0.1", "172.17.0.1", "192.168.1.20"]))

    def test_local_addresses_include_loopback(self):
        self.assertIn("127.0.0.1", local_ipv4_addresses())


class Reply(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class ClientRequests(unittest.TestCase):
    def client(self, body):
        self.urls = []

        def urlopen(url, timeout):
            self.urls.append(url)
            return Reply(json.dumps(body).encode())
        return GoProClient("172.28.136.51", urlopen=urlopen)

    def test_start_asks_for_resolution_and_lens(self):
        self.client({"status": 2, "error": 0}).start_webcam("720", "linear")
        self.assertEqual(self.urls, ["http://172.28.136.51:8080/gopro/webcam/start?res=7&fov=4"])

    def test_refused_start_is_an_error(self):
        with self.assertRaisesRegex(GoProError, "error 5"):
            self.client({"status": 1, "error": 5}).start_webcam("1080", "wide")

    def test_status_is_read_from_the_reply(self):
        self.assertEqual(self.client({"status": 2, "error": 0}).webcam_status(), 2)


class Stream(unittest.TestCase):
    def stream(self, client, captures, addresses=("172.28.136.52",), **cfg):
        captures = list(captures)
        self.opened = []

        def capture():
            self.opened.append(captures.pop(0) if captures else Capture(opened=False))
            return self.opened[-1]
        stream = GoProStream(cfg, addresses=lambda: list(addresses),
                             client=client, capture=capture)
        stream.RETRY_S = 0.01
        self.addCleanup(stream.stop)
        return stream

    def test_starts_webcam_then_publishes_frames(self):
        client = Client(status=1)
        stream = self.stream(client, [Capture()], resolution=1080, fov="wide")
        stream.start()
        self.assertTrue(wait_for(lambda: stream.latest()[1] is not None))
        self.assertEqual(client.calls[:3], [("host", "172.28.136.51"), ("status",),
                                            ("start", "1080", "wide")])
        self.assertEqual(stream.latest()[1].shape, (720, 1280, 3))

    def test_already_streaming_camera_is_not_restarted(self):
        client = Client(status=2)
        stream = self.stream(client, [Capture()])
        stream.start()
        self.assertTrue(wait_for(lambda: stream.latest()[1] is not None))
        self.assertNotIn("start", client.names())

    def test_stop_ends_webcam_mode_and_clears_the_frame(self):
        client = Client()
        stream = self.stream(client, [Capture()])
        stream.start()
        self.assertTrue(wait_for(lambda: stream.latest()[1] is not None))
        stream.stop()
        self.assertFalse(stream.running)
        self.assertEqual(client.names()[-1], "stop")
        self.assertIsNone(stream.latest()[1])
        self.assertTrue(self.opened[0].released)
        self.assertEqual(stream.status, "GoPro preview off")

    def test_stopped_video_reconnects(self):
        client = Client()
        stream = self.stream(client, [Capture(frames=3), Capture()])
        stream.start()
        self.assertTrue(wait_for(lambda: len(self.opened) == 2 and stream.latest()[1] is not None))
        self.assertTrue(self.opened[0].released)

    def test_missing_camera_is_reported_and_retried(self):
        addresses = []
        client = Client()
        stream = self.stream(client, [Capture()], addresses=addresses)
        stream.start()
        self.assertTrue(wait_for(lambda: "not found" in stream.status))
        addresses.append("172.28.136.52")
        self.assertTrue(wait_for(lambda: stream.latest()[1] is not None))

    def test_no_video_is_reported(self):
        client = Client()
        stream = self.stream(client, [Capture(opened=False)])
        stream.RETRY_S = 10.0
        stream.start()
        self.assertTrue(wait_for(lambda: "no video" in stream.status))
        self.assertIsNone(stream.latest()[1])

    def test_status_shows_size_and_rate(self):
        clock = [0.0]
        client = Client()
        stream = self.stream(client, [Capture()])
        stream._clock = lambda: clock[0]
        stream.start()
        self.assertTrue(wait_for(lambda: stream.latest()[0] > 5))
        clock[0] = 3.5          # past both the 1 s rate window and keep-alive
        self.assertTrue(wait_for(lambda: "1280x720" in stream.status))
        self.assertIn("keep_alive", client.names())

    def test_explicit_host_skips_discovery(self):
        client = Client()
        stream = self.stream(client, [Capture()], addresses=(), host="10.0.0.9")
        stream.start()
        self.assertTrue(wait_for(lambda: stream.latest()[1] is not None))
        self.assertEqual(client.calls[0], ("host", "10.0.0.9"))

    def test_restart_while_stopping_waits_for_the_old_worker(self):
        client = Client()
        stream = self.stream(client, [Capture(), Capture()])
        stream.start()
        self.assertTrue(wait_for(lambda: stream.latest()[1] is not None))
        stream.stop(timeout=0)
        stream.start()
        self.assertTrue(stream.running)
        self.assertTrue(wait_for(lambda: len(self.opened) == 2 and stream.latest()[1] is not None))

    def test_bad_settings_are_refused(self):
        with self.assertRaises(ValueError):
            GoProStream({"resolution": 480})
        with self.assertRaises(ValueError):
            GoProStream({"fov": "superview"})


if __name__ == "__main__":
    unittest.main()
