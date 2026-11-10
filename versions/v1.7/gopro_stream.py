"""GoPro webcam preview over USB, shown beside the 3D view.

A HERO10 on USB is a network adapter. Its Open GoPro API listens at
172.2X.1YZ.51:8080, where XYZ are the last three digits of its serial, and the
PC gets an address on the same /24. Webcam mode then sends MPEG-TS to the PC
on UDP 8554, about 5 Mbit/s at 720p.

Display only: nothing here reaches tracking or FreeD. Everything that can
block -- HTTP, opening the stream, decoding -- runs on one worker thread, and
the GUI only takes the newest frame.
"""
import fcntl
import json
import os
import re
import socket
import struct
import threading
import time
import urllib.request


API_PORT = 8080
WEBCAM_PORT = 8554
RESOLUTIONS = {"720": 7, "1080": 12}
FOVS = {"wide": 0, "narrow": 2, "linear": 4}
#: /gopro/webcam/status values that mean the camera is already sending.
STREAMING = (2, 3)        # high-power preview, low-power preview

# FFmpeg's default probe reads ~5 MB before the first frame: at the webcam's
# ~5 Mbit/s that is 8 s, longer than the open timeout, and the open fails
# although packets are arriving. Measured here: 1 MB opens in about 1 s.
CAPTURE_OPTIONS = "fflags;nobuffer|flags;low_delay|probesize;1000000|analyzeduration;1000000"

# The PC's end of the camera's USB network.
_USB_LINK = re.compile(r"172\.2\d\.1\d\d\.\d{1,3}$")


class GoProError(Exception):
    pass


def camera_host(addresses):
    """The camera's API address, from this PC's addresses; None if not linked."""
    for address in addresses:
        if _USB_LINK.match(address):
            return address.rsplit(".", 1)[0] + ".51"
    return None


def local_ipv4_addresses():
    """Every IPv4 address on this machine's interfaces (Linux)."""
    found = []
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        for _, name in socket.if_nameindex():
            try:
                reply = fcntl.ioctl(s.fileno(), 0x8915,        # SIOCGIFADDR
                                    struct.pack("256s", name[:15].encode()))
            except OSError:
                continue          # no IPv4 address on this interface
            found.append(socket.inet_ntoa(reply[20:24]))
    return found


class GoProClient:
    """The few Open GoPro HTTP calls the preview needs."""

    def __init__(self, host, timeout=4.0, urlopen=urllib.request.urlopen):
        self.base = f"http://{host}:{API_PORT}"
        self.timeout = timeout
        self._urlopen = urlopen

    def _get(self, path):
        with self._urlopen(self.base + path, timeout=self.timeout) as reply:
            body = reply.read()
        return json.loads(body) if body.strip() else {}

    def webcam_status(self):
        return int(self._get("/gopro/webcam/status").get("status", -1))

    def start_webcam(self, resolution, fov):
        reply = self._get(f"/gopro/webcam/start?res={RESOLUTIONS[resolution]}&fov={FOVS[fov]}")
        if reply.get("error", 0):
            raise GoProError(f"the camera refused webcam mode (error {reply['error']})")

    def stop_webcam(self):
        self._get("/gopro/webcam/stop")

    def keep_alive(self):
        self._get("/gopro/camera/keep_alive")


def open_capture(port=WEBCAM_PORT):
    """The webcam's UDP stream as an OpenCV capture."""
    import cv2
    # Read by OpenCV at open time, process-wide; nothing else here opens FFmpeg.
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = CAPTURE_OPTIONS
    return cv2.VideoCapture(f"udp://@0.0.0.0:{port}?overrun_nonfatal=1", cv2.CAP_FFMPEG,
                            [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000,
                             cv2.CAP_PROP_READ_TIMEOUT_MSEC, 2000])


class GoProStream:
    """Keep the newest GoPro frame, reconnecting until stopped."""

    RETRY_S = 3.0
    KEEP_ALIVE_S = 3.0

    def __init__(self, config=None, addresses=local_ipv4_addresses,
                 client=GoProClient, capture=open_capture, clock=time.monotonic):
        cfg = config or {}
        self.resolution = str(cfg.get("resolution", "720"))
        self.fov = str(cfg.get("fov", "linear"))
        if self.resolution not in RESOLUTIONS:
            raise ValueError(f"gopro.resolution must be one of {', '.join(RESOLUTIONS)}")
        if self.fov not in FOVS:
            raise ValueError(f"gopro.fov must be one of {', '.join(FOVS)}")
        self.host = str(cfg.get("host", "auto"))
        self._addresses = addresses
        self._client = client
        self._capture = capture
        self._clock = clock
        self.status = "GoPro preview off"
        self._lock = threading.Lock()
        self._frame = None
        self._seq = 0
        self._stop = threading.Event()
        self._thread = None

    @property
    def running(self):
        return self._thread is not None and self._thread.is_alive()

    def start(self):
        if self.running:
            if not self._stop.is_set():
                return
            self._thread.join()       # still leaving webcam mode, about a second
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="gopro-stream", daemon=True)
        self._thread.start()

    def stop(self, timeout=8.0):
        """Stop the preview and the camera's webcam mode, which heats it.

        timeout=0 returns at once and leaves the worker to finish: the GUI
        thread also sends FreeD, which must not stall while the camera answers.
        """
        if not self.running:
            return
        self.status = "Stopping GoPro webcam…"
        self._stop.set()
        self._thread.join(timeout)

    def latest(self):
        """(sequence, frame); the sequence changes whenever the frame does."""
        with self._lock:
            return self._seq, self._frame

    def _publish(self, frame):
        with self._lock:
            self._frame = frame
            self._seq += 1

    def _find_host(self):
        return camera_host(self._addresses()) if self.host == "auto" else self.host

    def _run(self):
        client = None
        try:
            while not self._stop.is_set():
                host = self._find_host()
                if host is None:
                    self.status = "GoPro not found on USB. Connect it and switch it on."
                else:
                    client = self._client(host)
                    try:
                        if client.webcam_status() not in STREAMING:
                            self.status = "Starting GoPro webcam…"
                            client.start_webcam(self.resolution, self.fov)
                        self.status = "Opening GoPro video…"
                        self._read(client)
                    except (OSError, ValueError, GoProError) as exc:
                        self.status = f"GoPro: {exc}"
                    self._publish(None)
                self._stop.wait(self.RETRY_S)
        finally:
            if client is not None:
                try:
                    client.stop_webcam()
                except (OSError, ValueError):
                    pass          # unplugged: nothing left to stop
            self._publish(None)
            self.status = "GoPro preview off"

    def _read(self, client):
        capture = self._capture()
        try:
            if not capture.isOpened():
                raise OSError("no video arrived from the camera")
            frames, since = 0, self._clock()
            next_keep_alive = since + self.KEEP_ALIVE_S
            while not self._stop.is_set():
                ok, frame = capture.read()
                if not ok:
                    raise OSError("the video stopped")
                self._publish(frame)
                frames += 1
                now = self._clock()
                if now - since >= 1.0:
                    h, w = frame.shape[:2]
                    self.status = f"GoPro · {w}x{h} · {frames / (now - since):.0f} fps · {self.fov}"
                    frames, since = 0, now
                if now >= next_keep_alive:
                    client.keep_alive()
                    next_keep_alive = now + self.KEEP_ALIVE_S
        finally:
            capture.release()
