"""Read-only ZED-F9P GGA/GST input. Altitude = MSL + geoid (ellipsoidal)."""
from dataclasses import dataclass
import math
import threading
import time
import numpy as np


@dataclass(frozen=True)
class GNSSFix:
    epoch: str
    latitude: float
    longitude: float
    altitude_m: float
    quality: int
    satellites: int
    hdop: float
    correction_age_s: float | None
    hacc_m: float | None
    vacc_m: float | None
    received: float


def wgs84_to_ecef(lat, lon, alt):
    lat, lon = math.radians(lat), math.radians(lon)
    e2 = 6.6943799901413165e-3
    n = 6378137.0 / math.sqrt(1 - e2 * math.sin(lat) ** 2)
    return np.array(((n + alt) * math.cos(lat) * math.cos(lon),
                     (n + alt) * math.cos(lat) * math.sin(lon),
                     (n * (1 - e2) + alt) * math.sin(lat)))


def local_position(fix, cfg):
    """Antenna position in displayed ROS world metres. No first-fix origin."""
    origin = cfg.get("origin_lla")
    if not cfg.get("calibrated", False) or origin is None:
        raise ValueError("GNSS needs a surveyed origin, ENU heading and antenna offset")
    lat, lon, alt = map(float, origin)
    if not all(map(math.isfinite, (lat, lon, alt))) or abs(lat) > 90 or abs(lon) > 180:
        raise ValueError("Invalid surveyed GNSS origin")
    a, b = math.radians(lat), math.radians(lon)
    delta = wgs84_to_ecef(fix.latitude, fix.longitude, fix.altitude_m) - wgs84_to_ecef(lat, lon, alt)
    enu = np.array(((-math.sin(b), math.cos(b), 0),
                    (-math.sin(a)*math.cos(b), -math.sin(a)*math.sin(b), math.cos(a)),
                    (math.cos(a)*math.cos(b), math.cos(a)*math.sin(b), math.sin(a)))) @ delta
    yaw = math.radians(float(cfg.get("enu_yaw_deg", 0)))
    rotation = np.array(((math.cos(yaw), -math.sin(yaw), 0),
                         (math.sin(yaw), math.cos(yaw), 0), (0, 0, 1)))
    return rotation @ enu + np.asarray(cfg.get("origin_world_m", [0, 0, 0]), dtype=float)


def quality_error(fix, cfg, now):
    if not 0 <= now - fix.received <= cfg.get("stale_timeout_s", 0.8):
        return "GNSS sample stale"
    if fix.quality != 4:
        return "RTK fixed required; check corrections and sky view"
    if fix.satellites < cfg.get("min_satellites", 8):
        return "Too few satellites"
    if not math.isfinite(fix.hdop) or not 0 < fix.hdop <= cfg.get("max_hdop", 2.0):
        return "Poor satellite geometry"
    if fix.correction_age_s is None or not 0 <= fix.correction_age_s <= cfg.get("max_correction_age_s", 2.0):
        return "RTCM correction age missing or too old"
    if fix.hacc_m is None or fix.vacc_m is None:
        return "Accuracy unavailable; enable NMEA GST alongside GGA"
    if not 0 < fix.hacc_m <= cfg.get("max_hacc_m", 0.05) or not 0 < fix.vacc_m <= cfg.get("max_vacc_m", 0.10):
        return "Position accuracy outside tracking limits"
    return ""


class NMEAParser:
    def __init__(self):
        self.gga = {}
        self.gst = {}
        self.last_epoch = None

    def feed(self, line, now=None):
        now = time.monotonic() if now is None else now
        try:
            if isinstance(line, bytes):
                line = line.decode("ascii")
            body, check = line.strip().split("*")
            if not body.startswith("$") or len(check) != 2:
                return None
            checksum = 0
            for c in body[1:]:
                checksum ^= ord(c)
            if checksum != int(check, 16):
                return None
            p = body[1:].split(",")
            kind, epoch = p[0][-3:], p[1]
            if not epoch or not math.isfinite(float(epoch)):
                return None
            epoch = f"{float(epoch):010.3f}"
            if kind == "GST":
                errors = [float(p[i]) for i in (6, 7, 8)]
                if not all(math.isfinite(e) and e > 0 for e in errors):
                    return None
                self.gst = {epoch: (math.hypot(*errors[:2]), errors[2])}
                if epoch not in self.gga:
                    return None
                p, now = self.gga[epoch]
            elif kind == "GGA":
                p, now = self.gga.get(epoch, (p, now))
                self.gga = {epoch: (p, now)}
            else:
                return None
            if epoch == self.last_epoch:
                return None
            quality = int(p[6] or 0)
            def coordinate(value, hemisphere, limit):
                degrees, minutes = divmod(float(value), 100)
                result = degrees + minutes / 60
                allowed = ("N", "S") if limit == 90 else ("E", "W")
                if hemisphere not in allowed or not 0 <= minutes < 60 or not 0 <= result <= limit:
                    raise ValueError("invalid coordinate")
                return -result if hemisphere in ("S", "W") else result
            if quality == 0:
                lat = lon = alt = 0.0
            else:
                lat = coordinate(p[2], p[3], 90)
                lon = coordinate(p[4], p[5], 180)
                if p[10] != "M" or p[12] != "M":
                    return None
                alt = float(p[9]) + float(p[11])
                if not all(map(math.isfinite, (lat, lon, alt))):
                    return None
            accuracy = self.gst.get(epoch, (None, None))
            if accuracy[0] is not None:
                self.last_epoch = epoch
            return GNSSFix(epoch, lat, lon, alt, quality, int(p[7] or 0),
                           float(p[8] or "inf"), float(p[13]) if len(p) > 13 and p[13] else None,
                           *accuracy, now)
        except (ValueError, IndexError, UnicodeError, OverflowError):
            return None


class GNSSReceiver(threading.Thread):
    def __init__(self, cfg, on_fix, log=print):
        super().__init__(name="gnss-receiver", daemon=True)
        self.cfg, self.on_fix, self.log = cfg, on_fix, log
        self.stop_event = threading.Event()
        self.status = "Connecting"

    def stop(self):
        self.stop_event.set()
        self.join(timeout=2.0)

    def run(self):
        try:
            import serial
            with serial.Serial(self.cfg["port"], int(self.cfg.get("baud", 115200)), timeout=0.25) as port:
                self.status = "Connected; waiting for NMEA GGA / GST"
                parser = NMEAParser()
                while not self.stop_event.is_set():
                    fix = parser.feed(port.readline(1024))
                    if fix is not None:
                        self.status = "Receiving NMEA"
                        self.on_fix(fix)
        except Exception as exc:
            self.status = f"GNSS unavailable: {exc}"
            self.log(self.status)
