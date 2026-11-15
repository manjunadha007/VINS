"""What the dashboard actually did, one JSON record per FreeD packet.

The testing arena (arena.py) records these beside the raw sensors, so every
take carries its own live result: what Unreal was sent, whether output was
Live / Held / Recovering, which evidence (scan, markers) supported it, and what
the Auto lineup button said. A replay of the same take publishes the same
records, so one scorer reads both and live-vs-replay is a like-for-like diff.

Two topics, both std_msgs/String holding JSON:

* STATE_TOPIC -- one record per FreeD packet sent, plus a heartbeat (with
  "freed": null) while nothing is being sent, so Waiting / Paused stretches are
  visible rather than silent gaps.
* SCENE_TOPIC -- the scene exactly as File > Save would write it, published
  transient-local whenever it changes, so a recorder that starts later still
  receives the scan alignment, locators and marker layout the take ran with.

* GNSS_NMEA_TOPIC -- every raw line the GNSS receiver sends, as read, with the
  host time it arrived. This is what a replay feeds back into NMEAParser.
* GNSS_FIX_TOPIC -- sensor_msgs/NavSatFix per parsed fix (fields from
  navsat_fields()), for viewers and maps. GGA quality (RTK fixed vs float) does
  not fit NavSatFix, so it travels in each state record's "gnss" instead.

Kept free of Qt and ROS so it can be unit-tested anywhere.
"""
import json
import math
import numbers

from freed import decode_freed_packet

STATE_TOPIC = "/hyperfusion/state"
SCENE_TOPIC = "/hyperfusion/scene"
GNSS_NMEA_TOPIC = "/hyperfusion/gnss/nmea"
GNSS_FIX_TOPIC = "/hyperfusion/gnss/fix"
STATE_FORMAT = "hyperfusion.arena_state"
STATE_VERSION = 1
NMEA_FORMAT = "hyperfusion.nmea"

# NMEA GGA fix quality.
GGA_QUALITY = {0: "no fix", 1: "GPS", 2: "DGPS", 3: "PPS", 4: "RTK fixed",
               5: "RTK float", 6: "dead reckoning", 7: "manual", 8: "simulation"}
# GGA quality -> sensor_msgs/NavSatStatus.status. RTK is ground-based
# augmentation (STATUS_GBAS_FIX); anything that is not a satellite fix is NO_FIX.
_NAVSAT_STATUS = {1: 0, 2: 1, 3: 0, 4: 2, 5: 2}
NAVSAT_NO_FIX = -1
COVARIANCE_TYPE_UNKNOWN = 0
COVARIANCE_TYPE_DIAGONAL_KNOWN = 2

# Evidence sources reported by FusionSupervisor.evaluate().
_SOURCES = ("scan", "markers", "gnss")


def _plain(value):
    """A JSON-safe copy of a scalar, or None for anything that is not one."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return float(value)
    return None


def _finite(value):
    """A float, or None for a missing / infinite / NaN reading."""
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def gnss_summary(fix, now_mono):
    """What the engine's latest GNSS fix said, for a state record.

    fix      -- gnss_receiver.GNSSFix (its `received` is time.monotonic())
    now_mono -- time.monotonic() at the time of the record
    """
    if fix is None:
        return None
    return {
        "quality": int(fix.quality),
        "label": GGA_QUALITY.get(fix.quality, f"quality {fix.quality}"),
        "satellites": int(fix.satellites),
        "hdop": _finite(fix.hdop),
        "hacc_m": _finite(fix.hacc_m),
        "vacc_m": _finite(fix.vacc_m),
        "correction_age_s": _finite(fix.correction_age_s),
        "age_s": round(now_mono - fix.received, 3),
        "epoch": fix.epoch,
        "lat": fix.latitude, "lon": fix.longitude, "alt_m": fix.altitude_m,
    }


def navsat_fields(fix):
    """sensor_msgs/NavSatFix field values for one GNSSFix.

    Altitude is already ellipsoidal (MSL + geoid), as NavSatFix requires. GST
    gives latitude and longitude sigmas, which the parser folds into one
    horizontal hacc = hypot(lat, lon); each axis gets half its square.
    """
    if fix.hacc_m is not None and fix.vacc_m is not None:
        h = fix.hacc_m ** 2 / 2
        cov, cov_type = [h, 0., 0., 0., h, 0., 0., 0., fix.vacc_m ** 2], COVARIANCE_TYPE_DIAGONAL_KNOWN
    else:
        cov, cov_type = [0.] * 9, COVARIANCE_TYPE_UNKNOWN
    return dict(status=_NAVSAT_STATUS.get(fix.quality, NAVSAT_NO_FIX),
                latitude=float(fix.latitude), longitude=float(fix.longitude),
                altitude=float(fix.altitude_m),
                position_covariance=cov, position_covariance_type=cov_type)


def encode_nmea(line, t):
    """One raw receiver line (bytes) and its host arrival time, as JSON."""
    if isinstance(line, bytes):
        line = line.decode("ascii", errors="replace")
    return json.dumps({"format": NMEA_FORMAT, "t": float(t), "line": line.strip("\r\n")},
                      separators=(",", ":"))


def _source_summary(source):
    """The scalar fields of one evidence source (ok, age_s, score, rank, ...)."""
    return {key: _plain(val) for key, val in source.items()
            if _plain(val) is not None or val is None}


def build_state(now, mode, lineup, status, packet=None, packets_sent=0,
                output_T=None, gnss=None):
    """One arena state record.

    now           -- host time.time() of this record (the bag's clock)
    mode          -- the dashboard's FreeD output mode: Live, Held, Recovering,
                     "Live · unverified", Waiting or Paused
    lineup        -- (checked, state, title) of the Auto lineup button, or None
    status        -- TrackingEngine.fusion_status(), or None
    packet        -- the FreeD packet just sent, or None for a heartbeat
    packets_sent  -- engine.packets_sent
    output_T      -- the engine's 4x4 output pose in the scan frame, or None
    gnss          -- gnss_summary() of the engine's latest fix, or None
    """
    record = {
        "format": STATE_FORMAT,
        "version": STATE_VERSION,
        "t": float(now),
        "mode": mode,
        "packets_sent": int(packets_sent),
        "freed": None,
        "output_T": None,
        "lineup": None,
        "fusion": None,
        "gnss": gnss,
    }
    if packet is not None:
        try:
            record["freed"] = decode_freed_packet(packet)
        except ValueError as exc:
            # Never let a bad packet take down the FreeD path; say what it was.
            record["freed"] = {"error": str(exc)}
    if output_T is not None:
        record["output_T"] = [round(float(v), 6) for row in output_T for v in row]
    if lineup is not None:
        checked, state, title = lineup
        record["lineup"] = {"enabled": bool(checked), "state": state, "title": title}
    if status is not None:
        sources = status.get("sources", {})
        record["fusion"] = {
            "state": status.get("state"),
            "reason": status.get("reason"),
            "validated": bool(status.get("validated", False)),
            "output_allowed": bool(status.get("output_allowed", False)),
            "supported_by": list(status.get("supported_by", [])),
            "sources": {name: _source_summary(sources[name])
                        for name in _SOURCES if name in sources},
        }
    return record


def encode(record):
    """Compact JSON for a std_msgs/String payload."""
    return json.dumps(record, separators=(",", ":"))


def decode(text):
    """Inverse of encode(); raises ValueError on a record of another format."""
    record = json.loads(text)
    if record.get("format") != STATE_FORMAT:
        raise ValueError(f"not an arena state record: {record.get('format')!r}")
    return record
