#!/usr/bin/env python3
"""
FreeD protocol encoder/decoder — the single source of truth for this project.

Every bridge in this repo (Isaac ROS dashboard, VINS-Fusion bridge, Spectacular AI
bridge, raw RealSense prototype, and the packet-injection test tool) encodes the
same 29-byte FreeD "D1" camera-position message. That encoder used to be
copy-pasted into five files; it now lives here.

Wire format (message 0xD1, 29 bytes, big-endian signed 24-bit fields):

    byte  0      0xD1            message type (camera position)
    byte  1      cam_id          camera identifier
    bytes 2-4    pan   (yaw)     int24, units of 1/32768 degree
    bytes 5-7    tilt  (pitch)   int24, units of 1/32768 degree
    bytes 8-10   roll            int24, units of 1/32768 degree
    bytes 11-13  X               int24, units of 1/64 mm (1 m = 64000)
    bytes 14-16  Y               int24, units of 1/64 mm
    bytes 17-19  Z               int24, units of 1/64 mm
    bytes 20-22  zoom            int24
    bytes 23-25  focus           int24
    bytes 26-27  user defined
    byte  28     checksum        (0x40 - sum(bytes 0..27)) mod 256

This is what Unreal Engine's LiveLinkFreeD plugin expects.

Two safety properties the old copies did not have:

  * Angles are wrapped into [-180, 180) before scaling. A signed 24-bit field
    holds +/-8388607, i.e. +/-256 degrees at 1/32768 scale, so an unbounded
    integrated yaw (the pure-gyro prototype drifts without limit) used to wrap
    around silently and teleport the virtual camera. Wrapping is lossless for
    angles -- -190 deg and +170 deg are the same rotation.
  * Translations are saturated at the int24 limit (~+/-131.07 m) instead of
    wrapping, so a diverging SLAM solution pins the camera at the edge of the
    volume rather than flipping it to the opposite side.
"""
import math

# Default Unreal Engine LiveLinkFreeD endpoint.
FREED_DEFAULT_IP   = "127.0.0.1"
FREED_DEFAULT_PORT = 11111

FREED_PACKET_SIZE = 29
FREED_MSG_CAMERA_POSITION = 0xD1

ANGLE_SCALE = 32768   # LSB = 1/32768 degree
TRANS_SCALE = 64000   # LSB = 1/64 mm, so 1 m = 64000 counts

_INT24_MAX = (1 << 23) - 1     # +8388607
_INT24_MIN = -(1 << 23)        # -8388608

#: Largest translation representable in a FreeD int24 field, in metres (~131.07).
MAX_TRANSLATION_M = _INT24_MAX / TRANS_SCALE


def wrap_deg(angle_deg):
    """Normalise an angle in degrees to [-180, 180).

    Rotations are periodic, so this never loses information -- it only keeps
    accumulated/drifting angles inside the int24 field.
    """
    return (float(angle_deg) + 180.0) % 360.0 - 180.0


def _int24_bytes(val):
    """Encode a value as three big-endian bytes of a signed 24-bit field."""
    val = int(val)
    if val > _INT24_MAX:
        val = _INT24_MAX
    elif val < _INT24_MIN:
        val = _INT24_MIN
    if val < 0:
        val += (1 << 24)
    return [(val >> 16) & 0xFF, (val >> 8) & 0xFF, val & 0xFF]


def _int24_from_bytes(b):
    """Decode three big-endian bytes as a signed 24-bit value."""
    val = (b[0] << 16) | (b[1] << 8) | b[2]
    if val & (1 << 23):
        val -= (1 << 24)
    return val


def freed_checksum(packet):
    """FreeD checksum over the first 28 bytes: (0x40 - sum) mod 256."""
    return (64 - (sum(packet[:28]) % 256)) % 256


def get_freed_packet(cam_id, pitch_deg, yaw_deg, roll_deg,
                     tx_m, ty_m, tz_m, zoom=0, focus=0):
    """Build a 29-byte FreeD D1 camera-position packet.

    Angles are in degrees, translations in metres. Argument order matches the
    original bridges (pitch, yaw, roll) even though the wire order is
    pan/tilt/roll, so existing call sites keep working.
    """
    packet = bytearray(FREED_PACKET_SIZE)
    packet[0] = FREED_MSG_CAMERA_POSITION
    packet[1] = cam_id & 0xFF

    packet[2:5]   = _int24_bytes(wrap_deg(yaw_deg)   * ANGLE_SCALE)   # pan
    packet[5:8]   = _int24_bytes(wrap_deg(pitch_deg) * ANGLE_SCALE)   # tilt
    packet[8:11]  = _int24_bytes(wrap_deg(roll_deg)  * ANGLE_SCALE)   # roll

    packet[11:14] = _int24_bytes(float(tx_m) * TRANS_SCALE)
    packet[14:17] = _int24_bytes(float(ty_m) * TRANS_SCALE)
    packet[17:20] = _int24_bytes(float(tz_m) * TRANS_SCALE)

    packet[20:23] = _int24_bytes(zoom)
    packet[23:26] = _int24_bytes(focus)
    packet[26:28] = [0, 0]   # user defined

    packet[28] = freed_checksum(packet)
    return bytes(packet)


def decode_freed_packet(packet):
    """Decode a FreeD D1 packet back into engineering units.

    Useful for tests and for sniffing what Unreal is actually being sent.
    Returns a dict; raises ValueError on a malformed packet.
    """
    if len(packet) != FREED_PACKET_SIZE:
        raise ValueError(f"expected {FREED_PACKET_SIZE} bytes, got {len(packet)}")
    packet = bytearray(packet)
    if packet[0] != FREED_MSG_CAMERA_POSITION:
        raise ValueError(f"not a D1 camera-position message: 0x{packet[0]:02X}")
    if packet[28] != freed_checksum(packet):
        raise ValueError("checksum mismatch")

    return {
        "cam_id":    packet[1],
        "yaw_deg":   _int24_from_bytes(packet[2:5])   / ANGLE_SCALE,
        "pitch_deg": _int24_from_bytes(packet[5:8])   / ANGLE_SCALE,
        "roll_deg":  _int24_from_bytes(packet[8:11])  / ANGLE_SCALE,
        "x_m":       _int24_from_bytes(packet[11:14]) / TRANS_SCALE,
        "y_m":       _int24_from_bytes(packet[14:17]) / TRANS_SCALE,
        "z_m":       _int24_from_bytes(packet[17:20]) / TRANS_SCALE,
        "zoom":      _int24_from_bytes(packet[20:23]),
        "focus":     _int24_from_bytes(packet[23:26]),
    }


def euler_from_quaternion(x, y, z, w):
    """Convert a quaternion to (roll, pitch, yaw) in radians, XYZ order.

    Kept dependency-free on purpose: the non-ROS bridges (Spectacular AI, raw
    RealSense) must not need tf_transformations just to format a packet.
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = max(-1.0, min(+1.0, t2))
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z
