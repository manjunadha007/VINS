#!/usr/bin/env python3
"""Route D455 exposures using hardware emitter metadata, never frame parity.

Only confirmed emitter-off stereo reaches tracking. Depth uses emitter-on
exposures, or passive exposures when the camera confirms the projector is off.
Missing, inconsistent or stale metadata drops the exposure.
"""
from collections import OrderedDict
import json
import time
from projector_control import ProjectorControl, RosProjectorParameters


def stamp_ns(msg):
    return msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec


def emitter_state(payload):
    try:
        value = json.loads(payload).get("frame_emitter_mode")
    except (ValueError, TypeError, AttributeError):
        return None
    return value if type(value) is int and value in (0, 1) else None


class FrameJoin:
    """Bounded exact-timestamp join, independent of ROS."""

    def __init__(self, required, max_pending=24, max_age_s=0.25):
        self.required = frozenset(required)
        self.max_pending = max_pending
        self.max_age_s = max_age_s
        self.pending = OrderedDict()
        self.last_stamp = -1
        self.dropped = 0

    def add(self, kind, stamp, value, now=None):
        now = time.monotonic() if now is None else now
        for key, (received, _) in list(self.pending.items()):
            if now - received > self.max_age_s or key <= self.last_stamp:
                self.pending.pop(key)
                self.dropped += 1
        if kind not in self.required or stamp <= self.last_stamp or stamp <= 0:
            return None
        _, parts = self.pending.setdefault(stamp, (now, {}))
        parts[kind] = value
        while len(self.pending) > self.max_pending:
            self.pending.popitem(last=False)
            self.dropped += 1
        if self.required.issubset(parts):
            self.pending.pop(stamp, None)
            self.last_stamp = stamp
            return parts
        return None


STREAMS = ("infra1", "infra2", "depth")
REQUIRED = tuple(STREAMS) + tuple(s + "_metadata" for s in STREAMS)
TRACKING_STREAMS = ("infra1", "infra2")
DEPTH_STREAMS = ("infra1", "depth")


def exposure_state(parts, streams=STREAMS):
    states = [emitter_state(parts[s + "_metadata"].json_data) for s in streams]
    if None in states or len(set(states)) != 1:
        return None
    return states[0]


def route_exposure(parts, projector_off=False):
    """Return tracking/depth destinations, retaining metadata checks on switches."""
    state = exposure_state(parts)
    return state == 0, state == 1 or (state == 0 and projector_off)


class ExposureRouter:
    """Independent, timestamp-matched stereo and depth/preview deliveries.

    Stereo requires both IR images and their emitter metadata. It never waits
    for depth. Depth retains its paired IR preview but does not wait for IR2.
    Each branch still rejects projected tracking, unknown or conflicting
    metadata, expired messages and backwards timestamps.
    """

    def __init__(self):
        self.joins = {
            name: FrameJoin(streams + tuple(s + "_metadata" for s in streams))
            for name, streams in (("tracking", TRACKING_STREAMS), ("depth", DEPTH_STREAMS))
        }
        self.bad_metadata = 0

    def add(self, kind, stamp, msg, projector_off=False, now=None):
        deliveries = {}
        for name, streams in (("tracking", TRACKING_STREAMS), ("depth", DEPTH_STREAMS)):
            join = self.joins[name]
            if kind not in join.required:
                continue
            parts = join.add(kind, stamp, msg, now)
            if parts is None:
                continue
            state = exposure_state(parts, streams)
            if state is None:
                self.bad_metadata += 1
            elif ((name == "tracking" and state == 0)
                  or (name == "depth" and (state == 1 or projector_off))):
                deliveries[name] = parts
        return deliveries


def main():
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Image
    from realsense2_camera_msgs.msg import Metadata

    class Splitter(Node):
        def __init__(self):
            super().__init__("vp_realsense_splitter")
            self.router = ExposureRouter()
            self.counts = [0, 0]
            self.previous = time.monotonic()
            self.projector = ProjectorControl(RosProjectorParameters(self))
            self.create_timer(0.1, self.projector.tick)
            # Reliable outputs serve existing dashboard subscriptions too.
            self.outputs = {
                s: self.create_publisher(Image, f"/camera/{s}/image_rect_raw", 5)
                for s in STREAMS
            }
            self.preview = self.create_publisher(
                Image, "/camera/depth_preview/image_rect_raw", 5)
            self.subscriptions_ = []
            for stream in STREAMS:
                for kind, suffix, msg_type in (
                    (stream, "image_rect_raw", Image),
                    (stream + "_metadata", "metadata", Metadata),
                ):
                    self.subscriptions_.append(self.create_subscription(
                        msg_type, f"/camera/raw/{stream}/{suffix}",
                        lambda msg, key=kind: self.receive(key, msg),
                        QoSProfile(depth=24, reliability=ReliabilityPolicy.BEST_EFFORT)))
            self.create_timer(5.0, self.report)
            self.get_logger().info(
                "Routing emitter-off stereo and active/passive depth; "
                "unknown metadata is rejected.")

        def receive(self, kind, msg):
            confirmed = self.projector.state
            deliveries = self.router.add(kind, stamp_ns(msg), msg,
                projector_off=confirmed is not None and not confirmed.enabled)
            if "tracking" in deliveries:
                parts = deliveries["tracking"]
                self.counts[0] += 1
                self.outputs["infra1"].publish(parts["infra1"])
                self.outputs["infra2"].publish(parts["infra2"])
            if "depth" in deliveries:
                parts = deliveries["depth"]
                self.counts[1] += 1
                self.outputs["depth"].publish(parts["depth"])
                self.preview.publish(parts["infra1"])

        def report(self):
            now = time.monotonic()
            dt = now - self.previous
            confirmed = self.projector.state
            passive = confirmed is not None and not confirmed.enabled
            depth_rate = self.counts[1] / dt
            depth_mode = "passive" if passive else "emitter-on"
            message = (f"Emitter-off stereo {self.counts[0] / dt:.1f} Hz; "
                       f"{depth_mode} depth {depth_rate:.1f} Hz; "
                       f"invalid metadata {self.router.bad_metadata}; "
                       f"incomplete/late stereo {self.router.joins['tracking'].dropped}; "
                       f"incomplete/late depth {self.router.joins['depth'].dropped}")
            if not self.counts[0] or not self.counts[1]:
                self.get_logger().warning(message + "; check emitter mode and metadata")
            else:
                self.get_logger().info(message)
            self.counts = [0, 0]
            self.previous = now

    rclpy.init()
    node = Splitter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.projector.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
