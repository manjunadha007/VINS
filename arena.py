#!/usr/bin/env python3
"""HyperFusion testing arena: record, replay and score scenario takes.

    ./arena.sh list                 scenarios, takes, references, last scores
    ./arena.sh record s2            guided take of scenario s2 (+ 3DE export)
    ./arena.sh info s2              latest take of s2: marks, notes, coverage
    ./arena.sh export s2            (re)write the 3DE plate for a take
    ./arena.sh test s2              replay through cuVSLAM + dashboard, then score
    ./arena.sh score s2 [--live]    score the latest replay (or the live take)

A take is named <scenario id>_<YYYYmmdd-HHMMSS>, and that name is used
everywhere: arena/inputs/<take>/ holds the recording, plate/<take>.#####.png
the footage to track, and arena/reference/<take>.abc is where the 3DE camera
goes. Replays and their scores land in arena/runs/<take>/<run>/.

arena/inputs/<take>/:
    bag/            rosbag2 (MCAP): the D455 streams the trackers consume (stereo,
                    depth, IMU), SLAM's output, GNSS, and the dashboard's records
                    (/hyperfusion/state: what Unreal was sent; /hyperfusion/scene)
    take.json       scenario, step marks and notes, per-stream coverage,
                    environment (VP_* launch settings), code and config hashes
    tracking.yaml   the live config at record time
    scene.json      the dashboard's scene at record time (File > Open-able)
    plate/ frames.csv plate.json survey_points.txt scan_3de.obj gnss.csv
    README_3DE.txt  the 3DE footage and its companions (arena_export.py)

Recording only subscribes -- it never opens the camera, changes parameters or
sends FreeD. The steps of a scenario (arena_scenarios.yaml) are confirmed with
Enter during the take; typed text plus Enter drops a timestamped note, and
q + Enter stops. Scoring starts at the first step mark.
"""
import argparse
import datetime
import hashlib
import json
import os
import re
import select
import shutil
import signal
import statistics
import struct
import subprocess
import sys
import threading
import time

import yaml

import arena_score
import arena_state

HERE = os.path.dirname(os.path.abspath(__file__))
SCENARIOS_FILE = os.path.join(HERE, "arena_scenarios.yaml")
ARENA_DIR = os.environ.get("VP_ARENA_DIR", "/workspaces/isaac_ros-dev/arena")
# The live deployment (launch_isaac.sh): what a recording's dashboard ran.
LIVE_SRC = "/workspaces/isaac_ros-dev/src"
LIVE_CONFIG = os.path.join(LIVE_SRC, "tracking.yaml")
DASHBOARD_LOG = "/tmp/hyperfusion-dashboard.log"
TAKE_FORMAT = "hyperfusion.arena_take"

# What a take records. The camera streams are the post-split topics the
# trackers actually consume -- in alternating-emitter mode realsense_streams.py
# publishes them from /camera/raw/*, which is deliberately not recorded twice.
# Compressed/theora/nitros variants, SLAM visualisation and the growing
# slam_path/vo_path are left out.
REQUIRED_TOPICS = [
    "/camera/infra1/image_rect_raw",
    "/camera/infra2/image_rect_raw",
    "/camera/imu",
    "/visual_slam/tracking/odometry",
]
# Published by the dashboard; the GNSS pair only while its receiver is connected.
DASHBOARD_TOPICS = [arena_state.STATE_TOPIC, arena_state.SCENE_TOPIC,
                    arena_state.GNSS_NMEA_TOPIC, arena_state.GNSS_FIX_TOPIC]
DEPTH_TOPIC = "/camera/depth/image_rect_raw"
OPTIONAL_TOPICS = [
    DEPTH_TOPIC,
    "/camera/depth/camera_info",
    "/camera/infra1/camera_info",
    "/camera/infra2/camera_info",
    "/camera/accel/imu_info",
    "/camera/gyro/imu_info",
    "/camera/extrinsics/depth_to_accel",
    "/camera/extrinsics/depth_to_gyro",
    "/camera/extrinsics/depth_to_infra1",
    "/camera/extrinsics/depth_to_infra2",
    "/tf",
    "/tf_static",
    "/visual_slam/tracking/vo_pose_covariance",
    "/visual_slam/tracking/vo_pose",
    "/visual_slam/status",
    "/rosout",
    "/diagnostics",
]
# Share of frames a sensor stream must keep for a faithful replay; a take warns
# below it. Stereo and IMU feed SLAM, so they must be near complete. Depth is
# consumed latest-frame-only by the scan fusion, a few times a second, so a few
# missing frames change nothing. Measured on this rig (2026-11-15): the camera /
# splitter already drops ~4% of stereo frames in pairs before anything records
# them (SLAM logs "Delta between current and previous frame"), and depth loses
# ~3% more. Odometry is reported, never warned about: SLAM skipping frames is
# its own behaviour, not a recording fault.
MIN_COVERAGE = {
    "/camera/infra1/image_rect_raw": 0.95,
    "/camera/infra2/image_rect_raw": 0.95,
    "/camera/imu": 0.95,
    DEPTH_TOPIC: 0.85,
}
SENSOR_TOPICS = list(MIN_COVERAGE)
STAMPED_STREAMS = SENSOR_TOPICS + ["/visual_slam/tracking/odometry",
                                   arena_state.GNSS_FIX_TOPIC]
STREAM_TOPICS = STAMPED_STREAMS + [arena_state.STATE_TOPIC, arena_state.GNSS_NMEA_TOPIC]
SETTLE_S = 1.0          # recorder start-up, excluded from frame coverage
PROBE_S = 3.0
CACHE_BYTES = 512 * 1024 * 1024


# ── Scenarios ────────────────────────────────────────────────────────────────
_ID = re.compile(r"^s\d+_[a-z0-9_]+$")


def load_scenarios(path=SCENARIOS_FILE):
    """The scenarios, with `defaults` merged in and the file checked."""
    with open(path) as fh:
        doc = yaml.safe_load(fh) or {}
    defaults = doc.get("defaults") or {}
    scenarios, seen = [], set()
    for raw in doc.get("scenarios") or []:
        sid = raw.get("id", "")
        where = f"{os.path.basename(path)}: scenario {sid or '?'}"
        if not _ID.match(sid):
            raise ValueError(f"{where}: id must look like s1_short_name")
        short = sid.split("_", 1)[0]
        if sid in seen or short in seen:
            raise ValueError(f"{where}: id or {short} is used twice")
        seen.update((sid, short))
        steps = raw.get("steps")
        if not steps or not all(isinstance(s, str) and s.strip() for s in steps):
            raise ValueError(f"{where}: needs a non-empty list of steps")
        if not isinstance(raw.get("title"), str):
            raise ValueError(f"{where}: needs a title")
        scenarios.append(dict(
            id=sid, short=short, title=raw["title"],
            description=" ".join(str(raw.get("description", "")).split()),
            exercises=list(raw.get("exercises") or []),
            setup=list(defaults.get("setup") or []) + list(raw.get("setup") or []),
            steps=list(steps),
            criteria=_criteria(defaults.get("pass"), raw.get("pass"), where),
        ))
    if not scenarios:
        raise ValueError(f"{path}: no scenarios defined")
    return scenarios


def _criteria(defaults, own, where):
    """A scenario's pass criteria: built-in defaults < file defaults < its own."""
    criteria = {**arena_score.DEFAULT_PASS, **(defaults or {}), **(own or {})}
    unknown = set(criteria) - set(arena_score.DEFAULT_PASS)
    if unknown:
        raise ValueError(f"{where}: unknown pass setting(s) {sorted(unknown)}")
    for key, value in criteria.items():
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"{where}: pass.{key} must be a positive number")
    if criteria["min_frames"] > 1:
        raise ValueError(f"{where}: pass.min_frames is a fraction, e.g. 0.90")
    return criteria


def resolve_scenario(scenarios, key):
    """A scenario by full id (s2_lidar_gap) or its short name (s2)."""
    key = key.strip().lower()
    for sc in scenarios:
        if key in (sc["id"], sc["short"]):
            return sc
    names = ", ".join(sc["short"] for sc in scenarios)
    raise KeyError(f"no scenario {key!r} (have {names})")


# ── Takes ────────────────────────────────────────────────────────────────────
def inputs_dir(root=ARENA_DIR):
    return os.path.join(root, "inputs")


def take_name(scenario_id, take_id):
    """One name for the recording, its plate, its .abc and its runs."""
    return f"{scenario_id}_{take_id}"


def list_takes(scenario_id, root=ARENA_DIR):
    """Take directories of one scenario, oldest first."""
    base = inputs_dir(root)
    if not os.path.isdir(base):
        return []
    return sorted(os.path.join(base, d) for d in os.listdir(base)
                  if d.startswith(scenario_id + "_")
                  and os.path.isfile(os.path.join(base, d, "take.json")))


def resolve_take(key, scenarios, root=ARENA_DIR):
    """A take directory from its name, a path, or a scenario (latest take)."""
    for candidate in (key, os.path.join(inputs_dir(root), key)):
        if os.path.isfile(os.path.join(candidate, "take.json")):
            return os.path.normpath(candidate)
    sc = resolve_scenario(scenarios, key)
    takes = list_takes(sc["id"], root)
    if not takes:
        raise KeyError(f"no takes recorded for {sc['id']} yet")
    return takes[-1]


def reference_abc(take_dir, root=ARENA_DIR):
    return os.path.join(root, "reference", os.path.basename(take_dir) + ".abc")


def runs_dir(take_dir, root=ARENA_DIR):
    return os.path.join(root, "runs", os.path.basename(take_dir))


def list_runs(take_dir, root=ARENA_DIR):
    base = runs_dir(take_dir, root)
    if not os.path.isdir(base):
        return []
    return sorted(os.path.join(base, d) for d in os.listdir(base)
                  if os.path.isfile(os.path.join(base, d, "run.json")))


def read_take(path):
    with open(os.path.join(path, "take.json")) as fh:
        return json.load(fh)


def header_stamp(raw):
    """header.stamp (s) of a CDR-serialised message whose first field is a
    std_msgs/Header (Image, Imu, Odometry, ...), without deserialising it."""
    order = "<" if raw[1] & 1 else ">"      # encapsulation: CDR_LE = 0x0001
    sec, nsec = struct.unpack_from(order + "iI", raw, 4)
    return sec + nsec * 1e-9


def stamp_timing(stamps):
    """Rate and missing frames of one stream, from its header stamps alone.

    The period is the median gap between consecutive stamps, so it is the
    stream's true rate even when frames are missing; any gap over 1.5 periods
    is a frame lost somewhere -- camera, splitter or recorder.
    """
    stamps = sorted(stamps)
    count = len(stamps)
    if count < 3:
        return dict(count=count)
    deltas = [b - a for a, b in zip(stamps, stamps[1:])]
    period = statistics.median(deltas)
    if period <= 0:
        return dict(count=count)
    expected = int(round((stamps[-1] - stamps[0]) / period)) + 1
    return dict(count=count, rate_hz=round(1.0 / period, 2), expected=expected,
                coverage=round(min(1.0, count / expected), 4),
                gaps=sum(1 for d in deltas if d > 1.5 * period),
                max_gap_ms=round(max(deltas) * 1000, 1))


def bag_stream_timing(bag_dir, topics, settle_s=None):
    """stamp_timing() for each of `topics` present in a recorded bag.

    The first `settle_s` of each stream are left out: while ros2 bag record is
    still subscribing it misses ~0.3 s of every stream at once (measured on
    this rig), which is the recorder starting, not frames lost from the take.
    """
    settle_s = SETTLE_S if settle_s is None else settle_s
    import rosbag2_py
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag_dir, storage_id="mcap"),
                rosbag2_py.ConverterOptions("", ""))
    reader.set_filter(rosbag2_py.StorageFilter(topics=list(topics)))
    stamps = {}
    while reader.has_next():
        topic, raw, _ = reader.read_next()
        stamps.setdefault(topic, []).append(header_stamp(raw))
    timing = {}
    for topic, s in stamps.items():
        first = min(s)
        timing[topic] = stamp_timing([t for t in s if t >= first + settle_s])
    return timing


def coverage_report(topics_meta, duration_s, timing, live_topics):
    """Per-topic counts and, for streams, frames missing from the bag.

    topics_meta -- {topic: {"type", "count"}} from the bag's metadata.yaml
    timing      -- {topic: stamp_timing(...)} for the stamped streams
    live_topics -- streams that were publishing just before recording
    Returns ({topic: {...}}, [warnings]).
    """
    report, warnings = {}, []
    for topic, meta in sorted(topics_meta.items()):
        count = int(meta.get("count", 0))
        entry = dict(type=meta.get("type"), count=count,
                     rate_hz=round(count / duration_s, 2) if duration_s > 0 else None)
        entry.update({k: v for k, v in timing.get(topic, {}).items() if k != "count"})
        cov = entry.get("coverage")
        if cov is not None and cov < MIN_COVERAGE.get(topic, 0.0):
            warnings.append(f"{topic}: {1 - cov:.1%} of frames missing "
                            f"({entry['gaps']} gaps, longest {entry['max_gap_ms']} ms)")
        report[topic] = entry
    for topic in live_topics:
        if topic not in topics_meta:
            warnings.append(f"{topic}: was live before recording but is not in the bag")
    return report, warnings


def read_bag_metadata(bag_dir):
    """(duration_s, {topic: {type, count}}) from rosbag2's metadata.yaml."""
    with open(os.path.join(bag_dir, "metadata.yaml")) as fh:
        info = yaml.safe_load(fh)["rosbag2_bagfile_information"]
    duration = info["duration"]["nanoseconds"] / 1e9
    topics = {t["topic_metadata"]["name"]: dict(type=t["topic_metadata"]["type"],
                                                count=t["message_count"])
              for t in info["topics_with_message_count"]}
    return duration, topics


def dir_bytes(path):
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def launch_environment():
    """VP_* settings the running SLAM launch was started with."""
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as fh:
                cmd = fh.read().split(b"\0")
            if b"launch" not in cmd or not any(b"isaac" in c or b"visual_slam" in c for c in cmd):
                continue
            with open(f"/proc/{pid}/environ", "rb") as fh:
                env = dict(kv.split(b"=", 1) for kv in fh.read().split(b"\0") if b"=" in kv)
            return {k.decode(): v.decode() for k, v in env.items() if k.startswith(b"VP_")}
        except OSError:
            continue
    return {}


def source_hashes(directory=LIVE_SRC):
    """sha256 (12 hex) of every deployed module, so a take says what code ran."""
    hashes = {}
    for name in sorted(os.listdir(directory)):
        if name.endswith((".py", ".yaml")):
            with open(os.path.join(directory, name), "rb") as fh:
                hashes[name] = hashlib.sha256(fh.read()).hexdigest()[:12]
    return hashes


def free_bytes(path):
    """Free space on the filesystem `path` will be created on."""
    while not os.path.isdir(path):
        path = os.path.dirname(path)
    return shutil.disk_usage(path).free


def gnss_enabled_in_config(path=None):
    """gnss.enabled in the live tracking.yaml, or None if it cannot be read."""
    path = path or os.environ.get("VP_TRACKING_CONFIG")
    try:
        with open(path) as fh:
            return bool(((yaml.safe_load(fh) or {}).get("gnss") or {}).get("enabled", False))
    except (OSError, TypeError, yaml.YAMLError):
        return None


def gnss_text(gnss, stale_s=3.0):
    """'RTK fixed, 18 sats, ±12 mm' from a state record's gnss summary."""
    if not gnss:
        return "no fix received"
    if gnss.get("age_s", 0) > stale_s:
        return f"silent {gnss['age_s']:.0f}s"
    acc = f", ±{gnss['hacc_m'] * 1000:.0f} mm" if gnss.get("hacc_m") is not None else ""
    return f"{gnss['label']}, {gnss['satellites']} sats{acc}"


def mmss(seconds):
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


# ── ROS side (imported lazily so `list`/`info` and the tests need no ROS) ────
class _Ros:
    def __init__(self):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        self.rclpy = rclpy
        rclpy.init(args=None)
        self.node = rclpy.create_node("hyperfusion_arena")
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self._running = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        self.state = None          # latest decoded /hyperfusion/state record
        self.scene = None          # latest /hyperfusion/scene JSON text

    def _spin(self):
        # spin_once in a loop, not spin(): close() can then stop and join this
        # thread before the node is destroyed. Destroying it under a blocked
        # spin() aborted the process ("terminate called without an active
        # exception").
        while self._running:
            try:
                self.executor.spin_once(timeout_sec=0.1)
            except Exception:
                if not self._running:
                    break

    def topic_types(self, settle_s=1.5):
        time.sleep(settle_s)   # let discovery fill in
        return {name: types[0] for name, types in self.node.get_topic_names_and_types()}

    def watch_dashboard(self):
        """Keep self.state / self.scene current (cheap: two small topics)."""
        from std_msgs.msg import String
        from rclpy.qos import (QoSProfile, DurabilityPolicy, ReliabilityPolicy,
                               qos_profile_sensor_data)

        def on_state(msg):
            try:
                self.state = arena_state.decode(msg.data)
            except ValueError:
                pass

        def on_scene(msg):
            self.scene = msg.data

        self.node.create_subscription(String, arena_state.STATE_TOPIC, on_state,
                                      qos_profile_sensor_data)
        self.node.create_subscription(
            String, arena_state.SCENE_TOPIC, on_scene,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                       reliability=ReliabilityPolicy.RELIABLE))

    def probe(self, topics, types, seconds=PROBE_S):
        """{topic: (hz, bytes_per_s)} for live topics, measured over `seconds`.

        Raw subscriptions: messages are sized, never deserialised. This Python
        node cannot keep up with three image streams plus 400 Hz IMU and drops
        some, so a stamped stream's rate comes from the median gap between its
        header stamps (true even with drops), not from how many arrived.
        Destroyed afterwards so nothing competes with the recorder.
        """
        from rclpy.qos import qos_profile_sensor_data
        from rosidl_runtime_py.utilities import get_message
        seen = {t: dict(n=0, size=0, stamps=[]) for t in topics if t in types}
        subs = []
        for topic in seen:
            stamped = topic in STAMPED_STREAMS

            def on_msg(raw, s=seen[topic], stamped=stamped):
                s["n"] += 1
                s["size"] += len(raw)
                if stamped:
                    s["stamps"].append(header_stamp(raw))
            subs.append(self.node.create_subscription(
                get_message(types[topic]), topic, on_msg, qos_profile_sensor_data, raw=True))
        time.sleep(seconds)
        for sub in subs:
            self.node.destroy_subscription(sub)
        result = {}
        for topic, s in seen.items():
            if not s["n"]:
                continue
            hz = stamp_timing(s["stamps"]).get("rate_hz") or s["n"] / seconds
            result[topic] = (hz, s["size"] / s["n"] * hz)
        return result

    def close(self):
        self._running = False
        self._thread.join(timeout=2.0)
        try:
            self.executor.shutdown(timeout_sec=1.0)
            self.node.destroy_node()
        finally:
            self.rclpy.try_shutdown()


# ── Commands ─────────────────────────────────────────────────────────────────
def _latest_score(take_dir, root):
    for run in reversed(list_runs(take_dir, root)):
        path = os.path.join(run, "score.json")
        if os.path.isfile(path):
            with open(path) as fh:
                return json.load(fh)
    return None


def cmd_list(args):
    for sc in load_scenarios(args.scenarios):
        takes = list_takes(sc["id"], args.arena)
        c = sc["criteria"]
        print(f"{sc['short']:>3}  {sc['title']}")
        print(f"     {sc['id']} -- {len(takes)} take(s); passes at {c['min_frames']:.0%} of "
              f"frames within {c['frame_mm']:g} mm / {c['frame_deg']:g}°")
        for path in takes[-3:]:
            t = read_take(path)
            parts = [mmss(t.get("duration_s", 0)),
                     f"{t.get('bag', {}).get('size_bytes', 0) / 1e9:.1f} GB",
                     "complete" if t.get("complete") else "incomplete",
                     "plate ✓" if os.path.isfile(os.path.join(path, "plate.json")) else "no plate",
                     "reference ✓" if os.path.isfile(reference_abc(path, args.arena)) else "awaiting .abc"]
            score = _latest_score(path, args.arena)
            if score:
                parts.append(f"{'PASS' if score['passed'] else 'FAIL'} {score['pass_ratio']:.0%}")
            print(f"       {os.path.basename(path)}  " + "  ".join(parts))
    return 0


def cmd_info(args):
    path = resolve_take(args.take, load_scenarios(args.scenarios), args.arena)
    t = read_take(path)
    print(f"{t['scenario']['id']} / {t['take']}  ({path})")
    print(f"  {t['scenario']['title']}")
    print(f"  {mmss(t['duration_s'])}, {t['bag']['size_bytes'] / 1e9:.1f} GB, "
          f"{'complete' if t['complete'] else 'INCOMPLETE'}; launch {t.get('environment') or '{}'}")
    for m in t["marks"]:
        print(f"  [{mmss(m['offset_s'])}] step {m['step']}: {m['label']}")
    for n in t["notes"]:
        print(f"  [{mmss(n['offset_s'])}] note: {n['text']}")
    if t.get("summary_note"):
        print(f"  notes: {t['summary_note']}")
    for topic, e in t["bag"]["topics"].items():
        cov = (f"  {e['coverage']:.1%} of frames, {e['gaps']} gaps"
               if e.get("coverage") is not None else "")
        print(f"    {topic:<44} {e['count']:>7}  {e['rate_hz'] or 0:>7.1f} Hz{cov}")
    for w in t.get("warnings", []):
        print(f"  ⚠️  {w}")
    return 0


def _print_card(sc):
    print("=" * 72)
    print(f" {sc['short'].upper()} -- {sc['title']}")
    print("=" * 72)
    print(f" {sc['description']}\n")
    print(" Setup:")
    for item in sc["setup"]:
        print(f"   [ ] {item}")
    print("\n Steps (press Enter at each):")
    for i, step in enumerate(sc["steps"], 1):
        print(f"   {i}. {step}")
    print("   Then Enter once more to stop. Text + Enter = note, q + Enter = stop now.\n")


def _status_line(ros, start, bag_dir, last, next_step):
    now = time.time()
    size = dir_bytes(bag_dir)
    rate = (size - last[1]) / max(1e-3, now - last[0]) / 1e6
    last[:] = [now, size]
    live = ""
    st = ros.state
    if st is not None:
        age = now - st["t"]
        lineup = (st.get("lineup") or {}).get("title", "")
        live = (f" | FreeD {st['mode']}" + (f" · {lineup}" if lineup else "")
                if age < 1.0 else f" | dashboard silent {age:.0f}s")
        if st.get("gnss"):
            live += f" | GNSS {gnss_text(st['gnss'])}"
    step = f" | next: {next_step}" if next_step else " | Enter to stop"
    text = f"● REC {mmss(now - start)}  {size / 1e9:5.1f} GB  {rate:5.0f} MB/s{live}{step}"
    # One terminal row at most: a wrapped line breaks the \r redraw.
    width = shutil.get_terminal_size((120, 24)).columns - 1
    return "\r\x1b[K" + (text if len(text) <= width else text[:width - 1] + "…")


def _stop_recorder(proc, timeout=60):
    if proc.poll() is not None:
        return proc.returncode
    # ros2 bag record finalises metadata.yaml on SIGINT; a kill would not.
    os.killpg(proc.pid, signal.SIGINT)
    try:
        return proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGTERM)
        return proc.wait(timeout=10)


def cmd_record(args):
    sc = resolve_scenario(load_scenarios(args.scenarios), args.scenario)
    _print_card(sc)
    interactive = sys.stdin.isatty()
    if not interactive and not args.duration:
        print("❌ No terminal: pass --duration SECONDS to record unattended.")
        return 2

    ros = _Ros()
    try:
        ros.watch_dashboard()
        types = ros.topic_types()
        wanted = REQUIRED_TOPICS + OPTIONAL_TOPICS + ([] if args.no_dashboard else DASHBOARD_TOPICS)
        print(f"Checking live streams ({PROBE_S:.0f} s)...")
        probe = ros.probe(STREAM_TOPICS, types)
        probe_hz = {t: hz for t, (hz, _) in probe.items()}
        missing = [t for t in REQUIRED_TOPICS if t not in probe_hz]
        if not args.no_dashboard and arena_state.STATE_TOPIC not in probe_hz:
            missing.append(f"{arena_state.STATE_TOPIC} (dashboard not running, or older "
                           f"than the arena: relaunch with ./launch_isaac.sh)")
        for t in STREAM_TOPICS:
            if t in probe:
                hz, bps = probe[t]
                print(f"   {t:<40} {hz:6.1f} Hz  {bps / 1e6:6.1f} MB/s")
            else:
                print(f"   {t:<40}      -- not publishing")
        if DEPTH_TOPIC not in probe_hz:
            print("   ⚠️  no depth: the LiDAR scan fusion cannot be replayed from this take")
        if not args.no_dashboard:
            if arena_state.GNSS_NMEA_TOPIC in probe_hz:
                print(f"   GNSS: {gnss_text((ros.state or {}).get('gnss'))}")
            elif gnss_enabled_in_config():
                print("   ⚠️  GNSS is enabled in tracking.yaml but no NMEA is arriving: connect "
                      "it in Setup > GNSS, or check the receiver's USB cable")
            else:
                print("   GNSS is off in tracking.yaml: this take will have no GNSS")
        if missing:
            print("\n❌ Not recording -- these are required and not publishing:")
            for m in missing:
                print(f"     {m}")
            return 1

        bytes_per_s = sum(bps for _, bps in probe.values())
        free = free_bytes(inputs_dir(args.arena))
        minutes = free / max(1.0, bytes_per_s) / 60
        print(f"\n   about {bytes_per_s * 60 / 1e9:.1f} GB per minute; "
              f"{free / 1e9:.0f} GB free = room for {minutes:.0f} min")
        if minutes < 5 and not args.force:
            print("❌ Less than 5 minutes of space. Free some, or pass --force.")
            return 1
        if ros.scene is None and not args.no_dashboard:
            print("   ⚠️  no scene from the dashboard yet; scene.json will be missing")

        if interactive and not args.yes:
            try:
                input("\nPress Enter to START recording (Ctrl-C to cancel) ")
            except (KeyboardInterrupt, EOFError):
                print("\nCancelled.")
                return 130

        take_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        name = take_name(sc["id"], take_id)
        take = os.path.join(inputs_dir(args.arena), name)
        bag_dir = os.path.join(take, "bag")
        os.makedirs(take)
        log_offset = os.path.getsize(DASHBOARD_LOG) if os.path.exists(DASHBOARD_LOG) else None
        rec_log = open(os.path.join(take, "record.log"), "w")
        cmd = ["ros2", "bag", "record", "-s", "mcap", "-o", bag_dir,
               "--max-cache-size", str(CACHE_BYTES)] + wanted
        # Own session: the terminal's Ctrl-C reaches this script, which then
        # stops the recorder cleanly (SIGINT, so metadata.yaml is written).
        # The finally below guarantees it is stopped however this loop ends --
        # an orphaned recorder would keep filling the disk.
        proc = subprocess.Popen(cmd, stdout=rec_log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True)
        start = time.time()
        start_state = ros.state
        marks, notes, stop_reason = [], [], None
        last = [start, 0]
        print(f"\nRecording {name}  ->  {take}\n")
        print(f"   step 1/{len(sc['steps'])}: {sc['steps'][0]}  -- press Enter when it is true")
        try:
            while True:
                if proc.poll() is not None:
                    stop_reason = f"recorder exited ({proc.returncode})"
                    break
                if args.duration and time.time() - start >= args.duration:
                    stop_reason = "duration reached"
                    break
                nxt = sc["steps"][len(marks)] if len(marks) < len(sc["steps"]) else None
                sys.stdout.write(_status_line(ros, start, bag_dir, last, nxt))
                sys.stdout.flush()
                if not interactive:
                    time.sleep(1.0)
                    continue
                ready, _, _ = select.select([sys.stdin], [], [], 1.0)
                if not ready:
                    continue
                line = sys.stdin.readline()
                now = time.time()
                if line == "":                    # EOF
                    stop_reason = "input closed"
                    break
                text = line.strip()
                if text.lower() == "q":
                    stop_reason = "stopped by operator"
                    break
                if text:
                    notes.append(dict(t=now, offset_s=round(now - start, 3), text=text))
                    print(f"   [{mmss(now - start)}] note: {text}")
                    continue
                if nxt is None:
                    stop_reason = "all steps done"
                    break
                marks.append(dict(t=now, offset_s=round(now - start, 3),
                                  step=len(marks) + 1, label=nxt))
                print(f"   [{mmss(now - start)}] ✓ step {len(marks)}: {nxt}")
                if len(marks) < len(sc["steps"]):
                    print(f"   step {len(marks) + 1}/{len(sc['steps'])}: "
                          f"{sc['steps'][len(marks)]}  -- press Enter when it is true")
                else:
                    print("   All steps marked. Press Enter to stop recording.")
        except KeyboardInterrupt:
            stop_reason = "Ctrl-C"
        finally:
            end = time.time()
            print(f"\n\nStopping ({stop_reason or 'error'}); finalising the bag...")
            rc = _stop_recorder(proc)
            rec_log.close()

        warnings = []
        try:
            duration, bag_topics = read_bag_metadata(bag_dir)
        except (OSError, KeyError, TypeError, yaml.YAMLError) as exc:
            duration, bag_topics = end - start, {}
            warnings.append(f"bag metadata unreadable ({exc}); see record.log")
        timing = {}
        if bag_topics:
            print("Checking the bag for missing frames...")
            try:
                timing = bag_stream_timing(bag_dir, [t for t in STAMPED_STREAMS if t in bag_topics])
            except Exception as exc:
                warnings.append(f"could not scan the bag for missing frames: {exc}")
        report, cov_warnings = coverage_report(bag_topics, duration, timing, list(probe_hz))
        warnings += cov_warnings
        if rc not in (0, None, -signal.SIGINT):
            warnings.append(f"ros2 bag record exited with {rc}; see record.log")
        if len(marks) < len(sc["steps"]):
            warnings.append(f"only {len(marks)} of {len(sc['steps'])} steps were marked")

        # Snapshots: what the take ran with.
        config = os.environ.get("VP_TRACKING_CONFIG")
        if config and os.path.isfile(config):
            shutil.copy2(config, os.path.join(take, "tracking.yaml"))
        else:
            warnings.append("live tracking.yaml not found (VP_TRACKING_CONFIG unset)")
        if ros.scene is not None:
            with open(os.path.join(take, "scene.json"), "w") as fh:
                json.dump(json.loads(ros.scene), fh, indent=2)
        if log_offset is not None:
            with open(DASHBOARD_LOG, "rb") as src, open(os.path.join(take, "dashboard.log"), "wb") as dst:
                src.seek(log_offset)
                dst.write(src.read())

        summary_note = ""
        if interactive:
            try:
                summary_note = input("Notes for this take (Enter to skip): ").strip()
            except (KeyboardInterrupt, EOFError):
                pass

        manifest = dict(
            format=TAKE_FORMAT, version=1,
            scenario={k: sc[k] for k in ("id", "title", "description", "steps", "criteria")},
            take=name, take_id=take_id,
            created=datetime.datetime.fromtimestamp(start).isoformat(timespec="seconds"),
            start_time=start, end_time=end, duration_s=round(duration, 3),
            stop_reason=stop_reason, complete=len(marks) == len(sc["steps"]),
            marks=marks, notes=notes, summary_note=summary_note,
            environment=launch_environment(),
            git_rev=os.environ.get("ARENA_GIT_REV"),
            sources=source_hashes(),
            start_state=start_state,
            scene_file="scene.json" if ros.scene is not None else None,
            bag=dict(path="bag", storage="mcap", size_bytes=dir_bytes(bag_dir),
                     coverage_excludes_first_s=SETTLE_S, topics=report),
            warnings=warnings,
        )
        with open(os.path.join(take, "take.json"), "w") as fh:
            json.dump(manifest, fh, indent=2)

        print(f"\n{'✅' if not warnings else '⚠️ '} Take saved: {take}")
        print(f"   {mmss(duration)}, {manifest['bag']['size_bytes'] / 1e9:.1f} GB, "
              f"{len(marks)}/{len(sc['steps'])} steps, {len(notes)} note(s)")
        for topic in STREAM_TOPICS:
            e = report.get(topic)
            if e:
                frames = (f"  {e['coverage']:.1%} of frames, {e['gaps']} gaps"
                          if e.get("coverage") is not None else "")
                print(f"   {topic:<40} {e['rate_hz'] or 0:6.1f} Hz{frames}")
        for w in warnings:
            print(f"   ⚠️  {w}")
    finally:
        ros.close()

    if not args.no_export:
        print("\nWriting the 3DE plate...")
        try:
            import arena_export
            arena_export.export_take(take)
            print(f"   Track plate/ in 3DE (see {name}/README_3DE.txt), then save the camera as\n"
                  f"   arena/reference/{name}.abc")
        except Exception as exc:
            warnings.append(f"3DE export failed: {exc}")
            print(f"   ⚠️  3DE export failed ({exc}); retry with ./arena.sh export {name}")
    return 0 if not warnings else 3


def cmd_export(args):
    import arena_export
    take = resolve_take(args.take, load_scenarios(args.scenarios), args.arena)
    arena_export.export_take(take)
    print(f"✅ {os.path.basename(take)}: 3DE plate written. Camera goes to "
          f"arena/reference/{os.path.basename(take)}.abc")
    return 0


def _criteria_for(take, scenarios):
    try:
        return resolve_scenario(scenarios, take["scenario"]["id"])["criteria"]
    except KeyError:
        return take["scenario"].get("criteria") or dict(arena_score.DEFAULT_PASS)


def _live_records(take_dir):
    """The dashboard's own state records from a take's bag (the live run)."""
    import rosbag2_py
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=os.path.join(take_dir, "bag"), storage_id="mcap"),
                rosbag2_py.ConverterOptions("", ""))
    reader.set_filter(rosbag2_py.StorageFilter(topics=[arena_state.STATE_TOPIC]))
    from rclpy.serialization import deserialize_message
    from std_msgs.msg import String
    records = []
    while reader.has_next():
        try:
            records.append(arena_state.decode(deserialize_message(reader.read_next()[1], String).data))
        except ValueError:
            pass
    return records


def score_take(take_dir, run_dir, scenarios, root):
    """Score one run (run_dir) or the live take (run_dir None). Returns report."""
    import tracking_config
    take = read_take(take_dir)
    name = os.path.basename(take_dir)
    ref_json = os.path.join(take_dir, "reference.json")
    if not os.path.isfile(ref_json):
        abc = reference_abc(take_dir, root)
        raise KeyError(f"no 3DE camera for {name} yet: save it as arena/reference/{name}.abc"
                       if not os.path.isfile(abc) else
                       f"{abc} has not been read yet: run this through ./arena.sh, which "
                       "converts it with Blender")
    reference = arena_score.load_json(ref_json)
    frames = arena_score.read_frames(os.path.join(take_dir, "frames.csv"))
    if run_dir is None:
        out_dir = os.path.join(take_dir, "live_score")
        os.makedirs(out_dir, exist_ok=True)
        records, offset, config = _live_records(take_dir), 0.0, os.path.join(take_dir, "tracking.yaml")
        if not records:
            raise KeyError(f"{name} has no live dashboard records (recorded with --no-dashboard, "
                           "or by a dashboard older than the arena): score a replay instead")
    else:
        out_dir = run_dir
        with open(os.path.join(run_dir, "state.jsonl")) as fh:
            records = [arena_state.decode(line) for line in fh if line.strip()]
        offset = float(arena_score.load_json(os.path.join(run_dir, "run.json"))["stamp_offset_s"])
        config = os.path.join(run_dir, "tracking.yaml")
    cfg = tracking_config.load_config(config, verbose=False)
    report, rows = arena_score.score(
        frames, reference, records, offset_s=offset,
        lead_s=float(cfg["tracking"].get("lead_time_s", 0.0)),
        rig_T=arena_score.rig_transform(cfg), marks=take.get("marks") or [],
        criteria=_criteria_for(take, scenarios))
    report.update(take=name, run=None if run_dir is None else os.path.basename(run_dir),
                  reference=dict(camera=reference.get("camera"), abc=reference.get("abc"),
                                 last_moving_frame=reference.get("last_moving_frame")))
    with open(os.path.join(out_dir, "score.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    arena_score.write_rows(os.path.join(out_dir, "errors.csv"), rows)
    label = name if run_dir is None else f"{name} / run {os.path.basename(run_dir)}"
    for line in arena_score.summary_lines(report, label + (" (live)" if run_dir is None else "")):
        print(line)
    # Frozen for the last second or more: the .abc probably ends before the
    # plate (Blender holds its last sample). A still camera can repeat a frame
    # or two by chance, so shorter runs are not flagged.
    last_frame = reference.get("first_frame", 1) + reference.get("frame_count", 0) - 1
    if reference.get("last_moving_frame", last_frame) < last_frame - reference.get("fps", 30):
        print(f"   ⚠️  the 3DE camera stops moving at frame {reference['last_moving_frame']}: "
              "is the .abc shorter than the plate?")
    print(f"   per-frame errors: {os.path.join(out_dir, 'errors.csv')}")
    return report


def cmd_score(args):
    scenarios = load_scenarios(args.scenarios)
    take = resolve_take(args.take, scenarios, args.arena)
    if args.live:
        run = None
    elif args.run:
        run = os.path.join(runs_dir(take, args.arena), args.run)
    else:
        runs = list_runs(take, args.arena)
        if not runs:
            raise KeyError(f"no replays of {os.path.basename(take)} yet: ./arena.sh test "
                           f"{os.path.basename(take)} (or score --live)")
        run = runs[-1]
    report = score_take(take, run, scenarios, args.arena)
    return 0 if report["passed"] else 1


def cmd_replay(args, then_score=False):
    import arena_replay
    scenarios = load_scenarios(args.scenarios)
    take = resolve_take(args.take, scenarios, args.arena)
    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run = os.path.join(runs_dir(take, args.arena), run_id)
    base = os.path.join(take, "tracking.yaml") if args.take_config else LIVE_CONFIG
    print(f"Replaying {os.path.basename(take)} -> runs/{os.path.basename(take)}/{run_id}")
    print(f"   code: {os.environ.get('ARENA_GIT_REV') or HERE}; config: "
          f"{'the take' if args.take_config else 'live'} ({base})")
    manifest = arena_replay.run_replay(take, run, code_dir=HERE, base_config=base,
                                       show=args.show, domain=args.domain)
    print(f"   replay done: {manifest.get('end_reason')}")
    if not then_score:
        return 0
    if not os.path.isfile(os.path.join(take, "reference.json")):
        print(f"   (not scored: no 3DE camera yet -- save it as "
              f"arena/reference/{os.path.basename(take)}.abc)")
        return 0
    report = score_take(take, run, scenarios, args.arena)
    return 0 if report["passed"] else 1


def cmd_test(args):
    return cmd_replay(args, then_score=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--arena", default=ARENA_DIR,
                        help="arena directory (inputs/, reference/, runs/)")
    parser.add_argument("--scenarios", default=SCENARIOS_FILE)
    sub = parser.add_subparsers(dest="command", required=True)
    take_help = "take name, or a scenario (s1..s4) for its latest take"
    sub.add_parser("list", help="scenarios, takes, references and latest scores")
    p = sub.add_parser("info", help="one take: marks, notes, per-stream coverage")
    p.add_argument("take", help=take_help)
    p = sub.add_parser("record", help="guided recording of one take")
    p.add_argument("scenario", help="s1..s4 or the full scenario id")
    p.add_argument("--duration", type=float, help="stop after this many seconds")
    p.add_argument("--yes", action="store_true", help="start without waiting for Enter")
    p.add_argument("--no-dashboard", action="store_true",
                   help="record without the dashboard's arena topics (sensors and SLAM only)")
    p.add_argument("--no-export", action="store_true", help="skip writing the 3DE plate")
    p.add_argument("--force", action="store_true", help="record even when disk is nearly full")
    p = sub.add_parser("export", help="(re)write a take's 3DE plate, survey and scan")
    p.add_argument("take", help=take_help)
    for name, text in (("replay", "replay a take through cuVSLAM + the dashboard"),
                       ("test", "replay a take, then score it against its 3DE camera")):
        p = sub.add_parser(name, help=text)
        p.add_argument("take", help=take_help)
        p.add_argument("--show", action="store_true", help="show the replay's dashboard window")
        p.add_argument("--take-config", action="store_true",
                       help="use the take's own tracking.yaml instead of the live one")
        p.add_argument("--domain", type=int, default=87, help="ROS domain for the replay")
    p = sub.add_parser("score", help="score the latest replay (or --live, the live take)")
    p.add_argument("take", help=take_help)
    p.add_argument("--run", help="a run id under runs/<take>/ (default: latest)")
    p.add_argument("--live", action="store_true", help="score what the dashboard did live")
    args = parser.parse_args(argv)
    commands = {"list": cmd_list, "info": cmd_info, "record": cmd_record, "export": cmd_export,
                "replay": cmd_replay, "test": cmd_test, "score": cmd_score}
    try:
        return commands[args.command](args)
    except (KeyError, ValueError) as exc:
        print(f"❌ {exc.args[0] if exc.args else exc}")
        return 2
    except KeyboardInterrupt:
        print("\nCancelled.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
