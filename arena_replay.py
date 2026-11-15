"""Testing arena: replay a take through cuVSLAM and the dashboard.

Everything runs in its own ROS domain (default 87), so a replay never meets
the live pipeline's topics (domain 0) -- the live camera, SLAM and dashboard
can keep running. In that domain:

* cuVSLAM, from isaac_vp.launch.py with VP_REPLAY=1: same parameters as live,
  no camera node, no emitter splitter.
* the dashboard under test (the code staged by arena.sh), with a replay-safe
  copy of the config: FreeD goes to an unused local port, the Unreal video
  plate, GNSS serial, GoPro and LTC are off. It opens the take's scene and
  arms Auto lineup by itself (VP_ARENA_SCENE), as the operator had it.
* this player: the take's camera, IMU and TF, published at their recorded
  pace with every header stamp shifted by one offset so they read as
  "now". The dashboard judges freshness against time.time(); without the
  shift every replayed pose would be stale and the output would read Held.

Not replayed: SLAM's own output (/visual_slam/*, /tf) -- cuVSLAM regenerates
it -- and the dashboard's records, which the dashboard under test publishes
afresh. GNSS is not fed back yet: it only gates fusion once gnss.calibrated is
true (RTK_SETUP.md), which it is not.
"""
import datetime
import json
import os
import signal
import struct
import subprocess
import threading
import time

import yaml

import arena_state

REPLAY_DOMAIN = 87
DEAD_FREED_PORT = 59111            # nothing listens here; FreeD must not reach Unreal
EXCLUDED = ("/tf", "/rosout", "/diagnostics", "/parameter_events")
EXCLUDED_PREFIXES = ("/visual_slam/", "/hyperfusion/")
READY_TIMEOUT_S = 150.0


# ── Pure helpers (unit-tested) ───────────────────────────────────────────────
def plays(topic):
    """Whether a recorded topic is replayed (tf_static is handled separately)."""
    return (topic not in EXCLUDED and topic != "/tf_static"
            and not topic.startswith(EXCLUDED_PREFIXES))


def restamp(raw, offset_ns):
    """Shift header.stamp of a CDR message whose first field is a Header."""
    order = "<" if raw[1] & 1 else ">"
    sec, nsec = struct.unpack_from(order + "iI", raw, 4)
    total = sec * 1_000_000_000 + nsec + int(offset_ns)
    buf = bytearray(raw)
    struct.pack_into(order + "iI", buf, 4, total // 1_000_000_000, total % 1_000_000_000)
    return bytes(buf)


def header_first(msg_cls):
    """True when the message type starts with a std_msgs/Header."""
    fields = iter(msg_cls.get_fields_and_field_types().items())
    first = next(fields, None)
    return first is not None and first == ("header", "std_msgs/Header")


def replay_config(base_path, out_path):
    """A copy of the config that cannot reach Unreal or real hardware."""
    with open(base_path) as fh:
        cfg = yaml.safe_load(fh) or {}

    def section(name):
        cfg[name] = cfg.get(name) or {}
        return cfg[name]

    section("network").update(unreal_ip="127.0.0.1", unreal_port=DEAD_FREED_PORT)
    section("unreal_video")["enabled"] = False
    section("gnss")["enabled"] = False
    section("gopro")["auto_start"] = False
    section("timecode")["enabled"] = False
    with open(out_path, "w") as fh:
        fh.write(f"# Replay-safe copy of {base_path}: FreeD -> 127.0.0.1:{DEAD_FREED_PORT}, "
                 "Unreal video, GNSS, GoPro and LTC off.\n")
        yaml.safe_dump(cfg, fh, sort_keys=False)
    return cfg


# ── Processes ────────────────────────────────────────────────────────────────
def _spawn(cmd, env, log_path):
    log = open(log_path, "w")
    return subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=True), log


def _stop(proc, first=signal.SIGINT, timeout=10.0):
    """Stop one process group we started -- never anything else."""
    if proc.poll() is not None:
        return
    for sig, wait in ((first, timeout), (signal.SIGKILL, 5.0)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=wait)
            return
        except subprocess.TimeoutExpired:
            continue


# ── Replay ───────────────────────────────────────────────────────────────────
def run_replay(take_dir, run_dir, *, code_dir, base_config, show=False,
               domain=REPLAY_DOMAIN, log=print):
    """Replay one take; writes state.jsonl, run.json and logs into run_dir."""
    import rclpy
    import rosbag2_py
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import (QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy)
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    from std_msgs.msg import String
    from tf2_msgs.msg import TFMessage

    with open(os.path.join(take_dir, "take.json")) as fh:
        take = json.load(fh)
    os.makedirs(run_dir)
    config_path = os.path.join(run_dir, "tracking.yaml")
    replay_config(base_config, config_path)
    scene_path = os.path.join(take_dir, "scene.json")
    scene = None
    if os.path.isfile(scene_path):
        with open(scene_path) as fh:
            scene = json.load(fh)

    env = dict(os.environ, ROS_DOMAIN_ID=str(domain), VP_TRACKING_CONFIG=config_path)
    env.pop("VP_ARENA_SCENE", None)
    rec_env = take.get("environment") or {}
    slam_env = dict(env, VP_REPLAY="1",
                    VP_ENABLE_DEPTH=rec_env.get("VP_ENABLE_DEPTH", "1"),
                    VP_EMITTER_MODE=rec_env.get("VP_EMITTER_MODE", "alternating"),
                    VP_CAMERA_FPS=rec_env.get("VP_CAMERA_FPS", "90"))
    gxf = ("export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(find /workspaces/isaac_ros-dev/install "
           "-type d -name lib | grep gxf | tr '\\n' ':')")
    slam, slam_log = _spawn(["/bin/bash", "-c", f"{gxf} && exec ros2 launch "
                             f"{os.path.join(code_dir, 'isaac_vp.launch.py')}"],
                            slam_env, os.path.join(run_dir, "slam.log"))
    dash_env = dict(env, PYTHONFAULTHANDLER="1")
    if scene is not None:
        dash_env["VP_ARENA_SCENE"] = scene_path
    if show and env.get("DISPLAY"):
        dash_env.setdefault("QT_XCB_GL_INTEGRATION", "xcb_egl")
    else:
        dash_env["QT_QPA_PLATFORM"] = "offscreen"
    dashboard, dash_log = _spawn(["python3", "-u", os.path.join(code_dir, "dashboard.py")],
                                 dash_env, os.path.join(run_dir, "dashboard.log"))

    os.environ["ROS_DOMAIN_ID"] = str(domain)      # this process joins the replay domain
    rclpy.init(args=None)
    node = rclpy.create_node("hyperfusion_arena_replay")
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    running = [True]

    def _spin():
        # spin_once in a loop so the finally below can stop and join it
        # before the node goes (see arena._Ros._spin).
        while running[0]:
            try:
                executor.spin_once(timeout_sec=0.1)
            except Exception:
                if not running[0]:
                    break

    spin = threading.Thread(target=_spin, daemon=True)
    spin.start()

    latest = dict(state=None, scene=None)
    state_file = open(os.path.join(run_dir, "state.jsonl"), "w")

    def on_state(msg):
        state_file.write(msg.data + "\n")
        try:
            latest["state"] = arena_state.decode(msg.data)
        except ValueError:
            pass

    reliable = QoSProfile(depth=200, reliability=ReliabilityPolicy.RELIABLE)
    latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         reliability=ReliabilityPolicy.RELIABLE)
    node.create_subscription(String, arena_state.STATE_TOPIC, on_state, reliable)
    node.create_subscription(String, arena_state.SCENE_TOPIC,
                             lambda m: latest.__setitem__("scene", m.data), latched)

    manifest = dict(format="hyperfusion.arena_run", version=1, take=os.path.basename(take_dir),
                    run=os.path.basename(run_dir),
                    created=datetime.datetime.now().isoformat(timespec="seconds"),
                    code_rev=os.environ.get("ARENA_GIT_REV"), base_config=base_config,
                    domain=domain, scene=bool(scene))
    try:
        bag_dir = os.path.join(take_dir, "bag")
        reader = rosbag2_py.SequentialReader()
        reader.open(rosbag2_py.StorageOptions(uri=bag_dir, storage_id="mcap"),
                    rosbag2_py.ConverterOptions("", ""))
        types = {t.name: t.type for t in reader.get_all_topics_and_types()}

        # Static TF first, merged into one latched message as tf2 expects.
        static = {}
        tf_reader = rosbag2_py.SequentialReader()
        tf_reader.open(rosbag2_py.StorageOptions(uri=bag_dir, storage_id="mcap"),
                       rosbag2_py.ConverterOptions("", ""))
        tf_reader.set_filter(rosbag2_py.StorageFilter(topics=["/tf_static"]))
        while tf_reader.has_next():
            for tr in deserialize_message(tf_reader.read_next()[1], TFMessage).transforms:
                static[tr.child_frame_id] = tr
        tf_pub = node.create_publisher(TFMessage, "/tf_static", latched)
        if static:
            tf_pub.publish(TFMessage(transforms=list(static.values())))
        else:
            log("   ⚠️  the take has no /tf_static; cuVSLAM may not find the camera frames")

        stream_qos = QoSProfile(depth=10, history=HistoryPolicy.KEEP_LAST,
                                reliability=ReliabilityPolicy.RELIABLE)
        pubs = {}
        for topic, type_name in types.items():
            if plays(topic):
                cls = get_message(type_name)
                pubs[topic] = (node.create_publisher(cls, topic, stream_qos), header_first(cls))
        reader.set_filter(rosbag2_py.StorageFilter(topics=list(pubs)))

        # Wait for cuVSLAM and the dashboard (scene loaded) before the clock starts.
        log(f"Starting cuVSLAM and the dashboard in ROS domain {domain}...")
        deadline = time.time() + READY_TIMEOUT_S
        infra1 = pubs.get("/camera/infra1/image_rect_raw", (None,))[0]
        infra2 = pubs.get("/camera/infra2/image_rect_raw", (None,))[0]
        want_map = (scene or {}).get("map_file")
        while True:
            for name, proc in (("cuVSLAM", slam), ("dashboard", dashboard)):
                if proc.poll() is not None:
                    raise RuntimeError(f"{name} exited during start-up ({proc.returncode}); "
                                       f"see {run_dir}/{'slam' if name == 'cuVSLAM' else 'dashboard'}.log")
            slam_ready = (infra1 is not None and infra1.get_subscription_count() >= 2
                          and infra2.get_subscription_count() >= 1)
            scene_ready = (not want_map or (latest["scene"] is not None
                                            and json.loads(latest["scene"]).get("map_file") == want_map))
            if slam_ready and scene_ready and latest["state"] is not None:
                break
            if time.time() > deadline:
                raise RuntimeError("cuVSLAM or the dashboard did not become ready in "
                                   f"{READY_TIMEOUT_S:.0f} s (slam={slam_ready}, scene={scene_ready})")
            time.sleep(0.5)

        # Play at the recorded pace; one offset for every stamp and the clock.
        counts = {t: 0 for t in pubs}
        lead_in = 0.5
        first = None
        offset_ns = 0
        max_late = 0.0
        late = 0
        last_print = 0.0
        duration = float(take.get("duration_s") or 0.0)
        log("Replaying...")
        while reader.has_next():
            topic, raw, t_recv = reader.read_next()
            if first is None:
                first = t_recv
                t0 = time.time() + lead_in
                offset_ns = int(t0 * 1e9) - first
                manifest["stamp_offset_s"] = offset_ns / 1e9
            target = t0 + (t_recv - first) / 1e9
            now = time.time()
            if target > now:
                time.sleep(target - now)
            elif now - target > 0.05:
                late += 1
                max_late = max(max_late, now - target)
            pub, stamped = pubs[topic]
            pub.publish(restamp(raw, offset_ns) if stamped else bytes(raw))
            counts[topic] += 1
            if now - last_print > 1.0:
                last_print = now
                st = latest["state"] or {}
                lineup = (st.get("lineup") or {}).get("title", "")
                print(f"\r\x1b[K▶ REPLAY {int(now - t0) // 60:02d}:{int(now - t0) % 60:02d}"
                      f" / {int(duration) // 60:02d}:{int(duration) % 60:02d}"
                      f" | FreeD {st.get('mode', '?')}" + (f" · {lineup}" if lineup else ""),
                      end="", flush=True)
        print()
        time.sleep(3.0)                  # let the last output records arrive
        manifest.update(played=counts, late_messages=late, max_late_ms=round(max_late * 1000, 1),
                        end_reason="take finished")
        if late:
            log(f"   ⚠️  the player fell behind {late} times (worst {max_late * 1000:.0f} ms): "
                "the machine was busy, so this replay is less faithful than usual")
    except KeyboardInterrupt:
        manifest["end_reason"] = "interrupted"
        raise
    except Exception as exc:
        manifest["end_reason"] = f"failed: {exc}"
        raise
    finally:
        running[0] = False
        spin.join(timeout=2.0)
        # The dashboard's SIGINT only shuts its ROS context down (rclpy's own
        # handler) and leaves Qt running, so it gets SIGTERM and a short wait.
        _stop(dashboard, first=signal.SIGTERM, timeout=3.0)
        _stop(slam)                      # ros2 launch: SIGINT stops it cleanly
        slam_log.close()
        dash_log.close()
        state_file.close()
        if latest["scene"] is not None:
            with open(os.path.join(run_dir, "scene.json"), "w") as fh:
                json.dump(json.loads(latest["scene"]), fh, indent=2)
        with open(os.path.join(run_dir, "run.json"), "w") as fh:
            json.dump(manifest, fh, indent=2)
        try:
            executor.shutdown(timeout_sec=1.0)
            node.destroy_node()
        finally:
            rclpy.try_shutdown()
    return manifest
