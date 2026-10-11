#!/usr/bin/env python3
"""
TrackingEngine — the front-end-agnostic core of the camera tracking bridge.

Everything that decides *where the virtual camera is* lives here: the pose
pipeline, ArUco marker solving, anchoring, FreeD encoding, and the command
protocol Unreal speaks. `dashboard.py` (GUI) and `bridge_headless.py` (no GUI)
are both thin front-ends over this class — they own ROS subscriptions, sockets
and widgets, and nothing else.

That split exists because the two front-ends had each grown their own copy of
the pose path and the anchoring maths, in subtly different code. Two copies of
a coordinate transform is two chances to be wrong, and only one of them gets
fixed when a bug is found.

No ROS, no Qt imports. OpenCV is imported lazily and only for marker solving,
so the pose path can be unit tested without it.

Anchoring, and why it takes a target
------------------------------------
The LiDAR-scan workflow is:

    a printed ArUco marker is taped to a known spot in the physical set
    the same set has been LiDAR-scanned and imported into Unreal
    an anchor Actor sits in Unreal at that same known spot
    the operator points the camera at the marker and presses anchor

At that instant the tracking output must start being expressed in the *scan's*
coordinate frame — because that is the frame the virtual set lives in. So the
marker has to map onto the anchor Actor's transform, not onto the origin.

Anchoring to the origin (what this used to do unconditionally) puts the virtual
camera off by exactly the anchor Actor's position. It looks perfect whenever
that Actor happens to sit near the origin, which is what makes it a trap.
"""
import json
from collections import deque
from fusion_supervisor import FusionSupervisor
from gnss_receiver import local_position, quality_error
import math
import threading
import time

import numpy as np

import freed
#: T_BODY_OPTICAL (camera optical -> camera_link body) is re-exported from
#: pose_transform, where it now lives: solvePnP here, the depth point cloud in
#: lidar_map_matcher.py and the marker solve all need it, and three separate
#: copies of the same 4x4 is three chances to fix only one of them.
from pose_transform import (T_BODY_OPTICAL, PoseGate, PosePredictor,
                            PoseSmoother, PoseTransformer, StationaryHold,
                            make_transform, rpy_to_matrix, matrix_to_quaternion,
                            se3_interpolate, so3_log)


class MarkerSolution:
    """Result of solving a marker sighting, including why it was rejected."""

    __slots__ = ("ok", "reason", "T_world_marker", "T_optical_marker",
                 "distance_pnp_m", "distance_depth_m", "depth_residual_m",
                 "used_depth")

    def __init__(self, ok=False, reason="", T_world_marker=None,
                 T_optical_marker=None, distance_pnp_m=None,
                 distance_depth_m=None, depth_residual_m=None,
                 used_depth=False):
        self.ok = ok
        self.reason = reason
        self.T_world_marker = T_world_marker
        self.T_optical_marker = T_optical_marker
        self.distance_pnp_m = distance_pnp_m
        self.distance_depth_m = distance_depth_m
        self.depth_residual_m = depth_residual_m
        self.used_depth = used_depth

    def __repr__(self):
        bits = [f"ok={self.ok}"]
        if self.distance_pnp_m is not None:
            bits.append(f"d_pnp={self.distance_pnp_m:.3f}m")
        if self.distance_depth_m is not None:
            bits.append(f"d_depth={self.distance_depth_m:.3f}m")
        if self.depth_residual_m is not None:
            bits.append(f"residual={self.depth_residual_m * 1000:.0f}mm")
        if self.reason:
            bits.append(f"reason={self.reason!r}")
        return f"MarkerSolution({', '.join(bits)})"


class TrackingEngine:
    """Pose pipeline + marker anchoring + FreeD output. No I/O of its own."""

    def __init__(self, config, log=print):
        self.cfg = config
        self.log = log

        trk = config["tracking"]
        net = config["network"]
        aru = config["aruco"]

        self.transformer = PoseTransformer(config["transform"])
        self.predictor = PosePredictor(
            lead_time_s=trk["lead_time_s"],
            max_extrapolation_s=trk["max_extrapolation_s"])
        self.smoother = PoseSmoother(alpha=trk["smoothing_alpha"])
        self.stale_timeout = float(trk["stale_pose_timeout_s"])
        self.gate = PoseGate(
            enabled=trk.get("gate_enabled", True),
            max_speed_m_s=trk.get("gate_max_speed_m_s", 3.0),
            max_angular_deg_s=trk.get("gate_max_angular_deg_s", 900.0),
            max_jump_m=trk.get("gate_max_jump_m", 0.05),
            max_consecutive_rejects=trk.get("gate_max_consecutive_rejects", 15))
        self.hold = StationaryHold(
            enabled=trk.get("hold_enabled", True),
            gyro_still_rad_s=trk.get("hold_gyro_still_rad_s", 0.05),
            gyro_release_rad_s=trk.get("hold_gyro_release_rad_s", 0.08),
            accel_still_std=trk.get("hold_accel_still_std", 0.15),
            window_samples=trk.get("hold_window_samples", 60),
            vo_release_m=trk.get("hold_vo_release_m", 0.003),
            vo_window_samples=trk.get("hold_vo_window_samples", 12))

        self.camera_id = int(net["camera_id"])
        self.marker_size = float(aru["marker_size_m"])

        # Depth cross-check on the marker solve. Four coplanar corners give a
        # two-fold pose ambiguity, so a mono solve can silently flip; an
        # independent distance measurement catches that before it becomes the
        # anchor for the whole scene.
        self.depth_check_enabled = bool(aru.get("depth_check_enabled", True))
        self.depth_tolerance_m = float(aru.get("depth_tolerance_m", 0.10))
        self.depth_refine = bool(aru.get("depth_refine", True))

        fb = aru["fallback_intrinsics"]
        self.aruco_K = np.array([[fb["fx"], 0.0, fb["cx"]],
                                 [0.0, fb["fy"], fb["cy"]],
                                 [0.0, 0.0, 1.0]])
        self.aruco_D = np.zeros((5, 1))
        self.intrinsics_from_camera_info = False

        self._depth_mm = None
        self._depth_K = None

        # Anchor state. `anchor_target` is the target transform an anchored
        # marker maps onto, expressed in the ROS canonical frame. It can come
        # from the dashboard's own locator gizmo (already ROS metres, see
        # set_anchor_target_ros) or from an external Unreal-units transform
        # (see set_anchor_target_from_unreal_units, used by bridge_headless.py
        # / the anchor_lidar-set_anchor command pair).
        self.anchor_requested = False
        self.anchor_target = None
        self.anchor_target_source = None      # for the status feed / UI
        self.last_marker_solution = None

        self.latest_pose = None               # (position, quaternion) in ROS
        self.latest_unreal = None             # last UnrealPose emitted
        self.slam_status = "WAITING FOR SLAM"
        self.last_vio_time = 0.0
        self.pose_updates = 0
        self.packets_sent = 0
        #: True once LiDAR-map fusion is driving ground zero (see
        #: apply_map_alignment). Distinguishes "anchored by marker, drifting"
        #: from "continuously localised against the scan".
        self.map_aligned = False
        self._lock = threading.RLock()
        self.fusion = FusionSupervisor(config.get("fusion", {}))
        self._pose_history = deque(maxlen=500)
        self._last_sensor_stamp = None
        self._last_pose_mono = None
        self._last_odometry_mono = None
        self._last_depth_mono = None
        self._depth_stamp = None
        self._depth_sample = None
        self._pose_rejected = False
        self._output_T = None
        self._output_pose = None
        self._output_time = None
        self._output_held = False
        self._relineup_required = False
        self.gnss_fix = None
        self.gnss_reason = "GNSS disabled; outdoor setup required"
        self._gnss_epoch = None

    # ── sensors in ───────────────────────────────────────────────────────────
    def update_imu(self, gyro, accel):
        self.hold.update_imu(gyro, accel)

    def set_intrinsics(self, K, D=None):
        """Real intrinsics from camera_info. First valid message wins."""
        if self.intrinsics_from_camera_info:
            return False
        K = np.asarray(K, dtype=float).reshape(3, 3)
        if not np.isfinite(K).all() or K[0, 0] <= 0.0 or K[1, 1] <= 0.0:
            return False
        self.aruco_K = K
        self.aruco_D = (np.asarray(D, dtype=float).reshape(-1, 1)
                        if D is not None and len(D) else np.zeros((5, 1)))
        self.intrinsics_from_camera_info = True
        self.log(f"[Intrinsics] from camera_info: fx={K[0,0]:.2f} fy={K[1,1]:.2f} "
                 f"cx={K[0,2]:.2f} cy={K[1,2]:.2f}")
        return True

    def set_depth(self, depth_mm, K=None, stamp=None):
        """Latest depth frame (16-bit millimetres) for the marker cross-check."""
        self._depth_mm = depth_mm
        self._last_depth_mono = time.monotonic()
        self._depth_stamp = stamp
        if K is not None:
            self._depth_K = np.asarray(K, dtype=float).reshape(3, 3)
        self._depth_sample = (depth_mm, stamp, self._last_depth_mono)

    def set_depth_intrinsics(self, K):
        """Depth stream intrinsics, from the depth camera_info topic.

        Separate from the colour/infra intrinsics: the marker cross-check and
        the LiDAR map fusion both deproject depth pixels, and doing that with
        another stream's focal length bends the point cloud.
        """
        K = np.asarray(K, dtype=float).reshape(3, 3)
        if not np.isfinite(K).all() or K[0, 0] <= 0.0 or K[1, 1] <= 0.0:
            return False
        self._depth_K = K
        return True

    @property
    def depth_intrinsics(self):
        """Depth K if the camera published it, else the best available."""
        return self._depth_K if self._depth_K is not None else self.aruco_K

    @property
    def has_depth(self):
        return self._depth_mm is not None

    # ── pose pipeline ────────────────────────────────────────────────────────
    def accept_pose(self, position, quaternion, linear, angular, stamp=None, *, source=None):
        """Single entry point for every incoming SLAM pose.

        Order matters: absorb tracker snaps first, then pin if the rig is
        physically still. Both stages work by accumulating an offset, so neither
        introduces a discontinuity in the output.
        """
        now = time.time()
        mono = time.monotonic()
        # Prefer odometry's velocity while that topic is fresh. A VO sample
        # must not suppress the next VO sample: using last_vio_time here
        # throttled a VO-only stream to one pose per second.
        if (source == "vo" and self._last_odometry_mono is not None
                and 0 <= mono - self._last_odometry_mono <= self.stale_timeout):
            return False
        values = np.concatenate([np.asarray(v, dtype=float).ravel()
                                 for v in (position, quaternion, linear, angular)])
        if values.size != 13 or not np.isfinite(values).all() or np.linalg.norm(quaternion) < 1e-9:
            self._pose_rejected = True
            return False
        if stamp is not None:
            if not math.isfinite(stamp) or (self._last_sensor_stamp is not None and stamp <= self._last_sensor_stamp):
                return False
            self._last_sensor_stamp = stamp
        quaternion = np.asarray(quaternion, dtype=float) / np.linalg.norm(quaternion)
        gate_t = stamp if stamp is not None else now

        position, quaternion, snapped, reason = self.gate.filter(
            position, quaternion, gate_t)
        if snapped:
            linear = (0.0, 0.0, 0.0)
            angular = (0.0, 0.0, 0.0)
            if self.gate.consecutive_rejects <= 1:
                self.log(f"[Gate] {reason} — holding pose")

        self._pose_rejected = snapped
        position, quaternion, held = self.hold.apply(position, quaternion)
        if held:
            linear = (0.0, 0.0, 0.0)
            angular = (0.0, 0.0, 0.0)

        with self._lock:
            self.latest_pose = (position, quaternion)
            self.predictor.update(position, quaternion, linear, angular, now)
            self.last_vio_time = now
            self.pose_updates += 1
            self._last_pose_mono = mono
            if not snapped:
                if source == "odometry":
                    self._last_odometry_mono = mono
                self._pose_history.append((stamp, mono, np.array(position, copy=True),
                                           np.array(quaternion, copy=True)))

        if self.slam_status != "TRACKING":
            self.slam_status = "TRACKING"
            self.log("[SLAM] ✅ Tracking — streaming 6DOF to Unreal Engine.")
        return True

    def camera_pose_now(self):
        """(UnrealPose, is_valid). Holds the last good pose when tracking is stale."""
        if self.fusion.enabled:
            return self._supervised_pose()
        now = time.time()
        with self._lock:
            if not self.predictor.has_pose:
                return None, False
            fresh = self.predictor.age(now) <= self.stale_timeout
            predicted = self.predictor.predict(now) if fresh else None
            fallback = self.latest_pose

        if not fresh and self._output_pose is not None:
            return self._output_pose, False
        position, quaternion = predicted if predicted is not None else fallback
        pose = self.transformer.ros_to_unreal(position, quaternion)
        if fresh:
            pose = self.smoother.apply(pose)
        self._output_pose = pose
        return pose, fresh

    def camera_pose_ros_corrected(self):
        """Latest raw SLAM pose with ground-zero applied, in ROS canonical
        metres -- the same frame as the marker solve and the dashboard's
        locator gizmo. Use this (not the raw latest_pose) to draw the camera
        in anchor-relative space, e.g. in the dashboard's 3D viewport, so it
        visually snaps next to the locator the moment an anchor is set.

        Unanchored, this returns the untouched raw pose (see
        PoseTransformer.pose_with_zero_ros), so nothing changes here until
        the operator actually anchors.
        """
        if self.fusion.enabled:
            self.camera_pose_now()
            if self._output_T is None:
                return None
            return self._output_T[:3, 3].copy(), matrix_to_quaternion(self._output_T[:3, :3])
        with self._lock:
            pose = self.latest_pose
        if pose is None:
            return None
        return self.transformer.pose_with_zero_ros(pose[0], pose[1])

    def freed_packet(self):
        """The 29-byte FreeD packet for right now, or None if nothing tracked."""
        pose, valid = self.camera_pose_now()
        if pose is None:
            return None
        if self.fusion.enabled:
            self.slam_status = self.fusion_status()["state"]
        if not self.fusion.enabled and not valid and self.slam_status == "TRACKING":
            self.slam_status = "TRACKING LOST — HOLDING"
            self.log("[SLAM] ⚠️  Pose stale — holding last known position.")
        self.latest_unreal = pose
        self.packets_sent += 1
        return freed.get_freed_packet(self.camera_id, pose.tilt, pose.pan,
                                      pose.roll, pose.x, pose.y, pose.z)

    def pose_at(self, stamp, *, host_time=False, tolerance=None):
        """Nearest accepted sample in the requested clock domain, within a bound.

        Never pair a depth image with an arbitrary latest pose. No conversion
        between ROS and host clocks is inferred here.
        """
        if stamp is None or not math.isfinite(stamp):
            return None
        tolerance = (self.cfg.get("fusion", {}).get("depth_pose_tolerance_s", 0.035)
                     if tolerance is None else tolerance)
        with self._lock:
            samples = [h for h in self._pose_history if host_time or h[0] is not None]
        if not samples:
            return None
        key = 1 if host_time else 0
        sample = min(samples, key=lambda h: abs(h[key] - stamp))
        if abs(sample[key] - stamp) > tolerance:
            return None
        return sample[2].copy(), sample[3].copy()

    def capture_depth(self):
        """One atomic image/timestamp reference with its matching accepted pose."""
        sample = self._depth_sample
        if sample is None or self._depth_K is None:
            return None
        frame, stamp, received = sample
        if time.monotonic() - received > self.cfg.get("fusion", {}).get("depth_timeout_s", 0.5):
            return None
        pose = self.pose_at(stamp)
        if pose is None:
            return None
        return frame, self._depth_K.copy(), pose, received

    def depth_pose(self):
        sample = self.capture_depth()
        return None if sample is None else sample[2]

    def observe_scan(self, measurement):
        if measurement is None or measurement.monotonic_timestamp is None:
            return
        ok = measurement.ok and measurement.observable_rank >= self.cfg.get("fusion", {}).get("min_scan_rank", 3)
        reason = measurement.reason
        if measurement.ok and not ok:
            reason = "Scan geometry does not constrain all required translation axes"
        self.fusion.observe("scan", ok, measurement.monotonic_timestamp, reason,
                            score=float(measurement.verify_score),
                            rank=int(measurement.observable_rank),
                            rmse_mm=float(measurement.rmse_m * 1000))

    def confirm_lineup(self):
        """Explicit operator realignment permits a new output origin."""
        with self._lock:
            self.fusion.reset()
            self._output_T = self._output_pose = None
            self._output_time = None
            self._output_held = False
            self._relineup_required = False
            self._gnss_epoch = None

    def update_gnss(self, fix):
        self.gnss_fix = fix
        cfg = self.cfg["gnss"]
        reason = quality_error(fix, cfg, time.monotonic())
        residual = None
        if not reason:
            try:
                antenna = local_position(fix, cfg)
                pose = self.pose_at(fix.received - cfg.get("latency_s", 0.0),
                                    host_time=True, tolerance=cfg.get("pose_tolerance_s", 0.05))
                if pose is None or not self.is_anchored:
                    raise ValueError("No time-paired, aligned camera pose for GNSS")
                # Lever arm is relative to the tracking camera, not the lens.
                T = self.transformer._zero @ make_transform(*pose)
                predicted = T[:3, 3] + T[:3, :3] @ np.asarray(cfg.get("antenna_offset_body_m", [0, 0, 0]))
                residual = float(np.linalg.norm(antenna - predicted))
                if not math.isfinite(residual) or residual > cfg.get("max_innovation_m", 0.20):
                    raise ValueError("GNSS disagrees with camera position; check alignment / multipath")
            except (ValueError, TypeError) as exc:
                reason = str(exc)
        self.gnss_reason = reason or "RTK position agrees with camera"
        # GGA can be upgraded by GST for the same epoch. Only quality-complete
        # fixes count; a plain GGA must not reset every consecutive good streak.
        if fix.hacc_m is not None or fix.quality != 4:
            if fix.epoch != self._gnss_epoch:
                self._gnss_epoch = fix.epoch
                self.fusion.observe("gnss", not reason, fix.received,
                                    self.gnss_reason, residual_m=residual)
        return not reason

    def fusion_status(self):
        fresh = self._last_pose_mono is not None and 0 <= time.monotonic() - self._last_pose_mono <= self.stale_timeout and not self._pose_rejected
        status = self.fusion.evaluate(time.monotonic(), fresh, self.is_anchored)
        if self._last_pose_mono is None:
            status.update(state="WAITING", reason="Waiting for D455 camera data",
                          output_allowed=False, validated=False)
        if self._relineup_required:
            status.update(state="RELINEUP", reason="Camera moved too far during hold; repeat lineup",
                          output_allowed=False, validated=False)
        status["enabled"] = self.fusion.enabled
        return status

    def _supervised_pose(self):
        with self._lock:
            now = time.monotonic()
            status = self.fusion_status()
            if not self.predictor.has_pose or not status["output_allowed"]:
                self._output_held = self._output_T is not None
                self._output_time = now
                return self._output_pose, False
            position, quaternion = self.predictor.predict(time.time())
            T = make_transform(position, quaternion) @ self.transformer._rig
            if self.transformer._zero is not None:
                T = self.transformer._zero @ T
            if self._output_held and self._output_T is not None:
                cfg = self.cfg.get("fusion", {})
                distance = float(np.linalg.norm(T[:3, 3] - self._output_T[:3, 3]))
                angle = math.degrees(float(np.linalg.norm(so3_log(self._output_T[:3, :3].T @ T[:3, :3]))))
                if distance > cfg.get("recovery_max_distance_m", 0.10) or angle > cfg.get("recovery_max_angle_deg", 5.0):
                    self._relineup_required = True
                    return self._output_pose, False
                dt = min(max(now - (self._output_time or now), 0.0), 0.05)
                alpha = min(1.0, cfg.get("recovery_slew_m_s", 0.10) * dt / max(distance, 1e-9),
                            cfg.get("recovery_slew_deg_s", 5.0) * dt / max(angle, 1e-9))
                T = se3_interpolate(self._output_T, T, alpha)
                self._output_held = alpha < 1.0
            self._output_T = T
            self._output_time = now
            self._output_pose = self.transformer.world_transform_to_unreal(T)
            return self._output_pose, status["validated"] and not self._output_held

    # ── marker solving ───────────────────────────────────────────────────────
    def solve_marker_pose(self, corners):
        """Solve a marker sighting into a pose in the VIO world frame.

        `corners` is one marker's 4x2 image points, in the ArUco corner order.

        Uses IPPE_SQUARE, which is built for square fiducials, rather than the
        general iterative solver: with only four coplanar points the iterative
        solution can settle on the mirrored pose, and a flipped anchor takes the
        whole virtual set with it.
        """
        import cv2

        with self._lock:
            latest = self.latest_pose
        if latest is None:
            return MarkerSolution(reason="no SLAM pose yet")

        s = self.marker_size / 2.0
        obj = np.array([[-s, s, 0.0], [s, s, 0.0], [s, -s, 0.0], [-s, -s, 0.0]],
                       dtype=np.float32)
        img = np.asarray(corners, dtype=np.float32).reshape(4, 2)

        flag = getattr(cv2, "SOLVEPNP_IPPE_SQUARE", cv2.SOLVEPNP_ITERATIVE)
        ok, rvec, tvec = cv2.solvePnP(obj, img, self.aruco_K, self.aruco_D,
                                      flags=flag)
        if not ok:
            return MarkerSolution(reason="solvePnP failed")

        tvec = np.asarray(tvec, dtype=float).reshape(3)
        d_pnp = float(np.linalg.norm(tvec))

        d_depth, residual, used_depth = None, None, False
        if self.depth_check_enabled and self._depth_mm is not None:
            d_depth = self._marker_depth_distance(img, tvec)
            if d_depth is not None:
                residual = abs(d_depth - d_pnp)
                used_depth = True
                if residual > self.depth_tolerance_m:
                    return MarkerSolution(
                        reason=(f"depth disagrees with PnP by "
                                f"{residual * 1000:.0f}mm "
                                f"(> {self.depth_tolerance_m * 1000:.0f}mm): "
                                f"PnP {d_pnp:.3f}m vs depth {d_depth:.3f}m"),
                        distance_pnp_m=d_pnp, distance_depth_m=d_depth,
                        depth_residual_m=residual, used_depth=True)
                if self.depth_refine and d_pnp > 1e-6:
                    # Distance is the weakest axis of a small-marker PnP solve;
                    # depth measures it directly. Rotation stays from PnP.
                    tvec = tvec * (d_depth / d_pnp)

        R_c_m, _ = cv2.Rodrigues(rvec)
        T_optical_marker = np.eye(4)
        T_optical_marker[:3, :3] = R_c_m
        T_optical_marker[:3, 3] = tvec

        position, quat = latest
        T_world_cam = make_transform(position, quat)
        T_world_marker = T_world_cam @ T_BODY_OPTICAL @ T_optical_marker

        return MarkerSolution(
            ok=True, T_world_marker=T_world_marker,
            T_optical_marker=T_optical_marker,
            distance_pnp_m=d_pnp, distance_depth_m=d_depth,
            depth_residual_m=residual, used_depth=used_depth)

    def _marker_depth_distance(self, img_corners, tvec):
        """Median depth over the marker quad, as a range in metres.

        Returns None when depth is unusable there (holes are common on printed
        markers, which are flat and often low-reflectivity in IR).
        """
        depth = self._depth_mm
        if depth is None:
            return None
        h, w = depth.shape[:2]
        K = self._depth_K if self._depth_K is not None else self.aruco_K

        cx_img = float(np.mean(img_corners[:, 0]))
        cy_img = float(np.mean(img_corners[:, 1]))
        # Sample a small patch at the marker centre rather than one pixel.
        half = max(2, int(0.15 * np.linalg.norm(
            img_corners[0] - img_corners[2]) / 2.0))
        x0, x1 = int(max(0, cx_img - half)), int(min(w, cx_img + half + 1))
        y0, y1 = int(max(0, cy_img - half)), int(min(h, cy_img + half + 1))
        if x1 <= x0 or y1 <= y0:
            return None

        patch = np.asarray(depth[y0:y1, x0:x1], dtype=float)
        valid = patch[(patch > 0) & np.isfinite(patch)]
        if valid.size < 8:
            return None
        z = float(np.median(valid)) / 1000.0        # 16UC1 millimetres -> metres
        if z <= 0.0:
            return None

        # Depth is Z along the optical axis; convert the centre pixel to a ray
        # so the comparison is range-to-range rather than Z-to-range.
        fx, fy = K[0, 0], K[1, 1]
        px, py = K[0, 2], K[1, 2]
        x = (cx_img - px) / fx * z
        y = (cy_img - py) / fy * z
        return float(np.linalg.norm((x, y, z)))

    # ── anchoring ────────────────────────────────────────────────────────────
    def set_anchor_target_from_unreal_units(self, position_unreal, rotation_deg,
                                            source="unreal"):
        """Record an anchor target given in Unreal world units (centimetres).

        `position_unreal` is in Unreal world units (centimetres) and
        `rotation_deg` is (pan, tilt, roll) — i.e. straight off an Actor. It is
        converted into the ROS canonical frame here, using the inverse of this
        instance's own output mapping, so a sign flipped in tracking.yaml moves
        the anchor and the live pose together.
        """
        self.anchor_target = self.transformer.unreal_to_ros_transform(
            position_unreal, rotation_deg)
        self.anchor_target_source = source
        self.log(f"[Anchor] target set from {source}: "
                 f"Unreal {tuple(round(float(v), 1) for v in position_unreal)} cm "
                 f"pan/tilt/roll {tuple(round(float(v), 1) for v in rotation_deg)} "
                 f"-> ROS {self.anchor_target[:3, 3].round(4)} m")
        return self.anchor_target

    def set_anchor_target_ros(self, T_ros, source="dashboard_locator"):
        """Record an anchor target already expressed in ROS canonical metres.

        Used by the dashboard's own 3D-viewport locator gizmo, which lives in
        the same frame as the marker solve (T_world_marker) -- no Unreal-unit
        conversion needed, unlike set_anchor_target_from_unreal_units().
        """
        self.anchor_target = np.asarray(T_ros, dtype=float).copy()
        self.anchor_target_source = source
        return self.anchor_target

    def clear_anchor_target(self):
        self.anchor_target = None
        self.anchor_target_source = None

    def request_anchor(self):
        """Arm anchoring; it fires on the next accepted marker sighting."""
        self.anchor_requested = True

    def try_anchor(self, corners, level_horizon=True):
        """Attempt to anchor on this marker sighting. Returns (ok, message)."""
        solution = self.solve_marker_pose(corners)
        self.last_marker_solution = solution
        if not solution.ok:
            return False, solution.reason

        with self._lock:
            latest = self.latest_pose
        camera_position = (np.zeros(3) if latest is None
                           else np.asarray(latest[0], dtype=float))

        T_marker = solution.T_world_marker
        if level_horizon:
            # Keep position and heading only. A marker taped down a degree or two
            # off level should not tilt the entire virtual set.
            #
            # Heading has to come from an axis that is actually horizontal. The
            # marker's +Y ("up" on the print) is horizontal for a marker lying
            # face-up on a table, but points straight at the sky for one taped
            # to a wall -- and there atan2(0, 0) is undefined, so the heading
            # came out as whatever floating-point noise decided: measured at
            # +0.1 deg and -180 deg for the same sighting under two different
            # intrinsics. A 180 deg error here puts the camera on the wrong
            # side of the anchor.
            #
            # So use +Y where it is horizontal (table markers, the previous
            # behaviour) and the marker's normal where it is not (wall markers).
            # One of the two is always well conditioned, because they are
            # perpendicular.
            R_marker = T_marker[:3, :3]
            marker_up = R_marker[:, 1]          # "up" on the printed marker
            marker_normal = R_marker[:, 2]      # perpendicular to its face
            if abs(marker_up[2]) < abs(marker_normal[2]):
                marker_fwd = marker_up                 # table marker
            else:
                # Wall marker: the normal is the horizontal axis. Its sign
                # depends on the detector's corner convention, which is not
                # worth relying on, so orient it away from the camera instead
                # -- a marker we can see is one we are in front of. That makes
                # a head-on marker read heading 0 whichever way the convention
                # runs.
                marker_fwd = np.asarray(marker_normal, dtype=float)
                to_marker = T_marker[:3, 3] - np.asarray(camera_position,
                                                         dtype=float)
                if float(np.dot(marker_fwd, to_marker)) < 0.0:
                    marker_fwd = -marker_fwd
            yaw = math.atan2(marker_fwd[1], marker_fwd[0])
            T_level = np.eye(4)
            T_level[:3, :3] = rpy_to_matrix(0.0, 0.0, yaw)
            T_level[:3, 3] = T_marker[:3, 3]
        else:
            T_level = T_marker
            yaw = math.atan2(T_marker[1, 0], T_marker[0, 0])

        self.transformer.set_ground_zero_from_transform(
            T_level, T_target=self.anchor_target)
        self.confirm_lineup()
        self.anchor_requested = False
        self.smoother.reset()

        where = ("origin" if self.anchor_target is None
                 else f"anchor target ({self.anchor_target_source})")
        detail = f"marker at {T_level[:3, 3].round(3)} m heading {math.degrees(yaw):+.1f}°"
        if solution.used_depth:
            detail += (f", depth-checked (PnP {solution.distance_pnp_m:.3f} m vs "
                       f"depth {solution.distance_depth_m:.3f} m, "
                       f"residual {solution.depth_residual_m * 1000:.0f} mm)")
        else:
            detail += f", distance {solution.distance_pnp_m:.3f} m (no depth)"
        msg = f"anchored to {where}: {detail}"
        self.log(f"[Anchor] ✅ {msg}")
        return True, msg

    def compute_snap_position(self):
        """Camera position in Unreal space right now.

        Called right after ``try_anchor()`` to produce the ``snap_camera``
        event.  The ground-zero transform was just set, so the current SLAM
        pose expressed through ``ros_to_unreal()`` is exactly where Unreal
        should teleport the CineCamera.

        Returns a dict ``{x, y, z, pan, tilt, roll}`` (metres + degrees) or
        None when nothing is tracked yet.
        """
        pose, valid = self.camera_pose_now()
        if pose is None:
            return None
        return {
            "x": pose.x, "y": pose.y, "z": pose.z,
            "pan": pose.pan, "tilt": pose.tilt, "roll": pose.roll,
        }

    def apply_map_alignment(self, T_slam_to_map, T_align=None):
        """Push a fused LiDAR-map localisation into the output pipeline.

        `T_slam_to_map` (from LidarMapMatcher) is in the map file's own native
        coordinates; `T_align` is the mesh-alignment transform the operator set
        in the 3D viewport, which carries those coordinates into the frame the
        locator and the on-screen camera live in. Their product is exactly the
        ground-zero transform an ArUco anchor would have produced, so it can be
        written straight in -- and from there the FreeD stream, the camera dot
        and `camera_pose_ros_corrected()` all pick it up for free.
        """
        T = np.asarray(T_slam_to_map, dtype=float)
        if T_align is not None:
            T = np.asarray(T_align, dtype=float) @ T
        with self._lock:
            self.map_aligned = True
            self.transformer.set_map_alignment(T)
        return T

    @property
    def is_anchored(self):
        return self.transformer.is_zeroed

    # ── command protocol (spoken by the Unreal plugin) ───────────────────────
    def handle_command(self, cmd):
        """Handle one JSON command. Returns a dict suitable for a reply."""
        if isinstance(cmd, (bytes, bytearray)):
            cmd = json.loads(cmd.decode("utf-8"))
        action = cmd.get("cmd", "")

        if action == "ping":
            return {"ok": True, "cmd": action, "reply": "pong"}

        if action == "anchor_aruco":
            # Anchor to the marker with no target: marker becomes the origin.
            self.clear_anchor_target()
            self.request_anchor()
            return {"ok": True, "cmd": action,
                    "message": "armed; anchoring marker to world origin"}

        if action in ("anchor_lidar", "set_anchor"):
            pos = cmd.get("marker_map_position", cmd.get("position", [0, 0, 0]))
            rot = cmd.get("marker_map_rotation", cmd.get("rotation", [0, 0, 0]))
            self.set_anchor_target_from_unreal_units(pos, rot, source=action)
            self.request_anchor()
            return {"ok": True, "cmd": action,
                    "message": "armed; anchoring marker to the anchor target",
                    "anchor_ros_m": [float(v) for v in self.anchor_target[:3, 3]]}

        if action == "reset_pose":
            self.confirm_lineup()
            self._pose_history.clear()
            self._last_sensor_stamp = None
            self._last_odometry_mono = None
            self.transformer.clear_ground_zero()
            self.clear_anchor_target()
            self.map_aligned = False
            self.gate.reset()
            self.hold.reset()
            self.smoother.reset()
            self.log("[Anchor] ground zero cleared")
            return {"ok": True, "cmd": action, "message": "ground zero cleared"}

        if action == "set_offset":
            T = np.eye(4)
            T[:3, :3] = rpy_to_matrix(
                math.radians(float(cmd.get("rx", 0.0))),
                math.radians(float(cmd.get("ry", 0.0))),
                math.radians(float(cmd.get("rz", 0.0))))
            T[:3, 3] = [float(cmd.get("tx", 0.0)), float(cmd.get("ty", 0.0)),
                        float(cmd.get("tz", 0.0))]
            self.transformer.set_ground_zero_from_transform(T)
            self.log(f"[Anchor] manual offset applied: {T[:3, 3].round(3)}")
            return {"ok": True, "cmd": action, "message": "offset applied"}

        return {"ok": False, "cmd": action, "error": "unknown command"}

    # ── status ───────────────────────────────────────────────────────────────
    def status_dict(self):
        pose = self.latest_unreal
        sol = self.last_marker_solution
        return {
            "fusion": self.fusion_status(),
            "gnss": {"reason": self.gnss_reason, "calibrated": self.cfg["gnss"].get("calibrated", False)},
            "slam_status": self.slam_status,
            "tracking": (time.time() - self.last_vio_time) < self.stale_timeout
                        if self.last_vio_time else False,
            "anchored": self.is_anchored,
            "map_aligned": self.map_aligned,
            "anchor_target": (None if self.transformer.anchor_target is None
                              else [float(v) for v in
                                    self.transformer.anchor_target[:3, 3]]),
            "anchor_requested": self.anchor_requested,
            "stationary_hold": self.hold.is_stationary,
            "gate_rejects": self.gate.total_rejects,
            "has_depth": self.has_depth,
            "intrinsics_from_camera_info": self.intrinsics_from_camera_info,
            "pose": (None if pose is None else {
                "x": pose.x, "y": pose.y, "z": pose.z,
                "pan": pose.pan, "tilt": pose.tilt, "roll": pose.roll}),
            "last_marker": (None if sol is None else {
                "ok": sol.ok, "reason": sol.reason,
                "distance_pnp_m": sol.distance_pnp_m,
                "distance_depth_m": sol.distance_depth_m,
                "used_depth": sol.used_depth}),
        }
