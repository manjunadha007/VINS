"""Monotonic-time evidence policy. Scan checks are not independent cameras."""
import threading


class FusionSupervisor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.enabled = bool(cfg.get("enabled", False))
        self._lock = threading.RLock()
        self.reset()

    def reset(self):
        with self._lock:
            self.evidence = {}
            self.last_supported = None

    def observe(self, source, ok, timestamp, reason="", **metrics):
        with self._lock:
            previous = self.evidence.get(source, {})
            if timestamp <= previous.get("timestamp", -float("inf")):
                return False
            count = previous.get("consecutive", 0)
            if timestamp - previous.get("timestamp", -float("inf")) > self.cfg.get(source + "_timeout_s", 0.8):
                count = 0
            self.evidence[source] = dict(ok=bool(ok), timestamp=float(timestamp),
                reason=reason, consecutive=count + 1 if ok else 0, **metrics)
            return True

    def evaluate(self, now, pose_fresh, aligned):
        with self._lock:
            sources = {}
            for name, item in self.evidence.items():
                age = now - item["timestamp"]
                fresh = 0 <= age <= self.cfg.get(name + "_timeout_s", 0.8)
                ready = fresh and item["ok"] and item["consecutive"] >= self.cfg.get("recovery_samples", 3)
                sources[name] = dict(item, age_s=max(0.0, age), fresh=fresh, ready=ready)
            supported = [s for s in ("scan", "markers", "gnss") if sources.get(s, {}).get("ready")]
            if not pose_fresh:
                state, reason = "HOLD", "Camera pose missing, stale, or rejected"
            elif self.cfg.get("require_lineup", True) and not aligned:
                state, reason = "ALIGNING", "Place the camera in the scan and complete lineup"
            elif supported:
                self.last_supported = now
                state = "VALIDATED" if "scan" in supported or "markers" in supported else "RTK_ASSISTED"
                reason = ("Stereo / IMU with depth-to-scan validation" if "scan" in supported
                          else "Stereo / IMU with surveyed markers and depth-to-scan validation" if "markers" in supported
                          else "Stereo / IMU with RTK position check; rotation is camera-only")
            elif aligned and self.cfg.get("allow_vio_fallback", True):
                # The scan estimates the world alignment; VIO supplies live
                # motion. A partial/noisy scan must not pin a fresh camera at
                # its lineup pose, including while the first checks accumulate.
                state = "DEGRADED"
                reason = "Stereo / IMU camera tracking active; scan validation unavailable or recovering"
            elif self.last_supported is None:
                state, reason = "VERIFYING", "Waiting for consecutive scan or calibrated RTK checks"
            elif now - self.last_supported <= max(0.0, self.cfg.get("single_source_grace_s", 1.0)):
                state, reason = "DEGRADED", "Temporary stereo / IMU fallback; external check unavailable"
            else:
                state, reason = "HOLD", "Fallback expired; reveal static geometry or repeat lineup"
            return dict(state=state, reason=reason, sources=sources, supported_by=supported,
                        output_allowed=state in ("VALIDATED", "RTK_ASSISTED", "DEGRADED"),
                        validated=state in ("VALIDATED", "RTK_ASSISTED"))
