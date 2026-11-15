"""Testing arena: score the dashboard's camera path against a 3DE solve.

For every plate frame of a take the 3DE camera (reference) and the dashboard's
output at the same instant are compared as the pose of the D455's left IR lens
-- the camera both of them describe:

    reference  Blender camera matrix (looks -Z, +Y up) @ diag(1, -1, -1)
    dashboard  output_T @ rig_offset^-1 @ T_BODY_OPTICAL

output_T is what Unreal is sent (ground zero @ SLAM pose @ rig offset), in the
dashboard's world; 3DE was solved in the same world (arena_export.py). The
output a FreeD packet carries is the pose at send time + lead_time_s, so a
frame at sensor stamp s is compared with the output interpolated at
s - lead + offset, where offset is the replay's clock shift (0 live).

A frame passes within `frame_mm` and `frame_deg`; a take passes when at least
`min_frames` of its scored frames do. Scoring starts at the first step mark
(the operator's "Locked at the start"). A frame with no output to compare
fails: Unreal had nothing then either.

Pure numpy; runs anywhere the arena runs.
"""
import bisect
import csv
import json
import math

import numpy as np

from pose_transform import (T_BODY_OPTICAL, invert_transform, rpy_to_matrix,
                            se3_interpolate, so3_log)

GL_TO_OPTICAL = np.diag([1.0, -1.0, -1.0, 1.0])
DEFAULT_PASS = dict(frame_mm=20.0, frame_deg=0.5, min_frames=0.90)
MAX_OUTPUT_GAP_S = 0.1     # output older than this around a frame = no output


# ── Poses ────────────────────────────────────────────────────────────────────
def rig_transform(cfg):
    """transform.rig_offset as PoseTransformer builds it."""
    rig = ((cfg or {}).get("transform") or {}).get("rig_offset") or {}
    T = np.eye(4)
    T[:3, :3] = rpy_to_matrix(*(math.radians(float(a))
                                for a in rig.get("rotation_deg", (0, 0, 0))))
    T[:3, 3] = [float(v) for v in rig.get("translation_m", (0, 0, 0))]
    return T


def reference_optical(T_blender_camera):
    return np.asarray(T_blender_camera, dtype=float) @ GL_TO_OPTICAL


def dashboard_optical(output_T, rig_T):
    return np.asarray(output_T, dtype=float) @ invert_transform(rig_T) @ T_BODY_OPTICAL


def pose_error(T_ref, T_test):
    """(position error mm, rotation error deg) between two 4x4 poses."""
    pos = float(np.linalg.norm(T_test[:3, 3] - T_ref[:3, 3])) * 1000.0
    rot = math.degrees(float(np.linalg.norm(so3_log(T_ref[:3, :3].T @ T_test[:3, :3]))))
    return pos, rot


class OutputTrack:
    """The dashboard's output over time, from arena state records."""

    def __init__(self, records):
        rows = sorted((r["t"], r) for r in records if r.get("t") is not None)
        self.times = [t for t, _ in rows]
        self.records = [r for _, r in rows]

    def at(self, t):
        """(4x4 output or None, mode) at host time t, interpolated."""
        i = bisect.bisect_left(self.times, t)
        if i < len(self.times) and self.times[i] == t:      # exactly on a sample
            rec = self.records[i]
            T = rec.get("output_T")
            return (None if T is None else np.asarray(T, dtype=float).reshape(4, 4)), rec.get("mode")
        before = self.records[i - 1] if i > 0 else None
        after = self.records[i] if i < len(self.records) else None
        mode = (before or after or {}).get("mode")
        if before is None or after is None:
            return None, mode
        if (before.get("output_T") is None or after.get("output_T") is None
                or self.times[i] - self.times[i - 1] > MAX_OUTPUT_GAP_S):
            return None, mode
        T0 = np.asarray(before["output_T"], dtype=float).reshape(4, 4)
        T1 = np.asarray(after["output_T"], dtype=float).reshape(4, 4)
        span = self.times[i] - self.times[i - 1]
        alpha = 0.0 if span <= 0 else (t - self.times[i - 1]) / span
        return se3_interpolate(T0, T1, alpha), mode


def umeyama(src, dst, with_scale=True):
    """Similarity (s, R, t) minimising |dst - (s R src + t)|."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(xd.T @ xs / len(src))
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = float(np.trace(np.diag(S) @ D) / xs.var(0).sum()) if with_scale else 1.0
    return s, R, mu_d - s * R @ mu_s


# ── Scoring ──────────────────────────────────────────────────────────────────
def read_frames(path):
    with open(path) as fh:
        return [dict(frame=int(r["frame"]), stamp=float(r["stamp_s"])) for r in csv.DictReader(fh)]


def score(frames, reference, records, *, offset_s=0.0, lead_s=0.0, rig_T=None,
          marks=(), criteria=None):
    """Score one run. Returns (report dict, per-frame rows)."""
    criteria = {**DEFAULT_PASS, **(criteria or {})}
    rig_T = np.eye(4) if rig_T is None else rig_T
    ref = {f["frame"]: reference_optical(f["T"]) for f in reference["frames"]}
    track = OutputTrack(records)
    start = marks[0]["t"] if marks else None
    rows = []
    for f in frames:
        if f["frame"] not in ref or (start is not None and f["stamp"] < start):
            continue
        T_out, mode = track.at(f["stamp"] - lead_s + offset_s)
        row = dict(frame=f["frame"], stamp=f["stamp"], mode=mode,
                   pos_mm=None, rot_deg=None, passed=False)
        if T_out is not None:
            pos, rot = pose_error(ref[f["frame"]], dashboard_optical(T_out, rig_T))
            row.update(pos_mm=pos, rot_deg=rot,
                       passed=pos <= criteria["frame_mm"] and rot <= criteria["frame_deg"])
        rows.append(row)

    compared = [r for r in rows if r["pos_mm"] is not None]
    report = dict(criteria=criteria, frames_scored=len(rows), frames_compared=len(compared),
                  frames_without_output=len(rows) - len(compared),
                  frames_passed=sum(r["passed"] for r in rows))
    report["pass_ratio"] = report["frames_passed"] / len(rows) if rows else 0.0
    report["passed"] = bool(rows) and report["pass_ratio"] >= criteria["min_frames"]
    if compared:
        pos = np.array([r["pos_mm"] for r in compared])
        rot = np.array([r["rot_deg"] for r in compared])
        report.update(pos_mm=_stats(pos), rot_deg=_stats(rot))
    report["phases"] = _phases(rows, marks, criteria)
    report["diagnostics"] = _diagnostics(frames, ref, track, compared, lead_s, offset_s, rig_T)
    return report, rows


def _stats(values):
    return dict(median=round(float(np.median(values)), 2),
                p95=round(float(np.percentile(values, 95)), 2),
                max=round(float(values.max()), 2))


def _phases(rows, marks, criteria):
    """Pass ratio between consecutive step marks (start, excursion, return)."""
    phases = []
    for i, mark in enumerate(marks):
        end = marks[i + 1]["t"] if i + 1 < len(marks) else float("inf")
        inside = [r for r in rows if mark["t"] <= r["stamp"] < end]
        if not inside:
            continue
        compared = [r["pos_mm"] for r in inside if r["pos_mm"] is not None]
        phases.append(dict(step=mark.get("step"), label=mark.get("label"), frames=len(inside),
                           pass_ratio=round(sum(r["passed"] for r in inside) / len(inside), 4),
                           median_mm=round(float(np.median(compared)), 2) if compared else None))
    return phases


def _diagnostics(frames, ref, track, compared, lead_s, offset_s, rig_T):
    """Why a take fails: constant offset vs drift, time lag, units, axes."""
    out = {}
    if len(compared) < 10:
        return out
    by_frame = {f["frame"]: f["stamp"] for f in frames}
    dash, refp = [], []
    for r in compared:
        T, _ = track.at(by_frame[r["frame"]] - lead_s + offset_s)
        dash.append(dashboard_optical(T, rig_T)[:3, 3])
        refp.append(ref[r["frame"]][:3, 3])
    dash, refp = np.array(dash), np.array(refp)
    extent = float(np.ptp(refp, axis=0).max())
    out["reference_extent_m"] = round(extent, 3)
    if extent > 0.2:
        s, R, t = umeyama(dash, refp)
        fitted = (s * (R @ dash.T)).T + t
        resid = np.linalg.norm(fitted - refp, axis=1) * 1000
        angle = math.degrees(float(np.linalg.norm(so3_log(R))))
        out["best_fit"] = dict(scale=round(s, 5), rotation_deg=round(angle, 3),
                               translation_mm=round(float(np.linalg.norm(t)) * 1000, 1),
                               median_mm_after_fit=round(float(np.median(resid)), 2))
        if not 0.05 < s < 20:
            out["warning"] = (f"the 3DE camera is {s:.3g}x the dashboard's size: the .abc "
                              "is probably not in metres (3DE units / Alembic export scale)")
        elif angle > 30:
            out["warning"] = (f"the 3DE world is turned {angle:.0f} deg from the dashboard's: "
                              "survey points were probably not used, or the axes were changed "
                              "on export")
    if extent <= 0.2:
        return out        # a camera that barely moves says nothing about lag
    # Best time shift: a clear optimum away from 0 means the output lags or
    # leads. Only reported when it cuts the error by a quarter or more.
    sample = compared[::max(1, len(compared) // 300)]

    def median_at(shift_ms):
        errs = []
        for r in sample:
            T, _ = track.at(by_frame[r["frame"]] - lead_s + offset_s + shift_ms / 1000)
            if T is not None:
                errs.append(np.linalg.norm(dashboard_optical(T, rig_T)[:3, 3]
                                           - ref[r["frame"]][:3, 3]) * 1000)
        return float(np.median(errs)) if errs else None

    at_zero = median_at(0)
    best = min(((s, median_at(s)) for s in range(-300, 301, 10)),
               key=lambda item: float("inf") if item[1] is None else item[1])
    if (at_zero is not None and best[1] is not None and best[0] != 0
            and best[1] <= 0.75 * at_zero):
        out["best_time_shift_ms"] = best[0]
        out["median_mm_at_best_shift"] = round(best[1], 2)
    return out


def write_rows(path, rows):
    with open(path, "w", newline="") as fh:
        out = csv.writer(fh)
        out.writerow(["frame", "stamp_s", "mode", "pos_err_mm", "rot_err_deg", "passed"])
        for r in rows:
            out.writerow([r["frame"], f"{r['stamp']:.6f}", r["mode"],
                          "" if r["pos_mm"] is None else f"{r['pos_mm']:.2f}",
                          "" if r["rot_deg"] is None else f"{r['rot_deg']:.4f}", int(r["passed"])])


def summary_lines(report, name):
    c = report["criteria"]
    verdict = "PASS" if report["passed"] else "FAIL"
    lines = [f"{verdict}  {name}: {report['pass_ratio']:.1%} of {report['frames_scored']} frames "
             f"within {c['frame_mm']:g} mm / {c['frame_deg']:g}° (needs {c['min_frames']:.0%})"]
    if report.get("pos_mm"):
        p, r = report["pos_mm"], report["rot_deg"]
        lines.append(f"   position error median {p['median']} mm, 95% {p['p95']} mm, max {p['max']} mm")
        lines.append(f"   rotation error median {r['median']}°, 95% {r['p95']}°, max {r['max']}°")
    if report["frames_without_output"]:
        lines.append(f"   {report['frames_without_output']} frames had no output (Waiting / gaps)")
    for ph in report["phases"]:
        lines.append(f"   step {ph['step']} {ph['label']!r}: {ph['pass_ratio']:.0%} of {ph['frames']} "
                     f"frames, median {ph['median_mm']} mm")
    d = report["diagnostics"]
    if "best_fit" in d:
        b = d["best_fit"]
        lines.append(f"   best-fit check: scale {b['scale']}, turn {b['rotation_deg']}°, shift "
                     f"{b['translation_mm']} mm -> median {b['median_mm_after_fit']} mm after fit")
    if "best_time_shift_ms" in d:
        shift = d["best_time_shift_ms"]
        lines.append(f"   time check: the output {'lags' if shift > 0 else 'leads'} the 3DE camera by "
                     f"{abs(shift)} ms (median {d['median_mm_at_best_shift']} mm when shifted back)")
    if "warning" in d:
        lines.append(f"   ⚠️  {d['warning']}")
    return lines


def load_json(path):
    with open(path) as fh:
        return json.load(fh)
