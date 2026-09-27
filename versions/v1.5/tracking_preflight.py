#!/usr/bin/env python3
"""Read-only project/input inventory. Does not open sensors or send packets."""
import argparse
import glob
import importlib.util
import json
import math
from pathlib import Path

from tracking_config import load_config


def inspect_scan(path, scale):
    low, high = [math.inf] * 3, [-math.inf] * 3
    vertices = faces = 0
    with Path(path).open() as stream:
        for line in stream:
            if line.startswith("v "):
                point = list(map(float, line.split()[1:4]))
                if len(point) != 3 or not all(map(math.isfinite, point)):
                    raise ValueError("Scan contains invalid vertices")
                low = [min(a, b) for a, b in zip(low, point)]
                high = [max(a, b) for a, b in zip(high, point)]
                vertices += 1
            elif line.startswith("f "):
                faces += 1
    if not vertices or not faces:
        raise ValueError("An OBJ mesh with vertices and faces is required")
    return dict(vertices=vertices, faces=faces, metres_per_unit=scale,
                extent_m=[round((b-a)*scale, 4) for a,b in zip(low,high)],
                bounds_min_m=[round(a*scale,4) for a in low],
                bounds_max_m=[round(a*scale,4) for a in high])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config, verbose=False)
    modules = ("numpy", "scipy", "cv2", "open3d", "PyQt5", "yaml", "serial", "rclpy", "cv_bridge")
    report = dict(config=cfg["_source"],
                  dependencies={m: importlib.util.find_spec(m) is not None for m in modules},
                  serial_devices=glob.glob("/dev/serial/by-id/*"),
                  video_devices=glob.glob("/dev/video*"),
                  scan_file=cfg["lidar"]["map_file"],
                  supervised_output=cfg["fusion"]["enabled"],
                  gnss_enabled=cfg["gnss"]["enabled"],
                  gnss_calibrated=cfg["gnss"]["calibrated"],
                  unreal=f'{cfg["network"]["unreal_ip"]}:{cfg["network"]["unreal_port"]}',
                  topics=cfg["topics"])
    try:
        report["scan"] = inspect_scan(report["scan_file"], cfg["lidar"]["map_scale"])
    except (OSError, ValueError) as exc:
        report["scan_error"] = str(exc)
    report["remaining_inputs"] = [
        "Live ROS pose, IR and depth timestamps plus camera_info",
        "Physical lineup and measured scan axis / scale check",
        "Lens nodal offset and capture-to-render latency measurement",
        "Outdoor RTK correction source, receiver GGA/GST configuration",
        "GNSS surveyed origin, ENU rotation and antenna lever arm",
        "Recorded occlusion / movement / sunlight validation"
    ]
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("HyperFusion input readiness")
        print(f'Config: {report["config"]}')
        print(f'Scan: {report.get("scan", report.get("scan_error"))}')
        print(f'Unreal: {report["unreal"]}')
        for name, present in report["dependencies"].items():
            print(f'  {name}: {"available" if present else "missing in this interpreter"}')
        print("Serial devices:", report["serial_devices"] or "none detected")
        print("Video devices:", report["video_devices"] or "none detected")
        print("This is a static inventory, not a camera accuracy or RTK fix test.")
    return 1 if "scan_error" in report else 0


if __name__ == "__main__":
    raise SystemExit(main())
