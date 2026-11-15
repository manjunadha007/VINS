"""Read a 3DEqualizer Alembic camera into JSON -- runs inside Blender.

    blender -b --python arena_abc.py -- <camera.abc> <plate.json> <out.json>

Run by ./arena.sh on the host (Blender is not in the Isaac container). Writes
the camera's world matrix for every plate frame:

* The scene is set to the plate's frame rate BEFORE import. Blender samples
  the Alembic by time, so a 30 fps camera read at Blender's default 24 fps
  comes back silently shifted (measured: 0.125 m off on frame 1).
* Blender converts Alembic's Y-up to its own Z-up on import, (x, y, z)_abc ->
  (x, -z, y). arena_export.py wrote the 3DE world as (x, z, -y) of the
  dashboard's, so the matrices read here are already in dashboard (ROS) axes.
* The camera matrix keeps Blender's camera convention: looks down -Z, +Y up.
"""
import json
import sys

import bpy


def main(abc_path, plate_path, out_path):
    with open(plate_path) as fh:
        plate = json.load(fh)
    fps = float(plate["fps"])
    first = int(plate["first_frame"])
    count = int(plate["frame_count"])

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.fps = int(round(fps))
    scene.render.fps_base = scene.render.fps / fps
    bpy.ops.wm.alembic_import(filepath=abc_path)

    cameras = [o for o in scene.objects if o.type == "CAMERA"]
    if not cameras:
        raise SystemExit(f"no camera in {abc_path}")
    if len(cameras) > 1:
        raise SystemExit(f"{len(cameras)} cameras in {abc_path} "
                         f"({', '.join(c.name for c in cameras)}): export only the solved camera")
    cam = cameras[0]

    frames = []
    for number in range(first, first + count):
        scene.frame_set(number)
        frames.append(dict(frame=number,
                           T=[[float(v) for v in row] for row in cam.matrix_world]))

    # A camera that stops moving before the plate ends usually means the .abc
    # covers fewer frames than the plate (Blender holds the last sample).
    last_change = first
    for a, b in zip(frames, frames[1:]):
        if any(abs(x - y) > 1e-9 for ra, rb in zip(a["T"], b["T"]) for x, y in zip(ra, rb)):
            last_change = b["frame"]
    with open(out_path, "w") as fh:
        json.dump(dict(format="hyperfusion.reference", version=1, abc=abc_path,
                       camera=cam.name, fps=fps, first_frame=first, frame_count=count,
                       last_moving_frame=last_change,
                       lens_mm=float(cam.data.lens), sensor_width_mm=float(cam.data.sensor_width),
                       blender=bpy.app.version_string, frames=frames), fh)
    print(f"[arena_abc] {cam.name}: {count} frames -> {out_path} "
          f"(moves until frame {last_change})")


if __name__ == "__main__":
    args = sys.argv[sys.argv.index("--") + 1:]
    main(*args[:3])
