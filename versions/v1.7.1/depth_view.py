"""Live depth-coverage preview for Auto lineup.

Auto lineup places the camera by fitting its depth points to the scan, so a
view needs two things: enough pixels with usable depth, and surfaces facing in
enough directions to pin down all three position axes (floor plus one wall
fixes two; the lineup can slide along the third). This reads the camera only,
with no scan and no pose, so it works before a lineup is attempted.

The axis count is the matcher's own observability test
(`LidarMapMatcher._observability`), run on the camera's surface normals instead
of the matched scan's. It is a preview: the scan must also contain those
surfaces for the real lineup to reach the same count.

The projector can add texture for stereo depth. These dots represent valid
measured depth, not visual features tracked by SLAM. Coverage and geometry are
a pre-lineup cue; they do not prove that the surfaces match the loaded scan.
"""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class DepthCoverage:
    shape: tuple          # (h, w) of the depth image
    coverage: float       # fraction of pixels with depth in range
    u: np.ndarray         # sampled pixel columns with depth
    v: np.ndarray         # sampled pixel rows with depth
    z: np.ndarray         # their depth, metres
    rank: int             # position axes the view's surfaces constrain, 0-3
    eigvals: np.ndarray   # constraint strength per axis, ascending, sums to 1
    n_normals: int


def depth_coverage(depth_mm, K, min_range_m=0.3, max_range_m=6.0,
                   observability_eps=0.04, step=4, normal_px=8):
    """Analyse one 16-bit millimetre depth frame. None if it is unusable."""
    depth = np.asarray(depth_mm)
    if depth.ndim != 2 or depth.size == 0:
        return None
    K = np.asarray(K, dtype=float).reshape(3, 3)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    if not (fx > 0 and fy > 0):
        return None
    h, w = depth.shape
    z = depth.astype(np.float32) * np.float32(0.001)
    valid = (z >= min_range_m) & (z <= max_range_m)

    rows = np.arange(step // 2, h, step)
    cols = np.arange(step // 2, w, step)
    vv, uu = np.meshgrid(rows, cols, indexing="ij")
    keep = valid[vv, uu]

    # Normals from box-smoothed depth. Raw stereo noise over an 8-pixel
    # difference tilts a flat wall's normals enough to fake a third axis.
    mask = valid.astype(np.float32)
    box = (normal_px + 1, normal_px + 1)
    support = cv2.boxFilter(mask, -1, box)
    zs = cv2.boxFilter(z * mask, -1, box) / np.maximum(support, 1e-6)
    solid = support >= 0.6

    d = normal_px
    r = rows[(rows >= d) & (rows < h - d)]
    c = cols[(cols >= d) & (cols < w - d)]
    rank, eig, n = 0, np.zeros(3), 0
    if len(r) and len(c):
        V, U = np.meshgrid(r, c, indexing="ij")
        ok = (valid[V, U] & solid[V, U] & solid[V, U + d] & solid[V, U - d]
              & solid[V + d, U] & solid[V - d, U])
        z0 = zs[V, U]
        # Off depth edges only: the sample must sit on the smoothed surface,
        # and opposite neighbours may differ by at most a ~80 degree slope.
        ok &= np.abs(z[V, U] - z0) < 0.03 * z0 + 0.01
        limit = 5.0 * (2 * d) * z0 / fx
        zr, zl, zd, zu = zs[V, U + d], zs[V, U - d], zs[V + d, U], zs[V - d, U]
        ok &= (np.abs(zr - zl) < limit) & (np.abs(zd - zu) < limit)
        if np.count_nonzero(ok) >= 50:
            U, V, zr, zl, zd, zu, z0 = (a[ok] for a in (U, V, zr, zl, zd, zu, z0))
            # Tangents across +-d pixels, back-projected; their cross product
            # is the surface normal. Elementwise only: a BLAS matmul here can
            # fan out across every core in the container's numpy.
            tux = (U + d - cx) / fx * zr - (U - d - cx) / fx * zl
            tuy = (V - cy) / fy * (zr - zl)
            tuz = zr - zl
            tvx = (U - cx) / fx * (zd - zu)
            tvy = (V + d - cy) / fy * zd - (V - d - cy) / fy * zu
            tvz = zd - zu
            nx = tuy * tvz - tuz * tvy
            ny = tuz * tvx - tux * tvz
            nz = tux * tvy - tuy * tvx
            norm = np.sqrt(nx * nx + ny * ny + nz * nz)
            good = norm > 1e-9
            nx, ny, nz = nx[good] / norm[good], ny[good] / norm[good], nz[good] / norm[good]
            # A pixel covers ~z^2 of surface; weighting by it mirrors the
            # matcher's 5 cm voxel thinning, which counts area, not pixels.
            wgt = z0[good] ** 2
            wgt = wgt / wgt.sum()
            M = np.array([[np.sum(wgt * a * b) for b in (nx, ny, nz)] for a in (nx, ny, nz)])
            eig = np.linalg.eigvalsh(M)
            rank = int(np.sum(eig >= observability_eps))
            n = int(len(nx))

    return DepthCoverage(shape=(h, w), coverage=float(valid.mean()),
                         u=uu[keep], v=vv[keep], z=z[vv, uu][keep],
                         rank=rank, eigvals=eig, n_normals=n)


GREEN, AMBER, RED = (179, 213, 130), (125, 192, 235), (154, 154, 238)   # BGR


def verdict(cov, need_rank=3):
    """(colour, headline, tip) for the operator."""
    if cov is None:
        return RED, "DEPTH  no recent depth frame", "Check the depth stream"
    head = f"DEPTH  {cov.coverage:.0%} of view  |  geometry {cov.rank}/3 axes"
    if cov.coverage < 0.10:
        return RED, head, "Too little depth: include nearer surfaces with usable depth"
    if cov.rank < need_rank:
        colour = AMBER if cov.rank == need_rank - 1 else RED
        return colour, head, "Flat view: include a corner or furniture at another depth"
    if cov.coverage < 0.30:
        return AMBER, head, "Limited coverage: include more surfaces with usable depth"
    return GREEN, head, "Candidate view: Auto lineup still needs scan agreement"


def draw_depth_points(image_bgr, cov, min_range_m=0.3, max_range_m=6.0, need_rank=3, draw_status=True):
    """The image with depth dots (red near, blue far) and a status bar."""
    out = np.array(image_bgr, dtype=np.uint8, copy=True)
    H, W = out.shape[:2]
    if cov is not None and len(cov.u):
        sx, sy = W / cov.shape[1], H / cov.shape[0]
        u = np.clip((cov.u * sx).astype(np.intp), 0, W - 2)
        v = np.clip((cov.v * sy).astype(np.intp), 0, H - 2)
        # Log scale: most of a room sits at 1-4 m, which a linear map squashes.
        t = np.log(np.maximum(cov.z, min_range_m) / min_range_m) / np.log(max_range_m / min_range_m)
        level = (255 * (1 - np.clip(t, 0, 1))).astype(np.uint8).reshape(-1, 1)
        colours = cv2.applyColorMap(level, cv2.COLORMAP_TURBO).reshape(-1, 3)
        for dy in (0, 1):
            for dx in (0, 1):
                out[v + dy, u + dx] = colours

    if not draw_status:
        return out
    colour, head, tip = verdict(cov, need_rank)
    scale = max(0.4, W / 1700)
    bar = int(52 * scale / 0.5)
    top = max(0, H - bar)
    out[top:] = (out[top:].astype(np.uint16) * 2 // 5).astype(np.uint8)
    line = int(20 * scale / 0.5)
    cv2.putText(out, head, (8, top + line), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)
    cv2.putText(out, tip + "   (near = red, far = blue)", (8, top + 2 * line),
                cv2.FONT_HERSHEY_SIMPLEX, scale * 0.85, (220, 228, 239), 1, cv2.LINE_AA)
    return out
