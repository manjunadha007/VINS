#!/usr/bin/env python3
"""
Line the camera up against the LiDAR using the live picture, not just depth.

This is the automatic version of a 3DEqualizer lineup: project the scan from
the current pose estimate, see where its edges land in the image, and move the
camera until they sit on the real edges. A matchmove artist does it by dragging
until the geometry "snaps" onto the plate; the same objective can be minimised
directly.

Why bother when depth-vs-scan ICP already exists
------------------------------------------------
Because the two fail in opposite places, which is what makes them worth having
together:

  * The D455's stereo depth is trustworthy to about 6 m and its error grows as
    z^2, so past a few metres it contributes noise. Image edges get *sharper*
    with distance, and a far corner constrains rotation far more tightly than
    anything in the near field -- a 1 px error on an edge 8 m away is a much
    smaller angular error than 1 px on a surface 1 m away.
  * ICP against one flat wall slides along it. Edges do not: a door frame or a
    table edge pins the two directions the wall leaves free.
  * Conversely, edges say nothing about a textureless surface filling the frame,
    which is exactly where depth is strongest.

So depth fixes near-field position and image edges fix orientation and the
far field, and each covers the other's blind spot.

Why edges rather than matching the picture directly
---------------------------------------------------
The scan carries photogrammetry texture; the live frame is the D455's infrared
imager with its projector off. Those are different modalities under different
lighting, so comparing intensities compares the lighting, not the geometry.
Edges are what survives: a wall/floor junction is an edge in both, whatever
either one's brightness happens to be.

The objective
-------------
Model edges are found by ray-casting the scan and taking depth discontinuities
(silhouettes) and normal discontinuities (creases). Each such pixel is a real 3D
point on the scan. Project it, look up how far it is from the nearest image edge
in a distance transform, and minimise the sum of those distances over the 6-DOF
pose by Gauss-Newton. Re-casting each iteration makes it an iterative-closest-
edge scheme: silhouettes move as the pose changes, so they have to be re-found,
not just re-projected.
"""
import math

import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

try:
    import open3d as o3d
    _HAS_O3D = True
except ImportError:
    _HAS_O3D = False

from pose_transform import (T_BODY_OPTICAL, invert_transform,
                            rpy_to_matrix, se3_exp)


class EdgeAlignResult:
    """Outcome of one lineup: the pose, and how well the picture agrees."""

    __slots__ = ("ok", "reason", "T_map_cam", "score", "median_px",
                 "n_model_edges", "n_image_edges", "moved_m", "moved_deg",
                 "iterations", "elapsed_ms")

    def __init__(self, ok=False, reason="", T_map_cam=None, score=0.0,
                 median_px=0.0, n_model_edges=0, n_image_edges=0, moved_m=0.0,
                 moved_deg=0.0, iterations=0, elapsed_ms=0.0):
        self.ok = ok
        self.reason = reason
        self.T_map_cam = T_map_cam
        #: fraction of projected scan edges sitting within `inlier_px` of a
        #: real image edge -- the "does the video line up" number.
        self.score = score
        self.median_px = median_px
        self.n_model_edges = n_model_edges
        self.n_image_edges = n_image_edges
        self.moved_m = moved_m
        self.moved_deg = moved_deg
        self.iterations = iterations
        self.elapsed_ms = elapsed_ms

    def __repr__(self):
        if not self.ok:
            return f"EdgeAlignResult(rejected: {self.reason})"
        return (f"EdgeAlignResult(score={self.score:.2f} "
                f"median={self.median_px:.1f}px "
                f"moved={self.moved_m * 1000:.0f}mm/{self.moved_deg:.2f}deg "
                f"iters={self.iterations})")


class LineupSession:
    """A frozen-frame manual lineup, in the 3DEqualizer sense.

    The operator parks the camera, freezes one frame, and nudges the virtual
    camera until the scan's projected edges sit on the picture. Committing says
    "the camera is *here*", and continuous fusion carries on from that pose.

    Frozen rather than live, deliberately. Lining up against a moving image is
    hopeless, and it is not necessary: the fusion corrects the SLAM-to-map
    *frame*, not individual samples, so a pose dialled against one frame stays
    valid for every pose that follows. What gets committed is

        T_slam_to_map = T_map_cam(dialled) @ inv(T_slam_cam(at freeze))

    which is the same composition the locator path uses -- only the operator's
    eye has replaced ICP as the source of T_map_cam.

    Deliberately free of Qt so the arithmetic can be tested on its own; the
    dashboard is a thin panel over this.
    """

    #: Nudge steps: (metres, degrees) for coarse and fine.
    COARSE = (0.020, 1.0)
    FINE = (0.002, 0.1)

    def __init__(self, gray, K, T_slam_cam, T_map_cam, aligner=None):
        self.gray = gray
        self.K = None if K is None else np.asarray(K, dtype=float)
        self.T_slam_cam = np.asarray(T_slam_cam, dtype=float).copy()
        self._initial = np.asarray(T_map_cam, dtype=float).copy()
        self.T_map_cam = self._initial.copy()
        self.aligner = aligner
        self.fine = False
        self.history = []

    # ── adjusting ────────────────────────────────────────────────────────────
    @property
    def step(self):
        return self.FINE if self.fine else self.COARSE

    def nudge(self, forward=0.0, left=0.0, up=0.0,
              pan=0.0, tilt=0.0, roll=0.0):
        """Move the virtual camera, in units of the current step size.

        Everything is in the camera's own axes -- dolly, truck, pedestal, pan,
        tilt, roll -- because that is how the movement reads on screen and how
        a matchmove artist thinks. Arguments are step *counts*, so nudge(
        forward=1) moves one step forward whatever the step is set to.
        """
        metres, degrees = self.step
        self.history.append(self.T_map_cam.copy())

        R = self.T_map_cam[:3, :3]
        delta_body = np.array([forward, left, up], dtype=float) * metres
        self.T_map_cam[:3, 3] = self.T_map_cam[:3, 3] + R @ delta_body

        if pan or tilt or roll:
            turn = rpy_to_matrix(math.radians(roll * degrees),
                                 math.radians(tilt * degrees),
                                 math.radians(pan * degrees))
            self.T_map_cam[:3, :3] = R @ turn
        return self.T_map_cam

    def undo(self):
        if self.history:
            self.T_map_cam = self.history.pop()
        return self.T_map_cam

    def reset(self):
        self.history.append(self.T_map_cam.copy())
        self.T_map_cam = self._initial.copy()
        return self.T_map_cam

    @property
    def moved_from_start(self):
        """(metres, degrees) the operator has dialled away from the seed."""
        moved = float(np.linalg.norm(self.T_map_cam[:3, 3] - self._initial[:3, 3]))
        R_rel = self._initial[:3, :3].T @ self.T_map_cam[:3, :3]
        cos = float(np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0))
        return moved, math.degrees(math.acos(cos))

    # ── looking at it ────────────────────────────────────────────────────────
    def overlay(self):
        """The frozen frame with the scan's edges drawn over it."""
        if self.aligner is None:
            return None
        return self.aligner.overlay(self.gray, self.K, self.T_map_cam)

    def score(self):
        """How well it lines up right now, or None."""
        if self.aligner is None:
            return None
        return self.aligner.score(self.gray, self.K, self.T_map_cam)

    def auto_refine(self):
        """Let the edge solver finish the job from where the operator got to.

        Optional and judged by eye: on a decimated scan the solver's objective
        can sit tens of millimetres off true (see the note in tracking.yaml),
        so this suggests a pose rather than imposing one -- the operator keeps
        it or undoes it.
        """
        if self.aligner is None or not self.aligner.available:
            return None
        result = self.aligner.align(self.gray, self.K, self.T_map_cam)
        if result.ok:
            self.history.append(self.T_map_cam.copy())
            self.T_map_cam = result.T_map_cam.copy()
        return result

    # ── committing ───────────────────────────────────────────────────────────
    def slam_to_map(self):
        """The transform to hand the fusion: SLAM world -> map."""
        return self.T_map_cam @ invert_transform(self.T_slam_cam)


class EdgeAligner:
    """Registers a camera image against a LiDAR mesh by its edges."""

    def __init__(self, config=None):
        cfg = config or {}
        self.enabled = bool(cfg.get("image_align_enabled", True))
        #: Working width. Everything scales from here: the raycast, the Canny
        #: and the distance transform. 320 keeps a full lineup inside ~30 ms.
        self.work_width = int(cfg.get("image_align_width", 320))
        self.max_iterations = int(cfg.get("image_align_iterations", 20))
        #: A projected edge this close to an image edge counts as lined up.
        self.inlier_px = float(cfg.get("image_align_inlier_px", 3.0))
        #: Beyond this there is no plausible correspondence, so the residual is
        #: dropped rather than dragging the pose toward an unrelated edge.
        self.max_search_px = float(cfg.get("image_align_search_px", 24.0))
        self.min_model_edges = int(cfg.get("image_align_min_edges", 150))
        self.min_score = float(cfg.get("image_align_min_score", 0.35))
        self.max_range_m = float(cfg.get("image_align_max_range_m", 15.0))
        #: Refuse a lineup that wants to move further than this; the seed comes
        #: from depth ICP, so a large move means the edges found a different
        #: part of the room rather than a better fit to the same part.
        self.max_move_m = float(cfg.get("image_align_max_move_m", 0.25))
        self.max_move_deg = float(cfg.get("image_align_max_move_deg", 10.0))
        #: cap on residuals per iteration; more only costs time
        self.max_edge_samples = int(cfg.get("image_align_max_edges", 4000))
        #: Include normal-discontinuity (crease) edges as well as silhouettes.
        self.use_creases = bool(cfg.get("image_align_use_creases", True))

        self._rng = np.random.default_rng(0)
        self._scene = None
        self._mesh_ready = False

    # ── the scan ─────────────────────────────────────────────────────────────
    @property
    def available(self):
        return _HAS_CV2 and _HAS_O3D and self._mesh_ready and self.enabled

    def set_mesh(self, vertices, triangles, scale=1.0):
        """Install the scan as a ray-castable surface, in metres.

        Takes the mesh rather than the sampled point cloud the ICP path uses:
        silhouette edges are a property of the *surface*, and a point cloud has
        no silhouettes to find.
        """
        if not (_HAS_O3D and _HAS_CV2):
            self._mesh_ready = False
            return False
        vertices = np.asarray(vertices, dtype=np.float64) * float(scale)
        triangles = np.asarray(triangles, dtype=np.int32)
        if len(vertices) == 0 or len(triangles) == 0:
            self._mesh_ready = False
            return False
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(vertices),
            o3d.utility.Vector3iVector(triangles))
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        self._scene = scene
        self._mesh_ready = True
        return True

    def clear_mesh(self):
        """Forget the scan surface -- it is gone, and a raycast scene built
        from it would answer questions about geometry no longer in the map."""
        self._scene = None
        self._mesh_ready = False

    # ── the image side ───────────────────────────────────────────────────────
    def _image_edges(self, gray, width):
        """(distance transform, gradient x, gradient y, edge count).

        The distance transform turns "find the nearest image edge" into a
        lookup, and its gradient gives the direction to move -- which is what
        makes a Gauss-Newton step possible at all.
        """
        h, w = gray.shape[:2]
        scale = width / float(w)
        small = cv2.resize(gray, (width, max(1, int(round(h * scale)))),
                           interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (5, 5), 0)

        # Auto thresholds from the median: the IR image's exposure is capped
        # and varies with the set, so fixed thresholds do not transfer.
        median = float(np.median(small))
        lo = int(max(0, 0.66 * median))
        hi = int(min(255, 1.33 * median))
        edges = cv2.Canny(small, lo, max(hi, lo + 1))
        n_edges = int(np.count_nonzero(edges))

        # DIST_MASK_PRECISE, not the 3x3 approximation: the gradient of this
        # transform is the only thing steering the fit, and the cheap mask
        # quantises it into steps that stall the solver.
        dt = cv2.distanceTransform(255 - edges, cv2.DIST_L2,
                                   cv2.DIST_MASK_PRECISE)
        smooth = cv2.GaussianBlur(dt, (5, 5), 0)
        gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3) / 8.0
        gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3) / 8.0
        return dt, gx, gy, n_edges, scale

    # ── the model side ───────────────────────────────────────────────────────
    def _model_edges(self, K, T_map_optical, width, height):
        """Ray-cast the scan and return its edge pixels as 3D camera points.

        Returns (pixels Nx2, points-in-camera Nx3).

        The edges are found with the *same* Canny used on the camera image,
        applied to two renders of the scan: inverse depth, whose discontinuities
        are silhouettes, and a headlight shading of the surface normals, whose
        discontinuities are creases. Doing it this way rather than thresholding
        the depth and normal steps directly matters on a real scan -- a
        photogrammetry mesh is locally noisy, so a plain threshold fires on
        reconstruction noise across whole surfaces. Measured here, that gave
        15 548 "edges" out of 57 600 pixels against the image's 3 311; Canny's
        hysteresis and thinning bring the two into the same range, which is what
        the residuals assume.
        """
        rays = self._scene.create_rays_pinhole(
            intrinsic_matrix=o3d.core.Tensor(np.ascontiguousarray(K)),
            extrinsic_matrix=o3d.core.Tensor(
                np.ascontiguousarray(invert_transform(T_map_optical))),
            width_px=width, height_px=height)
        ans = self._scene.cast_rays(rays)
        # Ray directions carry z == 1 in the optical frame, so t_hit is the
        # optical-axis depth directly.
        z = ans["t_hit"].numpy().astype(np.float32)
        hit = np.isfinite(z) & (z > 0.05) & (z < self.max_range_m)
        if np.count_nonzero(hit) < self.min_model_edges:
            return None, None
        z_filled = np.where(hit, z, self.max_range_m).astype(np.float32)

        # Silhouettes: inverse depth is bounded and its steps are occlusion
        # boundaries. (Depth itself is not: a far wall's noise swamps a near
        # object's edge.)
        inv = 1.0 / np.maximum(z_filled, 0.05)
        inv_lo, inv_hi = 1.0 / self.max_range_m, 1.0 / 0.3
        inv_u8 = np.clip((inv - inv_lo) / max(inv_hi - inv_lo, 1e-6) * 255.0,
                         0, 255).astype(np.uint8)
        edges = cv2.Canny(cv2.GaussianBlur(inv_u8, (3, 3), 0), 40, 120)

        # Creases: headlight shading turns a normal discontinuity into an
        # intensity discontinuity, which the same detector then picks up.
        if self.use_creases:
            normals = ans["primitive_normals"].numpy().astype(np.float32)
            dirs = rays.numpy()[:, :, 3:6]
            dirs = dirs / np.maximum(
                np.linalg.norm(dirs, axis=2, keepdims=True), 1e-9)
            lam = np.abs(np.sum(normals * dirs, axis=2))
            shade_u8 = np.clip(lam * 255.0, 0, 255).astype(np.uint8)
            shade_u8[~hit] = 0
            edges |= cv2.Canny(cv2.GaussianBlur(shade_u8, (3, 3), 0), 40, 120)

        edges[~hit] = 0
        vs, us = np.nonzero(edges)
        if len(us) < self.min_model_edges:
            return None, None

        # Cap the residual count: beyond a few thousand the solution stops
        # changing and only the cost grows.
        if len(us) > self.max_edge_samples:
            # Evenly spaced, not random: the cost is compared between poses, so
            # a different draw per render would add noise to that comparison.
            idx = np.linspace(0, len(us) - 1, self.max_edge_samples).astype(int)
            us, vs = us[idx], vs[idx]

        zz = z[vs, us].astype(np.float64)
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        pts = np.stack([(us - cx) / fx * zz, (vs - cy) / fy * zz, zz], axis=1)
        return np.stack([us, vs], axis=1).astype(np.float64), pts

    # ── the lineup ───────────────────────────────────────────────────────────
    def _evaluate(self, dt, Ks, T_map_optical, width, height):
        """One render + residual lookup at a pose. Returns None if unusable."""
        px, pts = self._model_edges(Ks, T_map_optical, width, height)
        if px is None:
            return None
        u_all = px[:, 0].astype(np.int32)
        v_all = px[:, 1].astype(np.int32)
        r_all = dt[v_all, u_all].astype(np.float64)

        # The cost counts EVERY model edge, with anything past the search
        # radius clamped to that radius rather than dropped. Averaging only
        # over the ones that found a match makes the cost incomparable between
        # poses -- and the solver exploits exactly that, moving to a pose that
        # renders fewer, easier edges and reporting a lower cost than the truth.
        # Clamping keeps a missing correspondence expensive; it simply stops
        # contributing gradient.
        clamped = np.minimum(r_all, self.max_search_px)
        delta = max(self.inlier_px, 1e-6)
        huber = np.where(clamped <= delta, 0.5 * clamped * clamped,
                         delta * (clamped - 0.5 * delta))
        cost = float(np.mean(huber))
        score = float(np.mean(r_all < self.inlier_px))
        median = float(np.median(r_all))

        keep = r_all < self.max_search_px
        if np.count_nonzero(keep) < self.min_model_edges:
            return None
        return {
            "u": u_all[keep], "v": v_all[keep], "r": r_all[keep],
            "P": pts[keep], "cost": cost, "score": score, "median": median,
            "n": len(px),
        }

    def align(self, gray, K, T_map_cam, trace=None):
        """Refine a camera pose so the scan's edges land on the image's.

        `T_map_cam` is the camera body pose in the map frame (x forward, y left,
        z up), the same convention the rest of the pipeline uses. Returns an
        EdgeAlignResult; `T_map_cam` on it is the refined body pose.

        Levenberg-Marquardt, not plain Gauss-Newton: a distance transform is
        piecewise linear with kinks at the Voronoi boundaries between edges, so
        a full Gauss-Newton step routinely overshoots into a neighbouring cell.
        Measured on a real scan, fixed damping halved the error in two
        iterations and then oscillated for eighteen more without improving.
        Accepting a step only when it lowers the cost, and stiffening the
        damping when it does not, is what turns that into convergence.
        """
        import time
        t0 = time.time()
        if not self.available:
            return EdgeAlignResult(reason="image alignment unavailable")
        if gray is None:
            return EdgeAlignResult(reason="no camera image")

        gray = np.asarray(gray)
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        if gray.dtype != np.uint8:
            gray = np.clip(gray, 0, 255).astype(np.uint8)

        dt, gx, gy, n_image_edges, scale = self._image_edges(gray,
                                                             self.work_width)
        if n_image_edges < self.min_model_edges:
            return EdgeAlignResult(
                reason=f"only {n_image_edges} edges in the image "
                       f"(too dark, flat or blurred to line up)",
                n_image_edges=n_image_edges,
                elapsed_ms=(time.time() - t0) * 1000)

        height, width = dt.shape[:2]
        Ks = np.asarray(K, dtype=float).copy()
        Ks[:2, :] *= scale

        T0 = np.asarray(T_map_cam, dtype=float)
        T = T0 @ T_BODY_OPTICAL
        state = self._evaluate(dt, Ks, T, width, height)
        if state is None:
            return EdgeAlignResult(
                reason="the scan shows no edges from here",
                n_image_edges=n_image_edges,
                elapsed_ms=(time.time() - t0) * 1000)

        damping = 1e-2
        iterations = 0
        for iterations in range(1, self.max_iterations + 1):
            if trace is not None:
                trace.append((iterations, state["cost"], state["score"],
                              state["median"]))
            J, g_vec = self._normal_equations(state, gx, gy, Ks)
            if J is None:
                break

            improved = False
            for _ in range(6):                      # damping retries
                H = J.copy()
                H[np.diag_indices_from(H)] *= (1.0 + damping)
                try:
                    xi = -np.linalg.solve(H, g_vec)
                except np.linalg.LinAlgError:
                    break
                if not np.all(np.isfinite(xi)):
                    break
                step = float(np.linalg.norm(xi[:3]))
                if step > 0.05:                      # never leap in one step
                    xi = xi * (0.05 / step)
                candidate = T @ invert_transform(se3_exp(xi))
                trial = self._evaluate(dt, Ks, candidate, width, height)
                if trial is not None and trial["cost"] < state["cost"]:
                    T, state = candidate, trial
                    damping = max(damping / 3.0, 1e-6)
                    improved = True
                    break
                damping *= 5.0
            if not improved or damping > 1e4:
                break

        T_refined = T @ invert_transform(T_BODY_OPTICAL)
        moved = float(np.linalg.norm(T_refined[:3, 3] - T0[:3, 3]))
        R_rel = T0[:3, :3].T @ T_refined[:3, :3]
        cos = float(np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0))
        moved_deg = math.degrees(math.acos(cos))
        elapsed = (time.time() - t0) * 1000.0

        if state["score"] < self.min_score:
            return EdgeAlignResult(
                reason=(f"only {state['score']:.0%} of the scan's edges line up "
                        f"with the picture (need {self.min_score:.0%})"),
                score=state["score"], median_px=state["median"],
                n_model_edges=state["n"], n_image_edges=n_image_edges,
                iterations=iterations, elapsed_ms=elapsed)
        if moved > self.max_move_m or moved_deg > self.max_move_deg:
            return EdgeAlignResult(
                reason=(f"lineup wanted to move {moved * 1000:.0f}mm / "
                        f"{moved_deg:.1f}° — further than a refinement should, "
                        f"so the edges matched a different part of the room"),
                score=state["score"], median_px=state["median"],
                n_model_edges=state["n"], n_image_edges=n_image_edges,
                moved_m=moved, moved_deg=moved_deg, iterations=iterations,
                elapsed_ms=elapsed)

        return EdgeAlignResult(
            ok=True, T_map_cam=T_refined, score=state["score"],
            median_px=state["median"], n_model_edges=state["n"],
            n_image_edges=n_image_edges, moved_m=moved, moved_deg=moved_deg,
            iterations=iterations, elapsed_ms=elapsed)

    def _normal_equations(self, state, gx, gy, Ks):
        """Huber-weighted (J^T W J, J^T W r) for the current residuals."""
        u, v, r, P = state["u"], state["v"], state["r"], state["P"]
        gux = gx[v, u].astype(np.float64)
        guy = gy[v, u].astype(np.float64)
        X, Y, Z = P[:, 0], P[:, 1], P[:, 2]
        Zs = np.where(np.abs(Z) < 1e-6, 1e-6, Z)
        fx, fy = float(Ks[0, 0]), float(Ks[1, 1])

        # d(residual)/dX for a pinhole, then dX/dxi = [I | -[X]x].
        a = gux * fx / Zs
        b = guy * fy / Zs
        c = -(gux * fx * X + guy * fy * Y) / (Zs * Zs)
        J = np.empty((len(r), 6))
        J[:, 0] = a
        J[:, 1] = b
        J[:, 2] = c
        J[:, 3] = -b * Z + c * Y
        J[:, 4] = a * Z - c * X
        J[:, 5] = -a * Y + b * X

        delta = max(self.inlier_px, 1e-6)
        w = np.where(r <= delta, 1.0, delta / np.maximum(r, 1e-9))
        JT_w = J.T * w
        H = JT_w @ J
        if not np.all(np.isfinite(H)):
            return None, None
        return H, JT_w @ r

    # ── read-only agreement, for the operator ────────────────────────────────
    def score(self, gray, K, T_map_cam):
        """How well the picture lines up right now, without moving anything."""
        if not self.available or gray is None:
            return None
        gray = np.asarray(gray)
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        dt, _, _, n_image_edges, scale = self._image_edges(gray,
                                                           self.work_width)
        height, width = dt.shape[:2]
        Ks = np.asarray(K, dtype=float).copy()
        Ks[:2, :] *= scale
        px, _ = self._model_edges(
            Ks, np.asarray(T_map_cam, dtype=float) @ T_BODY_OPTICAL,
            width, height)
        if px is None:
            return None
        r = dt[px[:, 1].astype(np.int32), px[:, 0].astype(np.int32)]
        return {
            "score": float(np.mean(r < self.inlier_px)),
            "median_px": float(np.median(r)),
            "n_model_edges": int(len(px)),
            "n_image_edges": int(n_image_edges),
        }

    # ── AR overlay, the thing an operator actually trusts ────────────────────
    def overlay(self, gray, K, T_map_cam, colour=(0, 255, 255)):
        """The camera frame with the scan's edges drawn over it, in colour.

        The numeric score says the lineup is good; this is how an operator
        confirms it -- the same way a matchmove artist checks a lineup by
        looking at it rather than at a residual.
        """
        if not self.available or gray is None:
            return None
        gray = np.asarray(gray)
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        scale = self.work_width / float(w)
        height = max(1, int(round(h * scale)))
        Ks = np.asarray(K, dtype=float).copy()
        Ks[:2, :] *= scale
        px, _ = self._model_edges(
            Ks, np.asarray(T_map_cam, dtype=float) @ T_BODY_OPTICAL,
            self.work_width, height)
        canvas = cv2.cvtColor(
            cv2.resize(gray, (self.work_width, height),
                       interpolation=cv2.INTER_AREA), cv2.COLOR_GRAY2BGR)
        if px is None:
            return canvas
        canvas[px[:, 1].astype(np.int32), px[:, 0].astype(np.int32)] = colour
        return canvas
