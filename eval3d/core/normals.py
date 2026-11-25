"""Rendered normal maps, per-pixel angular error and MAE between a reconstruction and the GT.

Normal maps are ray cast (embree through trimesh), one ray per pixel centre, so the GT and the
reconstruction are sampled at exactly the same rays: their maps overlap pixel for pixel. Normals are
the mesh vertex normals interpolated with the barycentric coordinates of the hit, expressed in the
camera frame with the OpenGL axes (x right, y up, z towards the camera), the convention of the
photometric-stereo normal maps ((n + 1) / 2 colour coding).

The pixel grid follows the camera dict: the centre of the top-left pixel is at ``pixel_centre``
(0.0 for AliceVision/OpenCV, ``cameras["pixel_centre"]``), so pixel (row, col) is the ray through
(col + pixel_centre, row + pixel_centre). There is no half-pixel shift to compensate, unlike the
OpenGL rasteriser of ``rendering.py`` which samples at (i + 0.5).
"""

from dataclasses import dataclass

import matplotlib
import numpy as np
import trimesh

# Colour-bar bound of the angular-error maps (degrees); errors above it are saturated.
ANGULAR_ERROR_MAX_DEG = 45.0
# Pixels on an excluded GT region (same light gray as the excluded vertices of colormap.py).
EXCLUDED_RGB = (200, 200, 200)
# Pixels seen by only one of the two meshes (no angle defined there).
SINGLE_RGB = (232, 232, 232)

_CV_TO_GL = np.array([1.0, -1.0, -1.0])


@dataclass
class NormalRender:
    """One ray-cast view. All arrays are (h, w[, 3]) over the rendered window (full image or bbox)."""

    normals: np.ndarray  # float32, unit camera-frame OpenGL normals, 0 where no hit; flipped on back faces (see render)
    mask: np.ndarray  # bool, ray hit the mesh
    points: np.ndarray  # float32 world hit points, NaN where no hit
    face_index: np.ndarray  # int32 hit face, -1 where no hit
    nearest_vertex: np.ndarray  # int32 vertex of the hit face closest to the hit (max barycentric), -1
    facing: np.ndarray  # float32 cos(interpolated mesh normal, direction to the camera), before any flip
    back_face: np.ndarray  # bool, the hit face is seen from behind (its winding turns away from the camera)


def scale_intrinsics(K: np.ndarray, scale: float, pixel_centre: float = 0.0) -> np.ndarray:
    """Intrinsics of the same camera on a grid resized by ``scale``.

    Pixel edges are at integers + (0.5 - pixel_centre); scaling the edges gives
    c_s = s * c + (s - 1) * (0.5 - pixel_centre), i.e. c_s = s * c - (1 - s) / 2 for AliceVision.
    """
    Ks = np.array(K, dtype=np.float64, copy=True)
    Ks[0, 0] *= scale
    Ks[1, 1] *= scale
    off = (scale - 1.0) * (0.5 - pixel_centre)
    Ks[0, 2] = scale * K[0, 2] + off
    Ks[1, 2] = scale * K[1, 2] + off
    return Ks


def _intersector(mesh: trimesh.Trimesh):
    try:
        from trimesh.ray.ray_pyembree import RayMeshIntersector
    except ImportError:  # pragma: no cover - embreex is a trimesh[easy] dependency
        from trimesh.ray.ray_triangle import RayMeshIntersector
    return RayMeshIntersector(mesh)


class NormalRenderer:
    """Ray-casts normal maps of one mesh; build once, render many views."""

    def __init__(self, mesh: trimesh.Trimesh, chunk_size: int = 2_000_000):
        self.mesh = mesh
        self.faces = np.asarray(mesh.faces, dtype=np.int64)
        self.vertices = np.asarray(mesh.vertices, dtype=np.float64)
        self.vertex_normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
        self.face_normals = np.asarray(mesh.face_normals, dtype=np.float64)
        self.chunk_size = chunk_size
        self._rmi = _intersector(mesh)

    def render(
        self,
        K: np.ndarray,
        Rt: np.ndarray,
        shape: tuple[int, int],
        pixel_centre: float = 0.0,
        bbox: tuple[int, int, int, int] | None = None,
        flip_back_faces: bool = True,
    ) -> NormalRender:
        """Normal map of the z-buffer surface (first hit of each pixel ray).

        With ``flip_back_faces``, a pixel whose hit face is seen from behind gets the flipped normal: the visible side
        of the surface is compared, whatever the winding of the mesh (as the renders draw it).
        """
        H, W = shape
        rmin, rmax, cmin, cmax = bbox if bbox is not None else (0, H - 1, 0, W - 1)
        h, w = rmax - rmin + 1, cmax - cmin + 1
        R = np.asarray(Rt, dtype=np.float64)[:3, :3]
        t = np.asarray(Rt, dtype=np.float64)[:3, 3]
        centre = -R.T @ t
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

        normals = np.zeros((h * w, 3), dtype=np.float32)
        points = np.full((h * w, 3), np.nan, dtype=np.float32)
        face_index = np.full(h * w, -1, dtype=np.int32)
        nearest_vertex = np.full(h * w, -1, dtype=np.int32)
        facing = np.zeros(h * w, dtype=np.float32)
        back_face = np.zeros(h * w, dtype=bool)

        xs = (np.arange(cmin, cmax + 1) + pixel_centre - cx) / fx
        ys = (np.arange(rmin, rmax + 1) + pixel_centre - cy) / fy
        n_pix = h * w
        for start in range(0, n_pix, self.chunk_size):
            ids = np.arange(start, min(start + self.chunk_size, n_pix))
            d_cam = np.stack([xs[ids % w], ys[ids // w], np.ones(len(ids))], axis=1)
            d_world = d_cam @ R  # rows of R^T d
            d_world /= np.linalg.norm(d_world, axis=1, keepdims=True)
            origins = np.broadcast_to(centre, d_world.shape)
            tri, ray, loc = self._rmi.intersects_id(
                origins, d_world, multiple_hits=False, return_locations=True
            )
            if len(tri) == 0:
                continue
            tri = np.asarray(tri, dtype=np.int64)
            corners = self.faces[tri]
            bary = trimesh.triangles.points_to_barycentric(self.vertices[corners], loc)
            n_w = np.einsum("nk,nkd->nd", bary, self.vertex_normals[corners])
            n_w /= np.maximum(np.linalg.norm(n_w, axis=1, keepdims=True), 1e-12)
            pix = ids[ray]
            facing[pix] = -np.einsum("nd,nd->n", n_w, d_world[ray])
            back = np.einsum("nd,nd->n", self.face_normals[tri], d_world[ray]) > 0
            back_face[pix] = back
            if flip_back_faces:
                n_w[back] *= -1.0
            normals[pix] = (n_w @ R.T) * _CV_TO_GL
            points[pix] = loc
            face_index[pix] = tri
            nearest_vertex[pix] = corners[np.arange(len(tri)), np.argmax(bary, axis=1)]

        return NormalRender(
            normals=normals.reshape(h, w, 3),
            mask=(face_index >= 0).reshape(h, w),
            points=points.reshape(h, w, 3),
            face_index=face_index.reshape(h, w),
            nearest_vertex=nearest_vertex.reshape(h, w),
            facing=facing.reshape(h, w),
            back_face=back_face.reshape(h, w),
        )


def render_normal_map(
    mesh: trimesh.Trimesh,
    K: np.ndarray,
    Rt: np.ndarray,
    shape: tuple[int, int],
    pixel_centre: float = 0.0,
    bbox: tuple[int, int, int, int] | None = None,
    flip_back_faces: bool = True,
) -> NormalRender:
    """Ray-cast the normal map of ``mesh`` seen by camera (K, Rt world->camera) on an (H, W) grid.

    ``bbox`` = (rmin, rmax, cmin, cmax), inclusive as ``rendering.crop_to_bbox``: only those pixels
    are cast and returned.
    """
    return NormalRenderer(mesh).render(K, Rt, shape, pixel_centre=pixel_centre, bbox=bbox,
                                       flip_back_faces=flip_back_faces)


def angular_error_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angle in degrees between vectors along the last axis.

    atan2(|a x b|, a . b): exact at small angles, where arccos of a float32 dot product has a
    ~0.01 deg noise floor.
    """
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    sin = np.linalg.norm(np.cross(a, b), axis=-1)
    cos = np.einsum("...d,...d->...", a, b)
    return np.degrees(np.arctan2(sin, cos))


def pixel_categories(gt_mask: np.ndarray, rec_mask: np.ndarray, excluded: np.ndarray) -> dict[str, np.ndarray]:
    """Split the pixels: ``valid`` (both meshes, not excluded: counted in the MAE), ``excluded``
    (seen by a mesh on an excluded region) and ``single`` (seen by one mesh only, not excluded)."""
    seen = gt_mask | rec_mask
    return {
        "valid": gt_mask & rec_mask & ~excluded,
        "excluded": seen & excluded,
        "single": (gt_mask ^ rec_mask) & ~excluded,
    }


def normal_error_stats(
    err_deg: np.ndarray, categories: dict[str, np.ndarray], gt_mask: np.ndarray, excluded: np.ndarray
) -> dict:
    """MAE and companions of one view, over the valid pixels."""
    valid = categories["valid"]
    e = err_deg[valid]
    n_gt = int((gt_mask & ~excluded).sum())
    return {
        "mae_deg": float(e.mean()) if e.size else float("nan"),
        "median_deg": float(np.median(e)) if e.size else float("nan"),
        "n_valid": int(e.size),
        "n_gt": n_gt,
        "n_single": int(categories["single"].sum()),
        "n_excluded": int(categories["excluded"].sum()),
        "gt_coverage": float(e.size / n_gt) if n_gt else float("nan"),
    }


def encode_normals_rgb(
    normals: np.ndarray,
    mask: np.ndarray,
    background: tuple[int, int, int] = (255, 255, 255),
    excluded: np.ndarray | None = None,
) -> np.ndarray:
    """(n + 1) / 2 colour coding (uint8), background outside ``mask``, gray on ``excluded``."""
    rgb = np.empty(mask.shape + (3,), dtype=np.uint8)
    rgb[:] = background
    rgb[mask] = np.clip(np.round((normals[mask] + 1.0) * 0.5 * 255.0), 0, 255).astype(np.uint8)
    if excluded is not None:
        rgb[mask & excluded] = EXCLUDED_RGB
    return rgb


def color_angular_error(
    err_deg: np.ndarray,
    categories: dict[str, np.ndarray],
    cmap: str = "viridis",
    background: tuple[int, int, int] = (255, 255, 255),
    vmax: float = ANGULAR_ERROR_MAX_DEG,
) -> tuple[np.ndarray, dict]:
    """Colour the angular-error map; returns (rgb uint8, colorbar params for rendering.add_colorbar)."""
    rgb = np.empty(err_deg.shape + (3,), dtype=np.uint8)
    rgb[:] = background
    valid = categories["valid"]
    t = np.clip(err_deg[valid] / vmax, 0.0, 1.0)
    rgb[valid] = np.round(matplotlib.colormaps[cmap](t)[:, :3] * 255.0).astype(np.uint8)
    rgb[categories["excluded"]] = EXCLUDED_RGB
    rgb[categories["single"]] = SINGLE_RGB
    return rgb, {"vmin": 0.0, "vmax": float(vmax), "cmap": cmap, "label": "Angular error (°)"}
