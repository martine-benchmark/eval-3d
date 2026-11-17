"""GT kit of a sparse-view evaluation: the base kit filtered by visibility in the custom views.

Nothing is resampled or recomputed. The base `gt_pcd.npy` (the published one, already cleaned with
all the evaluation views) is filtered by the views of the custom set (`paths.views` / `--views`)
with the same call as the second cleaning step of the reconstruction, `filter_by_visibility`: same
silhouettes, same dilation, same orientation test, no watertight masks; a point inside a view's
image but outside its dilated mask is removed (silhouette carving), the others are kept if at least
one view sees them unoccluded (ray casting on the GT mesh) and from their front (the normal of a GT
point is the vertex normal of the nearest vertex of the GT mesh, negated when the orientation check
of the usual cleanup finds the mesh normals inward).
Every per-point array of the kit (`gt_pcd.npy`, `attributes/*.npy`, `challenges/*.npy`, exclusion
masks included) is indexed by the same keep mask, so the attributes stay aligned; the other files of
the kit's scoring subtrees are copied. The result is written to `Groundtruth@<tag>/` with `keep.npy`
(indices into the base `gt_pcd.npy`) and `variant_provenance.json`.

The GT exclusion semantics do not change: `challenges/excluded.npy` is only subset like the others.
"""

import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

#: Sub-directories of a GT kit whose .npy files are indexed over the GT points.
PER_POINT_DIRS = ("attributes", "challenges")
#: Root-level kit files copied to the derived kit (provenance, notes). Meshes, camera files and
#: image directories are sources of the default kit, not scoring artefacts: not copied.
COPIED_ROOT_SUFFIXES = (".json", ".txt", ".md", ".yaml", ".yml", ".csv")

KEEP_FILE = "keep.npy"
PROVENANCE_FILE = "variant_provenance.json"


def require_view_masks(cameras: dict, masks_dir: Path) -> list[Path]:
    """Mask of every view of a custom camera set, looked up by view name; fail if one is missing."""
    names = cameras.get("mask_names")
    if names is None:
        raise ValueError("a custom camera set must be an SfMData file: its masks are looked up by view name")
    paths = [Path(masks_dir) / name for name in names]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} of {len(paths)} mask(s) missing in {masks_dir}: {', '.join(missing[:10])}"
        )
    return paths


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def visibility_settings(config, views_path: Path) -> dict:
    """What a second cleaning step depends on: the view file and the visibility settings."""
    c = config.cleanup
    return {
        "views": str(views_path),
        "views_md5": md5_file(views_path),
        "rule": "filter_by_visibility",
        "use_masks": bool(c.use_masks),
        "dilation_radius": c.dilation_radius if isinstance(c.dilation_radius, str) else int(c.dilation_radius),
        "dilation_ref_px": int(c.dilation_ref_px),
        "dilation_ref_n_pixels": int(c.dilation_ref_n_pixels),
        "views_max_normal_angle": float(c.views_max_normal_angle),
    }


def nearest_vertex_normals(points: np.ndarray, mesh) -> np.ndarray:
    """Normal of each point: the vertex normal of the nearest vertex of ``mesh``."""
    from scipy.spatial import cKDTree

    _, nearest = cKDTree(np.asarray(mesh.vertices)).query(np.asarray(points, dtype=np.float64), workers=-1)
    return np.asarray(mesh.vertex_normals, dtype=np.float64)[nearest]


def count_undefined_normals(normals: np.ndarray) -> int:
    return int((np.linalg.norm(normals, axis=1) == 0.0).sum())


def orient_normals(normals: np.ndarray, mesh, cameras: dict, what: str) -> tuple[np.ndarray, dict]:
    """Negate ``normals`` (taken from ``mesh``) when the orientation check of the usual cleanup
    (`detect_normal_flip`: samples seen from camera 0 of ``cameras``) finds the mesh normals
    inward. Returns the normals to use and what goes in the provenance."""
    from ..core.visibility import detect_normal_flip

    d = detect_normal_flip(mesh, cameras)
    frac = "n/a" if d["front_fraction"] is None else f"{d['front_fraction']:.3f}"
    print(f"  Normals of the {what}: front-facing fraction (cam0 samples)={frac} "
          f"({d['n_front']}/{d['n_total']}), flip={d['flip']}")
    if d["flip"]:
        print(f"  WARNING: the normals of the {what} point inward: flipped for the orientation test")
        normals = -normals
    return normals, {
        "normals_flipped": d["flip"],
        "front_facing_fraction": d["front_fraction"],
        "front_facing_samples": d["n_total"],
    }


def _kit_layout(base: Path, n_base: int) -> tuple[list[str], list[str], list[str]]:
    """(per-point arrays, copied files, skipped entries) of a base kit, as paths relative to it."""
    per_point, copied, skipped = [], [], []
    for p in sorted(base.iterdir()):
        if p.is_dir():
            if p.name not in PER_POINT_DIRS:
                skipped.append(p.name + "/")
            continue
        if p.suffix == ".npy" and p.name != KEEP_FILE:
            per_point.append(p.name)
        elif p.suffix in COPIED_ROOT_SUFFIXES and p.name != PROVENANCE_FILE:
            copied.append(p.name)
        elif p.name != "gt_pcd.ply":
            skipped.append(p.name)
    for sub in PER_POINT_DIRS:
        d = base / sub
        if not d.is_dir():
            continue
        for p in sorted(d.rglob("*")):
            if not p.is_file():
                continue
            rel = str(p.relative_to(base))
            (per_point if p.suffix == ".npy" else copied).append(rel)

    # Fail before any computation if an array is not aligned with gt_pcd.npy.
    for rel in per_point:
        n = np.load(base / rel, mmap_mode="r").shape[0]
        if n != n_base:
            raise ValueError(
                f"{base / rel} has {n} rows but gt_pcd.npy has {n_base} points: stale or foreign array, "
                "refusing to subset it"
            )
    return per_point, copied, skipped


def _occluder(config, object_name: str):
    """GT mesh for the ray casting: the one the default visibility_count uses (cleaned if present)."""
    import trimesh

    from ..core.mesh import load_mesh
    from .preprocess_gt import _raw_gt_parts

    cleaned = config.get_gt_mesh_path(object_name, cleaned=True)
    if cleaned.exists():
        return load_mesh(cleaned), [str(cleaned)]
    parts = _raw_gt_parts(config, object_name)
    mesh = trimesh.util.concatenate([load_mesh(p) for p in parts]) if len(parts) > 1 else load_mesh(parts[0])
    return mesh, [str(p) for p in parts]


def _dilation_per_view(config, mask_paths: list[Path]) -> list[int]:
    from PIL import Image

    from ..core.visibility import compute_auto_dilation

    c = config.cleanup
    if not c.use_masks:
        return [0] * len(mask_paths)
    if c.dilation_radius != "auto":
        return [int(c.dilation_radius)] * len(mask_paths)
    out = []
    for p in mask_paths:
        with Image.open(p) as im:
            w, h = im.size
        out.append(compute_auto_dilation(w * h, c.dilation_ref_px, c.dilation_ref_n_pixels))
    return out


def derive_variant_gt(config, object_name: str, force: bool = False) -> Path:
    """Write `Groundtruth@<tag>/` for `object_name`; returns its path.

    An existing derived kit is kept when its camera file and visibility settings are unchanged; a
    kit derived with other ones is refused unless `force`.
    """
    from ..core.camera import load_cameras_auto
    from ..core.visibility import filter_by_visibility

    if not config.is_variant:
        raise ValueError("derive_variant_gt needs a custom view set (--views)")

    base = config.get_base_gt_artifacts_dir(object_name)
    out = config.get_variant_gt_dir(object_name)
    cameras_path = config.get_views_path(object_name)
    masks_dir = config.get_views_masks_dir(object_name)
    c = config.cleanup

    if out.resolve() == base.resolve():
        raise ValueError(f"the derived kit would overwrite its base: {out}")
    if not cameras_path.exists():
        raise FileNotFoundError(f"Custom view set not found: {cameras_path}")
    if not (base / "gt_pcd.npy").exists():
        raise FileNotFoundError(f"GT point cloud not found: {base / 'gt_pcd.npy'}")

    settings = {**visibility_settings(config, cameras_path), "base_gt_dir": str(base)}

    prov_path = out / PROVENANCE_FILE
    if prov_path.exists() and not force:
        previous = json.loads(prov_path.read_text())
        changed = [k for k, v in settings.items() if previous.get(k) != v]
        if not changed:
            print(f"  Sparse-view GT exists, skipping: {out}")
            return out
        raise ValueError(
            f"{out} was derived with other settings ({', '.join(changed)}); pass --force to rebuild it"
        )

    print(f"Sparse-view GT for {object_name} ({config.variant}): {cameras_path}")
    gt_pcd = np.load(base / "gt_pcd.npy")
    n_base = len(gt_pcd)
    per_point, copied, skipped = _kit_layout(base, n_base)

    cameras = load_cameras_auto(cameras_path)
    mask_paths = require_view_masks(cameras, masks_dir)
    views = [Path(n).stem for n in cameras["mask_names"]]
    print(f"  {len(views)} view(s): {', '.join(views)}")

    mesh, mesh_paths = _occluder(config, object_name)
    print(f"  Occluder: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces")

    # Same call as the second cleaning step of the reconstruction, on the GT points.
    normals, orientation = orient_normals(nearest_vertex_normals(gt_pcd, mesh), mesh, cameras, "GT mesh")
    keep = filter_by_visibility(
        gt_pcd, mesh, cameras, masks_dir,
        dilation_radius=c.dilation_radius,
        show_progress=True,
        use_masks=c.use_masks,
        dilation_ref_px=c.dilation_ref_px,
        dilation_ref_n_pixels=c.dilation_ref_n_pixels,
        normals=normals,
        max_normal_angle=c.views_max_normal_angle,
    )
    n_kept = int(keep.sum())
    print(f"  Kept {n_kept:,} / {n_base:,} GT points ({100.0 * n_kept / n_base:.2f}%)")
    if n_kept == 0:
        raise ValueError(f"no GT point of {object_name} is seen by the views of {cameras_path}")

    tmp = out.parent / f".{out.name}.tmp-{os.getpid()}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        for rel in per_point:
            dst = tmp / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            np.save(dst, np.load(base / rel)[keep])
        for rel in copied:
            dst = tmp / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(base / rel, dst)
        if (base / "gt_pcd.ply").exists():
            import trimesh

            trimesh.PointCloud(gt_pcd[keep]).export(tmp / "gt_pcd.ply")
            per_point.append("gt_pcd.ply")
        np.save(tmp / KEEP_FILE, np.flatnonzero(keep).astype(np.int64))

        prov = {
            "variant": config.variant,
            **settings,
            "views_list": views,
            "n_views": len(views),
            "masks_dir": str(masks_dir),
            "dilation_px_per_view": _dilation_per_view(config, mask_paths),
            "occluder_mesh": mesh_paths,
            "normals": "vertex normal of the nearest vertex of the occluder mesh",
            "n_undefined_normals": count_undefined_normals(normals),
            **orientation,
            "n_base": n_base,
            "n_kept": n_kept,
            "kept_fraction": n_kept / n_base,
            "subset_files": per_point,
            "copied_files": copied,
            "not_copied": skipped,
            "written": datetime.now().isoformat(timespec="seconds"),
        }
        (tmp / PROVENANCE_FILE).write_text(json.dumps(prov, indent=2))
        if out.exists():
            shutil.rmtree(out)
        tmp.rename(out)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)
    print(f"  Saved {out}")
    return out
