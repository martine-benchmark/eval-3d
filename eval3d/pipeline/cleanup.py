"""Mesh cleanup via visibility-based filtering."""


import numpy as np

from ..config import Config
from ..core.camera import load_cameras_auto
from ..core.mesh import filter_mesh_by_vertex_mask, load_mesh, save_mesh
from ..core.visibility import filter_by_issue_watertight, filter_by_visibility
from .variant_gt import require_view_masks, visibility_settings


def cleanup_mesh(
    config: Config,
    object_name: str,
    method_name: str,
    force: bool = False,
) -> None:
    """Clean up a reconstructed mesh by removing non-visible vertices.

    Algorithm:
    1. Load raw mesh from results_raw/mesh.ply
    2. For each camera view:
       - Project vertices to image
       - Check if inside (dilated) mask
       - Check visibility via ray tracing
    3. Keep vertices visible from at least one view
    4. Save cleaned mesh to results_cleaned/mesh.ply

    Args:
        config: Pipeline configuration.
        object_name: Object name.
        method_name: Method name.
        force: Overwrite existing output.
    """
    if config.is_variant:
        # Sparse-view evaluation: the usual cleanup, then a second step on the custom views.
        cleanup_mesh_views(config, object_name, method_name, force)
        return

    raw_mesh_path = config.get_raw_mesh_path(object_name, method_name)
    cleaned_mesh_path = config.get_cleaned_mesh_path(object_name, method_name)
    cameras_path = config.get_cameras_path(object_name, method_name)
    masks_dir = config.get_masks_dir(object_name, method_name)

    if not raw_mesh_path.exists():
        raise FileNotFoundError(f"Raw mesh not found: {raw_mesh_path}")

    if cleaned_mesh_path.exists() and not force:
        print(f"  Cleaned mesh exists, skipping: {cleaned_mesh_path}")
        return

    print(f"Cleaning mesh: {object_name}/{method_name}")

    mesh = load_mesh(raw_mesh_path)
    print(f"  Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        # A failed reconstruction (e.g. COLMAP on too-few views writes an empty PLY).
        # Persist an empty cleaned mesh instead of returning early, so evaluate() finds
        # its input and records a failure (NaN) cell rather than raising FileNotFound.
        print(f"  Warning: {object_name}/{method_name} - Empty mesh (0 vertices/faces) -> writing empty cleaned mesh (scored as failure)")
        cleaned_mesh_path.parent.mkdir(parents=True, exist_ok=True)
        import trimesh
        save_mesh(trimesh.Trimesh(), cleaned_mesh_path)
        return

    if not cameras_path.exists():
        raise FileNotFoundError(f"Cameras not found: {cameras_path}")
    if config.cleanup.use_masks and not masks_dir.exists():
        raise FileNotFoundError(f"Masks directory not found: {masks_dir}")

    # Apply z_threshold FIRST (before visibility) - matching reference implementation
    if config.cleanup.z_threshold is not None:
        z_mask = mesh.vertices[:, 2] >= config.cleanup.z_threshold
        n_before = len(mesh.vertices)
        mesh = filter_mesh_by_vertex_mask(mesh, z_mask)
        print(f"  After z-threshold ({config.cleanup.z_threshold}): {len(mesh.vertices)} vertices ({n_before - len(mesh.vertices)} removed)")

    cameras = load_cameras_auto(cameras_path)
    print(f"  Loaded {len(cameras['P'])} camera views")

    print(f"  Filtering by visibility (dilation={config.cleanup.dilation_radius}, use_masks={config.cleanup.use_masks})")
    visible_mask = filter_by_visibility(
        mesh.vertices,
        mesh,
        cameras,
        masks_dir,
        dilation_radius=config.cleanup.dilation_radius,
        show_progress=True,
        use_masks=config.cleanup.use_masks,
        dilation_ref_px=config.cleanup.dilation_ref_px,
        dilation_ref_n_pixels=config.cleanup.dilation_ref_n_pixels,
    )

    keep_mask = visible_mask
    n_visible = int(np.sum(visible_mask))

    # Robin's watertight culling: additionally drop visible, front-facing vertices that
    # project into the per-view hole-region masks (surfaces filling the GT's non-watertight
    # Optional third mask family. Its presence is the feature flag: published
    # kits without it retain the ordinary silhouette cleanup.
    issue_dir = config.get_watertight_masks_dir(object_name)
    if issue_dir.exists() and any(issue_dir.glob("*.png")):
        print(f"  Watertight culling using: {issue_dir}")
        issue_counts = filter_by_issue_watertight(
            mesh.vertices,
            mesh,
            cameras,
            issue_dir,
            visible_mask=visible_mask,
            orientation_ratio_threshold=config.cleanup.watertight_orientation_ratio,
            show_progress=True,
        )
        keep_mask = visible_mask & (issue_counts == 0)
        n_culled = n_visible - int(np.sum(keep_mask))
        pct = (100.0 * n_culled / n_visible) if n_visible else 0.0
        print(f"  Watertight culling removed {n_culled} extra vertices "
              f"({pct:.2f}% of the {n_visible} visible)")

    n_kept = int(np.sum(keep_mask))
    n_removed = len(mesh.vertices) - n_kept
    print(f"  Visible vertices: {n_visible} ({100*n_visible/len(mesh.vertices):.1f}%)")
    print(f"  Kept after culling: {n_kept}  |  Removed total: {n_removed}")

    cleaned_mesh = filter_mesh_by_vertex_mask(mesh, keep_mask)
    print(f"  Cleaned mesh: {len(cleaned_mesh.vertices)} vertices, {len(cleaned_mesh.faces)} faces")

    cleaned_mesh_path.parent.mkdir(parents=True, exist_ok=True)
    save_mesh(cleaned_mesh, cleaned_mesh_path)
    print(f"  Saved: {cleaned_mesh_path}")


VIEWS_PROVENANCE = "views_provenance.json"


def cleanup_mesh_views(
    config: Config,
    object_name: str,
    method_name: str,
    force: bool = False,
) -> None:
    """Sparse-view cleanup of a reconstruction, in two steps.

    1. The usual cleanup (all evaluation views, silhouettes, watertight masks) to
       results_cleaned/mesh.ply, reused when it exists.
    2. The same visibility filter (`filter_by_visibility`, same dilation) on that cleaned mesh
       with the cameras and silhouettes of the custom view set, to results_cleaned@<tag>/mesh.ply.
       No watertight masks here: they only remove vertices, and step 1 has applied them.

    `force` rebuilds step 2 only.
    """
    import json

    base = config.default_view()
    cleanup_mesh(base, object_name, method_name)
    source_path = base.get_cleaned_mesh_path(object_name, method_name)
    out_path = config.get_cleaned_mesh_path(object_name, method_name)
    views_path = config.get_views_path(object_name, method_name)
    masks_dir = config.get_views_masks_dir(object_name, method_name)
    prov_path = out_path.parent / VIEWS_PROVENANCE

    if not views_path.exists():
        raise FileNotFoundError(f"Custom view set not found: {views_path}")
    settings = visibility_settings(config, views_path)

    if out_path.exists() and not force:
        previous = json.loads(prov_path.read_text()) if prov_path.exists() else {}
        changed = [k for k, v in settings.items() if previous.get(k) != v]
        if changed:
            raise ValueError(
                f"{out_path} was cleaned with other settings ({', '.join(changed)}); pass --force to rebuild it"
            )
        print(f"  Sparse-view cleaned mesh exists, skipping: {out_path}")
        return

    print(f"Sparse-view cleanup ({config.variant}): {object_name}/{method_name}")
    mesh = load_mesh(source_path)
    n_source = len(mesh.vertices)
    print(f"  Loaded cleaned mesh: {n_source} vertices, {len(mesh.faces)} faces")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if n_source == 0 or len(mesh.faces) == 0:
        # Failed reconstruction: keep the empty mesh, scored as a failure by evaluate().
        import trimesh
        save_mesh(trimesh.Trimesh(), out_path)
        views, n_kept = [], 0
    else:
        cameras = load_cameras_auto(views_path)
        require_view_masks(cameras, masks_dir)
        views = [name.rsplit(".", 1)[0] for name in cameras["mask_names"]]
        print(f"  {len(views)} view(s): {', '.join(views)}")
        keep_mask = filter_by_visibility(
            mesh.vertices,
            mesh,
            cameras,
            masks_dir,
            dilation_radius=config.cleanup.dilation_radius,
            show_progress=True,
            use_masks=config.cleanup.use_masks,
            dilation_ref_px=config.cleanup.dilation_ref_px,
            dilation_ref_n_pixels=config.cleanup.dilation_ref_n_pixels,
        )
        n_kept = int(np.sum(keep_mask))
        print(f"  Kept {n_kept} / {n_source} vertices of the cleaned mesh ({100 * n_kept / n_source:.1f}%)")
        cleaned_mesh = filter_mesh_by_vertex_mask(mesh, keep_mask)
        print(f"  Sparse-view mesh: {len(cleaned_mesh.vertices)} vertices, {len(cleaned_mesh.faces)} faces")
        save_mesh(cleaned_mesh, out_path)

    prov_path.write_text(json.dumps({
        "variant": config.variant,
        **settings,
        "views_list": views,
        "n_views": len(views),
        "masks_dir": str(masks_dir),
        "source_mesh": str(source_path),
        "n_source_vertices": n_source,
        "n_kept_vertices": n_kept,
    }, indent=2))
    print(f"  Saved: {out_path}")
