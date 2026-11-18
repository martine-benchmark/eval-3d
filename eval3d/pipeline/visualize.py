"""Visualization pipeline: render meshes with metric coloring."""

import json
import os
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from scipy.spatial import cKDTree

from ..config import Config
from ..core.camera import load_cameras_auto
from ..core.colormap import (
    color_by_accuracy,
    color_by_completeness,
    color_by_curvature,
    color_by_visibility,
    color_uniform,
)
from ..core.mesh import load_mesh
from ..core.nearest import NearestPointIndex
from ..core.rendering import (
    add_colorbar,
    check_pyrender,
    render_views,
)


def points_to_mesh(points: np.ndarray, colors: np.ndarray, point_size: float = 0.5) -> trimesh.Trimesh:
    """Convert colored point cloud to a mesh of small spheres for rendering.

    For efficiency, creates billboards (camera-facing quads) instead of spheres.
    Actually, creates a simple point cloud mesh that pyrender can render.
    """
    # Create a PointCloud that can be rendered
    cloud = trimesh.PointCloud(points)
    cloud.colors = colors
    return cloud


def visualize_method(
    config: Config,
    object_name: str,
    method_name: str,
    metrics: list[str],
    view_indices: list[int] | None = None,
    scale: float | None = None,
    crop: bool | None = None,
    crop_margin: int | None = None,
    cmap_overrides: dict | None = None,
    max_dist: float | None = None,
    force: bool = False,
    decimation_target: int = 10_000_000,
    exclude_mode: str | None = None,
    integrated_colorbar: bool | None = None,
    excluded_override_path: Path | None = None,
) -> dict[str, list[Path]]:
    """Generate all visualization renders for a method.

    Metric assignment:
    - GT mesh only: visibility, curvature (GT attributes)
    - Method mesh only: accuracy, completeness (distance metrics)
    - Both: uniform (plain gray render)
    - Normals (ray cast, both meshes): normals (GT and method normal maps), angular_error
      (per-pixel angle between them, plus angular_error/mae.json). Excluded regions follow
      exclude_mode as for the other metrics: 'gray' paints them gray (out of the MAE),
      'remove' ray casts the meshes with the excluded faces removed.

    Args:
        config: Pipeline configuration
        object_name: Object name
        method_name: Method name
        metrics: List of metrics to visualize
        view_indices: View indices to render (None = use config default)
        scale: Resolution scale factor (None = use config)
        crop: Crop to bbox (None = use config)
        crop_margin: Margin around bbox (None = use config)
        cmap_overrides: Override colormaps per metric
        max_dist: Max distance for error coloring (None = use config)
        force: Overwrite existing files
        decimation_target: Max number of faces for visualization (decimate if exceeded)
        exclude_mode: How to handle excluded regions: 'none', 'gray', 'remove' (None = use config)
        integrated_colorbar: If True, composite the lateral colorbar onto each rendered
            view and save it as view_XXX_cb.png (in addition to the raw view_XXX.png).
            If False (default), keep the current behavior of saving a standalone
            colorbar.png. None = use config (visualization.integrated_colorbar).
        excluded_override_path: Optional path to a boolean .npy exclusion mask (indexed
            like gt_pcd, True = excluded) to load INSTEAD of challenges/excluded.npy.
            Only the source of the mask changes; the coloring logic (gray for excluded,
            max_dist cap) is untouched.

    Returns:
        Dict mapping metric name to list of generated image paths
    """
    check_pyrender()

    # Metric categories
    GT_ONLY_METRICS = {"visibility", "curvature"}
    METHOD_ONLY_METRICS = {"accuracy", "completeness"}
    BOTH_METRICS = {"uniform"}
    NORMAL_METRICS = {"normals", "angular_error"}

    # Get paths
    method_dir = config.get_method_dir(object_name, method_name)
    cleaned_mesh_path = config.get_cleaned_mesh_path(object_name, method_name)
    cameras_path = config.get_cameras_path(object_name, method_name)
    eval_dir = config.get_eval_dir(object_name, method_name)
    gt_dir = config.get_gt_artifacts_dir(object_name)

    if exclude_mode is None:
        exclude_mode = config.visualization.exclude_mode
    if integrated_colorbar is None:
        integrated_colorbar = config.visualization.integrated_colorbar

    vis_suffix = "visualizations_remove" if exclude_mode == "remove" else "visualizations"
    vis_dir = method_dir / vis_suffix
    # GT renders/bbox.json are DERIVED: write them to the workspace Groundtruth dir,
    # never into gt_dir, which may be the (read-only by contract) published GT kit.
    gt_vis_dir = config.get_gt_dir(object_name) / vis_suffix

    # Use config defaults if not specified
    if view_indices is None:
        view_indices = config.visualization.default_views
    if scale is None:
        scale = config.visualization.scale
    if crop is None:
        crop = config.visualization.crop
    if crop_margin is None:
        crop_margin = config.visualization.crop_margin
    if max_dist is None:
        max_dist = config.visualization.max_dist

    decimation_target = decimation_target or config.visualization.decimation_target
    min_component_size = config.visualization.min_component_size

    colormaps = config.visualization.colormaps.copy()
    if cmap_overrides:
        colormaps.update(cmap_overrides)

    # Load cameras
    if not cameras_path.exists():
        raise FileNotFoundError(f"Cameras not found: {cameras_path}")
    cameras = load_cameras_auto(cameras_path)

    # Load meshes (only load method mesh if needed by requested metrics)
    needs_method_mesh = bool(set(metrics) & (METHOD_ONLY_METRICS | BOTH_METRICS | NORMAL_METRICS))
    method_mesh = None
    if needs_method_mesh and cleaned_mesh_path.exists():
        method_mesh = load_mesh(cleaned_mesh_path)
        if len(method_mesh.faces) > decimation_target:
            print(f"  Decimating method mesh: {len(method_mesh.faces)} -> {decimation_target} faces")
            method_mesh = method_mesh.simplify_quadric_decimation(face_count=decimation_target)

    gt_mesh_path = config.get_gt_mesh_path(object_name, cleaned=True)
    if not gt_mesh_path.exists():
        gt_mesh_path = config.get_gt_mesh_path(object_name, cleaned=False)
    gt_mesh = load_mesh(gt_mesh_path) if gt_mesh_path.exists() else None
    gt_mesh_original_vertices = gt_mesh.vertices.copy() if gt_mesh is not None else None
    if gt_mesh is not None and len(gt_mesh.faces) > decimation_target:
        print(f"  Decimating GT mesh: {len(gt_mesh.faces)} -> {decimation_target} faces")
        gt_mesh = gt_mesh.simplify_quadric_decimation(face_count=decimation_target)

    # Load GT point cloud (needed for attribute mapping to decimated meshes)
    gt_pcd_path = gt_dir / "gt_pcd.npy"
    gt_pcd = np.load(gt_pcd_path) if gt_pcd_path.exists() else None

    # Load and apply exclusion mask according to exclude_mode.
    # excluded_override_path lets a caller substitute a different boolean mask for the
    # default challenges/excluded.npy. Only the SOURCE of the mask changes here; the
    # downstream coloring (gray + max_dist cap) is untouched.
    excluded_npy_path = Path(excluded_override_path) if excluded_override_path is not None \
        else gt_dir / "challenges" / "excluded.npy"
    gt_excluded_mask = None
    method_excluded_mask = None

    if exclude_mode != "none" and excluded_npy_path.exists() and gt_pcd is not None:
        print(f"  Loading exclusion mask from {excluded_npy_path.name} (mode={exclude_mode})")
        excluded_pcd = np.load(excluded_npy_path)

        if len(excluded_pcd) != len(gt_pcd):
            print(f"  WARNING: excluded.npy ({len(excluded_pcd)}) != gt_pcd ({len(gt_pcd)}), skipping exclusion mask")
        elif exclude_mode == "remove":
            from ..core.mesh import filter_mesh_by_vertex_mask
            # Remove excluded faces from meshes
            nearest_gt = NearestPointIndex(gt_pcd)
            if gt_mesh is not None:
                indices = nearest_gt.query(gt_mesh.vertices)
                keep_mask = ~excluded_pcd[indices].astype(bool)
                n_before = len(gt_mesh.vertices)
                gt_mesh = filter_mesh_by_vertex_mask(gt_mesh, keep_mask)
                print(f"  Removed excluded from GT mesh: {n_before} -> {len(gt_mesh.vertices)} vertices")
            if method_mesh is not None:
                indices = nearest_gt.query(method_mesh.vertices)
                keep_mask = ~excluded_pcd[indices].astype(bool)
                n_before = len(method_mesh.vertices)
                method_mesh = filter_mesh_by_vertex_mask(method_mesh, keep_mask)
                print(f"  Removed excluded from method mesh: {n_before} -> {len(method_mesh.vertices)} vertices")
        else:
            # mode == "gray": pass masks to colormap functions
            nearest_gt = NearestPointIndex(gt_pcd)
            if gt_mesh is not None:
                indices = nearest_gt.query(gt_mesh.vertices)
                gt_excluded_mask = excluded_pcd[indices]
            if method_mesh is not None:
                indices = nearest_gt.query(method_mesh.vertices)
                method_excluded_mask = excluded_pcd[indices]

    print(f"Visualizing: {object_name}/{method_name} (exclude_mode={exclude_mode})")
    if method_mesh:
        print(f"  Method mesh: {len(method_mesh.vertices)} vertices")
    if gt_mesh:
        print(f"  GT mesh: {len(gt_mesh.vertices)} vertices")
    print(f"  Cameras: {len(cameras['K'])} views")
    print(f"  Metrics: {metrics}")

    results = {}

    # Compute uniform crop bboxes from GT mesh first (largest mesh = most inclusive bbox)
    crop_bboxes = None
    gt_uniform_rendered = False
    bbox_path = gt_vis_dir / "bbox.json"

    if crop and bbox_path.exists() and not force:
        with open(bbox_path) as f:
            raw = json.load(f)
        crop_bboxes = {int(k): tuple(v) for k, v in raw.items()}
        print(f"  Loaded crop bboxes from {bbox_path} ({len(crop_bboxes)} views)")

    if crop and crop_bboxes is None and gt_mesh is not None:
        gt_output = gt_vis_dir / "uniform"
        gt_output.mkdir(parents=True, exist_ok=True)

        print("  Computing uniform crop bboxes from GT mesh...")
        colored_gt = color_uniform(gt_mesh, excluded_mask=gt_excluded_mask)
        _, crop_bboxes = render_views(
            colored_gt, cameras, view_indices, gt_output,
            scale=scale, crop=True, crop_margin=crop_margin, crop_bboxes=None,
            lighting_mode="uniform", min_component_size=min_component_size
        )
        print(f"    Computed bboxes for {len(crop_bboxes)} views")

        # Atomic write: the viz jobs of one object run concurrently and may read bbox.json
        # while another job (re)writes it; a truncated read would lock the combo.
        tmp_bbox = bbox_path.with_name(f"{bbox_path.name}.tmp{os.getpid()}")
        tmp_bbox.write_text(json.dumps({str(k): [int(x) for x in v] for k, v in crop_bboxes.items()}, indent=2))
        os.replace(tmp_bbox, bbox_path)
        print(f"    Saved bboxes to {bbox_path}")

        results["uniform_gt"] = list(gt_output.glob("view_*.png"))
        gt_uniform_rendered = True

    for metric in metrics:
        # Determine which mesh to use and output directory
        target_excluded_mask = None
        if metric in GT_ONLY_METRICS:
            if gt_mesh is None:
                print(f"  Skipping {metric}: GT mesh not found")
                continue
            target_mesh = gt_mesh
            target_excluded_mask = gt_excluded_mask
            output_dir = gt_vis_dir / metric
        elif metric in METHOD_ONLY_METRICS:
            if method_mesh is None:
                print(f"  Skipping {metric}: method mesh not found")
                continue
            target_mesh = method_mesh
            target_excluded_mask = method_excluded_mask
            output_dir = vis_dir / metric
        elif metric in BOTH_METRICS:
            pass  # Handled specially below
        elif metric in NORMAL_METRICS:
            continue  # Handled once for both metrics after this loop
        else:
            print(f"  Unknown metric: {metric}")
            continue

        # Special handling for "uniform" - render both meshes
        if metric == "uniform":
            # Render method mesh
            if method_mesh is not None:
                method_output = vis_dir / "uniform"
                if method_output.exists() and not force:
                    existing = list(method_output.glob("view_*.png"))
                    if existing:
                        print("  uniform (method): already exists, skipping")
                        results["uniform_method"] = existing
                    else:
                        results["uniform_method"] = []
                else:
                    print("  Generating uniform (method) visualizations...")
                    colored = color_uniform(method_mesh, excluded_mask=method_excluded_mask)
                    paths, _ = render_views(colored, cameras, view_indices, method_output,
                                        scale=scale, crop=crop, crop_margin=crop_margin,
                                        crop_bboxes=crop_bboxes, lighting_mode="uniform",
                                        min_component_size=min_component_size)
                    results["uniform_method"] = paths
                    print(f"    Generated {len(paths)} images")

            # GT uniform - check if already rendered during bbox computation
            if gt_mesh is not None and not gt_uniform_rendered:
                gt_output = gt_vis_dir / "uniform"
                if gt_output.exists() and not force:
                    existing = list(gt_output.glob("view_*.png"))
                    if existing:
                        print("  uniform (GT): already exists, skipping")
                        results["uniform_gt"] = existing
                    else:
                        results["uniform_gt"] = []
                else:
                    print("  Generating uniform (GT) visualizations...")
                    colored = color_uniform(gt_mesh, excluded_mask=gt_excluded_mask)
                    paths, computed_bboxes = render_views(colored, cameras, view_indices, gt_output,
                                        scale=scale, crop=crop, crop_margin=crop_margin,
                                        crop_bboxes=crop_bboxes, lighting_mode="uniform",
                                        min_component_size=min_component_size)
                    results["uniform_gt"] = paths
                    if crop_bboxes is None and crop:
                        crop_bboxes = computed_bboxes
                    print(f"    Generated {len(paths)} images")
            elif gt_uniform_rendered:
                print("  uniform (GT): already rendered for bbox computation")
            continue

        # Check if already done
        if output_dir.exists() and not force:
            existing = list(output_dir.glob("view_*.png"))
            if existing:
                print(f"  {metric}: already exists, skipping")
                results[metric] = existing
                continue

        print(f"  Generating {metric} visualizations...")

        # Color the mesh based on metric
        colorbar_params = None

        if metric == "accuracy":
            dist_path = eval_dir / "distances" / "data2gt_dist.npy"
            points_path = eval_dir / "distances" / "data_points.npy"
            if not dist_path.exists():
                print(f"    Skipping {metric}: {dist_path} not found")
                continue
            dist = np.load(dist_path)
            # Map distances from sampled points to mesh vertices using KDTree
            if points_path.exists() and method_mesh is not None:
                points = np.load(points_path)
                if len(dist) == len(points):
                    print(f"    Mapping {len(points)} sampled points to {len(method_mesh.vertices)} mesh vertices")
                    tree = cKDTree(points)
                    _, indices = tree.query(method_mesh.vertices, k=1)
                    vertex_dist = dist[indices]
                    colored_mesh, colorbar_params = color_by_accuracy(
                        method_mesh, vertex_dist, max_dist=max_dist, 
                        cmap=colormaps.get("accuracy", "jet"),
                        excluded_mask=method_excluded_mask
                    )
                else:
                    print(f"    Skipping {metric}: points/dist size mismatch")
                    continue
            elif len(dist) == len(target_mesh.vertices):
                colored_mesh, colorbar_params = color_by_accuracy(
                    target_mesh, dist, max_dist=max_dist, 
                    cmap=colormaps.get("accuracy", "jet"),
                    excluded_mask=target_excluded_mask
                )
            else:
                print(f"    Skipping {metric}: distance array size mismatch ({len(dist)} vs {len(target_mesh.vertices)})")
                continue

        elif metric == "completeness":
            dist_path = eval_dir / "distances" / "gt2data_dist.npy"
            points_path = eval_dir / "distances" / "gt_points.npy"
            if not dist_path.exists():
                print(f"    Skipping {metric}: {dist_path} not found")
                continue
            dist = np.load(dist_path)
            # Map distances from sampled GT points to GT mesh vertices using KDTree
            if points_path.exists() and gt_mesh is not None:
                points = np.load(points_path)
                if len(dist) == len(points):
                    print(f"    Mapping {len(points)} sampled GT points to {len(gt_mesh.vertices)} GT mesh vertices")
                    tree = cKDTree(points)
                    _, indices = tree.query(gt_mesh.vertices, k=1)
                    vertex_dist = dist[indices]
                    colored_mesh, colorbar_params = color_by_completeness(
                        gt_mesh, vertex_dist, max_dist=max_dist, 
                        cmap=colormaps.get("completeness", "jet"),
                        excluded_mask=gt_excluded_mask
                    )
                else:
                    print(f"    Skipping {metric}: points/dist size mismatch")
                    continue
            elif gt_mesh is not None and len(dist) == len(gt_mesh.vertices):
                colored_mesh, colorbar_params = color_by_completeness(
                    gt_mesh, dist, max_dist=max_dist, 
                    cmap=colormaps.get("completeness", "jet"),
                    excluded_mask=gt_excluded_mask
                )
            else:
                if gt_mesh is None:
                    print(f"    Skipping {metric}: GT mesh not found")
                else:
                    print(f"    Skipping {metric}: distance array size mismatch ({len(dist)} vs {len(gt_mesh.vertices)})")
                continue
            output_dir = vis_dir / metric  # Still save under method's vis dir

        elif metric == "visibility":
            vis_path = gt_dir / "attributes" / "visibility_count.npy"
            if not vis_path.exists():
                print(f"    Skipping {metric}: {vis_path} not found")
                continue
            vis_count = np.load(vis_path)
            if len(vis_count) != len(target_mesh.vertices):
                # Find the right source points for KNN mapping
                source_pts = None
                if gt_pcd is not None and len(vis_count) == len(gt_pcd):
                    source_pts = gt_pcd
                elif gt_mesh_original_vertices is not None and len(vis_count) == len(gt_mesh_original_vertices):
                    source_pts = gt_mesh_original_vertices
                if source_pts is not None:
                    print(f"    Mapping {len(vis_count)} attribute values to {len(target_mesh.vertices)} mesh vertices via KNN")
                    tree = cKDTree(source_pts)
                    _, indices = tree.query(target_mesh.vertices, k=1)
                    vis_count = vis_count[indices]

            if len(vis_count) != len(target_mesh.vertices):
                print(f"    Skipping {metric}: visibility array size mismatch ({len(vis_count)} vs {len(target_mesh.vertices)})")
                continue
            colored_mesh, colorbar_params = color_by_visibility(
                target_mesh, vis_count, cmap=colormaps.get("visibility", "viridis"),
                excluded_mask=target_excluded_mask
            )

        elif metric == "curvature":
            curv_path = gt_dir / "attributes" / "curvature_values.npy"
            if not curv_path.exists():
                print(f"    Skipping {metric}: {curv_path} not found")
                continue
            curvature = np.load(curv_path)
            if len(curvature) != len(target_mesh.vertices):
                source_pts = None
                if gt_pcd is not None and len(curvature) == len(gt_pcd):
                    source_pts = gt_pcd
                elif gt_mesh_original_vertices is not None and len(curvature) == len(gt_mesh_original_vertices):
                    source_pts = gt_mesh_original_vertices
                if source_pts is not None:
                    print(f"    Mapping {len(curvature)} attribute values to {len(target_mesh.vertices)} mesh vertices via KNN")
                    tree = cKDTree(source_pts)
                    _, indices = tree.query(target_mesh.vertices, k=1)
                    curvature = curvature[indices]

            if len(curvature) != len(target_mesh.vertices):
                print(f"    Skipping {metric}: curvature array size mismatch ({len(curvature)} vs {len(target_mesh.vertices)})")
                continue
            colored_mesh, colorbar_params = color_by_curvature(
                target_mesh, curvature, cmap=colormaps.get("curvature", "coolwarm"),
                excluded_mask=target_excluded_mask
            )

        # Render views
        paths, _ = render_views(
            colored_mesh,
            cameras,
            view_indices,
            output_dir,
            scale=scale,
            crop=crop,
            crop_margin=crop_margin,
            crop_bboxes=crop_bboxes,
            min_component_size=min_component_size
        )

        # Generate and save colorbar if needed
        if colorbar_params is not None:
            if integrated_colorbar:
                # Composite the lateral colorbar directly onto each rendered view
                # (view_XXX_cb.png), reserving fixed label space so wide views don't clip.
                for view_path in paths:
                    view_img = np.array(Image.open(view_path).convert("RGB"))
                    composite = add_colorbar(
                        view_img,
                        colorbar_params["vmin"],
                        colorbar_params["vmax"],
                        colorbar_params["cmap"],
                        colorbar_params["label"],
                        position="right",
                        reserve_label_space=True,
                    )
                    cb_view_path = view_path.with_name(f"{view_path.stem}_cb.png")
                    Image.fromarray(composite).save(cb_view_path)
                print(f"    Saved {len(paths)} integrated-colorbar views (*_cb.png)")
            else:
                dummy = np.full((100, 100, 3), 255, dtype=np.uint8)
                with_cb = add_colorbar(
                    dummy,
                    colorbar_params["vmin"],
                    colorbar_params["vmax"],
                    colorbar_params["cmap"],
                    colorbar_params["label"],
                )
                colorbar_img = with_cb[:, 100:]
                cb_path = output_dir / "colorbar.png"
                Image.fromarray(colorbar_img).save(cb_path)
                print(f"    Saved colorbar to {cb_path}")

        results[metric] = paths
        print(f"    Generated {len(paths)} images")

    if set(metrics) & NORMAL_METRICS:
        results.update(visualize_normals(
            gt_mesh, method_mesh, cameras, view_indices,
            metrics=[m for m in metrics if m in NORMAL_METRICS],
            gt_out=gt_vis_dir / "normals",
            method_out=vis_dir / "normals",
            error_out=vis_dir / "angular_error",
            scale=scale,
            crop_bboxes=crop_bboxes if crop else None,
            gt_excluded=None if gt_excluded_mask is None else np.asarray(gt_excluded_mask, dtype=bool),
            method_excluded=None if method_excluded_mask is None else np.asarray(method_excluded_mask, dtype=bool),
            cmap=colormaps.get("angular_error", "viridis"),
            integrated_colorbar=integrated_colorbar,
            force=force,
            info={"object": object_name, "method": method_name, "exclude_mode": exclude_mode},
        ))

    return results


def _fit(image: np.ndarray, max_resolution: int | None) -> np.ndarray:
    """Downsize like render_views (longest side <= max_resolution, LANCZOS)."""
    h, w = image.shape[:2]
    if max_resolution is None or max(h, w) <= max_resolution:
        return image
    ratio = max_resolution / max(h, w)
    return np.array(Image.fromarray(image).resize((int(w * ratio), int(h * ratio)), Image.LANCZOS))


def _save_png(image: np.ndarray, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)
    return path


def visualize_normals(
    gt_mesh: trimesh.Trimesh | None,
    method_mesh: trimesh.Trimesh | None,
    cameras: dict,
    view_indices: list[int],
    metrics: list[str],
    gt_out: Path,
    method_out: Path,
    error_out: Path,
    scale: float = 1.0,
    crop_bboxes: dict[int, tuple[int, int, int, int]] | None = None,
    gt_excluded: np.ndarray | None = None,
    method_excluded: np.ndarray | None = None,
    cmap: str = "viridis",
    integrated_colorbar: bool = False,
    force: bool = False,
    max_resolution: int | None = 1000,
    info: dict | None = None,
) -> dict:
    """GT / method normal maps (``normals``), angular-error maps and mae.json (``angular_error``).

    Both meshes are ray cast through the same pixel centres (full image at ``scale``, restricted to
    the crop bbox of the view when given), so the maps overlap pixel for pixel and with the other
    renders cropped by the same bbox. The MAE is computed at that resolution, before downsizing.
    With ``*_excluded`` (per-vertex booleans of the meshes given, exclude_mode 'gray'), pixels whose
    GT hit (or, with no GT hit, method hit) lands on an excluded vertex are gray in all three maps
    and out of the MAE. For 'remove', pass the meshes with the excluded faces already removed and
    no flags. Pixels seen by one mesh only are out of the MAE (light gray in the error map).
    """
    import time

    from ..core.camera import PIXEL_CENTRE_TRUNCATE
    from ..core.normals import (
        ANGULAR_ERROR_MAX_DEG,
        NormalRenderer,
        angular_error_deg,
        color_angular_error,
        encode_normals_rgb,
        normal_error_stats,
        pixel_categories,
        scale_intrinsics,
    )
    from ..core.rendering import _BG

    if gt_mesh is None or method_mesh is None:
        print("  Skipping normals/angular_error: " + ("GT" if gt_mesh is None else "method") + " mesh not found")
        return {}
    want_normals = "normals" in metrics
    want_error = "angular_error" in metrics
    mae_path = error_out / "mae.json"
    names = [f"view_{v:03d}.png" for v in view_indices]
    done = (not want_error or mae_path.exists()) and (
        not want_normals or all((d / n).exists() for d in (gt_out, method_out) for n in names))
    if done and not force:
        print("  normals/angular_error: already exist, skipping")
        out = {}
        if want_normals:
            out["normals_gt"] = [gt_out / n for n in names]
            out["normals_method"] = [method_out / n for n in names]
        if want_error:
            out["angular_error"] = [error_out / n for n in names]
        return out

    bg = tuple(int(round(c * 255)) for c in _BG[:3])
    pixel_centre = cameras.get("pixel_centre", PIXEL_CENTRE_TRUNCATE)
    H = int(cameras["image_height"] * scale)
    W = int(cameras["image_width"] * scale)
    print(f"  Ray casting normals at {W}x{H} (scale={scale}, pixel centre {pixel_centre})")
    t0 = time.perf_counter()
    gt_r, rec_r = NormalRenderer(gt_mesh), NormalRenderer(method_mesh)
    t_build = time.perf_counter() - t0

    out = {"normals_gt": [], "normals_method": [], "angular_error": []}
    per_view = {}
    colorbar_params = None
    for idx, name in zip(view_indices, names):
        t0 = time.perf_counter()
        K = scale_intrinsics(cameras["K"][idx], scale, pixel_centre)
        bbox = crop_bboxes.get(idx) if crop_bboxes else None
        g = gt_r.render(K, cameras["Rt"][idx], (H, W), pixel_centre=pixel_centre, bbox=bbox)
        r = rec_r.render(K, cameras["Rt"][idx], (H, W), pixel_centre=pixel_centre, bbox=bbox)

        excluded = np.zeros(g.mask.shape, dtype=bool)
        if gt_excluded is not None:
            excluded[g.mask] = gt_excluded[g.nearest_vertex[g.mask]]
        if method_excluded is not None:
            only_rec = r.mask & ~g.mask
            excluded[only_rec] = method_excluded[r.nearest_vertex[only_rec]]

        err = angular_error_deg(g.normals, r.normals)
        cat = pixel_categories(g.mask, r.mask, excluded)
        stats = normal_error_stats(err, cat, g.mask, excluded)
        valid = cat["valid"]
        stats["gt_back_facing"] = float((g.facing[valid] < 0).mean()) if valid.any() else float("nan")
        stats["method_back_facing"] = float((r.facing[valid] < 0).mean()) if valid.any() else float("nan")
        stats["window"] = [int(x) for x in (bbox if bbox is not None else (0, H - 1, 0, W - 1))]
        stats["time_s"] = round(time.perf_counter() - t0, 2)
        per_view[str(idx)] = stats

        if want_normals:
            out["normals_gt"].append(_save_png(
                _fit(encode_normals_rgb(g.normals, g.mask, bg, excluded), max_resolution), gt_out / name))
            out["normals_method"].append(_save_png(
                _fit(encode_normals_rgb(r.normals, r.mask, bg, excluded), max_resolution), method_out / name))
        if want_error:
            rgb, colorbar_params = color_angular_error(err, cat, cmap=cmap, background=bg)
            path = _save_png(_fit(rgb, max_resolution), error_out / name)
            out["angular_error"].append(path)
            if integrated_colorbar:
                composite = add_colorbar(
                    np.array(Image.open(path).convert("RGB")), colorbar_params["vmin"], colorbar_params["vmax"],
                    colorbar_params["cmap"], colorbar_params["label"], position="right", reserve_label_space=True)
                _save_png(composite, path.with_name(f"{path.stem}_cb.png"))
        print(f"    view {idx}: MAE {stats['mae_deg']:.2f} deg over {stats['n_valid']} px "
              f"(GT coverage {stats['gt_coverage']:.3f}), {stats['time_s']} s")

    if want_error:
        if colorbar_params is not None and not integrated_colorbar:
            with_cb = add_colorbar(np.full((100, 100, 3), 255, dtype=np.uint8), colorbar_params["vmin"],
                                   colorbar_params["vmax"], colorbar_params["cmap"], colorbar_params["label"])
            _save_png(with_cb[:, 100:], error_out / "colorbar.png")
        n = np.array([s["n_valid"] for s in per_view.values()], dtype=float)
        maes = np.array([s["mae_deg"] for s in per_view.values()], dtype=float)
        ok = n > 0
        summary = {
            **(info or {}),
            "definition": "per-pixel angle between the ray-cast GT and method normal maps (interpolated vertex "
                          "normals), over the pixels seen by both meshes and not excluded",
            "vmax_deg": ANGULAR_ERROR_MAX_DEG,
            "scale": scale,
            "pixel_centre": pixel_centre,
            "n_faces": {"gt": int(len(gt_mesh.faces)), "method": int(len(method_mesh.faces))},
            "build_s": round(t_build, 2),
            "views": per_view,
            "cell": {
                "mae_deg": float((maes[ok] * n[ok]).sum() / n[ok].sum()) if ok.any() else float("nan"),
                "mean_view_mae_deg": float(maes[ok].mean()) if ok.any() else float("nan"),
                "n_valid": int(n.sum()),
                "n_views": int(ok.sum()),
            },
        }
        error_out.mkdir(parents=True, exist_ok=True)
        tmp = mae_path.with_name(f"{mae_path.name}.tmp{os.getpid()}")
        tmp.write_text(json.dumps(summary, indent=2))
        os.replace(tmp, mae_path)
        print(f"    Saved {mae_path} (cell MAE {summary['cell']['mae_deg']:.2f} deg)")
    return {k: v for k, v in out.items() if v}
