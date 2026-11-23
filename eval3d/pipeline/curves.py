"""Compute dense precision/recall/F-score curves from saved distances."""


import numpy as np

from ..config import Config
from ..core.metrics import compute_fscore_curve
from .metrics import has_distances, mask_distances, read_distances


def compute_curves(
    config: Config,
    object_name: str,
    method_name: str,
    n_thresholds: int = 100,
    max_threshold: float = 1.5,
    force: bool = False,
    extra_exclude: list[str] | None = None,
) -> dict | None:
    """Compute dense P/R/F curves from pre-computed distances.

    Requires the distances stage to have run first (distances/*.npy must exist).

    Args:
        config: Pipeline configuration.
        object_name: Object name.
        method_name: Method name.
        n_thresholds: Number of thresholds to sample.
        max_threshold: Maximum threshold value (mm).
        force: Overwrite existing curves.
        extra_exclude: Additional GT exclusion masks (see `core.masking`).

    Returns:
        Dict with thresholds/precision/recall/fscore arrays, or None if skipped.
    """
    eval_dir = config.get_eval_dir(object_name, method_name)
    cleaned_mesh = config.get_cleaned_mesh_path(object_name, method_name)

    if not cleaned_mesh.exists():
        return None

    if not has_distances(eval_dir):
        return None

    out_dir = eval_dir / "curves"
    marker = out_dir / "thresholds.npy"
    if marker.exists() and not force:
        existing_t = np.load(marker)
        if len(existing_t) >= n_thresholds:
            return None

    print(f"  Computing dense curves: {object_name}/{method_name}")

    # Same GT exclusion masks as the metrics stage, or the curves would contradict the metrics.json next to them.
    masked = mask_distances(*read_distances(eval_dir), eval_dir, config.get_gt_artifacts_dir(object_name), extra_exclude)
    if masked is None:
        return None
    dist_data2gt, dist_gt2data, _applied = masked

    thresholds = np.linspace(0, max_threshold, n_thresholds)
    result = compute_fscore_curve(dist_data2gt, dist_gt2data, thresholds)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "thresholds.npy", result["thresholds"])
    np.save(out_dir / "precision.npy", result["precision"])
    np.save(out_dir / "recall.npy", result["recall"])
    np.save(out_dir / "fscore.npy", result["fscore"])

    print(f"    {n_thresholds} thresholds [0, {max_threshold}] -> {out_dir}")
    return result
