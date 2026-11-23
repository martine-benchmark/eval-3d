"""Evaluation of a cell: the distances stage, then the metrics stage.

    cleaned mesh + GT cloud --[pipeline/distances.py]--> distances/*.npy
                            --[pipeline/metrics.py]----> metrics.json, curves/

Each stage also runs alone (`eval-pipeline distances`, `eval-pipeline metrics`).
"""

import json

from ..config import Config
from ..core.masking import apply_exclude, load_gt_exclude
from .distances import compute_cell_distances, save_cell_distances
from .metrics import build_metrics, write_metrics


def evaluate(
    config: Config,
    object_name: str,
    method_name: str,
    force: bool = False,
    seed: int = 42,
    extra_exclude: list[str] | None = None,
) -> dict:
    """Evaluate a reconstructed mesh against ground truth.

    Computes bidirectional distances and stores:
        - distances/gt2data_dist.npy, gt2data_idx.npy
        - distances/data2gt_dist.npy, data2gt_idx.npy
        - distances/gt_points.npy, data_points.npy
        - curves/thresholds.npy, precision.npy, recall.npy, fscore.npy
        - metrics.json

    An empty reconstruction is recorded as a failure: metrics.json (NaN, `empty_reconstruction`) and curves, no
    distances.

    Args:
        config: Pipeline configuration.
        object_name: Object name.
        method_name: Method name.
        force: Overwrite existing output.
        seed: Random seed for reproducibility.
        extra_exclude: Additional GT exclusion masks (see `core.masking`).

    Returns:
        Dictionary with computed metrics.
    """
    gt_dir = config.get_gt_artifacts_dir(object_name)
    eval_dir = config.get_eval_dir(object_name, method_name)
    metrics_path = eval_dir / "metrics.json"

    if metrics_path.exists() and not force:
        print(f"  Metrics exist, loading: {metrics_path}")
        with open(metrics_path) as f:
            return json.load(f)

    print(f"Evaluating: {object_name}/{method_name}")

    # Stage 1: distances, at full length.
    cell = compute_cell_distances(config, object_name, method_name, seed)

    if cell.empty:
        # Failure cell: NaN metrics (the empty-set contract of compute_metrics), kept out of means and rankings.
        print("  Empty cleaned mesh -> recording failure metrics (NaN)")
        metrics = build_metrics(
            config, cell.data2gt_dist, cell.gt2data_dist,
            applied_masks=[], n_gt_points=len(cell.gt_points), n_data_points=0, seed=seed, empty_mesh=True,
        )
        write_metrics(eval_dir, metrics, cell.data2gt_dist, cell.gt2data_dist, fresh=True)
        print("  Chamfer: nan (empty reconstruction, scored as failure)")
        return metrics

    # Stage 2: metrics, under the GT exclusion masks. Nothing is written before the metrics are computed, so a
    # mask that cannot be applied leaves the cell untouched.
    exclude, applied_masks = load_gt_exclude(gt_dir, n_gt=len(cell.gt_points), extra_exclude=extra_exclude)
    dist_data2gt, dist_gt2data = apply_exclude(cell.data2gt_dist, cell.gt2data_dist, cell.data2gt_idx, exclude)

    print("  Computing metrics")
    metrics = build_metrics(
        config, dist_data2gt, dist_gt2data,
        applied_masks=applied_masks, n_gt_points=len(cell.gt_points), n_data_points=len(cell.data_points), seed=seed,
    )

    save_cell_distances(eval_dir, cell)
    write_metrics(eval_dir, metrics, dist_data2gt, dist_gt2data, fresh=True)

    print(f"  Chamfer: {metrics['chamfer']:.4f}")
    # Find closest threshold to 1.0 for display
    thresholds = metrics["thresholds"]
    if 1.0 in thresholds:
        fscore_1 = metrics["fscore"][thresholds.index(1.0)]
    else:
        closest_idx = min(range(len(thresholds)), key=lambda i: abs(thresholds[i] - 1.0))
        fscore_1 = metrics["fscore"][closest_idx]
    print(f"  F-score@1.0: {fscore_1:.4f}")
    print(f"  Saved results to {eval_dir}")

    return metrics
