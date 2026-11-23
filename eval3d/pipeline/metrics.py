"""Metrics stage: distances of a cell -> `metrics.json` and `curves/`.

Second stage of the evaluation of a cell (object, method), after the distances stage (`pipeline/distances.py`):

    distances (full length, before any mask) -> GT exclusion masks -> Chamfer, F-score -> metrics.json, curves/

`evaluate` chains the two stages. This stage also runs alone (`eval-pipeline metrics`) from the saved
`distances/*.npy`: after a change of `max_dist`, of the thresholds or of a mask, the metrics are rebuilt in seconds,
without the nearest-neighbour search, and equal those of a full evaluation.

The two entry points share `build_metrics` and `write_metrics`. They differ by `fresh` only:

- `fresh=True` (the distances were just computed): whatever the cell held is out of date, `metrics.json` and the
  curves are written anew.
- `fresh=False` (stage run alone): the keys other stages added to `metrics.json` are kept, and so is a dense
  threshold grid written by `curves`, recomputed under the current masks.
"""

import json
from pathlib import Path

import numpy as np

from ..config import Config
from ..core.masking import apply_exclude, load_gt_exclude
from ..core.metrics import FAILURE_KEYS, compute_fscore_curve, compute_metrics, mark_failure_if_no_data_kept

# Seed of the reconstruction-side sampling, recorded in metrics.json. The distances stage has always run with it.
DEFAULT_SEED = 42


def has_distances(eval_dir: Path) -> bool:
    """Whether the distances stage has written the two distance arrays of a cell."""
    distances_dir = eval_dir / "distances"
    return (distances_dir / "data2gt_dist.npy").exists() and (distances_dir / "gt2data_dist.npy").exists()


def read_distances(eval_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Saved (data2gt, gt2data) distances of a cell, at full length; None if the distances stage has not run."""
    if not has_distances(eval_dir):
        return None
    return np.load(eval_dir / "distances" / "data2gt_dist.npy"), np.load(eval_dir / "distances" / "gt2data_dist.npy")


def mask_distances(
    dist_data2gt: np.ndarray,
    dist_gt2data: np.ndarray,
    eval_dir: Path,
    gt_dir: Path,
    extra_exclude: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
    """Apply the GT exclusion masks (`core.masking`) to saved distances.

    Returns:
        (data2gt, gt2data, applied mask paths), or None when a mask applies but `data2gt_idx.npy`, which carries
        the exclusion over to the reconstruction side, is missing.
    """
    exclude, applied = load_gt_exclude(gt_dir, n_gt=len(dist_gt2data), extra_exclude=extra_exclude)
    if exclude is None:
        return dist_data2gt, dist_gt2data, applied
    idx_path = eval_dir / "distances" / "data2gt_idx.npy"
    if not idx_path.exists():
        print("    Cannot apply exclusion mask: data2gt_idx.npy missing — skipping cell")
        return None
    dist_data2gt, dist_gt2data = apply_exclude(dist_data2gt, dist_gt2data, np.load(idx_path), exclude)
    return dist_data2gt, dist_gt2data, applied


def build_metrics(
    config: Config,
    dist_data2gt: np.ndarray,
    dist_gt2data: np.ndarray,
    *,
    applied_masks: list[str],
    n_gt_points: int | None,
    n_data_points: int | None,
    seed: int = DEFAULT_SEED,
    empty_mesh: bool = False,
) -> dict:
    """Content of `metrics.json` from masked distances. The single place that lays its keys out.

    Args:
        config: Pipeline configuration (`evaluation.fscore_thresholds`, `evaluation.max_dist`).
        dist_data2gt: Data-to-GT distances, after exclusion.
        dist_gt2data: GT-to-data distances, after exclusion.
        applied_masks: Paths of the exclusion masks applied, recorded by file name.
        n_gt_points: Size of the GT cloud before exclusion; None leaves the key out.
        n_data_points: Size of the reconstruction cloud before exclusion; None leaves the key out.
        seed: Seed of the reconstruction-side sampling.
        empty_mesh: The reconstruction has no face: the distances are degenerate and the cell is a failure.
    """
    metrics = compute_metrics(
        dist_data2gt, dist_gt2data, config.evaluation.fscore_thresholds, config.evaluation.max_dist
    )
    if not empty_mesh:
        mark_failure_if_no_data_kept(metrics, len(dist_data2gt))
    metrics["seed"] = seed
    if n_gt_points is not None:
        metrics["n_gt_points"] = n_gt_points
    if n_data_points is not None:
        metrics["n_data_points"] = n_data_points
    metrics["exclude_masks"] = [Path(p).name for p in applied_masks]
    metrics["max_dist"] = config.evaluation.max_dist
    if empty_mesh:
        metrics["empty_reconstruction"] = True
    return metrics


def write_metrics(
    eval_dir: Path, metrics: dict, dist_data2gt: np.ndarray, dist_gt2data: np.ndarray, fresh: bool
) -> dict:
    """Write `metrics.json` and `curves/*.npy` of a cell; see the module docstring for `fresh`.

    Args:
        eval_dir: `eval_results` directory of the cell.
        metrics: Output of `build_metrics`; completed in place with the kept keys when not `fresh`.
        dist_data2gt: Masked data-to-GT distances the metrics come from (for a dense curve).
        dist_gt2data: Masked GT-to-data distances the metrics come from.
        fresh: The distances were just computed.

    Returns:
        The metrics as written.
    """
    eval_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = eval_dir / "metrics.json"
    curves_dir = eval_dir / "curves"
    curves_dir.mkdir(exist_ok=True)

    # Keys written by other stages are not recomputable from the distances: keep them. The failure flags are
    # recomputable, so a stale one is dropped.
    if not fresh and metrics_path.exists():
        try:
            with open(metrics_path) as f:
                previous = json.load(f)
            for key, value in previous.items():
                if key not in metrics and key not in FAILURE_KEYS:
                    metrics[key] = value
        except (json.JSONDecodeError, OSError) as e:
            print(f"    Could not merge previous metrics.json ({e}), writing fresh")

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # The curves are rewritten with the metrics, or the arrays would contradict the metrics.json next to them.
    # `curves` writes a dense grid (100 thresholds) where the metrics hold the report grid (a few): when the cell
    # has a denser grid, recompute that one under the same masks rather than downgrade the cell.
    curve = {key: np.array(metrics[key]) for key in ("thresholds", "precision", "recall", "fscore")}
    thresholds_path = curves_dir / "thresholds.npy"
    if not fresh and thresholds_path.exists():
        existing = np.load(thresholds_path)
        if len(existing) > len(curve["thresholds"]):
            dense = compute_fscore_curve(dist_data2gt, dist_gt2data, existing)
            curve = {key: np.asarray(value) for key, value in dense.items()}
            print(f"    preserved dense curve resolution: {len(existing)} thresholds")
    for name, arr in curve.items():
        np.save(curves_dir / f"{name}.npy", arr)

    return metrics


def compute_cell_metrics(
    config: Config,
    object_name: str,
    method_name: str,
    extra_exclude: list[str] | None = None,
    strict: bool = True,
) -> dict | None:
    """Run the metrics stage alone: `metrics.json` and curves of a cell from its saved distances.

    Args:
        config: Pipeline configuration.
        object_name: Object name.
        method_name: Method name.
        extra_exclude: Additional GT exclusion masks (see `core.masking`).
        strict: Skip a cell whose saved distances no longer match the current GT point count: they were measured
            against a GT that changed, the distances stage has to run again.

    Returns:
        The metrics, or None for a skipped cell (no saved distances, stale distances, missing index).
    """
    eval_dir = config.get_eval_dir(object_name, method_name)
    gt_dir = config.get_gt_artifacts_dir(object_name)
    distances_dir = eval_dir / "distances"

    saved = read_distances(eval_dir)
    if saved is None:
        return None
    dist_data2gt, dist_gt2data = saved

    print(f"  Metrics: {object_name}/{method_name}")

    # The saved gt2data array has one entry per GT point: a different length means the GT changed since the
    # distances were computed. More reliable than mtimes on NFS.
    gt_pcd_path = gt_dir / "gt_pcd.npy"
    n_gt_current = None
    if gt_pcd_path.exists():
        n_gt_current = int(np.load(gt_pcd_path, mmap_mode="r").shape[0])
        if n_gt_current != len(dist_gt2data):
            print(
                f"    STALE DISTANCES: gt2data_dist has {len(dist_gt2data):,} entries but "
                f"current GT has {n_gt_current:,} points — run the distances stage again"
            )
            if strict:
                return None

    masked = mask_distances(dist_data2gt, dist_gt2data, eval_dir, gt_dir, extra_exclude)
    if masked is None:
        return None
    dist_data2gt, dist_gt2data, applied_masks = masked

    gt_points_path = distances_dir / "gt_points.npy"
    data_points_path = distances_dir / "data_points.npy"
    n_gt_points = n_gt_current
    if gt_points_path.exists():
        n_gt_points = int(np.load(gt_points_path, mmap_mode="r").shape[0])
    n_data_points = None
    if data_points_path.exists():
        n_data_points = int(np.load(data_points_path, mmap_mode="r").shape[0])

    metrics = build_metrics(
        config, dist_data2gt, dist_gt2data,
        applied_masks=applied_masks, n_gt_points=n_gt_points, n_data_points=n_data_points,
    )
    write_metrics(eval_dir, metrics, dist_data2gt, dist_gt2data, fresh=False)

    print(
        f"    chamfer={metrics['chamfer']:.4f} coverage={metrics['coverage']:.4f} "
        f"(max_dist={config.evaluation.max_dist})"
    )
    return metrics
