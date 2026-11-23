"""Distances stage: GT point cloud + cleaned mesh -> bidirectional nearest-neighbour distances.

First stage of the evaluation of a cell (object, method), and the expensive one: the mesh is sampled, then two
KD-tree queries run over the two clouds. Its output, `distances/*.npy`, is stored at full length, before any mask,
so that every later stage (`metrics`, `curves`, `apply-masks`, `visualize`) reads it instead of searching again.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import Config
from ..core.mesh import load_mesh
from ..core.metrics import compute_distances
from ..core.sampling import downsample_pcd, remove_nan_points, upsample_mesh


@dataclass
class CellDistances:
    """Distances of a cell, at full length (no mask applied).

    For an empty reconstruction (`empty`), the arrays are the degenerate ones `compute_metrics` turns into NaN: no
    data point, every GT point at an infinite distance, no index.
    """

    gt_points: np.ndarray
    data_points: np.ndarray
    gt2data_dist: np.ndarray
    gt2data_idx: np.ndarray | None
    data2gt_dist: np.ndarray
    data2gt_idx: np.ndarray | None

    @property
    def empty(self) -> bool:
        return self.data2gt_idx is None


def compute_cell_distances(config: Config, object_name: str, method_name: str, seed: int = 42) -> CellDistances:
    """Sample the cleaned mesh and measure the distances to and from the GT point cloud. Writes nothing.

    Args:
        config: Pipeline configuration.
        object_name: Object name.
        method_name: Method name.
        seed: Random seed of the reconstruction-side sampling.

    Raises:
        FileNotFoundError: no GT point cloud or no cleaned mesh.
    """
    gt_pcd_path = config.get_gt_artifacts_dir(object_name) / "gt_pcd.npy"
    cleaned_mesh_path = config.get_cleaned_mesh_path(object_name, method_name)

    if not gt_pcd_path.exists():
        raise FileNotFoundError(f"GT point cloud not found: {gt_pcd_path}")
    if not cleaned_mesh_path.exists():
        raise FileNotFoundError(f"Cleaned mesh not found: {cleaned_mesh_path}")

    gt_pcd = np.load(gt_pcd_path)
    print(f"  Loaded GT point cloud: {len(gt_pcd)} points")

    data_mesh = load_mesh(cleaned_mesh_path)
    print(f"  Loaded cleaned mesh: {len(data_mesh.vertices)} vertices")

    if len(data_mesh.vertices) == 0 or len(data_mesh.faces) == 0:
        # Empty reconstruction (failed method). NearestNeighbors cannot fit 0 points: build the degenerate
        # distances directly.
        return CellDistances(
            gt_points=gt_pcd,
            data_points=np.empty((0, 3), dtype=gt_pcd.dtype),
            gt2data_dist=np.full(len(gt_pcd), np.inf, dtype=np.float32),
            gt2data_idx=None,
            data2gt_dist=np.empty(0, dtype=np.float32),
            data2gt_idx=None,
        )

    density = config.evaluation.downsample_density

    print(f"  Upsampling data mesh (density={density})")
    data_pcd = upsample_mesh(data_mesh.vertices, data_mesh.faces, density)
    print(f"  Upsampled to {len(data_pcd)} points")

    # Numerical hygiene only: drop NaN points. No relative height cut is
    # applied any more, so reconstructed points below the GT floor (bases,
    # tables, ...) are kept and count in the distance/F-score metrics.
    data_pcd = remove_nan_points(data_pcd)
    print(f"  After NaN removal: {len(data_pcd)} points")

    print("  Downsampling data point cloud")
    data_down = downsample_pcd(data_pcd, density, shuffle=True, seed=seed)
    print(f"  Downsampled to {len(data_down)} points")

    print("  Computing distances")
    dist_data2gt, idx_data2gt = compute_distances(data_down, gt_pcd)
    dist_gt2data, idx_gt2data = compute_distances(gt_pcd, data_down)

    return CellDistances(
        gt_points=gt_pcd,
        data_points=data_down,
        gt2data_dist=dist_gt2data,
        gt2data_idx=idx_gt2data,
        data2gt_dist=dist_data2gt,
        data2gt_idx=idx_data2gt,
    )


def save_cell_distances(eval_dir: Path, cell: CellDistances) -> None:
    """Write `distances/*.npy`. An empty reconstruction has no distances to store: nothing is written."""
    if cell.empty:
        return
    distances_dir = eval_dir / "distances"
    distances_dir.mkdir(parents=True, exist_ok=True)
    np.save(distances_dir / "gt2data_dist.npy", cell.gt2data_dist)
    np.save(distances_dir / "gt2data_idx.npy", cell.gt2data_idx)
    np.save(distances_dir / "data2gt_dist.npy", cell.data2gt_dist)
    np.save(distances_dir / "data2gt_idx.npy", cell.data2gt_idx)
    np.save(distances_dir / "gt_points.npy", cell.gt_points)
    np.save(distances_dir / "data_points.npy", cell.data_points)


def distances_stage(
    config: Config, object_name: str, method_name: str, force: bool = False, seed: int = 42
) -> bool:
    """Run the distances stage alone: compute and write `distances/*.npy` of a cell.

    The metrics on disk are left as they are: run the metrics stage next (`evaluate` does both).

    Returns:
        True if distances were written; False if they exist (and not `force`) or the reconstruction is empty.
    """
    eval_dir = config.get_eval_dir(object_name, method_name)
    if (eval_dir / "distances" / "gt2data_dist.npy").exists() and not force:
        print(f"  Distances exist: {eval_dir / 'distances'}")
        return False

    print(f"Distances: {object_name}/{method_name}")
    cell = compute_cell_distances(config, object_name, method_name, seed)
    if cell.empty:
        print("  Empty cleaned mesh -> no distances (evaluate records the failure)")
        return False
    save_cell_distances(eval_dir, cell)
    print(f"  Saved distances to {eval_dir / 'distances'}")
    return True
