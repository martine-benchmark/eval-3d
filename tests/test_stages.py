"""The two stages of the evaluation of a cell: `distances`, then `metrics`; `evaluate` chains them.

What is checked: the stages chained by hand write the files of `evaluate` byte for byte, and the metrics stage run
alone (from saved distances) differs from a fresh evaluation only by what it keeps of the cell: the keys other
stages added to `metrics.json` and a dense threshold grid written by `curves`.
"""
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import trimesh

from eval3d.config import Config
from eval3d.pipeline.curves import compute_curves
from eval3d.pipeline.distances import distances_stage
from eval3d.pipeline.evaluate import evaluate
from eval3d.pipeline.metrics import compute_cell_metrics

OBJ, METHOD = "obj1", "m"
N_GT = 3000
DISTANCE_FILES = {f"distances/{n}.npy" for n in (
    "gt2data_dist", "gt2data_idx", "data2gt_dist", "data2gt_idx", "gt_points", "data_points")}
CURVE_FILES = {f"curves/{n}.npy" for n in ("thresholds", "precision", "recall", "fscore")}
METRICS_KEYS_TAIL = ["seed", "n_gt_points", "n_data_points", "exclude_masks", "max_dist"]
EMPTY_PLY = "ply\nformat ascii 1.0\nelement vertex 0\nproperty float x\nproperty float y\nproperty float z\n" \
            "element face 0\nproperty list uchar int vertex_indices\nend_header\n"


def _config(root: Path, **evaluation) -> Config:
    ev = {"max_dist": 0.2, "fscore_thresholds": [0.02, 0.05, 0.1], "downsample_density": 0.05, **evaluation}
    return Config(
        dataset={"name": "t", "objects": [OBJ], "methods": [METHOD]},
        paths={"data_root": str(root / "data/{object}"), "eval_root": str(root / "eval/{object}"),
               "gt_artifacts_root": str(root / "gt/{object}")},
        evaluation=ev,
    )


def _build(root: Path, excluded: bool = True, empty_mesh: bool = False) -> Config:
    """A cell ready for `evaluate`: sampled GT (a sphere), optional `excluded.npy`, cleaned mesh (a bumped sphere)."""
    config = _config(root)
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(N_GT, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    gt_dir = config.get_gt_artifacts_dir(OBJ)
    (gt_dir / "challenges").mkdir(parents=True)
    np.save(gt_dir / "gt_pcd.npy", pts)
    if excluded:
        np.save(gt_dir / "challenges" / "excluded.npy", pts[:, 2] > 0.6)
    np.save(gt_dir / "challenges" / "holes.npy", pts[:, 0] < -0.7)

    mesh_path = config.get_cleaned_mesh_path(OBJ, METHOD)
    mesh_path.parent.mkdir(parents=True)
    if empty_mesh:
        mesh_path.write_text(EMPTY_PLY)
    else:
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
        v = mesh.vertices.copy()
        v *= 1.0 + 0.08 * np.sin(5 * v[:, [0]])  # distances spread over the thresholds
        trimesh.Trimesh(v, mesh.faces, process=False).export(mesh_path)
    return config


@pytest.fixture
def cell(tmp_path) -> Config:
    """An evaluated cell, with an exclusion mask."""
    config = _build(tmp_path / "cell")
    evaluate(config, OBJ, METHOD)
    return config


def _eval_dir(config: Config) -> Path:
    return config.get_eval_dir(OBJ, METHOD)


def _digests(config: Config) -> dict[str, str]:
    eval_dir = _eval_dir(config)
    return {str(p.relative_to(eval_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(eval_dir.rglob("*")) if p.is_file()}


def _metrics(config: Config) -> dict:
    return json.loads((_eval_dir(config) / "metrics.json").read_text())


def _changed(config: Config, **evaluation) -> Config:
    """The same cell under another configuration: the reason to run the metrics stage alone."""
    return _config(Path(config.paths.eval_root).parents[1], **evaluation)


def _add_keys(config: Config, **keys) -> None:
    path = _eval_dir(config) / "metrics.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), **keys}))


# --- evaluate = distances + metrics ---------------------------------------------------------------------------------

def test_evaluate_writes_distances_metrics_and_curves(cell):
    assert set(_digests(cell)) == DISTANCE_FILES | CURVE_FILES | {"metrics.json"}
    m = _metrics(cell)
    assert list(m)[-len(METRICS_KEYS_TAIL):] == METRICS_KEYS_TAIL
    assert m["exclude_masks"] == ["excluded.npy"] and m["n_gt_points"] == N_GT and m["seed"] == 42


@pytest.mark.parametrize("excluded", [True, False])
def test_distances_then_metrics_equals_evaluate(tmp_path, excluded):
    """The two stages run one by one write the files of `evaluate`, byte for byte."""
    whole, staged = _build(tmp_path / "whole", excluded=excluded), _build(tmp_path / "staged", excluded=excluded)
    evaluate(whole, OBJ, METHOD)

    assert distances_stage(staged, OBJ, METHOD) is True
    assert set(_digests(staged)) == DISTANCE_FILES
    assert compute_cell_metrics(staged, OBJ, METHOD) is not None
    assert _digests(staged) == _digests(whole)

    assert distances_stage(staged, OBJ, METHOD) is False  # distances exist
    assert distances_stage(staged, OBJ, METHOD, force=True) is True
    assert _digests(staged) == _digests(whole)


def test_metrics_alone_gives_back_the_files_of_evaluate(cell):
    before = _digests(cell)
    compute_cell_metrics(cell, OBJ, METHOD)
    assert _digests(cell) == before


def test_evaluate_without_force_keeps_the_cell(cell):
    before = _digests(cell)
    assert evaluate(_changed(cell, max_dist=0.5), OBJ, METHOD)["max_dist"] == 0.2
    assert _digests(cell) == before


def test_evaluate_missing_inputs(tmp_path):
    config = _build(tmp_path / "cell")
    config.get_cleaned_mesh_path(OBJ, METHOD).unlink()
    for stage in (evaluate, distances_stage):
        with pytest.raises(FileNotFoundError, match="Cleaned mesh not found"):
            stage(config, OBJ, METHOD)
    assert not _eval_dir(config).exists()


def test_empty_mesh_is_a_failure_without_distances(tmp_path):
    config = _build(tmp_path / "cell", empty_mesh=True)
    assert distances_stage(config, OBJ, METHOD) is False
    assert not _eval_dir(config).exists()

    m = evaluate(config, OBJ, METHOD)
    assert set(_digests(config)) == CURVE_FILES | {"metrics.json"}
    assert m["empty_reconstruction"] is True and "empty_reason" not in m and m["n_data_points"] == 0
    assert np.isnan(m["chamfer"])
    # the metrics stage alone has no distances to read: the failure record stays as evaluate wrote it
    before = _digests(config)
    assert compute_cell_metrics(config, OBJ, METHOD) is None
    assert _digests(config) == before


# --- fresh evaluation vs metrics stage alone ------------------------------------------------------------------------

def _dense_cell_with_added_keys(cell: Config) -> None:
    assert compute_curves(cell, OBJ, METHOD, n_thresholds=40, max_threshold=0.15) is not None
    _add_keys(cell, timing_s=3, empty_reconstruction=True, empty_reason="no_data_point_outside_excluded")


def test_metrics_alone_keeps_the_added_keys_and_the_dense_grid(cell):
    _dense_cell_with_added_keys(cell)
    compute_cell_metrics(_changed(cell, max_dist=0.05), OBJ, METHOD)
    m = _metrics(cell)
    assert m["max_dist"] == 0.05 and m["timing_s"] == 3
    assert "empty_reconstruction" not in m and "empty_reason" not in m  # a stale failure flag is not kept
    assert len(m["thresholds"]) == 3
    assert len(np.load(_eval_dir(cell) / "curves" / "thresholds.npy")) == 40


def test_fresh_evaluation_keeps_nothing_of_the_cell(cell):
    reference = _digests(cell)
    _dense_cell_with_added_keys(cell)
    evaluate(cell, OBJ, METHOD, force=True)
    assert _digests(cell) == reference


def test_metrics_alone_leaves_dense_curves_as_curves_wrote_them(cell):
    compute_curves(cell, OBJ, METHOD, n_thresholds=40, max_threshold=0.15)
    before = _digests(cell)
    compute_cell_metrics(cell, OBJ, METHOD)
    assert _digests(cell) == before


def test_metrics_with_extra_exclude(cell):
    reference = _metrics(cell)
    compute_cell_metrics(cell, OBJ, METHOD, extra_exclude=["holes"])
    m = _metrics(cell)
    assert m["exclude_masks"] == ["excluded.npy", "holes.npy"]
    assert m["n_kept_a2b"] < reference["n_kept_a2b"] and m["n_gt_points"] == N_GT
    # same result as a full evaluation under that mask
    evaluate(cell, OBJ, METHOD, force=True, extra_exclude=["holes"])
    assert _metrics(cell) == m


def test_metrics_all_data_points_excluded_is_a_failure(cell):
    np.save(cell.get_gt_artifacts_dir(OBJ) / "challenges" / "excluded.npy", np.ones(N_GT, bool))
    compute_cell_metrics(cell, OBJ, METHOD)
    m = _metrics(cell)
    assert m["empty_reconstruction"] is True and m["empty_reason"] == "no_data_point_outside_excluded"


def test_metrics_without_the_saved_point_clouds(cell):
    (_eval_dir(cell) / "distances" / "gt_points.npy").unlink()
    (_eval_dir(cell) / "distances" / "data_points.npy").unlink()
    compute_cell_metrics(cell, OBJ, METHOD)
    m = _metrics(cell)
    assert m["n_gt_points"] == N_GT  # read from the current GT
    assert m["n_data_points"] > 0  # kept from the previous metrics.json


def test_metrics_over_an_unreadable_metrics_json(cell):
    reference = _metrics(cell)
    (_eval_dir(cell) / "metrics.json").write_text("{not json")
    compute_cell_metrics(cell, OBJ, METHOD)
    assert _metrics(cell) == reference


def _stale_gt(config: Config) -> None:
    np.save(config.get_gt_artifacts_dir(OBJ) / "gt_pcd.npy", np.zeros((N_GT + 5, 3)))


def _no_index(config: Config) -> None:
    (_eval_dir(config) / "distances" / "data2gt_idx.npy").unlink()


def _no_distances(config: Config) -> None:
    shutil.rmtree(_eval_dir(config) / "distances")


@pytest.mark.parametrize("prepare", [_stale_gt, _no_index, _no_distances])
def test_metrics_skips_a_cell_it_cannot_score(cell, prepare):
    prepare(cell)
    before = _digests(cell)
    assert compute_cell_metrics(_changed(cell, max_dist=0.05), OBJ, METHOD) is None
    assert _digests(cell) == before


def test_metrics_needs_no_index_without_a_mask(tmp_path):
    config = _build(tmp_path / "cell", excluded=False)
    evaluate(config, OBJ, METHOD)
    _no_index(config)
    assert compute_cell_metrics(_changed(config, max_dist=0.05), OBJ, METHOD)["max_dist"] == 0.05


# --- dense curves ---------------------------------------------------------------------------------------------------

def test_curves_follow_the_masks_of_the_metrics(cell):
    """At the report thresholds, the dense curve equals the F-scores of metrics.json (same masks)."""
    m = _metrics(cell)
    out = compute_curves(cell, OBJ, METHOD, n_thresholds=40, max_threshold=0.15)
    assert len(out["thresholds"]) == 40
    unmasked = _build(Path(cell.paths.eval_root).parents[1].parent / "unmasked", excluded=False)
    evaluate(unmasked, OBJ, METHOD)
    assert _metrics(unmasked)["fscore"] != m["fscore"]  # the mask matters on this cell

    assert compute_curves(cell, OBJ, METHOD, n_thresholds=30) is None  # already dense enough
    forced = compute_curves(cell, OBJ, METHOD, n_thresholds=3, max_threshold=0.1, force=True)
    # linspace(0, 0.1, 3) = [0, 0.05, 0.1]: the last two are report thresholds
    assert forced["fscore"][1:].tolist() == pytest.approx(m["fscore"][1:], abs=1e-6)


def test_curves_skips_a_masked_cell_without_index(cell):
    _no_index(cell)
    before = _digests(cell)
    assert compute_curves(cell, OBJ, METHOD, n_thresholds=40) is None
    assert _digests(cell) == before
