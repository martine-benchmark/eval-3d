"""A cell with no reconstructed point left once the `excluded` GT zones are removed is a failure.

Rule (Baptiste, 2026-11-23, grid/colmap_d4_v10 on the v3 kit): every data point's nearest GT point lies in
`excluded.npy`, so no data point is scored and the metrics are NaN. Like an empty mesh, the cell is recorded as
`empty_reconstruction` (`empty_reason` tells the two apart), which the figures count as a failure of the method.
"""
import json
from pathlib import Path

import numpy as np

from eval3d.config import Config
from eval3d.core.metrics import mark_failure_if_no_data_kept
from eval3d.pipeline.metrics import compute_cell_metrics as recompute_metrics

N_GT, N_DATA = 40, 10


def _setup(tmp_path: Path, excluded: np.ndarray, previous: dict | None = None):
    config = Config(
        dataset={"name": "t", "objects": ["obj1"], "methods": ["m"]},
        paths={"data_root": str(tmp_path / "data/{object}"), "eval_root": str(tmp_path / "eval/{object}"),
               "gt_artifacts_root": str(tmp_path / "gt/{object}")},
        evaluation={"max_dist": 4.0, "fscore_thresholds": [0.5, 1.0]},
    )
    ch = config.get_gt_artifacts_dir("obj1") / "challenges"
    ch.mkdir(parents=True)
    np.save(ch / "excluded.npy", excluded)
    eval_dir = config.get_eval_dir("obj1", "m")
    dist = eval_dir / "distances"
    dist.mkdir(parents=True)
    np.save(dist / "gt2data_dist.npy", np.full(N_GT, 0.2, np.float32))
    np.save(dist / "data2gt_dist.npy", np.full(N_DATA, 0.2, np.float32))
    np.save(dist / "data2gt_idx.npy", np.arange(N_DATA))  # data point j -> GT point j
    (eval_dir / "metrics.json").write_text(json.dumps(previous or {}))
    return config, eval_dir


def test_all_data_points_excluded_is_a_failure(tmp_path):
    excluded = np.zeros(N_GT, bool)
    excluded[:N_DATA] = True  # the GT points owning every data point
    config, eval_dir = _setup(tmp_path, excluded)
    recompute_metrics(config, "obj1", "m")
    m = json.loads((eval_dir / "metrics.json").read_text())
    assert m["empty_reconstruction"] is True
    assert m["empty_reason"] == "no_data_point_outside_excluded"
    assert m["n_kept_a2b"] == 0


def test_some_data_points_kept_is_not_a_failure(tmp_path):
    excluded = np.zeros(N_GT, bool)
    excluded[: N_DATA - 1] = True  # one data point survives
    config, eval_dir = _setup(tmp_path, excluded)
    recompute_metrics(config, "obj1", "m")
    m = json.loads((eval_dir / "metrics.json").read_text())
    assert not m.get("empty_reconstruction", False) and "empty_reason" not in m


def test_stale_failure_flag_is_not_carried_over(tmp_path):
    # recompute keeps the additive keys of the previous metrics.json; a failure flag set by an older kit is not one
    config, eval_dir = _setup(tmp_path, np.zeros(N_GT, bool),
                              previous={"empty_reconstruction": True, "empty_reason": "no_data_point_outside_excluded",
                                        "timing_s": 3})
    recompute_metrics(config, "obj1", "m")
    m = json.loads((eval_dir / "metrics.json").read_text())
    assert not m.get("empty_reconstruction", False) and "empty_reason" not in m
    assert m["timing_s"] == 3  # other additive keys still preserved


def test_mark_failure_leaves_an_empty_mesh_reason_alone():
    m = {"empty_reconstruction": True, "empty_reason": "empty_mesh"}
    assert mark_failure_if_no_data_kept(m, 0) == {"empty_reconstruction": True, "empty_reason": "empty_mesh"}
    assert mark_failure_if_no_data_kept({}, 3) == {}
