"""Zone scores and zone renders: exclusion mask, P/R/F of a zone, pooling, visible share of a zone in a render."""

import json

import numpy as np
import pytest

from eval3d.figures.supplementary import pick_view, zones

from ..helpers import make_cell, make_config, make_gt, make_spec


def test_override_grays_everything_outside_the_zone_and_the_excluded_points():
    zone = np.array([True, True, False, False])
    excluded = np.array([False, True, False, True])
    assert zones.override(zone, excluded).tolist() == [False, True, True, True]
    assert zones.override(zone, None).tolist() == [False, False, True, True]


def test_prf_at_counts_strictly_below_the_threshold():
    p, r, f = zones.prf_at(np.array([0.1, 0.5, 0.9, 2.0]), np.array([0.2, 0.4]), 0.5)
    assert (p, r) == (25.0, 100.0) and abs(f - 40.0) < 1e-9


def test_prf_at_of_a_failed_object_is_zero():
    assert zones.prf_at(np.empty(0), np.full(3, np.inf), 0.5) == (0.0, 0.0, 0.0)


def test_pool_concatenates_points_so_objects_weigh_by_their_size():
    pairs = [(np.array([0.1]), np.array([0.1, 0.1, 0.1])), (np.empty(0), np.full(1, np.inf))]
    d, g = zones.pool(pairs)
    assert len(d) == 1 and len(g) == 4
    assert zones.prf_at(d, g, 0.5)[:2] == (100.0, 75.0)


def test_colored_fraction_ignores_grey_and_white_pixels():
    img = np.full((2, 2, 3), 255, np.uint8)
    img[0, 0] = (128, 128, 128)
    img[0, 1] = (84, 1, 68)
    frac, n = pick_view.colored_fraction(img)
    assert n == 1 and frac == 0.5


RADAR = {"methods": ["alpha"], "tier": "d4", "threshold": 0.5, "properties": ["lambertian", "specular"],
         "attributes": [{"key": "low_vis", "attribute": "visibility_count", "ge": 1, "le": 3}], "output": "radar.pdf"}
PRF = {"methods": ["alpha"], "tier": "d4", "max_threshold": 1.0, "n_thresholds": 3, "output": "prf.pdf"}


def test_scores_follow_the_radar_rules_and_feed_the_pooled_curves(tmp_path):
    """Same synthetic objects as the radar test: zone selection, excluded points, a failed object as missed GT."""
    config = make_config(tmp_path)
    make_gt(config, "obj1", 4, excluded=[0, 1, 0, 0], challenges={"lambertian": [1, 1, 1, 0]},
            attributes={"visibility_count": [0, 2, 3, 5]})
    make_cell(config, "obj1", "alpha_d4", d2g=[0.1, 0.9, 0.2, 0.1], g2d=[0.1, 0.1, 0.9, 0.1], idx=[0, 1, 2, 3])
    make_gt(config, "obj2", 2, challenges={"lambertian": [1, 1]}, attributes={"visibility_count": [0, 9]})
    spec = make_spec(radar=RADAR, prf=PRF)
    out = tmp_path / "zone_scores.json"
    zones.scores(config, spec, out)
    res = json.loads(out.read_text())
    assert list(res["axes"]) == ["lambertian", "specular", "low_vis"] and res["thresholds"] == [0.0, 0.5, 1.0]
    lam = res["axes"]["lambertian"]
    assert lam["ids"] == ["obj1", "obj2"] and lam["points"] == {"obj1": 2, "obj2": 2}
    assert lam["object"]["obj1"]["alpha"] == pytest.approx([100.0, 50.0, 100 * 2 / 3])
    assert lam["object"]["obj2"]["alpha"] is None                       # failed: no score, its GT points are missed
    assert lam["pooled"]["alpha"] == pytest.approx([100.0, 25.0, 40.0])
    assert lam["curves"]["alpha"]["recall"] == pytest.approx([0.0, 25.0, 50.0])
    assert res["axes"]["specular"]["ids"] == [] and res["axes"]["low_vis"]["pooled"]["alpha"][1] == 0.0


def test_plot_zones_writes_one_pooled_figure_per_axis(tmp_path):
    from eval3d.figures.supplementary import plot_zones
    spec = make_spec(prf=PRF)
    (tmp_path / "spec.yaml").write_text(json.dumps(spec.model_dump(mode="json")))
    curve = {k: [0.0, 50.0, 100.0] for k in ("precision", "recall", "fscore")}
    data = {"thresholds": [0.0, 0.5, 1.0], "axes": {"specular": {"curves": {"alpha": curve}}}}
    (tmp_path / "scores.json").write_text(json.dumps(data))
    plot_zones.main([str(tmp_path / "spec.yaml"), str(tmp_path / "scores.json"), str(tmp_path / "out")])
    assert (tmp_path / "out" / "prf_specular.pdf").stat().st_size > 0
