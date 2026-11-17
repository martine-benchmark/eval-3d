"""`visualize` with the normal metrics: GT / method normal maps, angular-error map and mae.json.

Runs on the lego fixture without pyrender: the crop bboxes are read from a pre-written bbox.json
(the normal maps are ray cast, only the bbox computation would need the OpenGL renderer).
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import trimesh
import yaml
from PIL import Image

import eval3d.pipeline.visualize as viz
from eval3d.config import load_config
from eval3d.core.camera import load_cameras_auto
from eval3d.core.normals import EXCLUDED_RGB

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests" / "data" / "lego"
VIEWS = [0, 5]


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO)
    monkeypatch.setattr(viz, "check_pyrender", lambda: None)
    cfg = yaml.safe_load((REPO / "config" / "example.yaml").read_text())
    cfg["paths"]["eval_root"] = str(tmp_path / "runs" / "{object}")
    cfg_path = tmp_path / "example.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    config = load_config(cfg_path)

    mesh = trimesh.load(FIXTURE / "pgsr_d4.ply", process=False)
    gt_dir = config.get_gt_dir("lego")
    gt_dir.mkdir(parents=True)
    mesh.export(gt_dir / "gt_cleaned.ply")
    cleaned = config.get_cleaned_mesh_path("lego", "m")
    cleaned.parent.mkdir(parents=True)
    shutil.copy(gt_dir / "gt_cleaned.ply", cleaned)

    # Crop bboxes as the GT uniform render would have saved them
    cams = load_cameras_auto(config.get_cameras_path("lego"))
    H, W = cams["image_height"], cams["image_width"]
    bboxes = {str(v): [100, H - 101, 150, W - 151] for v in VIEWS}
    (gt_dir / "visualizations").mkdir()
    (gt_dir / "visualizations" / "bbox.json").write_text(json.dumps(bboxes))
    return config, tmp_path, (H - 200, W - 300)


def _run(config, **kw):
    kw.setdefault("exclude_mode", "none")
    return viz.visualize_method(
        config, "lego", "m", metrics=["normals", "angular_error"], view_indices=VIEWS,
        crop=True, integrated_colorbar=False, **kw,
    )


def test_identical_meshes_give_zero_error_and_aligned_maps(workspace):
    config, _, (h, w) = workspace
    res = _run(config)
    gt_vis = config.get_gt_dir("lego") / "visualizations"
    m_vis = config.get_method_dir("lego", "m") / "visualizations"
    for v in VIEWS:
        gt_png = np.array(Image.open(gt_vis / "normals" / f"view_{v:03d}.png"))
        rec_png = np.array(Image.open(m_vis / "normals" / f"view_{v:03d}.png"))
        err_png = np.array(Image.open(m_vis / "angular_error" / f"view_{v:03d}.png"))
        assert gt_png.shape == rec_png.shape == err_png.shape == (h, w, 3)
        np.testing.assert_array_equal(gt_png, rec_png)
    assert (m_vis / "angular_error" / "colorbar.png").is_file()
    mae = json.loads((m_vis / "angular_error" / "mae.json").read_text())
    for v in VIEWS:
        s = mae["views"][str(v)]
        assert s["n_valid"] > 1000
        assert s["mae_deg"] < 1e-3
        assert s["gt_coverage"] == pytest.approx(1.0)
        assert s["n_single"] == 0
    assert mae["cell"]["mae_deg"] < 1e-3
    assert mae["vmax_deg"] == 45.0
    assert set(res) >= {"normals_gt", "normals_method", "angular_error"}


def test_excluded_region_is_gray_and_out_of_the_mae(workspace):
    config, tmp_path, _ = workspace
    gt_pcd = np.load(FIXTURE / "gt" / "gt_pcd.npy")
    excluded = gt_pcd[:, 0] > np.median(gt_pcd[:, 0])
    np.save(tmp_path / "excluded.npy", excluded)
    _run(config, exclude_mode="gray", excluded_override_path=tmp_path / "excluded.npy")
    m_vis = config.get_method_dir("lego", "m") / "visualizations"
    mae = json.loads((m_vis / "angular_error" / "mae.json").read_text())
    s = mae["views"][str(VIEWS[0])]
    assert s["n_excluded"] > 1000 and s["n_valid"] > 1000
    err_png = np.array(Image.open(m_vis / "angular_error" / f"view_{VIEWS[0]:03d}.png"))
    assert (err_png == EXCLUDED_RGB).all(axis=-1).sum() == s["n_excluded"]


def test_perturbed_reconstruction_has_positive_mae(workspace):
    config, _, _ = workspace
    cleaned = config.get_cleaned_mesh_path("lego", "m")
    mesh = trimesh.load(cleaned, process=False)
    rng = np.random.default_rng(0)
    scale = mesh.scale * 2e-3
    mesh.vertices = mesh.vertices + rng.normal(scale=scale, size=mesh.vertices.shape)
    mesh.export(cleaned)
    _run(config)
    mae = json.loads((config.get_method_dir("lego", "m") / "visualizations" / "angular_error" / "mae.json").read_text())
    assert mae["cell"]["mae_deg"] > 1.0
