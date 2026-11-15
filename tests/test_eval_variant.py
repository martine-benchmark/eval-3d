"""Evaluation variant: an overriding evaluation camera set for the two visibility cleanings.

`--eval-cameras <custom.sfm> --variant <tag>` (or `paths.eval_cameras_override` + `variant` in the
config) evaluates a reconstruction only on the surface seen by the custom views:

- the reconstruction cleanup (`cleanup`) casts rays and reads masks from the custom set
  (masks: `<dir of custom sfm>/masks/`, watertight masks: `<dir>/masks_watertight/` if present);
- the GT kit is restricted to the published points seen by the custom views (`preprocess-gt` in
  variant mode): `Groundtruth@<tag>/` holds every per-point array indexed by the same keep mask;
- outputs go to `results_cleaned@<tag>/`, `eval_results@<tag>/`, `Groundtruth@<tag>/`; nothing of
  the default tree is written.

Synthetic scene: two square plates (z = +1 facing +z, z = -1 facing -z) and two cameras, A above
(world z = +10, looking down) and B below (z = -10, looking up). Each plate occludes the other, so
camera A sees only the top plate and camera B only the bottom one.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import trimesh
import yaml
from click.testing import CliRunner
from PIL import Image

from eval3d.cli import main
from eval3d.config import Config, load_config
from eval3d.core.camera import load_cameras_auto
from eval3d.core.visibility import compute_auto_dilation, compute_visibility_count

W, H = 64, 48
OBJ, METHOD, TAG = "obj", "m", "front"
N_GRID = 21  # points per side and per plate


# --------------------------------------------------------------------------- scene builders


def _plate(z, n, up):
    xs = np.linspace(-1.0, 1.0, n)
    xx, yy = np.meshgrid(xs, xs, indexing="ij")
    verts = np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, z)])
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b, c, d = i * n + j, (i + 1) * n + j, (i + 1) * n + j + 1, i * n + j + 1
            faces += [[a, b, c], [a, c, d]] if up else [[a, c, b], [a, d, c]]
    return trimesh.Trimesh(verts, np.array(faces), process=False)


def _plates(n=N_GRID, dz=0.0):
    """Top plate (normal +z) and bottom plate (normal -z)."""
    return trimesh.util.concatenate([_plate(1.0 + dz, n, up=True), _plate(-1.0 - dz, n, up=False)])


# AliceVision applies diag(1, -1, -1) to the raw pose: a raw identity rotation looks down world -z.
_CAMS = {
    "A": {"rotation": ["1", "0", "0", "0", "1", "0", "0", "0", "1"], "center": ["0", "0", "-10"]},
    "B": {"rotation": ["1", "0", "0", "0", "-1", "0", "0", "0", "-1"], "center": ["0", "0", "10"]},
}


def _write_sfm(path, views, w=W, h=H):
    """views: list of (stem, camera key). Focal 100 mm on a 36 mm sensor: the plates fill the centre."""
    sfm = {
        "views": [
            {"viewId": str(i + 1), "poseId": str(i + 1), "intrinsicId": "1", "path": f"masks_eval/{stem}.png"}
            for i, (stem, _) in enumerate(views)
        ],
        "intrinsics": [{
            "intrinsicId": "1", "width": str(w), "height": str(h),
            "focalLength": "100", "sensorWidth": "36", "principalPoint": ["0", "0"],
        }],
        "poses": [{"poseId": str(i + 1), "pose": {"transform": _CAMS[cam]}} for i, (_, cam) in enumerate(views)],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sfm))
    return path


def _write_masks(masks_dir, stems, w=W, h=H, value=255):
    masks_dir.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        Image.fromarray(np.full((h, w), value, dtype=np.uint8)).save(masks_dir / f"{stem}.png")


def _write_override(root, views, name=TAG, w=W, h=H, masks=True):
    sfm = _write_sfm(root / "overrides" / name / "cameras.sfm", views, w, h)
    if masks:
        _write_masks(sfm.parent / "masks", [s for s, _ in views], w, h)
    return sfm


def _scene(tmp_path, dilation="auto"):
    """Base (default) set: two views, 000 = A and 001 = B. Returns the config path."""
    data = tmp_path / "data" / OBJ
    _write_sfm(data / "cameras.sfm", [("000", "A"), ("001", "B")])
    _write_masks(data / "masks", ["000", "001"])

    gt_src = tmp_path / "gt" / OBJ
    gt_src.mkdir(parents=True)
    _plates().export(gt_src / "gt.ply")

    kit = tmp_path / "kit" / OBJ
    pcd = _plates(n=41).vertices.astype(np.float64)
    n = len(pcd)
    (kit / "attributes").mkdir(parents=True)
    (kit / "challenges").mkdir()
    np.save(kit / "gt_pcd.npy", pcd)
    np.save(kit / "attributes" / "visibility_count.npy", np.where(pcd[:, 2] > 0, 2, 1).astype(np.int32))
    np.save(kit / "attributes" / "curvature_values.npy", np.arange(n, dtype=np.float32))
    np.savez(kit / "attributes" / "proxy_curvature_r1_R0.5_e0.15.npz", points=np.zeros((3, 3)), curvature=np.zeros(3))
    np.save(kit / "challenges" / "lambertian.npy", np.ones(n, dtype=bool))
    excluded = np.zeros(n, dtype=bool)
    excluded[::7] = True
    np.save(kit / "challenges" / "excluded.npy", excluded)
    (kit / "gt_pcd_provenance.json").write_text(json.dumps({"n_points": n}))

    raw = tmp_path / "eval" / OBJ / METHOD / "results_raw"
    raw.mkdir(parents=True)
    _plates(dz=0.01).export(raw / "mesh.ply")

    cfg = {
        "dataset": {"name": "t", "objects": [OBJ], "methods": [METHOD]},
        "paths": {
            "data_root": str(tmp_path / "data" / "{object}"),
            "gt_root": str(tmp_path / "gt" / "{object}"),
            "gt_artifacts_root": str(tmp_path / "kit" / "{object}"),
            "eval_root": str(tmp_path / "eval" / "{object}"),
        },
        "cleanup": {"dilation_radius": dilation, "dilation_ref_px": 4, "dilation_ref_n_pixels": W * H},
        "evaluation": {"fscore_thresholds": [0.5, 1.0]},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def _cli(config_path, *args, variant=None):
    pre = ["-c", str(config_path)]
    if variant is not None:
        sfm, tag = variant
        pre += ["--eval-cameras", str(sfm), "--variant", tag]
    result = CliRunner().invoke(main, pre + list(args), catch_exceptions=False)
    return result


def _tree_hashes(root):
    return {
        str(p.relative_to(root)): hashlib.md5(p.read_bytes()).hexdigest()
        for p in sorted(Path(root).rglob("*")) if p.is_file()
    }


def _variant_config(config_path, sfm, tag=TAG):
    cfg = load_config(config_path)
    cfg.apply_variant(tag, str(sfm))
    return cfg


# --------------------------------------------------------------------------- config


def test_default_accessors_are_unchanged(tmp_path):
    cfg = load_config(_scene(tmp_path))
    assert cfg.variant is None and cfg.paths.eval_cameras_override is None
    assert cfg.get_cleaning_cameras_path(OBJ, METHOD) == cfg.get_cameras_path(OBJ, METHOD)
    assert cfg.get_cleaning_masks_dir(OBJ, METHOD) == cfg.get_masks_dir(OBJ, METHOD)
    assert cfg.get_cleaning_watertight_dir(OBJ) == cfg.get_watertight_masks_dir(OBJ)
    method_dir = cfg.get_method_dir(OBJ, METHOD)
    assert cfg.get_cleaned_mesh_path(OBJ, METHOD) == method_dir / "results_cleaned" / "mesh.ply"
    assert cfg.get_eval_dir(OBJ, METHOD) == method_dir / "eval_results"
    assert cfg.get_gt_artifacts_dir(OBJ) == cfg.get_base_gt_artifacts_dir(OBJ) == tmp_path / "kit" / OBJ
    assert cfg.tagged("visualizations") == "visualizations"


def test_variant_accessors(tmp_path):
    config_path = _scene(tmp_path)
    sfm = _write_override(tmp_path, [("000", "A")])
    cfg = _variant_config(config_path, sfm)
    base = load_config(config_path)
    # visibility_count / challenges / visualize keep the default (84-view) set
    assert cfg.get_cameras_path(OBJ, METHOD) == base.get_cameras_path(OBJ, METHOD)
    assert cfg.get_masks_dir(OBJ, METHOD) == base.get_masks_dir(OBJ, METHOD)
    # the two cleanings follow the override
    assert cfg.get_cleaning_cameras_path(OBJ, METHOD) == sfm.resolve()
    assert cfg.get_cleaning_masks_dir(OBJ, METHOD) == sfm.parent.resolve() / "masks"
    assert cfg.get_cleaning_watertight_dir(OBJ) == sfm.parent.resolve() / "masks_watertight"
    method_dir = cfg.get_method_dir(OBJ, METHOD)
    assert cfg.get_raw_mesh_path(OBJ, METHOD) == base.get_raw_mesh_path(OBJ, METHOD)
    assert cfg.get_cleaned_mesh_path(OBJ, METHOD) == method_dir / f"results_cleaned@{TAG}" / "mesh.ply"
    assert cfg.get_eval_dir(OBJ, METHOD) == method_dir / f"eval_results@{TAG}"
    assert cfg.get_base_gt_artifacts_dir(OBJ) == tmp_path / "kit" / OBJ
    assert cfg.get_gt_artifacts_dir(OBJ) == cfg.get_eval_root(OBJ) / f"Groundtruth@{TAG}"
    assert cfg.get_lock_path(OBJ, METHOD, "cleanup") == method_dir / f".cleanup_failed@{TAG}"
    assert base.get_lock_path(OBJ, METHOD, "cleanup") == method_dir / ".cleanup_failed"


def test_variant_from_config_file(tmp_path):
    config_path = _scene(tmp_path)
    sfm = _write_override(tmp_path, [("000", "A")])
    raw = yaml.safe_load(config_path.read_text())
    raw["variant"] = TAG
    raw["paths"]["eval_cameras_override"] = str(sfm)
    config_path.write_text(yaml.safe_dump(raw))
    cfg = load_config(config_path)
    assert cfg.variant == TAG
    assert cfg.get_cleaning_cameras_path(OBJ) == sfm.resolve()


@pytest.mark.parametrize("missing", ["variant", "eval_cameras_override"])
def test_config_options_come_together(tmp_path, missing):
    config_path = _scene(tmp_path)
    raw = yaml.safe_load(config_path.read_text())
    raw["variant"] = TAG
    raw["paths"]["eval_cameras_override"] = str(tmp_path / "x.sfm")
    if missing == "variant":
        del raw["variant"]
    else:
        del raw["paths"]["eval_cameras_override"]
    with pytest.raises(ValueError, match="together"):
        Config(**raw)


@pytest.mark.parametrize("args", [["--variant", TAG], ["--eval-cameras", "{sfm}"]])
def test_cli_options_come_together(tmp_path, args):
    config_path = _scene(tmp_path)
    sfm = _write_override(tmp_path, [("000", "A")])
    args = [a.format(sfm=sfm) for a in args]
    result = CliRunner().invoke(main, ["-c", str(config_path)] + args + ["cleanup"])
    assert result.exit_code != 0
    assert "together" in result.output


@pytest.mark.parametrize("tag", ["", "a/b", "a@b", "x y"])
def test_variant_tag_is_a_plain_name(tmp_path, tag):
    cfg = load_config(_scene(tmp_path))
    with pytest.raises(ValueError):
        cfg.apply_variant(tag, str(tmp_path / "x.sfm"))


# --------------------------------------------------------------------------- visibility count, auto dilation


def test_compute_visibility_count_auto_dilation_matches_explicit_radius(tmp_path):
    sfm = _write_sfm(tmp_path / "c.sfm", [("000", "A")])
    cams = load_cameras_auto(sfm)
    masks = tmp_path / "masks"
    masks.mkdir()
    m = np.zeros((H, W), dtype=np.uint8)
    m[H // 2 - 2:H // 2 + 2, W // 2 - 2:W // 2 + 2] = 255   # a small square: dilation matters
    Image.fromarray(m).save(masks / "000.png")
    pts = _plates(n=41).vertices
    mesh = _plates()
    radius = compute_auto_dilation(W * H, 4, W * H)
    assert radius == 4
    explicit = compute_visibility_count(pts, mesh, cams, masks, dilation_radius=radius, show_progress=False)
    auto = compute_visibility_count(pts, mesh, cams, masks, dilation_radius="auto", dilation_ref_px=4,
                                    dilation_ref_n_pixels=W * H, show_progress=False)
    undilated = compute_visibility_count(pts, mesh, cams, masks, show_progress=False)
    np.testing.assert_array_equal(auto, explicit)
    assert explicit.sum() > undilated.sum()


# --------------------------------------------------------------------------- GT restriction (derived kit)


def test_derive_keeps_points_seen_by_the_override(tmp_path):
    from eval3d.pipeline.variant_gt import derive_variant_gt

    config_path = _scene(tmp_path)
    sfm = _write_override(tmp_path, [("000", "A")])
    cfg = _variant_config(config_path, sfm)
    out = derive_variant_gt(cfg, OBJ)

    kit = tmp_path / "kit" / OBJ
    base_pcd = np.load(kit / "gt_pcd.npy")
    expected = base_pcd[:, 2] > 0   # top plate only: the bottom one is occluded from A
    assert out == cfg.get_eval_root(OBJ) / f"Groundtruth@{TAG}"
    keep = np.load(out / "keep.npy")
    np.testing.assert_array_equal(keep, np.flatnonzero(expected))
    np.testing.assert_array_equal(np.load(out / "gt_pcd.npy"), base_pcd[expected])
    for rel in ["attributes/visibility_count.npy", "attributes/curvature_values.npy",
                "challenges/lambertian.npy", "challenges/excluded.npy"]:
        np.testing.assert_array_equal(np.load(out / rel), np.load(kit / rel)[expected], err_msg=rel)
    # non-per-point files are copied as they are
    for rel in ["attributes/proxy_curvature_r1_R0.5_e0.15.npz", "gt_pcd_provenance.json"]:
        assert (out / rel).read_bytes() == (kit / rel).read_bytes(), rel

    prov = json.loads((out / "variant_provenance.json").read_text())
    assert prov["variant"] == TAG
    assert prov["eval_cameras"] == str(sfm.resolve())
    assert prov["eval_cameras_md5"] == hashlib.md5(sfm.read_bytes()).hexdigest()
    assert prov["views"] == ["000"]
    assert prov["n_base"] == len(base_pcd) and prov["n_kept"] == int(expected.sum())
    assert prov["kept_fraction"] == pytest.approx(expected.mean())
    assert prov["dilation_radius"] == "auto" and prov["use_masks"] is True
    assert prov["base_gt_dir"] == str(kit)


def test_derive_with_both_cameras_keeps_everything(tmp_path):
    from eval3d.pipeline.variant_gt import derive_variant_gt

    config_path = _scene(tmp_path)
    sfm = _write_override(tmp_path, [("000", "A"), ("001", "B")], name="both")
    out = derive_variant_gt(_variant_config(config_path, sfm, "both"), OBJ)
    assert len(np.load(out / "keep.npy")) == len(np.load(tmp_path / "kit" / OBJ / "gt_pcd.npy"))


def test_derive_resolves_masks_by_name_for_a_subset(tmp_path):
    """A subset holding only view 001 (camera B): its mask is 001.png, not the first mask on disk."""
    from eval3d.pipeline.variant_gt import derive_variant_gt

    config_path = _scene(tmp_path)
    sfm = _write_override(tmp_path, [("001", "B")], name="below")
    # a decoy 000.png that would hide everything if masks were looked up by index
    _write_masks(sfm.parent / "masks", ["000"], value=0)
    out = derive_variant_gt(_variant_config(config_path, sfm, "below"), OBJ)
    pcd = np.load(tmp_path / "kit" / OBJ / "gt_pcd.npy")
    np.testing.assert_array_equal(np.load(out / "keep.npy"), np.flatnonzero(pcd[:, 2] < 0))


def test_derive_fails_on_a_missing_mask(tmp_path):
    from eval3d.pipeline.variant_gt import derive_variant_gt

    config_path = _scene(tmp_path)
    sfm = _write_override(tmp_path, [("000", "A")], masks=False)
    with pytest.raises(FileNotFoundError, match="000.png"):
        derive_variant_gt(_variant_config(config_path, sfm), OBJ)


def test_derive_refuses_a_misaligned_per_point_array(tmp_path):
    from eval3d.pipeline.variant_gt import derive_variant_gt

    config_path = _scene(tmp_path)
    np.save(tmp_path / "kit" / OBJ / "challenges" / "stale.npy", np.ones(5, dtype=bool))
    sfm = _write_override(tmp_path, [("000", "A")])
    with pytest.raises(ValueError, match="stale.npy"):
        derive_variant_gt(_variant_config(config_path, sfm), OBJ)
    assert not (tmp_path / "eval" / OBJ / f"Groundtruth@{TAG}").exists()


def test_derive_low_resolution_set_with_auto_dilation(tmp_path):
    from eval3d.pipeline.variant_gt import derive_variant_gt

    config_path = _scene(tmp_path, dilation="auto")
    sfm = _write_override(tmp_path, [("000", "A")], name="low", w=W // 2, h=H // 2)
    out = derive_variant_gt(_variant_config(config_path, sfm, "low"), OBJ)
    pcd = np.load(tmp_path / "kit" / OBJ / "gt_pcd.npy")
    np.testing.assert_array_equal(np.load(out / "keep.npy"), np.flatnonzero(pcd[:, 2] > 0))
    prov = json.loads((out / "variant_provenance.json").read_text())
    assert prov["dilation_radius"] == "auto"
    assert prov["dilation_px_per_view"] == [compute_auto_dilation((W // 2) * (H // 2), 4, W * H)]


def test_derive_refuses_a_changed_camera_file_without_force(tmp_path):
    from eval3d.pipeline.variant_gt import derive_variant_gt

    config_path = _scene(tmp_path)
    sfm = _write_override(tmp_path, [("000", "A")])
    cfg = _variant_config(config_path, sfm)
    derive_variant_gt(cfg, OBJ)
    _write_sfm(sfm, [("001", "B")])
    _write_masks(sfm.parent / "masks", ["001"])
    with pytest.raises(ValueError, match="force"):
        derive_variant_gt(cfg, OBJ)
    out = derive_variant_gt(cfg, OBJ, force=True)
    assert json.loads((out / "variant_provenance.json").read_text())["views"] == ["001"]


# --------------------------------------------------------------------------- reconstruction cleanup


def _cleaned_vertices(cfg):
    return trimesh.load(cfg.get_cleaned_mesh_path(OBJ, METHOD), process=False, force="mesh").vertices


def test_cleanup_follows_the_override(tmp_path):
    from eval3d.pipeline.cleanup import cleanup_mesh

    config_path = _scene(tmp_path)
    base = load_config(config_path)
    cleanup_mesh(base, OBJ, METHOD)
    v = _cleaned_vertices(base)
    assert (v[:, 2] > 0).any() and (v[:, 2] < 0).any()

    sfm = _write_override(tmp_path, [("000", "A")])
    cfg = _variant_config(config_path, sfm)
    cleanup_mesh(cfg, OBJ, METHOD)
    v = _cleaned_vertices(cfg)
    assert len(v) == N_GRID * N_GRID and (v[:, 2] > 0).all()


def test_no_watertight_dir_in_the_override_means_no_culling(tmp_path):
    from eval3d.pipeline.cleanup import cleanup_mesh

    config_path = _scene(tmp_path)
    # default watertight masks (white everywhere): cull every visible, front-facing vertex
    _write_masks(tmp_path / "gt" / OBJ / "masks_watertight", ["000", "001"])
    base = load_config(config_path)
    cleanup_mesh(base, OBJ, METHOD)
    n_base = len(_cleaned_vertices(base))

    sfm = _write_override(tmp_path, [("000", "A")])
    cfg = _variant_config(config_path, sfm)
    cleanup_mesh(cfg, OBJ, METHOD)
    assert n_base < N_GRID * N_GRID
    assert len(_cleaned_vertices(cfg)) == N_GRID * N_GRID


def test_override_watertight_masks_are_paired_by_name(tmp_path):
    """Views B then A; watertight 000.png (camera A) culls, 001.png (camera B) does not.
    Paired by glob order, 000.png would land on camera B and cull the bottom plate instead."""
    from eval3d.pipeline.cleanup import cleanup_mesh

    config_path = _scene(tmp_path)
    sfm = _write_override(tmp_path, [("001", "B"), ("000", "A")], name="both")
    wt = sfm.parent / "masks_watertight"
    _write_masks(wt, ["000"], value=255)
    _write_masks(wt, ["001"], value=0)
    cfg = _variant_config(config_path, sfm, "both")
    cleanup_mesh(cfg, OBJ, METHOD)
    v = _cleaned_vertices(cfg)
    assert (v[:, 2] < 0).all() and len(v) == N_GRID * N_GRID


# --------------------------------------------------------------------------- end to end, CLI


def test_variant_chain_writes_only_tagged_outputs(tmp_path):
    config_path = _scene(tmp_path)
    # the default tree first: cleaned mesh + eval_results
    assert _cli(config_path, "cleanup").exit_code == 0
    assert _cli(config_path, "evaluate").exit_code == 0
    sfm = _write_override(tmp_path, [("000", "A")])
    before = _tree_hashes(tmp_path)

    v = (sfm, TAG)
    for args in (["preprocess-gt"], ["cleanup"], ["evaluate"], ["apply-masks"], ["curves"], ["recompute"]):
        result = _cli(config_path, *args, variant=v)
        assert result.exit_code == 0, (args, result.output)

    after = _tree_hashes(tmp_path)
    for rel, digest in before.items():
        assert after.get(rel) == digest, f"modified or removed: {rel}"
    new = sorted(set(after) - set(before))
    assert new, "nothing written"
    for rel in new:
        assert any(part.endswith(f"@{TAG}") for part in Path(rel).parts[:-1]), f"untagged output: {rel}"

    cfg = _variant_config(config_path, sfm)
    metrics = json.loads((cfg.get_eval_dir(OBJ, METHOD) / "metrics.json").read_text())
    n_top = int((np.load(tmp_path / "kit" / OBJ / "gt_pcd.npy")[:, 2] > 0).sum())
    assert metrics["n_gt_points"] == n_top
    assert metrics["exclude_masks"] == ["excluded.npy"]
    base_metrics = json.loads((load_config(config_path).get_eval_dir(OBJ, METHOD) / "metrics.json").read_text())
    assert base_metrics["n_gt_points"] > n_top


def test_out_of_scope_commands_refuse_variant_mode(tmp_path):
    config_path = _scene(tmp_path)
    sfm = _write_override(tmp_path, [("000", "A")])
    before = _tree_hashes(tmp_path)
    for args in (["aggregate"], ["latex"], ["proxy-curvature"], ["preprocess-challenges"]):
        result = CliRunner().invoke(main, ["-c", str(config_path), "--eval-cameras", str(sfm),
                                           "--variant", TAG] + args)
        assert result.exit_code != 0, args
        assert "variant" in result.output, args
    assert _tree_hashes(tmp_path) == before


def test_local_runner_carries_the_variant(tmp_path):
    from eval3d.runner.base import Job
    from eval3d.runner.local import LocalRunner, _run_job

    config_path = _scene(tmp_path)
    sfm = _write_override(tmp_path, [("000", "A")])
    cfg = _variant_config(config_path, sfm)
    cfg._config_path = config_path
    runner = LocalRunner(cfg, max_workers=1)
    job_id = runner.submit(Job(OBJ, METHOD, "cleanup", str(config_path)))
    _, ok = _run_job(runner._pending_jobs[job_id])
    assert ok
    assert cfg.get_cleaned_mesh_path(OBJ, METHOD).exists()
    assert not load_config(config_path).get_cleaned_mesh_path(OBJ, METHOD).exists()


# --------------------------------------------------------------------------- watcher


def _slurm_config(config_path):
    raw = yaml.safe_load(config_path.read_text())
    raw["execution"] = {"mode": "slurm", "slurm": {"account": "acc", "partition": "part"}}
    config_path.write_text(yaml.safe_dump(raw))


def _submitted(monkeypatch, cfg, config_path, tmp_path):
    import subprocess

    import eval3d.watcher.scanner as scanner

    calls = []

    class _Done:
        stdout = "42"
        returncode = 0

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return _Done()

    monkeypatch.setattr(scanner.subprocess, "run", fake_run)
    monkeypatch.setattr(subprocess, "run", fake_run)
    sub = scanner.SlurmSubmitter(cfg, config_path, venv_path=tmp_path / "venv")
    calls.clear()
    job = scanner.EvalJob(OBJ, METHOD, "mesh.ply")
    assert sub.submit_pipeline(job) is True
    return calls


def test_watcher_default_commands_unchanged(tmp_path, monkeypatch):
    config_path = _scene(tmp_path)
    _slurm_config(config_path)
    cfg = load_config(config_path)
    calls = _submitted(monkeypatch, cfg, config_path, tmp_path)
    wraps = [c[c.index("--wrap") + 1] for c in calls]
    assert wraps[0].endswith(f"eval-pipeline -c {config_path} cleanup --object {OBJ} --method {METHOD}")
    assert wraps[1].endswith(f"eval-pipeline -c {config_path} evaluate --object {OBJ} --method {METHOD}")
    assert f"--job-name=cleanup_{OBJ}_{METHOD}" in calls[0]
    method_dir = cfg.get_method_dir(OBJ, METHOD)
    assert f"--output={method_dir / 'slurm_logs' / 'cleanup_%j.out'}" in calls[0]


def test_watcher_variant_commands(tmp_path, monkeypatch):
    config_path = _scene(tmp_path)
    _slurm_config(config_path)
    sfm = _write_override(tmp_path, [("000", "A")])
    cfg = _variant_config(config_path, sfm)
    (cfg.get_gt_artifacts_dir(OBJ)).mkdir(parents=True)
    np.save(cfg.get_gt_artifacts_dir(OBJ) / "gt_pcd.npy", np.zeros((1, 3)))
    calls = _submitted(monkeypatch, cfg, config_path, tmp_path)
    wraps = [c[c.index("--wrap") + 1] for c in calls]
    opts = f"--eval-cameras {sfm} --variant {TAG}"
    assert wraps[0].endswith(f"eval-pipeline -c {config_path} {opts} cleanup --object {OBJ} --method {METHOD}")
    assert wraps[1].endswith(f"eval-pipeline -c {config_path} {opts} evaluate --object {OBJ} --method {METHOD}")
    assert any(a.startswith(f"--job-name=cleanup@{TAG}_") for a in calls[0])
    method_dir = cfg.get_method_dir(OBJ, METHOD)
    assert f"--output={method_dir / f'slurm_logs@{TAG}' / 'cleanup_%j.out'}" in calls[0]
    assert not (method_dir / "slurm_logs").exists()


def test_watcher_state_file_is_tagged(tmp_path):
    from eval3d.watcher.scanner import default_state_path

    config_path = _scene(tmp_path)
    sfm = _write_override(tmp_path, [("000", "A")])
    assert default_state_path(config_path, load_config(config_path)) == tmp_path / ".config_state.json"
    assert default_state_path(config_path, _variant_config(config_path, sfm)) == tmp_path / f".config@{TAG}_state.json"
