"""Sparse-view evaluation: `run --views <custom.sfm>` (or `paths.views` in the config).

Two cleaning steps:

1. the usual cleanup (all evaluation views, silhouettes, watertight masks) -> `results_cleaned/`;
2. the same `filter_by_visibility` on the step-1 mesh with the cameras and silhouettes of the custom
   file (`<its dir>/masks/`), without watertight masks -> `results_cleaned@<tag>/`; a view counts
   for a point only if it sees it from its front (`cleanup.views_max_normal_angle`, default 90);
3. the same step 2 on the published GT point cloud -> `Groundtruth@<tag>/` (every per-point array
   indexed by the same keep mask, `keep.npy`);
4. evaluation -> `eval_results@<tag>/`.

The tag is the file name of the custom file (`--variant` sets another one). Without `--views`
nothing changes. A custom file holding all the evaluation views gives the default result only when
every vertex kept by the usual cleanup and every GT point is seen from its front by one of them, as
in this synthetic scene: the orientation test and the GT filter belong to the `--views` mode, the
usual cleanup has neither.

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
from eval3d.core.visibility import check_points_in_mask, compute_auto_dilation, filter_by_visibility

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
    sfm = _write_sfm(root / "overrides" / name / f"{name}.sfm", views, w, h)
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
        pre += ["--views", str(sfm), "--variant", tag]
    result = CliRunner().invoke(main, pre + list(args), catch_exceptions=False)
    return result


def _tree_hashes(root):
    return {
        str(p.relative_to(root)): hashlib.md5(p.read_bytes()).hexdigest()
        for p in sorted(Path(root).rglob("*")) if p.is_file()
    }


def _variant_config(config_path, sfm, tag=TAG):
    cfg = load_config(config_path)
    cfg.apply_views(str(sfm), tag)
    return cfg


# --------------------------------------------------------------------------- config


def test_default_accessors_are_unchanged(tmp_path):
    cfg = load_config(_scene(tmp_path))
    assert cfg.variant is None and cfg.paths.views is None and not cfg.is_variant
    assert cfg.default_view() is cfg
    with pytest.raises(ValueError, match="--views"):
        cfg.get_views_path(OBJ)
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
    # the second cleaning step follows the custom file
    assert cfg.get_views_path(OBJ, METHOD) == sfm.resolve()
    assert cfg.get_views_masks_dir(OBJ, METHOD) == sfm.parent.resolve() / "masks"
    # the first step is the default config
    first = cfg.default_view()
    assert not first.is_variant and cfg.is_variant
    assert first.get_cleaned_mesh_path(OBJ, METHOD) == base.get_cleaned_mesh_path(OBJ, METHOD)
    method_dir = cfg.get_method_dir(OBJ, METHOD)
    assert cfg.get_raw_mesh_path(OBJ, METHOD) == base.get_raw_mesh_path(OBJ, METHOD)
    assert cfg.get_cleaned_mesh_path(OBJ, METHOD) == method_dir / f"results_cleaned@{TAG}" / "mesh.ply"
    assert cfg.get_eval_dir(OBJ, METHOD) == method_dir / f"eval_results@{TAG}"
    assert cfg.get_base_gt_artifacts_dir(OBJ) == tmp_path / "kit" / OBJ
    assert cfg.get_gt_artifacts_dir(OBJ) == cfg.get_eval_root(OBJ) / f"Groundtruth@{TAG}"
    assert cfg.get_lock_path(OBJ, METHOD, "cleanup") == method_dir / f".cleanup_failed@{TAG}"
    assert base.get_lock_path(OBJ, METHOD, "cleanup") == method_dir / ".cleanup_failed"


def test_views_from_config_file(tmp_path):
    config_path = _scene(tmp_path)
    sfm = _write_override(tmp_path, [("000", "A")])
    raw = yaml.safe_load(config_path.read_text())
    raw["paths"]["views"] = str(sfm)
    config_path.write_text(yaml.safe_dump(raw))
    cfg = load_config(config_path)
    assert cfg.variant == TAG   # the file name
    assert cfg.get_views_path(OBJ) == sfm.resolve()
    raw["variant"] = "other"
    assert Config(**raw).variant == "other"


def test_tag_without_views_is_an_error(tmp_path):
    config_path = _scene(tmp_path)
    raw = yaml.safe_load(config_path.read_text())
    raw["variant"] = TAG
    with pytest.raises(ValueError, match="views"):
        Config(**raw)
    result = CliRunner().invoke(main, ["-c", str(config_path), "--variant", TAG, "cleanup"])
    assert result.exit_code != 0 and "--views" in result.output
    result = CliRunner().invoke(main, ["-c", str(config_path), "run", "--variant", TAG])
    assert result.exit_code != 0 and "--views" in result.output


def test_tag_defaults_to_the_file_name(tmp_path):
    cfg = load_config(_scene(tmp_path))
    assert cfg.apply_views(str(tmp_path / "sets" / "360_6.sfm")).variant == "360_6"
    assert cfg.apply_views(str(tmp_path / "sets" / "360_6.sfm"), "six").variant == "six"
    with pytest.raises(ValueError, match="--variant"):
        cfg.apply_views(str(tmp_path / "{object}" / "{object}.sfm"))   # a template: no usable name


@pytest.mark.parametrize("tag", ["", "a/b", "a@b", "x y"])
def test_variant_tag_is_a_plain_name(tmp_path, tag):
    cfg = load_config(_scene(tmp_path))
    with pytest.raises(ValueError):
        cfg.apply_views(str(tmp_path / "x.sfm"), tag)


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
    assert prov["views"] == str(sfm.resolve())
    assert prov["views_md5"] == hashlib.md5(sfm.read_bytes()).hexdigest()
    assert prov["views_list"] == ["000"]
    assert prov["n_base"] == len(base_pcd) and prov["n_kept"] == int(expected.sum())
    assert prov["kept_fraction"] == pytest.approx(expected.mean())
    assert prov["dilation_radius"] == "auto" and prov["use_masks"] is True
    assert prov["base_gt_dir"] == str(kit)


def test_derive_carves_by_silhouette_like_cleanup(tmp_path):
    """Same rule as cleanup: a GT point inside a view's image but outside its (dilated) mask is removed,
    even when another view sees it unoccluded."""
    from eval3d.pipeline.variant_gt import derive_variant_gt

    config_path = _scene(tmp_path)
    sfm = _write_override(tmp_path, [("000", "A"), ("001", "B")], name="carve")
    half = np.full((H, W), 255, dtype=np.uint8)
    half[:, : W // 2] = 0   # camera A: left half of the image outside the silhouette
    Image.fromarray(half).save(sfm.parent / "masks" / "000.png")
    cfg = _variant_config(config_path, sfm, "carve")
    out = derive_variant_gt(cfg, OBJ)

    pcd = np.load(tmp_path / "kit" / OBJ / "gt_pcd.npy")
    cams = load_cameras_auto(sfm)
    radius = compute_auto_dilation(W * H, 4, W * H)
    _, in_a = check_points_in_mask(pcd, cams["P"][0], half > 0, radius, cams["pixel_centre"])
    bottom_carved = (pcd[:, 2] < 0) & ~in_a      # seen by B, but outside A's silhouette
    assert bottom_carved.any() and in_a.any() and not in_a.all()
    keep = np.zeros(len(pcd), dtype=bool)
    keep[np.load(out / "keep.npy")] = True
    assert not keep[bottom_carved].any()
    np.testing.assert_array_equal(keep, in_a)    # every point is seen by A (top) or B (bottom)

    # and it is exactly cleanup's rule
    expected = filter_by_visibility(
        pcd, trimesh.load(tmp_path / "gt" / OBJ / "gt.ply", process=False), cams, sfm.parent / "masks",
        dilation_radius="auto", show_progress=False, use_masks=True, dilation_ref_px=4, dilation_ref_n_pixels=W * H,
    )
    np.testing.assert_array_equal(keep, expected)
    assert json.loads((out / "variant_provenance.json").read_text())["rule"] == "filter_by_visibility"


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
    assert json.loads((out / "variant_provenance.json").read_text())["views_list"] == ["001"]


# --------------------------------------------------------------------------- reconstruction cleanup


def _cleaned_vertices(cfg):
    return trimesh.load(cfg.get_cleaned_mesh_path(OBJ, METHOD), process=False, force="mesh").vertices


def _md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def test_cleanup_runs_the_usual_step_then_the_custom_views(tmp_path):
    from eval3d.pipeline.cleanup import cleanup_mesh

    config_path = _scene(tmp_path)
    base = load_config(config_path)
    sfm = _write_override(tmp_path, [("000", "A")])
    cfg = _variant_config(config_path, sfm)
    assert not base.get_cleaned_mesh_path(OBJ, METHOD).exists()
    cleanup_mesh(cfg, OBJ, METHOD)

    # step 1: the default cleaned mesh, both plates
    v1 = _cleaned_vertices(base)
    assert (v1[:, 2] > 0).any() and (v1[:, 2] < 0).any()
    # step 2: only the plate seen by camera A
    v2 = _cleaned_vertices(cfg)
    assert len(v2) == N_GRID * N_GRID and (v2[:, 2] > 0).all()
    prov = json.loads((cfg.get_cleaned_mesh_path(OBJ, METHOD).parent / "views_provenance.json").read_text())
    assert prov["views"] == str(sfm.resolve()) and prov["views_md5"] == _md5(sfm)
    assert prov["views_list"] == ["000"]
    assert prov["n_source_vertices"] == len(v1) and prov["n_kept_vertices"] == len(v2)
    assert prov["source_mesh"] == str(base.get_cleaned_mesh_path(OBJ, METHOD))


def test_step_one_is_the_default_cleanup_and_is_reused(tmp_path):
    from eval3d.pipeline.cleanup import cleanup_mesh

    config_path = _scene(tmp_path)
    base = load_config(config_path)
    cleanup_mesh(base, OBJ, METHOD)
    default_md5 = _md5(base.get_cleaned_mesh_path(OBJ, METHOD))
    mtime = base.get_cleaned_mesh_path(OBJ, METHOD).stat().st_mtime_ns

    other = tmp_path / "other"
    other_cfg = _scene(other)
    sfm = _write_override(other, [("000", "A")])
    cleanup_mesh(_variant_config(other_cfg, sfm), OBJ, METHOD)
    # step 1 run by the sparse-view cleanup = the default cleanup, byte for byte
    assert _md5(load_config(other_cfg).get_cleaned_mesh_path(OBJ, METHOD)) == default_md5

    sfm = _write_override(tmp_path, [("000", "A")])
    cfg = _variant_config(config_path, sfm)
    cleanup_mesh(cfg, OBJ, METHOD)
    cleanup_mesh(cfg, OBJ, METHOD, force=True)   # force rebuilds step 2 only
    assert base.get_cleaned_mesh_path(OBJ, METHOD).stat().st_mtime_ns == mtime
    assert _md5(base.get_cleaned_mesh_path(OBJ, METHOD)) == default_md5


def test_second_step_starts_from_the_first_and_uses_no_watertight_masks(tmp_path):
    """Watertight masks act at step 1 only: what they removed stays removed, and a masks_watertight/
    directory beside the custom file is not read."""
    from eval3d.pipeline.cleanup import cleanup_mesh

    config_path = _scene(tmp_path)
    plain = tmp_path / "plain"
    plain_cfg = load_config(_scene(plain))
    cleanup_mesh(plain_cfg, OBJ, METHOD)
    n_plain = len(_cleaned_vertices(plain_cfg))

    # default watertight masks, white for view 000 only: step 1 culls what camera A sees front-facing
    _write_masks(tmp_path / "gt" / OBJ / "masks_watertight", ["000"])
    _write_masks(tmp_path / "gt" / OBJ / "masks_watertight", ["001"], value=0)
    base = load_config(config_path)
    sfm = _write_override(tmp_path, [("001", "B")], name="below")
    _write_masks(sfm.parent / "masks_watertight", ["001"])   # would cull again if it were read
    cfg = _variant_config(config_path, sfm, "below")
    cleanup_mesh(cfg, OBJ, METHOD)

    step1 = trimesh.load(base.get_cleaned_mesh_path(OBJ, METHOD), process=False, force="mesh")
    assert 0 < len(step1.vertices) < n_plain
    expected = filter_by_visibility(
        step1.vertices, step1, load_cameras_auto(sfm), sfm.parent / "masks",
        dilation_radius="auto", show_progress=False, use_masks=True, dilation_ref_px=4, dilation_ref_n_pixels=W * H,
    )
    v2 = _cleaned_vertices(cfg)
    np.testing.assert_array_equal(v2, step1.vertices[expected])
    assert len(v2) == int(expected.sum()) > 0


def test_second_step_fails_on_a_missing_silhouette(tmp_path):
    from eval3d.pipeline.cleanup import cleanup_mesh

    config_path = _scene(tmp_path)
    sfm = _write_override(tmp_path, [("000", "A")], masks=False)
    (sfm.parent / "masks").mkdir()
    cfg = _variant_config(config_path, sfm)
    with pytest.raises(FileNotFoundError, match="000.png"):
        cleanup_mesh(cfg, OBJ, METHOD)
    assert not cfg.get_cleaned_mesh_path(OBJ, METHOD).exists()


def test_second_step_refuses_a_changed_view_file_without_force(tmp_path):
    from eval3d.pipeline.cleanup import cleanup_mesh

    config_path = _scene(tmp_path)
    sfm = _write_override(tmp_path, [("000", "A")])
    cfg = _variant_config(config_path, sfm)
    cleanup_mesh(cfg, OBJ, METHOD)
    cleanup_mesh(cfg, OBJ, METHOD)   # unchanged: skipped
    _write_sfm(sfm, [("001", "B")])
    _write_masks(sfm.parent / "masks", ["001"])
    with pytest.raises(ValueError, match="force"):
        cleanup_mesh(cfg, OBJ, METHOD)
    cleanup_mesh(cfg, OBJ, METHOD, force=True)
    assert (_cleaned_vertices(cfg)[:, 2] < 0).all()


# --------------------------------------------------------------------------- orientation test


def _bottom_plate_seen_from_above(tmp_path):
    """The bottom plate alone (normal -z): camera A, above, sees its back; camera B, below, its front."""
    mesh = _plate(-1.0, N_GRID, up=False)
    sfm = _write_sfm(tmp_path / "ab" / "ab.sfm", [("000", "A"), ("001", "B")])
    _write_masks(sfm.parent / "masks", ["000", "001"])
    return mesh, load_cameras_auto(sfm), sfm.parent / "masks"


def _one_camera(cams, i):
    out = dict(cams)
    for k in ("K", "K_inv", "P", "P_inv", "Rt", "scale", "scale_inv"):
        if k in out:
            out[k] = out[k][i:i + 1]
    out["mask_names"] = cams["mask_names"][i:i + 1]
    return out


def test_filter_by_visibility_orientation_test(tmp_path):
    mesh, cams, masks = _bottom_plate_seen_from_above(tmp_path)
    kw = dict(dilation_radius=0, show_progress=False)
    normals = np.asarray(mesh.vertex_normals)
    assert (normals[:, 2] < -0.99).all()
    above, below = _one_camera(cams, 0), _one_camera(cams, 1)
    # without normals (the usual cleanup): seen is enough
    assert filter_by_visibility(mesh.vertices, mesh, above, masks, **kw).all()
    # with normals: the view above sees only the back of the plate, the view below its front
    assert not filter_by_visibility(mesh.vertices, mesh, above, masks, normals=normals, **kw).any()
    assert filter_by_visibility(mesh.vertices, mesh, below, masks, normals=normals, **kw).all()
    assert filter_by_visibility(mesh.vertices, mesh, cams, masks, normals=normals, **kw).all()
    # an undefined normal faces no camera
    broken = normals.copy()
    broken[0] = 0.0
    keep = filter_by_visibility(mesh.vertices, mesh, below, masks, normals=broken, **kw)
    assert not keep[0] and keep[1:].all()


def test_facing_camera_threshold():
    from eval3d.core.visibility import facing_camera

    cam = np.array([0.0, 0.0, 10.0])
    pts = np.zeros((5, 3))
    ang = np.deg2rad([0.0, 30.0, 89.0, 90.0, 120.0])       # angle between the normal and the camera direction
    normals = np.column_stack([np.sin(ang), np.zeros(5), np.cos(ang)])
    normals[3] = [1.0, 0.0, 0.0]                             # exactly 90 degrees
    np.testing.assert_array_equal(facing_camera(pts, normals, cam), [True, True, True, False, False])   # strict
    np.testing.assert_array_equal(facing_camera(pts, normals, cam, 45.0), [True, True, False, False, False])
    np.testing.assert_array_equal(facing_camera(pts, normals, cam, 150.0), [True, True, True, True, True])


def _holed_plates(dz=0.0):
    """The two plates, the top one with a strip missing (a fifth of it): through that hole, camera A
    (above) sees the back of the bottom plate. Most of what A sees is still a front side, so the
    global orientation check does not take the mesh for an inverted one."""
    top = _plate(1.0 + dz, N_GRID, up=True)
    top.update_faces(top.triangles_center[:, 0] > -0.6)
    top.remove_unreferenced_vertices()
    return trimesh.util.concatenate([top, _plate(-1.0 - dz, N_GRID, up=False)])


def _holed_scene(tmp_path):
    config_path = _scene(tmp_path)
    _holed_plates(dz=0.01).export(tmp_path / "eval" / OBJ / METHOD / "results_raw" / "mesh.ply")
    return config_path


def _bottom(vertices):
    return vertices[vertices[:, 2] < 0]


def test_vertex_seen_from_behind_through_a_hole_is_removed(tmp_path):
    from eval3d.pipeline.cleanup import cleanup_mesh

    config_path = _holed_scene(tmp_path)
    base = load_config(config_path)
    sfm_a = _write_override(tmp_path, [("000", "A")], name="above")
    cfg_a = _variant_config(config_path, sfm_a, "above")
    cleanup_mesh(cfg_a, OBJ, METHOD)
    step1 = trimesh.load(base.get_cleaned_mesh_path(OBJ, METHOD), process=False, force="mesh")
    n_top = int((step1.vertices[:, 2] > 0).sum())
    assert n_top and len(_bottom(step1.vertices)) == N_GRID * N_GRID      # the usual cleanup keeps both
    # A sees a part of the bottom plate unoccluded, through the hole...
    seen = filter_by_visibility(step1.vertices, step1, load_cameras_auto(sfm_a), sfm_a.parent / "masks",
                                dilation_radius="auto", show_progress=False, dilation_ref_px=4,
                                dilation_ref_n_pixels=W * H)
    assert (seen & (step1.vertices[:, 2] < 0)).sum() > 20
    # ... but from behind: only the top plate is kept
    kept = _cleaned_vertices(cfg_a)
    assert len(kept) == n_top and len(_bottom(kept)) == 0
    prov = json.loads((cfg_a.get_cleaned_mesh_path(OBJ, METHOD).parent / "views_provenance.json").read_text())
    assert prov["views_max_normal_angle"] == 90.0 and prov["normals_flipped"] is False
    assert 0.5 < prov["front_facing_fraction"] < 1.0

    # a view on the other side sees the bottom plate from its front: kept
    sfm_b = _write_override(tmp_path, [("001", "B")], name="below")
    cfg_b = _variant_config(config_path, sfm_b, "below")
    cleanup_mesh(cfg_b, OBJ, METHOD)
    assert len(_bottom(_cleaned_vertices(cfg_b))) == N_GRID * N_GRID


def test_orientation_threshold_is_read_from_the_config(tmp_path):
    from eval3d.pipeline.cleanup import cleanup_mesh

    config_path = _holed_scene(tmp_path)
    sfm = _write_override(tmp_path, [("000", "A")], name="above")
    raw = yaml.safe_load(config_path.read_text())
    raw["cleanup"]["views_max_normal_angle"] = 179.0     # nearly no orientation test
    config_path.write_text(yaml.safe_dump(raw))
    cfg = _variant_config(config_path, sfm, "above")
    assert cfg.cleanup.views_max_normal_angle == 179.0
    cleanup_mesh(cfg, OBJ, METHOD)
    # the bottom plate under the hole is seen from behind at 171 to 180 degrees: kept below 179
    assert len(_bottom(_cleaned_vertices(cfg))) > 20
    prov_path = cfg.get_cleaned_mesh_path(OBJ, METHOD).parent / "views_provenance.json"
    assert json.loads(prov_path.read_text())["views_max_normal_angle"] == 179.0

    # another threshold is another setting: refused without --force
    raw["cleanup"]["views_max_normal_angle"] = 90.0
    config_path.write_text(yaml.safe_dump(raw))
    strict = _variant_config(config_path, sfm, "above")
    with pytest.raises(ValueError, match="views_max_normal_angle"):
        cleanup_mesh(strict, OBJ, METHOD)
    cleanup_mesh(strict, OBJ, METHOD, force=True)
    assert len(_bottom(_cleaned_vertices(strict))) == 0

    for bad in (0.0, -10.0, 181.0):
        raw["cleanup"]["views_max_normal_angle"] = bad
        with pytest.raises(ValueError):
            Config(**raw)


def test_gt_point_seen_from_behind_is_removed(tmp_path):
    """Same rule for the GT: the normal of a GT point is the vertex normal of the nearest vertex of the
    GT mesh. GT mesh = the plates with the hole, GT points = the bottom plate only: camera A sees some
    of them, through the hole, and only from behind."""
    from eval3d.pipeline.variant_gt import derive_variant_gt, nearest_vertex_normals

    config_path = _scene(tmp_path)
    gt_mesh = _holed_plates()
    gt_mesh.export(tmp_path / "gt" / OBJ / "gt.ply")
    kit = tmp_path / "kit" / OBJ
    pcd = np.load(kit / "gt_pcd.npy")
    bottom = pcd[:, 2] < 0
    for rel in ["gt_pcd.npy", "attributes/visibility_count.npy", "attributes/curvature_values.npy",
                "challenges/lambertian.npy", "challenges/excluded.npy"]:
        np.save(kit / rel, np.load(kit / rel)[bottom])
    pcd = pcd[bottom]
    assert (nearest_vertex_normals(pcd, gt_mesh)[:, 2] < -0.99).all()

    sfm_a = _write_override(tmp_path, [("000", "A")], name="above")
    with pytest.raises(ValueError, match="no GT point"):          # seen by A, but only from behind
        derive_variant_gt(_variant_config(config_path, sfm_a, "above"), OBJ)
    sfm_b = _write_override(tmp_path, [("001", "B")], name="below")
    out = derive_variant_gt(_variant_config(config_path, sfm_b, "below"), OBJ)
    np.testing.assert_array_equal(np.load(out / "keep.npy"), np.arange(len(pcd)))
    prov = json.loads((out / "variant_provenance.json").read_text())
    assert prov["views_max_normal_angle"] == 90.0 and prov["n_undefined_normals"] == 0
    assert "nearest vertex" in prov["normals"] and prov["normals_flipped"] is False

    # the threshold is the config's
    raw = yaml.safe_load(config_path.read_text())
    raw["cleanup"]["views_max_normal_angle"] = 179.0
    config_path.write_text(yaml.safe_dump(raw))
    out = derive_variant_gt(_variant_config(config_path, sfm_a, "above"), OBJ)
    assert 20 < len(np.load(out / "keep.npy")) < len(pcd)


def test_surface_seen_only_from_behind_is_taken_for_an_inverted_mesh(tmp_path):
    """Limit of the global orientation check, as in the usual cleanup: when the first view sees only
    back sides (here the bottom plate alone, from above), the normals are taken for inward ones and
    negated, and what the view sees is kept."""
    mesh, cams, masks = _bottom_plate_seen_from_above(tmp_path)
    from eval3d.core.visibility import detect_normal_flip

    d = detect_normal_flip(mesh, _one_camera(cams, 0))
    assert d["flip"] is True and d["front_fraction"] == 0.0
    assert detect_normal_flip(mesh, _one_camera(cams, 1))["flip"] is False


def _inverted(path):
    mesh = trimesh.load(path, process=False, force="mesh")
    trimesh.Trimesh(mesh.vertices, mesh.faces[:, ::-1], process=False).export(path)


def test_mesh_with_all_faces_inverted_keeps_the_same_points(tmp_path):
    """Inward normals are detected (orientation check of the usual cleanup) and negated before the
    orientation test: same kept vertices and same kept GT points as with the mesh the right way round."""
    from eval3d.core.visibility import detect_normal_flip
    from eval3d.pipeline.cleanup import cleanup_mesh
    from eval3d.pipeline.variant_gt import derive_variant_gt

    kept, keep_gt, prov = {}, {}, {}
    for name in ("straight", "inverted"):
        root = tmp_path / name
        root.mkdir()
        config_path = _scene(root)
        if name == "inverted":
            _inverted(root / "eval" / OBJ / METHOD / "results_raw" / "mesh.ply")
            _inverted(root / "gt" / OBJ / "gt.ply")
        sfm = _write_override(root, [("000", "A")], name="above")
        cfg = _variant_config(config_path, sfm, "above")
        cleanup_mesh(cfg, OBJ, METHOD)
        kept[name] = _cleaned_vertices(cfg)
        out = derive_variant_gt(cfg, OBJ)
        keep_gt[name] = np.load(out / "keep.npy")
        prov[name] = (
            json.loads((cfg.get_cleaned_mesh_path(OBJ, METHOD).parent / "views_provenance.json").read_text()),
            json.loads((out / "variant_provenance.json").read_text()),
        )
        step1 = trimesh.load(load_config(config_path).get_cleaned_mesh_path(OBJ, METHOD), process=False, force="mesh")
        assert detect_normal_flip(step1, load_cameras_auto(sfm))["flip"] is (name == "inverted")

    assert len(kept["straight"]) == N_GRID * N_GRID                 # the top plate, seen from its front by A
    np.testing.assert_array_equal(kept["inverted"], kept["straight"])
    assert len(keep_gt["straight"]) > 0
    np.testing.assert_array_equal(keep_gt["inverted"], keep_gt["straight"])
    for p in prov["straight"]:
        assert p["normals_flipped"] is False and p["front_facing_fraction"] == 1.0
    for p in prov["inverted"]:
        assert p["normals_flipped"] is True and p["front_facing_fraction"] == 0.0


def test_all_evaluation_views_still_remove_what_they_see_only_from_behind(tmp_path):
    """The control with all the evaluation views is not the default result in general: here the usual
    cleanup keeps the bottom plate (B sees it), and so does the second step; but with the bottom plate
    turned inside out (normals up), every view that sees it sees its back, and the second step removes
    it while the usual cleanup, which has no orientation test, keeps it."""
    from eval3d.pipeline.cleanup import cleanup_mesh

    config_path = _scene(tmp_path)
    raw_mesh = tmp_path / "eval" / OBJ / METHOD / "results_raw" / "mesh.ply"
    trimesh.util.concatenate([_plate(1.01, N_GRID, up=True), _plate(-1.01, N_GRID, up=True)]).export(raw_mesh)
    base = load_config(config_path)
    sfm = _write_override(tmp_path, [("000", "A"), ("001", "B")], name="all")
    cfg = _variant_config(config_path, sfm, "all")
    cleanup_mesh(cfg, OBJ, METHOD)
    v1 = _cleaned_vertices(base)
    assert (v1[:, 2] > 0).any() and (v1[:, 2] < 0).any()        # usual cleanup: both plates
    v2 = _cleaned_vertices(cfg)
    assert len(v2) == N_GRID * N_GRID and (v2[:, 2] > 0).all()   # second step: the inside-out plate is gone


# --------------------------------------------------------------------------- control: all the evaluation views


def test_all_evaluation_views_give_the_default_result(tmp_path):
    """Control: a custom file holding the evaluation views themselves changes nothing here, for the
    reconstruction and for the GT, because every vertex and GT point of this scene is seen from its
    front by one of them. It is not a general property: the second step also removes what the views
    do not see (test_gt_points_unseen_by_all_the_views_are_removed) or see only from behind
    (test_all_evaluation_views_still_remove_what_they_see_only_from_behind)."""
    config_path = _scene(tmp_path)
    base = load_config(config_path)
    assert _cli(config_path, "run", "-w", "1").exit_code == 0
    sfm = _write_override(tmp_path, [("000", "A"), ("001", "B")], name="all")
    before = _tree_hashes(tmp_path)
    result = _cli(config_path, "run", "-w", "1", "--views", str(sfm))
    assert result.exit_code == 0, result.output
    after = _tree_hashes(tmp_path)
    assert all(after.get(rel) == digest for rel, digest in before.items())

    cfg = _variant_config(config_path, sfm, "all")
    # reconstruction: same cleaned mesh, byte for byte
    assert _md5(cfg.get_cleaned_mesh_path(OBJ, METHOD)) == _md5(base.get_cleaned_mesh_path(OBJ, METHOD))
    # GT: every point kept, per-point arrays identical byte for byte
    kit, out = tmp_path / "kit" / OBJ, cfg.get_gt_artifacts_dir(OBJ)
    n = len(np.load(kit / "gt_pcd.npy"))
    np.testing.assert_array_equal(np.load(out / "keep.npy"), np.arange(n))
    for rel in ["gt_pcd.npy", "attributes/visibility_count.npy", "attributes/curvature_values.npy",
                "challenges/lambertian.npy", "challenges/excluded.npy"]:
        assert _md5(out / rel) == _md5(kit / rel), rel
    # evaluation: same files, byte for byte
    d0, d1 = base.get_eval_dir(OBJ, METHOD), cfg.get_eval_dir(OBJ, METHOD)
    assert d1.name == "eval_results@all"
    h0, h1 = _tree_hashes(d0), _tree_hashes(d1)
    assert h0 and h0 == h1


def test_gt_points_unseen_by_all_the_views_are_removed(tmp_path):
    """The GT has no step 1: points of the published cloud that no evaluation view sees (here one
    between the two plates) are removed at step 3, even with all the evaluation views."""
    from eval3d.pipeline.variant_gt import derive_variant_gt

    config_path = _scene(tmp_path)
    kit = tmp_path / "kit" / OBJ
    pcd = np.load(kit / "gt_pcd.npy")
    hidden = len(pcd) // 2
    pcd[hidden] = [0.0, 0.0, 0.0]   # between the plates: occluded from A and from B
    np.save(kit / "gt_pcd.npy", pcd)
    sfm = _write_override(tmp_path, [("000", "A"), ("001", "B")], name="all")
    out = derive_variant_gt(_variant_config(config_path, sfm, "all"), OBJ)
    np.testing.assert_array_equal(np.load(out / "keep.npy"), np.delete(np.arange(len(pcd)), hidden))
    assert len(np.load(out / "attributes" / "visibility_count.npy")) == len(pcd) - 1


def test_run_views_does_the_four_steps(tmp_path):
    config_path = _scene(tmp_path)
    sfm = _write_override(tmp_path, [("000", "A")])
    result = _cli(config_path, "run", "-w", "1", "-o", OBJ, "-m", METHOD, "--views", str(sfm))
    assert result.exit_code == 0, result.output
    method_dir = tmp_path / "eval" / OBJ / METHOD
    assert (method_dir / "results_cleaned" / "mesh.ply").exists()               # step 1
    assert (method_dir / f"results_cleaned@{TAG}" / "mesh.ply").exists()        # step 2
    gt = tmp_path / "eval" / OBJ / f"Groundtruth@{TAG}"
    pcd = np.load(tmp_path / "kit" / OBJ / "gt_pcd.npy")
    np.testing.assert_array_equal(np.load(gt / "keep.npy"), np.flatnonzero(pcd[:, 2] > 0))   # step 3
    metrics = json.loads((method_dir / f"eval_results@{TAG}" / "metrics.json").read_text())  # step 4
    assert metrics["n_gt_points"] == int((pcd[:, 2] > 0).sum())
    assert not (method_dir / "eval_results").exists()   # the default evaluation is not run
    # --variant renames the outputs
    result = _cli(config_path, "run", "-w", "1", "--views", str(sfm), "--variant", "three")
    assert result.exit_code == 0, result.output
    assert (method_dir / "eval_results@three" / "metrics.json").exists()


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
        result = CliRunner().invoke(main, ["-c", str(config_path), "--views", str(sfm)] + args)
        assert result.exit_code != 0, args
        assert "--views" in result.output, args
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
    assert load_config(config_path).get_cleaned_mesh_path(OBJ, METHOD).exists()   # step 1


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
    opts = f"--views {sfm} --variant {TAG}"
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
