"""LaTeX generator of the per-object supplementary report, and its image helpers (photos, montage)."""

import json

import numpy as np

from eval3d.figures.supplementary import document as supp
from eval3d.figures.supplementary import montage, photos

SPEC = {
    "methods": {
        "realityscan": {"display": "RealityScan", "color": "#003d5c"},
        "meshroom": {"display": "Meshroom", "color": "#003d5c", "hollow": True},
        "colmap": {"display": "COLMAP", "color": "#5749a5"},
        "supernormal": {"display": "SuperNormal", "color": "#f8a700", "hollow": True},
    },
    "cells": {"tiers": ["d1", "d2", "d4", "d8"], "views_tier": "d4", "full_views": 84, "declared": {
        "realityscan": {"tiers": ["d1", "d2", "d4", "d8"], "views": [40, 20, 10, 5]},
        "meshroom": {"tiers": ["d1", "d4"]},
        "colmap": {"tiers": ["d2", "d4", "d8"], "views": [40, 20, 10, 5]},
        "supernormal": {"tiers": ["d4"]}}},
    "resolution_views": {"threshold": 1.0, "methods": ["realityscan", "colmap"], "views": [5, 10, 20, 40, 84], "tiers": ["d1", "d2", "d4", "d8"],
                         "tier_labels": {"d1": "Full\n9568x6376", "d2": "Half\n4784x3188", "d4": "Quarter\n2392x1594",
                                         "d8": "Eighth\n1196x797"}},
}


def row(status="OK", f05="0.982011", cd="0.101571"):
    return {"status": status, "f05": f05 if status == "OK" else "", "cd": cd if status == "OK" else ""}


METRICS = {("01_rock", c): row() for c in
           ["realityscan_d1", "realityscan_d2", "realityscan_d4", "realityscan_d8", "realityscan_d4_v40", "realityscan_d4_v20",
            "realityscan_d4_v10", "realityscan_d4_v5", "meshroom_d4", "colmap_d2", "colmap_d8", "colmap_d4_v40", "colmap_d4_v20"]}
METRICS[("01_rock", "colmap_d4")] = row("SPARSE_FAIL")
METRICS[("01_rock", "colmap_d4_v10")] = row("EMPTY")
METRICS[("01_rock", "colmap_d4_v5")] = row("MISSING")
METRICS[("01_rock", "supernormal_d4")] = row(f05="0.5", cd="1.23456")
OBJ = dict(obj_id="01_rock", name="rock", view=77)


def test_scores_format():
    assert supp.scores(row()) == r"F@0.5 = 98.2 \textbar{} CD = 0.10\,mm"
    assert supp.scores(row(f05="0.5", cd="1.23456")) == r"F@0.5 = 50.0 \textbar{} CD = 1.23\,mm"
    assert supp.scores(row(), stacked=True) == r"\shortstack{F@0.5 = 98.2\\CD = 0.10\,mm}"


def test_fail_labels():
    assert supp.fail_label("SPARSE_FAIL") == "SfM failed"
    assert supp.fail_label("EMPTY") == "empty mesh"
    assert supp.fail_label("MISSING") == "run failed"


def test_cell_name():
    assert supp.cell_name("colmap", "d4", 84, SPEC) == "colmap_d4"
    assert supp.cell_name("colmap", "d4", 10, SPEC) == "colmap_d4_v10"
    assert supp.cell_name("colmap", "d2", 84, SPEC) == "colmap_d2"


def test_image_path():
    assert supp.image("01_rock", "colmap_d4", "accuracy", 77) == "vizf/01_rock/colmap_d4/visualizations/accuracy/view_077.png"
    assert supp.image("01_rock", "colmap_d4", "normals", 77) == "vizn/01_rock/colmap_d4/visualizations/normals/view_077.png"
    assert supp.image("01_rock", "colmap_d4", "angular_error", 77) == "angular/01_rock/colmap_d4/visualizations/angular_error/view_077.png"


def test_object_page_rows_follow_spec_order_and_use_d4():
    tex = supp.object_page(OBJ, SPEC, METRICS)
    order = [tex.index(n) for n in ["RealityScan", "Meshroom", "COLMAP", "SuperNormal"]]
    assert order == sorted(order)
    for metric in ["uniform", "accuracy", "completeness"]:
        assert f"vizf/01_rock/realityscan_d4/visualizations/{metric}/view_077.png" in tex
    assert "realityscan_d2" not in tex and "meshroom_d1" not in tex
    assert tex.startswith(r"\noindent\obj{rock}\par")


def test_object_page_failure_spans_the_row_without_image_or_score():
    tex = supp.object_page(OBJ, SPEC, METRICS)
    assert "colmap_d4/" not in tex
    assert "SfM failed" in tex
    assert tex.count(r"F@0.5 = 98.2 \textbar{} CD = 0.10\,mm") == 2      # realityscan, meshroom
    assert tex.count(r"F@0.5 = 50.0 \textbar{} CD = 1.23\,mm") == 1


def test_object_page_hollow_marker():
    tex = supp.object_page(OBJ, SPEC, METRICS)
    assert r"\mthh{Meshroom}{003d5c}" in tex and r"\mth{RealityScan}{003d5c}" in tex


def test_object_page_reference_row():
    tex = supp.object_page(OBJ, SPEC, METRICS)
    assert "vizf/01_rock/Groundtruth/visualizations/uniform/view_077.png" in tex
    assert "photos/rock_077.jpg" in tex


def test_tiers_page_shows_the_resolution_methods_of_the_paper_only():
    tex = supp.tiers_page(OBJ, SPEC, METRICS)
    assert "Meshroom" not in tex and "SuperNormal" not in tex
    assert tex.index("RealityScan") < tex.index("COLMAP")
    assert "montage/01_rock/realityscan_d1/view_077.jpg" in tex
    assert "colmap_d1" not in tex                      # not declared: empty cell
    assert "SfM failed" in tex                         # colmap_d4
    assert tex.count(r"\includegraphics") == 4 + 2


def test_views_page_keeps_methods_with_view_subsets_only():
    tex = supp.views_page(OBJ, SPEC, METRICS)
    assert "Meshroom" not in tex and "SuperNormal" not in tex


def test_curves_pages_one_full_width_figure_per_object_two_per_page():
    tex = supp.curves_pages(["rock", "vase", "hammer_head"])
    assert tex.count(r"\includegraphics[width=\textwidth]") == 3 and tex.count(r"\clearpage") == 2
    assert "curves/curves_hammer_head.pdf" in tex and r"\obj{hammer head}" in tex


def test_every_cell_has_the_row_height_and_is_vertically_centred():
    for tex, h in [(supp.tiers_page(OBJ, SPEC, METRICS), "3.0cm"), (supp.views_page(OBJ, SPEC, METRICS), "3.0cm"),
                   (supp.object_page(OBJ, SPEC, METRICS), "1.6cm")]:
        assert tex.count(r"\includegraphics") == tex.count(r"\cell{") - tex.count(r"\rotatebox") > 0
        assert tex.count(rf"{{{h}}}{{\rotatebox") == tex.count(r"\rotatebox")          # labels at the row height
        assert all(f"{{{h}}}" in line for line in tex.splitlines() if r"\failbox" in line)


def test_tiers_page_headers_are_resolutions_and_title_names_the_maps():
    tex = supp.tiers_page(OBJ, SPEC, METRICS)
    assert r"9568$\times$6376" in tex and r"1196$\times$797" in tex and "d1 (full)" not in tex
    for t in [tex, supp.views_page(OBJ, SPEC, METRICS)]:
        assert "accuracy maps" not in t and "visualizations" not in t


def test_preamble_drops_the_conference_footer():
    assert r"\renewcommand{\@noticestring}{}" in supp.PRE


def test_guide_lists_methods_in_order_and_explains_the_maps():
    tex = supp.guide(SPEC)
    order = [tex.index(n) for n in ["RealityScan", "Meshroom", "COLMAP", "SuperNormal"]]
    assert order == sorted(order)
    for word in ["Accuracy", "Completeness", "F@0.5", "CD", "SfM failed", "empty mesh", "run failed"]:
        assert word in tex


def test_views_page_columns_from_all_views_down():
    tex = supp.views_page(OBJ, SPEC, METRICS)
    head = [tex.index(f"{{{n}}}") for n in ["84 views", "40 views", "20 views", "10 views", "5 views"]]
    assert head == sorted(head)
    assert "montage/01_rock/realityscan_d4_v5/view_077.jpg" in tex
    assert "empty mesh" in tex and "run failed" in tex and "SfM failed" in tex
    assert tex.count(r"\includegraphics") == 5 + 2


def test_prf_pages_three_per_page_then_pooled_mean():
    tex = supp.prf_pages(["rock", "vase", "candle", "can"])
    assert tex.count(r"\includegraphics") == 5
    assert "prf/prf_can.pdf" in tex
    assert tex.index("prf/prf_mean.pdf") > tex.index("prf/prf_can.pdf") and r"\textsc{mean}" in tex
    assert tex.index(r"\rule{\textwidth}") < tex.index("prf/prf_mean.pdf")


def test_referenced_files():
    tex = r"\includegraphics[width=1cm]{viz/a/b.png} x \includegraphics[height=2cm]{prf/prf_can.pdf}"
    assert supp.referenced_files(tex) == ["viz/a/b.png", "prf/prf_can.pdf"]


def test_photo_crop_box_scales_and_clamps():
    ys, xs = photos.crop_box([2172, 4086, 3056, 6460], 4, (1594, 2392))
    assert (ys.start, ys.stop, xs.start, xs.stop) == (543, 1022, 764, 1615)
    ys, xs = photos.crop_box([-8, 7000, -4, 9600], 4, (1594, 2392))
    assert (ys.start, ys.stop, xs.start, xs.stop) == (0, 1594, 0, 2392)


def test_object_names_are_written_with_spaces_as_in_the_paper():
    o = dict(obj_id="01_rock", name="hammer_head", view=77)
    for tex in [supp.object_page(o, SPEC, METRICS), supp.tiers_page(o, SPEC, METRICS), supp.prf_pages(["hammer_head"])]:
        assert r"\obj{hammer head}" in tex and r"\obj{hammer_head}" not in tex
    assert "photos/hammer_head_077.jpg" in supp.object_page(o, SPEC, METRICS)
    assert "prf/prf_hammer_head.pdf" in supp.prf_pages(["hammer_head"])


def test_referenced_files_ignores_macro_arguments():
    assert supp.referenced_files(r"\newcommand{\cbar}[1]{\includegraphics[width=1cm]{#1}} \cbar{photos/colorbar.png}") == ["photos/colorbar.png"]


MAE = {("01_rock", "realityscan_d4"): 9.5285, ("01_rock", "meshroom_d4"): 3.04, ("01_rock", "supernormal_d4"): 12.0}


def test_object_page_with_normals_has_five_columns_from_the_normals_root():
    tex = supp.object_page(OBJ, SPEC, METRICS, mae=MAE)
    for metric in ["uniform", "accuracy", "completeness"]:                  # flat renders of the maps
        assert f"vizf/01_rock/realityscan_d4/visualizations/{metric}/view_077.png" in tex
    assert "vizn/01_rock/realityscan_d4/visualizations/normals/view_077.png" in tex
    assert "vizf/01_rock/Groundtruth/visualizations/uniform/view_077.png" in tex
    assert "angular/01_rock/realityscan_d4/visualizations/angular_error/view_077.png" in tex      # rendered again in magma
    assert "visualizations_remove" not in tex
    head = [tex.index(t) for t in ["Reconstruction", "Accuracy", "Completeness", "Normals", "Angular error"]]
    assert head == sorted(head)
    assert r"MAE 9.5$^\circ$" in tex and r"MAE 3.0$^\circ$" in tex
    assert tex.count(r"F@0.5 = 98.2 \textbar{} CD = 0.10\,mm") == 2
    assert "vizn/01_rock/Groundtruth/visualizations/normals/view_077.png" in tex
    assert "SfM failed" in tex and "colmap_d4/" not in tex
    assert r"\cbars{photos/colorbar.png}{photos/colorbar_angular.png}" in tex      # the two colorbars side by side
    assert "vizn/01_rock/realityscan_d4/visualizations_magma/angular_error/view_077.png" not in tex


def test_object_page_without_normals_has_three_columns():
    tex = supp.object_page(OBJ, SPEC, METRICS)
    assert "Normals" not in tex and "MAE" not in tex and "normals/" not in tex
    assert "vizf/01_rock/realityscan_d4/visualizations/accuracy/view_077.png" in tex and "visualizations_remove" not in tex


def test_load_mae_reads_the_displayed_view(tmp_path):
    d = tmp_path / "01_rock" / "realityscan_d4" / "visualizations" / "angular_error"
    d.mkdir(parents=True)
    (d / "mae.json").write_text(json.dumps({"views": {"77": {"mae_deg": 9.5}, "3": {"mae_deg": 1.0}}, "cell": {"mae_deg": 5.0}}))
    assert supp.load_mae(tmp_path, [OBJ], "visualizations") == {("01_rock", "realityscan_d4"): 9.5}


def test_reference_row_shows_unimsps_normals():
    tex = supp.object_page(OBJ, SPEC, METRICS)
    assert "photos/rock_077_unimsps.jpg" in tex and "Uni-MS-PS normals" in tex
    tex = supp.object_page(OBJ, SPEC, METRICS, mae=MAE)
    assert tex.index("photos/rock_077.jpg") < tex.index("photos/rock_077_mvps.jpg") < tex.index("vizn/01_rock/Groundtruth/visualizations/normals") < tex.index("photos/rock_077_unimsps.jpg")
    assert "MVPS image" in tex
    assert tex.index("Ground-truth normals") < tex.index("Uni-MS-PS normals")


def test_support_box_adds_a_margin_and_clamps():
    m = np.zeros((100, 200), bool)
    m[10:40, 50:120] = True
    ys, xs = photos.support_box(m, 4)
    assert (ys.start, ys.stop, xs.start, xs.stop) == (6, 44, 46, 124)
    m[0, 0] = True
    ys, xs = photos.support_box(m, 4)
    assert (ys.start, xs.start) == (0, 0)


def test_montage_split_left_mesh_right_accuracy_with_an_oblique_line():
    mesh = np.full((100, 200, 3), 200, np.uint8)
    acc = np.full((100, 200, 3), 50, np.uint8)
    mesh[20:80, 60:140] = 180                       # support of the object: centre at x = 99.5
    out = montage.split(mesh, acc, lean=0.1, line=2, color=(0, 0, 0))
    assert out.shape == mesh.shape
    assert (out[50, 60] == 180).all() and (out[50, 139] == 50).all()
    assert (out[0, 0] == 200).all() and (out[99, 199] == 50).all()
    assert (out[50, 99] == 0).all() or (out[50, 100] == 0).all()        # the cut goes through the centre of the support
    top, bottom = np.nonzero((out[0] == 0).all(-1))[0].mean(), np.nonzero((out[99] == 0).all(-1))[0].mean()
    assert top - bottom > 15                         # oblique: leaning to the right at the top (2 x 0.1 x 100)


def test_montage_split_without_support_cuts_in_the_middle():
    white = np.full((10, 20, 3), 255, np.uint8)
    out = montage.split(white, np.zeros_like(white), lean=0.0, line=1, color=(9, 9, 9))
    assert (out[:, :9] == 255).all() and (out[:, 11:] == 0).all()


def test_every_page_starts_with_the_object_name_alone_in_small_caps():
    o = dict(obj_id="01_rock", name="hammer_head", view=77)
    for tex in [supp.object_page(o, SPEC, METRICS), supp.tiers_page(o, SPEC, METRICS), supp.views_page(o, SPEC, METRICS),
                supp.prf_pages(["hammer_head"]), supp.curves_pages(["hammer_head"])]:
        assert tex.startswith(r"\noindent\obj{hammer head}\par") and r"\subsection" not in tex and r"\textbf{\obj" not in tex


def test_every_section_has_a_title_then_a_short_text():
    objs = [OBJ]
    tex = supp.document(objs, SPEC, METRICS)
    parts = tex.split(r"\section{")[1:]
    assert len(parts) == 5
    for part in parts:
        intro = part.split("\n")[1]
        assert intro.startswith(r"\noindent ") and intro.endswith(r"\par\medskip") and 80 < len(intro) < 330
    assert "left" in parts[1].split("\n")[1] and "right" in parts[1].split("\n")[1]      # the montage is explained
    assert "0.5\\,mm" in parts[3].split("\n")[1] and "1\\,mm" not in parts[3].split("\n")[1]   # threshold of part 4


def test_jpeg_paths_rewrites_only_the_png_renders():
    tex = r"\includegraphics[width=1cm]{vizn/a/b/view_001.png} \includegraphics[width=1cm]{prf/prf_can.pdf} \cbar{photos/colorbar.png} {montage/a/b/view_001.jpg}"
    out, pairs = supp.jpeg_paths(tex)
    assert r"{jpg/vizn/a/b/view_001.jpg}" in out and "vizn/a/b/view_001.png" not in out
    assert "prf/prf_can.pdf" in out and "photos/colorbar.png" in out and "{montage/a/b/view_001.jpg}" in out
    assert pairs == [("vizn/a/b/view_001.png", "jpg/vizn/a/b/view_001.jpg")]


def test_reference_row_names_the_two_modalities():
    tex = supp.object_page(OBJ, SPEC, METRICS, mae=MAE)
    assert r"\scriptsize MVS image" in tex
    assert "Input image" not in tex and "one light" not in tex
    assert tex.index(r"\scriptsize MVS image") < tex.index(r"\scriptsize MVPS image")


def test_no_wider_gap_before_the_normal_columns():
    tex = supp.object_page(OBJ, SPEC, METRICS, mae=MAE)
    assert r"\hspace{10pt}" not in tex and tex.count(r"@{\hspace{2pt}}") == 5


ZONE = {"ids": ["01_rock"], "objects": ["rock"],
        "object": {"01_rock": {"realityscan": [90.123, 80.0, 84.7], "meshroom": [1.0, 2.0, 3.0], "colmap": None, "supernormal": [5, 6, 7]}},
        "pooled": {"realityscan": [91.0, 81.0, 85.7], "meshroom": [1.5, 2.5, 3.5], "colmap": [0.0, 0.0, 0.0], "supernormal": [5, 6, 7]}}


def test_chunks_are_balanced_and_at_most_six():
    assert [len(c) for c in supp.chunks(list(range(12)), 6)] == [6, 6]
    assert [len(c) for c in supp.chunks(list(range(26)), 6)] == [6, 5, 5, 5, 5]
    assert [len(c) for c in supp.chunks(list(range(4)), 6)] == [4]
    assert sum(supp.chunks(list(range(11)), 6), []) == list(range(11))


def test_taxo_page_has_one_row_per_method_and_prf_below_each_map():
    tex = supp.taxo_pages("transparent", ZONE, {"01_rock": OBJ}, SPEC, METRICS)
    assert tex.count(r"\clearpage") == 1 and r"\obj{Transparent}" in tex
    assert "vizz/transparent/01_rock/realityscan_d4/visualizations/accuracy/view_077.png" in tex
    assert r"\scriptsize 90.1 / 80.0 / 84.7" in tex
    assert tex.index("RealityScan") < tex.index("Meshroom") < tex.index("SuperNormal") < tex.index("photos/rock_077.jpg")
    assert "SfM failed" in tex and "colmap_d4/visualizations" not in tex        # failure: grey box, no image


def test_taxo_page_numbers_its_pages_and_takes_a_view_override():
    zone = {"ids": [f"{i:02d}_o" for i in range(7)], "object": {f"{i:02d}_o": {m: [1, 2, 3] for m in SPEC["methods"]} for i in range(7)}}
    objs = {f"{i:02d}_o": dict(obj_id=f"{i:02d}_o", name=f"o{i}", view=3) for i in range(7)}
    metrics = {(o, f"{m}_d4"): row() for o in objs for m in SPEC["methods"]}
    tex = supp.taxo_pages("specular", zone, objs, SPEC, metrics, views={("specular", "00_o"): 9})
    assert tex.count(r"\clearpage") == 2 and r"\obj{Specular (1/2)}" in tex and r"\obj{Specular (2/2)}" in tex
    assert "vizz/specular/00_o/realityscan_d4/visualizations/accuracy/view_009.png" in tex and "photos/o0_009.jpg" in tex
    assert "vizz/specular/01_o/realityscan_d4/visualizations/accuracy/view_003.png" in tex


def test_pooled_pages_give_the_curves_and_the_pooled_numbers_of_each_axis():
    tex = supp.pooled_pages({"transparent": ZONE, "high_curvature": ZONE}, SPEC)
    assert "taxo/prf_transparent.pdf" in tex and "taxo/prf_high_curvature.pdf" in tex
    assert r"\obj{High curvature}" in tex and tex.count(r"\clearpage") == 1
    assert "tabular" not in tex and "Precision &" not in tex      # no table below the curves
    seven = supp.pooled_pages({a: ZONE for a in supp.TAXO_LABELS}, SPEC)
    assert seven.count(r"\clearpage") == 3                         # 3 axes per page, like part 5


def test_document_adds_the_taxonomy_parts_only_with_zone_scores():
    tex = supp.document([OBJ], SPEC, METRICS, sections=["taxo", "pooled"], zones={"axes": {"transparent": ZONE}})
    assert tex.count(r"\section{") == 2 and "taxo/prf_transparent.pdf" in tex
    assert "taxo" not in supp.document([OBJ], SPEC, METRICS)


def test_zone_views_map_axis_and_object_id_to_a_view(tmp_path):
    f = tmp_path / "views.tsv"
    f.write_text("# axis\tobject\tview\twhy\ntransparent\trock\t20\tseen from the front\n\n")
    assert supp.load_zone_views(f, [OBJ]) == {("transparent", "01_rock"): 20}


def test_main_writes_the_document_and_the_renders_to_convert(tmp_path, monkeypatch, capsys):
    import yaml
    (tmp_path / "spec.yaml").write_text(yaml.safe_dump(SPEC, sort_keys=False))
    cols = ["obj_id", "object", "cell", "status", "f05", "cd"]
    rows = [[o, "rock", c, r["status"], r["f05"], r["cd"]] for (o, c), r in METRICS.items()]
    (tmp_path / "metrics.tsv").write_text("".join("\t".join(r) + "\n" for r in [cols] + rows))
    (tmp_path / "views.tsv").write_text("object\tview\tsource\nrock\t77,3\tpaper\n")
    for var in ["ZONES", "TAXO_VIEWS", "SECTIONS"]:
        monkeypatch.delenv(var, raising=False)
    out = tmp_path / "build"
    out.mkdir()
    supp.main([str(tmp_path / f) for f in ["spec.yaml", "metrics.tsv", "views.tsv"]] + [str(out)])
    tex = (out / "supp.tex").read_text()
    assert tex == supp.jpeg_paths(supp.document([OBJ], SPEC, METRICS))[0]
    pairs = [line.split("\t") for line in (out / "jpeg_pairs.tsv").read_text().splitlines()]
    assert ["vizf/01_rock/realityscan_d4/visualizations/accuracy/view_077.png",
            "jpg/vizf/01_rock/realityscan_d4/visualizations/accuracy/view_077.jpg"] in pairs
    assert "missing" in capsys.readouterr().out
