# Paper figures and tables

[← back to README](../README.md)

`eval-pipeline figures` builds the figures and tables of a paper from the eval outputs. A YAML
specification lists the methods, the configurations that were run and the parameters of each output.

```bash
eval-pipeline -c config/martine.yaml figures --spec config/figures/martine_neurips2026.yaml --out figures_out/
eval-pipeline -c config/martine.yaml figures --spec ... --out figures_out/ --only radar --only tables
```

**The Martine specification is an example.** The reconstructions of the benchmarked methods and their
evaluations are not distributed with the dataset (too heavy). `config/figures/martine_neurips2026.yaml`
is the specification of the Martine paper's figures. To get the same figures, run your methods, evaluate
them with this pipeline, and then point a copy of that file at your own eval outputs.

## Inputs

Everything is found through the eval config. The specification contains no path.

| read from | file | used by |
|---|---|---|
| `<eval_root>/<object>/<cell>/eval_results/` | `metrics.json` | tables, views/resolution |
| same | `distances/{data2gt_dist,gt2data_dist,data2gt_idx}.npy` | PRF curves, radar |
| `<gt_artifacts_root>` (the dataset's `gt/`) | `challenges/<property>.npy`, `challenges/excluded.npy` | radar, every pooled score |
| same | `attributes/{curvature_values,visibility_count}.npy` | radar attribute axes |
| `<eval_root>/<object>/<cell>/<render_dir>/` | renders of `eval-pipeline visualize` | qualitative crops |

A **cell** is one configuration of a method: `<method>_<tier>` (all views) or
`<method>_<tier>_v<views>` (a subset of views at `cells.views_tier`). The directory patterns are
`cells.dir` and `cells.views_dir`.

- A cell listed under `cells.declared` but without `metrics.json` for an object, or whose
  `metrics.json` records an empty reconstruction (`empty_reconstruction`: a mesh with no vertex), is
  a **failure** of the method on that object. It is drawn as a cross, written as `--` in the Chamfer table and listed
  in the failures table. In every pooled score (PRF, radar, F of the Chamfer table's Mean row) it counts as one
  more piece of the single pooled mesh with nothing reconstructed: no data point, and its GT points stay in the pool,
  all missed. It lowers recall and F, not precision; beyond `evaluation.max_dist` its points leave the pooled
  Chamfer distance unchanged.
- A cell that is not declared was **not run**. It is written as `--` in the coverage table and not
  drawn, and it is never counted as a failure.
- The GT points of `challenges/excluded.npy` are removed from every pooled score through
  `core.masking`, as in `evaluate`.

## Outputs

| section | output | what it shows |
|---|---|---|
| `prf` | `prf_curves.pdf` | precision, recall and F-score against the threshold τ, pooled over all objects |
| `radar` | `radar_chart_*.pdf` | F-score at one threshold per surface property and per attribute range |
| `resolution_views` | `resolution_views.pdf` | F-score of one object against the number of views and the image resolution |
| `tables.coverage` | `.tex` | objects reconstructed per method and configuration |
| `tables.chamfer` | `.tex` | Chamfer distance per object and method × resolution (best in bold, second underlined); optionally the F-score at one threshold (`fscore`), as 'CD / F' cells or a second block (`layout`), each metric ranked on its own; `method_tiers` restricts a method's tiers; `mean: pooled` pools the objects' point distances for the Mean row; `split` cuts the methods into sub-tables, one tabular each stacked in an invisible tabular (one scale; cell layout; ranks over the whole row); `centred` lists the single-tier methods centred under their name (default all); `sideways` for a landscape page; a list gives several versions |
| `tables.failures` | `.tex` | failed configurations and their objects |
| `qualitative` | `qualitative/<object>_<method>.png`, colorbar PDF | error renders padded for the paper |

`numbers.json` keeps every value behind the figures at full precision: the curves, the radar values
and the objects pooled on each axis, and the views/resolution points.

**Pooling.** The point-level distances of all objects are concatenated before counting, so each
object weighs by its number of points. A point counts when its distance is strictly below the
threshold. The ratios are computed in float64.

**Radar axes.** An object counts on a property axis if and only if its GT has
`challenges/<property>.npy`. An object whose whole surface has the property ships a mask of ones.
Three cases are errors: a mask name that is not a declared property, an object without
`challenges/`, and a missing file that a pool expects. An attribute axis selects GT points with
`lt` (strict upper bound) and/or `ge`/`le` (closed bounds) on `attributes/<attribute>.npy`, or on its
absolute value with `abs: true`.

**Qualitative renders** are made beforehand with `visualize`, for each object × method cell. The
command for the Martine paper is in the comment above the `qualitative` section of the
specification.

## Style

`style` holds matplotlib rcParams applied to every figure, and a figure can override them (`prf.rc`).
Each method has a colour, a marker, a line style and `hollow` (white marker face). The Martine
example uses Computer Modern, one hue per family, and a dashed line with hollow markers for the second
method of a family.

## Non-regression

`tests/figures/test_paper_regression.py` rebuilds the Martine paper's outputs and compares them with
the published ones: numbers equal to 1e-9, `.tex` tables identical byte for byte, qualitative crops
identical pixel for pixel. It needs the eval outputs of the methods, so it is skipped unless
`EVAL_FIGURES_REF`, `EVAL_FIGURES_CONFIG` and `EVAL_FIGURES_VIZ_CONFIG` are set (see its docstring).

## Supplementary material (per-object report)

`eval3d/figures/supplementary/` holds the tools that built the per-object supplementary material of the Martine
paper: one PDF with the reconstructions of every method on every object, their error maps and their scores. It reuses
the figures specification (methods, order, colours, declared cells) and the code of the figures above. The tools
produce images, numbers and one `supp.tex`; the LaTeX class file (NeurIPS style, `neurips_2026.sty`) is not shipped
and must be put next to `supp.tex` before compiling.

**This is the tooling of one document, not a general report generator.** The page layout, the front-page text
(26 objects, 84 views) and the layout of the dataset read by `photos` are those of Martine. To reuse it, start from
`document.py`.

### Content

| part | pages | source |
|---|---|---|
| 1. Per-object results at quarter resolution | one page per object, one row per method: reconstruction, accuracy, completeness (and normals, angular error with `normals`), scores below, reference row | `visualize` renders, `photos` |
| 2. Image resolution | one page per object: mesh \| accuracy montage of each method of `resolution_views` at each tier | `montage` |
| 3. Number of views | same, for each number of input views at `cells.views_tier` | `montage` |
| 4. P/R/F against resolution and views | one figure per object, at one threshold | `curves` |
| 5. P/R/F against the threshold | one figure per object, then all objects pooled (the PRF figure of the paper) | `prf_objects` |
| 6. Surface properties | per property of the radar: accuracy maps restricted to the zone, for each object that has it, with the P/R/F of the zone | `zones` |
| 7. Surface properties, pooled | per property: P/R/F curves pooled over the zone of its objects | `zones scores`, `plot_zones` |

Parts 6 and 7 are added when the zone scores are given. A failed cell is a grey box with its reason; a cell that was
not run is left empty.

### Tools

Each one runs with `python -m eval3d.figures.supplementary.<tool>`; its docstring gives the arguments and the files
read and written.

| tool | what it does |
|---|---|
| `tasks` | lists, per object, its reference view(s) and its evaluated cells (TSV driving the render loop) |
| `zones render` | accuracy map of one cell with everything outside a zone of the radar greyed |
| `pick_view` | ranks the rendered views of a cell by visible zone area, to choose the view of a zone |
| `zones scores` | P/R/F per zone, per object and pooled, and the pooled curves (`zone_scores.json`) |
| `plot_zones` | pooled PRF curves of each zone, drawn like the PRF figure |
| `curves` | P/R/F of each object against image resolution and number of views |
| `prf_objects` | PRF curves of each object; with `mean`, of all objects pooled |
| `photos` | reference photos cropped to the frame of the renders, Uni-MS-PS normal maps, colorbars |
| `montage` | one image per cell: mesh on the left, accuracy map on the right, oblique cut |
| `document` | writes `supp.tex` and the list of PNG renders to convert |
| `jpeg` | converts those renders to JPEG (lighter PDF) |

### Steps

`document` reads everything in one build directory, under fixed names (directories or symbolic links):

| name | content | written by |
|---|---|---|
| `vizf/<object>/<cell>/visualizations/{uniform,accuracy,completeness}/` | mesh and distance maps | `visualize` |
| `vizn/<object>/<cell>/visualizations/{normals,angular_error}/` | normal maps and `mae.json` | `visualize` |
| `angular/<object>/<cell>/visualizations/angular_error/` | angular error in magma | `visualize` |
| `vizz/<axis>/<object>/<cell>/visualizations/accuracy/` | zone renders | `zones render` |
| `photos/`, `montage/`, `curves/`, `prf/`, `taxo/` | see the tools | `photos`, `montage`, `curves`, `prf_objects`, `plot_zones` |

`visualize` writes in `<eval_root>/<object>/<cell>/visualizations/`, so each of `vizf`, `vizn`, `angular` and
`vizz/<axis>` is the `paths.eval_root` of its own copy of the eval config (for Martine: a root of symbolic links to
the evaluated cells, so that nothing is written in the eval outputs). `Groundtruth` is rendered like a cell.

```bash
S="python -m eval3d.figures.supplementary"; SPEC=config/figures/martine_neurips2026.yaml; B=build

# 1. Task list: eval name, published name, reference view(s), evaluated cells
$S.tasks <eval_root> config/martine.yaml views.tsv tasks.tsv

# 2. Renders at the reference view, excluded regions in gray, no colorbar on the images; for each object and cell
eval-pipeline -c vizf.yaml visualize -o <object> -m <cell> --metrics uniform,accuracy,completeness --views <view> \
    --cmap viridis --max-dist 4 --exclude-mode gray --no-integrated-colorbar --force
eval-pipeline -c vizn.yaml visualize -o <object> -m <cell> --metrics normals,angular_error --views <view> \
    --exclude-mode gray --no-integrated-colorbar --force
eval-pipeline -c angular.yaml visualize -o <object> -m <cell> --metrics angular_error --views <view> \
    --cmap magma --exclude-mode gray --no-integrated-colorbar --force

# 3. Zone renders: each axis of the radar, each object that has it, each method at radar.tier (and Groundtruth)
$S.zones render vizz_<axis>.yaml $SPEC <object> <cell> <axis> <view> vizz/<axis>/_masks
$S.pick_view vizz/<axis>/<object>/<cell>/visualizations/accuracy     # optional: after rendering all views of a cell

# 4. Scores and curves (the eval config with the real eval_root)
$S.zones scores config/martine.yaml $SPEC $B/taxo/zone_scores.json
$S.plot_zones $SPEC $B/taxo/zone_scores.json $B/taxo
$S.curves config/martine.yaml $SPEC $B/curves 0.5
$S.prf_objects config/martine.yaml $SPEC $B/prf
$S.prf_objects config/martine.yaml $SPEC $B/prf mean

# 5. Photos and montages
$S.photos <dataset_root> $B/vizf views.tsv $B/photos --tasks tasks.tsv --normals-root <unimsps_root> \
    --zone-views zone_views.tsv --zone-root $B/vizz
$S.montage $B/vizf visualizations $B/montage

# 6. Document, JPEG conversion, compilation
ZONES=$B/taxo/zone_scores.json TAXO_VIEWS=zone_views.tsv $S.document $SPEC metrics.tsv views.tsv $B normals
$S.jpeg $B
cp <path>/neurips_2026.sty $B/ && (cd $B && pdflatex supp.tex && pdflatex supp.tex)
```

`views.tsv` (`object`, `view`, `source`) gives the reference view of each object by its published name, and
`zone_views.tsv` (`axis`, `object`, `view`) the zones shown from another view. `metrics.tsv` has one row per object
and cell: `obj_id`, `object`, `cell`, `status` (`OK`, `SPARSE_FAIL`, `EMPTY`, `MISSING`), `f05`, `cd`. `document`
prints the referenced files that are missing. The threshold of part 4 is `CURVES_THRESHOLD` in `document.py`
(0.5 mm): pass the same value to `curves`. The renders and `zones render` need an OpenGL backend; the other tools
do not.
