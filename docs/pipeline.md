# Running the pipeline end to end

The command sequence from a GT mesh and raw reconstructions to `metrics.json` and curves, and the other
stages.

[← back to README](../README.md)

```bash
CFG=config/martine.yaml     # or config/skoltech3d.yaml

# 1. Ground truth: clean the mesh, sample it, compute per-point attributes.
#    Writes Groundtruth/{gt_cleaned.ply,gt_pcd.npy,gt_pcd_provenance.json,attributes/}.
#    Skip it for the published Martine dataset: its gt/ kit already ships these files
#    and config/martine.yaml reads them in place (paths.gt_artifacts_root).
eval-pipeline -c $CFG preprocess-gt --clean-gt

# 2. Carve reconstructed vertices falling outside every silhouette.
#    Reads results_raw/mesh.ply, writes results_cleaned/mesh.ply.
eval-pipeline -c $CFG cleanup

# 3. Distances, Chamfer, F-score at the reporting thresholds.
#    Writes eval_results/metrics.json and eval_results/distances/.
eval-pipeline -c $CFG evaluate

# 4. Dense PR curves from the saved distances (no re-query).
#    Writes eval_results/curves/.
eval-pipeline -c $CFG curves -n 100

# Optional: rebuild metrics.json + curves from saved distances, e.g. after changing
# max_dist or adding a mask. Never re-runs the nearest-neighbour search.
eval-pipeline -c $CFG recompute
```

Scope any stage with `-o/--object` and `-m/--method`; re-run with `-f/--force`.

`preprocess-gt --clean-gt` removes ground-truth points that **no camera sees**, and resamples
from the cleaned surface. For a solid object this changes almost nothing; for an open,
foliage-like object it removes the entire unobservable interior. That is intended (scoring a
reconstruction on surfaces the rig never observed measures nothing), but it means GT point
counts are not comparable across the change. `gt_pcd_provenance.json` records which mesh the
cloud came from, the sampling density and the resulting count.

## Other stages

`eval-pipeline --help` lists every subcommand. Beyond the core flow above:

- `run`: orchestrate cleanup → evaluate → curves over all objects and methods in one call
  (`--slurm` to dispatch on a cluster; see [running on a cluster](cluster.md)).
- `aggregate`: collect per-cell `metrics.json` into summary tables across objects and methods.
- `latex`: emit a LaTeX results table from the aggregated metrics.
- `figures`: build a paper's figures and tables from a YAML specification ([figures.md](figures.md)).
- `visualize`: render per-point metric colourings onto the reconstruction (needs OSMesa; see
  the visualization extra in [install](../README.md#install)).
- `apply-masks`, `preprocess-challenges`: build and apply the optional
  per-point challenge masks described in [configuration](configuration.md#masks).

## Sparse-view evaluation

A reconstruction made from a few views (3, 6, n) is scored on the surface these views see. Give
`run` the SfMData file of the views that were used:

```bash
eval-pipeline -c cfg.yaml run -o rock -m my_method                               # usual evaluation
eval-pipeline -c cfg.yaml run -o rock -m my_method --views sparse/front3.sfm     # sparse-view evaluation
```

Without `--views`, nothing changes. With `--views`, the cleaning is done in two steps:

1. **Usual cleanup**, as without `--views`: all the evaluation views, their silhouettes and, for
   the objects that have them, the watertight masks. Output: `results_cleaned/` (reused when it
   exists).
2. **Second cleanup of the reconstruction** on the custom views: the same visibility filter
   (`filter_by_visibility`, same dilation) applied to the mesh of step 1 with the cameras and
   silhouettes of the custom file. Output: `results_cleaned@<tag>/` (`views_provenance.json`
   records the file and the settings).
3. **Same second cleanup of the GT point cloud**: the same call on the published `gt_pcd.npy`
   (occluder: the GT mesh). Output: `Groundtruth@<tag>/`, with every per-point array of
   `attributes/` and `challenges/` subset by the same mask, `keep.npy` (indices into the published
   cloud) and `variant_provenance.json`. Nothing is resampled: the attributes, including
   `visibility_count`, keep their values on the full view set.
4. **Evaluation** of `results_cleaned@<tag>/` against `Groundtruth@<tag>/`. Output:
   `eval_results@<tag>/`.

**The custom file.** An SfMData file (`.sfm` or `.json`) in the frame of the evaluation cameras,
with one view per camera. The silhouette of a view is `<dir of the file>/masks/<stem>.png`, where
`<stem>` is the file name of the view's `path` without its extension; a view without its silhouette
is an error.

**The tag** is the file name without its extension (`front3.sfm` gives `@front3`); `--variant <tag>`
sets another one. One tag per view set: an existing `@<tag>` output made from another file or other
settings is refused, `--force` (on `cleanup` and `preprocess-gt`) rebuilds it.

**Watertight masks are applied at step 1 only.** They mark the holes of the GT (e.g. the inside of
a shoe) and only remove vertices of the reconstruction. Step 1 has removed them, so the custom view
set needs no watertight masks, and the GT, which has no surface there, is not concerned.

**The GT has no step 1**: the published `gt_pcd.npy` is already cleaned with all the evaluation
views. Control: a custom file holding all the evaluation views gives the result of the usual
evaluation, for the reconstruction and for the GT.

`--views` (and `--variant`) are also global options, for the commands run one by one:

```bash
V="--views sparse/front3.sfm"
eval-pipeline -c cfg.yaml $V preprocess-gt -o rock           # step 3
eval-pipeline -c cfg.yaml $V cleanup -o rock -m my_method    # steps 1 and 2
eval-pipeline -c cfg.yaml $V evaluate -o rock -m my_method   # step 4
```

`apply-masks`, `curves`, `recompute`, `visualize` and `watch` then read and write the tagged
directories. `aggregate`, `latex`, `figures`, `proxy-curvature` and `preprocess-challenges` refuse
`--views`. The config keys are `paths.views` and `variant`.
