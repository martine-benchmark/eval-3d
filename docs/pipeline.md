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

## Evaluation variants: a custom camera set

To score a reconstruction only on the surface seen by a subset of views (e.g. a sparse-view
reconstruction from 3 or 6 input views), give the two visibility cleanings another camera file:

```bash
V="--eval-cameras presets/front3/cameras.sfm --variant front3"
eval-pipeline -c cfg.yaml $V preprocess-gt -o rock      # Groundtruth@front3/
eval-pipeline -c cfg.yaml $V run -o rock -m my_method   # results_cleaned@front3/, eval_results@front3/
```

The two options go together (or `paths.eval_cameras_override` and `variant` in the config).
The custom file is an SfMData file; its masks are read from `<its dir>/masks/` by view name, its
watertight masks from `<its dir>/masks_watertight/` (by view name too) when that directory
exists, otherwise there is no watertight culling.

- `cleanup` carves the reconstruction with the custom views (`results_raw/` is shared).
- `preprocess-gt` only derives `Groundtruth@<tag>/`: the base kit's `gt_pcd.npy` restricted to
  the points seen by at least one custom view (ray casting on the GT mesh, masks dilated as in
  `cleanup`), with every per-point array of `attributes/` and `challenges/` subset by the same
  mask, `keep.npy` (indices into the base cloud) and `variant_provenance.json`. Nothing is
  resampled: the attributes, including `visibility_count`, keep their values on the default set.
- `evaluate`, `apply-masks`, `curves`, `recompute`, `visualize` and `watch` read and write the
  tagged directories; the default tree is never written. `aggregate`, `latex`, `figures`,
  `proxy-curvature` and `preprocess-challenges` refuse variant mode.
