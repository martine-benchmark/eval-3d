# Running on a cluster

Running the stages as SLURM jobs from the bundled templates.

[← back to README](../README.md)

Set `execution.mode: "slurm"` and fill `execution.slurm.account` / `partition`. Job templates in
`slurm/templates/` (`cleanup.sh.j2`, `eval.sh.j2`) are fully parameterised: nothing
site-specific is baked into `eval3d/`.

Machine-specific runtime setup (for example a headless OSMesa library and the C++ runtime it needs
on `LD_LIBRARY_PATH`, for `visualize`) goes in a shell script named by
`execution.slurm.setup_script`; every job of `run --slurm` and of the watcher sources it first. Keep that script out of git:
copy `config/setup_env.example.sh` to `config/setup_env.local.sh` (git-ignored) and point your
`*.local.yaml` to it.

Dispatch the whole flow with `run --slurm`, or submit individual stages. `run --slurm` submits one
`cleanup` job and one `evaluate` job (after it) per object and method, then waits for them; the jobs
activate the virtualenv that submitted them, the scripts go to `<output_root>/slurm_scripts/` and
the logs to `<output_root>/slurm_logs/`. With `--views`, one more job per object derives `Groundtruth@<tag>` and the `evaluate` jobs wait for
it too; scripts and logs go to `slurm_scripts@<tag>/` and `slurm_logs@<tag>/`. It reads the templates from the source tree, so it needs
an editable install (`pip install -e .`). `watch` polls for new
meshes and submits jobs as they appear; `watch-status` reports on them.
