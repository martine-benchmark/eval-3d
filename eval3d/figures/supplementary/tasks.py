"""Render task list: one row per object (eval name, published name, views, evaluated cells separated by commas).

Usage: python -m eval3d.figures.supplementary.tasks <eval_root> <eval_config.yaml> <views.tsv> <out.tsv>

- ``eval_root``: directory holding ``<object>/<cell>/`` (the eval outputs). A cell is listed when it has
  ``eval_results/`` and a non-empty ``results_cleaned/``.
- ``eval_config.yaml``: its ``dataset.object_aliases`` maps eval names to published names.
- ``views.tsv``: header line, then ``published object<TAB>view[,view...]<TAB>source`` per object.

The TSV is read by a render loop (one `eval-pipeline visualize` per cell) and by `photos`.
"""

import sys
from pathlib import Path


def main(argv=None):
    import yaml
    argv = sys.argv[1:] if argv is None else argv
    eval_root, cfg, views_tsv, out = Path(argv[0]), argv[1], argv[2], argv[3]
    with open(cfg) as f:
        alias = yaml.safe_load(f)["dataset"]["object_aliases"]
    views = {}
    for line in Path(views_tsv).read_text().splitlines()[1:]:
        name, view, source = line.split("\t")
        views[name] = (view, source)
    rows = []
    for obj, pub in alias.items():
        cells = sorted(d.name for d in (eval_root / obj).iterdir()
                       if d.name != "Groundtruth" and (d / "eval_results").exists()
                       and any((d / "results_cleaned").glob("*")))
        view, source = views[pub]
        rows.append("\t".join([obj, pub, view, ",".join(cells)]))
        print(f"{pub:18s} view {view:6s} {len(cells):2d} cells  ({source})")
    Path(out).write_text("\n".join(rows) + "\n")
    print("total cells", sum(len(r.split("\t")[3].split(",")) for r in rows))


if __name__ == "__main__":
    main()
