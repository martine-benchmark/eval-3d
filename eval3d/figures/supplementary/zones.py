"""Surface properties: scores and renders restricted to a zone (the axes of the paper's radar).

Same rules as the radar (eval3d.figures.radar): an object has a property if its GT has challenges/<property>.npy;
the attribute axes (curvature, visibility) select GT points by attribute; the `excluded` points are always removed; a
failed object keeps its GT points in the pool, all missed. Methods are those of `prf.methods`, at `radar.tier`.

Usage:
  python -m eval3d.figures.supplementary.zones scores <eval_config.yaml> <figures_spec.yaml> <out.json>
  python -m eval3d.figures.supplementary.zones render <eval_config.yaml> <figures_spec.yaml> <object> <cell> <axis>
      <view[,view...]> <mask_dir>

`scores` writes, per axis: the objects, the precision / recall / F (%) of each object and method at
`radar.threshold`, the pooled values and the pooled curves over the thresholds of the PRF figure.
`render` writes the accuracy map of one cell with everything outside the zone greyed (viridis, 0-4 mm, no colorbar)
in <eval_root>/<object>/<cell>/visualizations/accuracy/, like `eval-pipeline visualize --exclude-mode gray` with the
exclusion mask replaced; the mask is saved as <mask_dir>/<object>_<axis>.npy. It needs an OpenGL backend. Point
`paths.eval_root` at a separate root per axis: every axis writes to the same file names.
"""

import json
import sys
from pathlib import Path

import numpy as np


def override(zone, excluded):
    """Exclusion mask of the render: everything outside the zone, plus the excluded points of the GT."""
    out = ~np.asarray(zone, bool)
    return out if excluded is None else out | np.asarray(excluded, bool)


def prf_at(data2gt, gt2data, threshold):
    """(precision, recall, F) in %, distance strictly below the threshold; without reconstructed point, precision 0."""
    p = float(np.mean(data2gt < threshold)) if len(data2gt) else 0.0
    r = float(np.mean(gt2data < threshold)) if len(gt2data) else 0.0
    f = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return 100 * p, 100 * r, 100 * f


def pool(pairs):
    """Concatenate the distances of several objects: an object weighs by its number of points."""
    d = [d for d, _ in pairs if len(d)]
    g = [g for _, g in pairs if len(g)]
    return (np.concatenate(d) if d else np.empty(0)), (np.concatenate(g) if g else np.empty(0))


def axes(config, spec, objects):
    """[(axis, objects, selection(object) -> boolean mask of the GT points)], in the order of the radar."""
    from .. import radar
    s = spec.radar
    props = radar.discover_properties(config, objects, s.properties)
    out = [(p, props[p], lambda obj, p=p: np.load(radar._gt_file(config, obj, "challenges", p)).astype(bool))
           for p in s.properties]
    out += [(a.key, list(objects), lambda obj, a=a: radar._attribute_mask(
        a, np.load(radar._gt_file(config, obj, "attributes", a.attribute)))) for a in s.attributes]
    return out


def scores(config, spec, out):
    from ..cells import Cell, Results
    from ..pooling import load_distances, missed_gt, prf_curves
    results = Results(config, spec)
    t = spec.radar.threshold
    taus = np.linspace(0.0, spec.prf.max_threshold, spec.prf.n_thresholds)
    alias = config.dataset.object_aliases
    res = {"threshold": t, "thresholds": taus.tolist(), "methods": list(spec.prf.methods), "axes": {}}
    for key, objects, select in axes(config, spec, results.objects):
        entry = {"objects": [alias.get(o, o) for o in objects], "ids": list(objects), "points": {}, "object": {},
                 "pooled": {}, "curves": {}}
        masks = {o: select(o) for o in objects}
        for m in spec.prf.methods:
            cell, pairs = Cell(m, spec.radar.tier), []
            for o in objects:
                if results.ok(o, cell):
                    pair = load_distances(config, o, spec.cell_dir(cell), select=masks[o])
                    entry["object"].setdefault(o, {})[m] = prf_at(*pair, t)
                else:
                    pair = missed_gt(config, o, select=masks[o])
                    entry["object"].setdefault(o, {})[m] = None
                entry["points"][o] = len(pair[1])
                pairs.append(pair)
            d, g = pool(pairs)
            entry["pooled"][m] = prf_at(d, g, t)
            p, r, f = prf_curves(d, g, taus) if len(d) else (np.zeros(len(taus)),) * 3
            entry["curves"][m] = {"precision": (100 * p).tolist(), "recall": (100 * r).tolist(),
                                  "fscore": (100 * f).tolist()}
            print(f"{key:15s} {m:13s} F={entry['pooled'][m][2]:.2f}", flush=True)
        res["axes"][key] = entry
    Path(out).write_text(json.dumps(res))


def render(config, spec, obj, cell, key, view, mask_dir):
    from ...core.masking import load_gt_exclude
    from ...pipeline.visualize import visualize_method
    select = dict((k, s) for k, _, s in axes(config, spec, [obj]))[key](obj)
    excluded, _ = load_gt_exclude(config.get_gt_artifacts_dir(obj), len(select), verbose=False)
    path = Path(mask_dir) / f"{obj}_{key}.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, override(select, excluded))
    visualize_method(config, obj, cell, metrics=["accuracy"], view_indices=[int(v) for v in str(view).split(",")],
                     cmap_overrides={"accuracy": "viridis"}, max_dist=4.0, force=True, exclude_mode="gray",
                     integrated_colorbar=False, excluded_override_path=path)


def main(argv=None):
    from ...config import load_config
    from ..spec import load_spec
    argv = sys.argv[1:] if argv is None else argv
    mode, config, spec = argv[0], load_config(argv[1]), load_spec(argv[2])
    if mode == "scores":
        scores(config, spec, argv[3])
    elif mode == "render":
        render(config, spec, *argv[3:8])
    else:
        raise SystemExit(f"unknown mode {mode!r}: scores or render")
    print("OK", mode)


if __name__ == "__main__":
    main()
