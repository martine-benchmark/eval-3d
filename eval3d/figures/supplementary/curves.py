"""Precision, recall and F-score of each object against image resolution (top row) and number of views (bottom row).

Methods, views and tiers are those of the `resolution_views` section of the specification; the style and the one-line
legend below are those of its PRF figure. A failed cell is drawn as a cross at 0.

Usage: python -m eval3d.figures.supplementary.curves <eval_config.yaml> <figures_spec.yaml> <out_dir> [threshold_mm]
The threshold defaults to `resolution_views.threshold`; it must be one of the thresholds of metrics.json.
Writes <out_dir>/curves_<published object>.pdf (+ .png) and curves_prf.json (numbers).
"""

import json
import sys
from pathlib import Path

KEYS = ["precision", "recall", "fscore"]


def prf_at(metrics, threshold):
    """(precision, recall, F) in % at the given threshold, or None for a cell without result (failure)."""
    if metrics is None:
        return None
    i = metrics["thresholds"].index(threshold)
    return tuple(100.0 * float(metrics[k][i]) for k in KEYS)


def series(methods, declared, metrics_of, tiers, views, views_tier, threshold):
    """{method: {"res": [(tier index, prf or None)], "views": [(n views, prf or None)]}} over the declared cells.
    declared(method, tier, views) -> bool; metrics_of(method, tier, views) -> metrics.json or None."""
    out = {}
    for m in methods:
        res = [(i, prf_at(metrics_of(m, t, None), threshold)) for i, t in enumerate(tiers) if declared(m, t, None)]
        has_views = any(declared(m, views_tier, v) for v in views[:-1])
        vs = [(v, prf_at(metrics_of(m, views_tier, v), threshold)) for v in views if has_views and declared(m, views_tier, v)]
        out[m] = {"res": res, "views": vs}
    return out


def plot(spec, data, path, threshold):
    import matplotlib
    import matplotlib.pyplot as plt

    from ..style import apply_style, method_style
    s, rv = spec.prf, spec.resolution_views
    apply_style(spec, s.rc)
    width = s.figsize[0]
    height = 5.6
    fig, axes = plt.subplots(2, 3, figsize=(width, height))
    methods = list(data)
    spread = {m: (i - (len(methods) - 1) / 2) * 0.05 for i, m in enumerate(methods)}
    titles = ["Precision (%)", "Recall (%)", "F-score (%)"]
    for r, kind in enumerate(["res", "views"]):
        for c in range(3):
            ax = axes[r, c]
            for m in methods:
                style = method_style(spec, m, markersize=4.2, markeredgewidth=0.9)
                pts = data[m][kind]
                ok = [(x, p[c]) for x, p in pts if p is not None]
                if ok:
                    ax.plot(*zip(*ok), label=spec.methods[m].display if (r, c) == (0, 0) else None, **style)
                fail = [x + spread[m] if kind == "res" else x * (1 + spread[m]) for x, p in pts if p is None]
                if fail:
                    ax.plot(fail, [0] * len(fail), linestyle="none", marker="x", markersize=5, markeredgewidth=1.2,
                            color=style["color"], clip_on=False, zorder=5)
            if kind == "views":
                ax.set_xscale("log")
                ax.set_xticks(rv.views)
                ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
                ax.minorticks_off()
                ax.set_xlim(4, 100)
            else:
                ax.set_xticks(range(len(rv.tiers)))
                ax.set_xticklabels([rv.tier_labels.get(t, t) for t in rv.tiers])
                ax.set_xlim(-0.3, len(rv.tiers) - 0.7)
            ax.set_ylim(0, 100)
            ax.set_yticks(range(0, 101, 25))
            if c > 0:
                ax.set_yticklabels([])
            ax.grid(True, linestyle=":", alpha=0.8)
            ax.tick_params(axis="both", labelsize=7, pad=1)
            if r == 0:
                ax.set_title(titles[c])
        axes[r, 1].set_xlabel("Image resolution" if kind == "res" else "Number of views (log scale)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    cross, = axes[0, 0].plot([], [], linestyle="none", marker="x", color="#555555")
    present = {spec.methods[m].display for m in methods if data[m]["res"] or data[m]["views"]}
    keep = [(h, name) for h, name in zip(handles, labels) if name in present] + [(cross, "Failure")]
    fig.legend(*zip(*keep), loc="lower center", ncol=len(keep), frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.01),
               columnspacing=1.2, handletextpad=0.4)
    plt.subplots_adjust(wspace=0.08, hspace=0.42, left=0.06, right=0.99, top=1 - 0.22 / height, bottom=0.75 / height)
    fig.savefig(path)
    fig.savefig(Path(path).with_suffix(".png"), dpi=150)
    plt.close(fig)


def main(argv=None):
    from ...config import load_config
    from ..cells import Results
    from ..spec import Cell, load_spec
    argv = sys.argv[1:] if argv is None else argv
    config, spec, out = load_config(argv[0]), load_spec(argv[1]), Path(argv[2])
    out.mkdir(parents=True, exist_ok=True)
    results, rv, alias = Results(config, spec), spec.resolution_views, config.dataset.object_aliases
    threshold = float(argv[3]) if len(argv) > 3 else rv.threshold
    numbers = {}
    for obj in spec.object_list(config):
        pub = alias.get(obj, obj)
        data = series(list(rv.methods), lambda m, t, v: spec.is_declared(Cell(m, t, v)),
                      lambda m, t, v, obj=obj: results.metrics(obj, Cell(m, t, v)),
                      rv.tiers, rv.views, spec.cells.views_tier, threshold)
        plot(spec, data, out / f"curves_{pub}.pdf", threshold)
        numbers[pub] = data
        print(f"== {pub}", flush=True)
    (out / "curves_prf.json").write_text(json.dumps({"threshold": threshold, "objects": numbers}))
    print("OK", len(numbers))


if __name__ == "__main__":
    main()
