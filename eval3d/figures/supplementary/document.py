"""LaTeX generator of the per-object supplementary material (the one of the Martine paper).

Parts: per-object pages at the views tier, image resolution, number of views, P/R/F against resolution and views,
P/R/F against the threshold, then (with zone scores) surface properties per property and pooled.

Image paths are relative to the build directory, which holds the inputs under fixed names (directories or links):
``vizf/`` (uniform, accuracy and completeness renders), ``vizn/`` (normals, with ``angular_error/mae.json``),
``angular/`` (angular error), ``vizz/<axis>/`` (zone renders), ``photos/``, ``montage/``, ``prf/``, ``curves/``,
``taxo/``. The methods, their order, their colours and the declared cells come from the figures specification.

Usage: python -m eval3d.figures.supplementary.document <spec.yaml> <metrics.tsv> <views.tsv> <build_dir> [normals]

- ``metrics.tsv``: one row per object and cell, columns ``obj_id``, ``object`` (published name), ``cell``,
  ``status`` (OK, SPARSE_FAIL, EMPTY or MISSING), ``f05`` (F-score at 0.5 mm, in [0, 1]) and ``cd`` (mm).
- ``views.tsv``: header ``object<TAB>view``, one row per object; the first view listed is its reference view.
- ``normals``: five-column object pages, with the normal renders read in ``<build_dir>/vizn`` and the MAE below.

Environment:
- ``ZONES``: zone scores (``zones scores``); adds the two surface-property parts.
- ``TAXO_VIEWS``: TSV ``axis<TAB>published object<TAB>view`` of the views that differ from the reference view.
- ``SECTIONS``: ``+``-separated subset of the parts (object, tiers, views, curves, prf, taxo, pooled).

Writes ``<build_dir>/supp.tex`` and ``jpeg_pairs.tsv`` (the PNG renders to convert, see ``jpeg``), and prints the
referenced files that are missing.
"""

import csv
import json
import os
import re
import sys
from pathlib import Path

FAIL = {"SPARSE_FAIL": "SfM failed", "EMPTY": "empty mesh", "MISSING": "run failed"}
CBAR = r"\cbar{photos/colorbar.png}"
CBARS = r"\cbars{photos/colorbar.png}{photos/colorbar_angular.png}"
COLS_N = [("Normals", "normals"), ("Angular error", "angular_error")]
COLS_B = [("Reconstruction", "uniform"), ("Accuracy", "accuracy"), ("Completeness", "completeness")]

PRE = r"""\documentclass{article}
\usepackage[eandd, final]{neurips_2026}
\usepackage[utf8]{inputenc}\usepackage[T1]{fontenc}
\usepackage{booktabs,array,graphicx,xcolor,microtype,caption}
\usepackage[colorlinks,linkcolor=black]{hyperref}
\newcommand{\obj}[1]{\textsc{#1}}
\newcommand{\martine}{\textsc{Martine}}
\setlength{\fboxsep}{0pt}\setlength{\fboxrule}{0.3pt}
\newcommand{\mth}[2]{\textcolor[HTML]{#2}{\rule{5pt}{5pt}}\,#1}
\newcommand{\mthh}[2]{{\color[HTML]{#2}\setlength{\fboxrule}{1.2pt}\fbox{\rule{0pt}{2.6pt}\rule{2.6pt}{0pt}}}\,#1}
\makeatletter\renewcommand{\@noticestring}{}\makeatother
% \cell{width}{height}{content}: box of fixed height, content centred both ways
\newcommand{\cell}[3]{\parbox[c][#2][c]{#1}{\centering #3}}
\newcommand{\failbox}[3]{\fcolorbox{black!25}{black!4}{\parbox[c][#2][c]{#1}{\centering\scriptsize\textit{#3}}}}
\newcommand{\cbar}[1]{\par\smallskip{\centering\scriptsize Distance: 0\,mm\ \raisebox{-1pt}{\includegraphics[width=0.4\textwidth,height=5pt]{#1}}\ $\geq$\,4\,mm\par}}
\newcommand{\cbars}[2]{\par\smallskip{\centering\scriptsize Distance: 0\,mm\ \raisebox{-1pt}{\includegraphics[width=0.22\textwidth,height=5pt]{#1}}\ $\geq$\,4\,mm\hspace{2.5em}Angular error: 0$^\circ$\ \raisebox{-1pt}{\includegraphics[width=0.22\textwidth,height=5pt]{#2}}\ $\geq$\,45$^\circ$\par}}
\newcolumntype{M}[1]{>{\centering\arraybackslash}m{#1}}
\title{\martine{}: per-object results\\[3pt]\large Supplementary material --- all objects, all methods}
\author{}
\begin{document}
\maketitle
"""


def scores(row, stacked=False):
    f, cd = rf"F@0.5 = {100 * float(row['f05']):.1f}", rf"CD = {float(row['cd']):.2f}\,mm"
    return rf"\shortstack{{{f}\\{cd}}}" if stacked else rf"{f} \textbar{{}} {cd}"


def fail_label(status):
    return FAIL[status]


def cell_name(method, tier, views, spec):
    return f"{method}_{tier}" if views == spec["cells"]["full_views"] else f"{method}_{tier}_v{views}"


# Root of the renders of each map (in the build directory): mesh and distance maps in vizf, unless listed here.
ROOTS = {"normals": "vizn", "angular_error": "angular"}


def image(obj_id, cell, metric, view):
    return f"{ROOTS.get(metric, 'vizf')}/{obj_id}/{cell}/visualizations/{metric}/view_{view:03d}.png"


def label(method, spec, h):
    m = spec["methods"][method]
    name = rf"\{'mthh' if m.get('hollow') else 'mth'}{{{m['display']}}}{{{m['color'].lstrip('#')}}}"
    return side(name, h)


def side(text, h):
    return rf"\cell{{0.05\textwidth}}{{{h}}}{{\rotatebox{{90}}{{\scriptsize {text}}}}}"


def graphic(path, w, h):
    return rf"\cell{{{w}}}{{{h}}}{{\includegraphics[width={w},height={h},keepaspectratio]{{{path}}}}}"


def head(name):
    return rf"\noindent\obj{{{name.replace('_', ' ')}}}\par\smallskip" + "\n"


def tabular(ncols, w):
    return r"\noindent\begin{tabular}{@{}M{0.05\textwidth}" + rf"@{{\hspace{{2pt}}}}M{{{w}}}" * ncols + "@{}}\n"


def object_page(obj, spec, metrics, mae=None, w=None, h="1.6cm"):
    """One row per method at the views tier. With `mae` ({(object, cell): degrees}), two more columns: normals and
    angular error, with the MAE below."""
    oid, view, tier = obj["obj_id"], obj["view"], spec["cells"]["views_tier"]
    cols = COLS_B + (COLS_N if mae is not None else [])
    w = w or (r"0.175\textwidth" if mae is not None else r"0.27\textwidth")
    n, gap = len(cols), "2pt"

    def img(cell, metric):
        return graphic(image(oid, cell, metric, view), w, h)

    tex = head(obj["name"])
    tex += r"\noindent\begin{tabular}{@{}M{0.05\textwidth}" + rf"@{{\hspace{{2pt}}}}M{{{w}}}" * len(COLS_B)
    tex += rf"@{{\hspace{{{gap}}}}}M{{{w}}}@{{\hspace{{2pt}}}}M{{{w}}}" * (mae is not None) + "@{}}\n"
    tex += " & " + " & ".join(rf"\footnotesize {t}" for t, _ in cols) + r" \\" + "\n"
    for method in spec["methods"]:
        cell = cell_name(method, tier, spec["cells"]["full_views"], spec)
        row = metrics.get((oid, cell))
        if row is None:
            continue
        if row["status"] != "OK":
            span = rf"\dimexpr {n}\dimexpr{w}\relax+{2 * (n - 1)}pt" + (rf"+{gap}-2pt" if mae is not None else "") + r"\relax"
            tex += rf"{label(method, spec, h)} & \multicolumn{{{n}}}{{c}}{{\failbox{{{span}}}{{{h}}}{{{fail_label(row['status'])}}}}} \\[2pt]" + "\n"
            continue
        tex += label(method, spec, h) + " & " + " & ".join(img(cell, m) for _, m in cols) + r" \\[-1pt]" + "\n"
        tex += rf" & & \multicolumn{{2}}{{c}}{{\scriptsize {scores(row)}}}"
        if mae is not None:
            tex += " & & " + (rf"\scriptsize MAE {mae[(oid, cell)]:.1f}$^\circ$" if (oid, cell) in mae else "")
        tex += r" \\[1pt]" + "\n"
    tex += r"\midrule" + "\n"
    uni = graphic(f"photos/{obj['name']}_{view:03d}_unimsps.jpg", w, h)
    ref = [img("Groundtruth", "uniform"), graphic(f"photos/{obj['name']}_{view:03d}.jpg", w, h)]
    names = [r"\scriptsize Ground-truth mesh", r"\scriptsize MVS image"]
    if mae is not None:
        ref += [graphic(f"photos/{obj['name']}_{view:03d}_mvps.jpg", w, h), img("Groundtruth", "normals"), uni]
        names += [r"\scriptsize MVPS image", r"\scriptsize Ground-truth normals", r"\scriptsize Uni-MS-PS normals"]
    else:
        ref += [uni]
        names += [r"\scriptsize Uni-MS-PS normals"]
    tex += side("Reference", h) + " & " + " & ".join(ref) + r" \\" + "\n"
    tex += " & " + " & ".join(names) + r" \\" + "\n" + r"\end{tabular}" + (CBARS if mae is not None else CBAR)
    return tex + "\n\\clearpage\n"


# Labels of the 7 axes of the taxonomy: those of the paper's radar (`radar.labels` of the Martine specification).
TAXO_LABELS = {"lambertian": "Lambertian", "specular": "Specular", "transparent": "Transparent", "not_textured": "Textureless",
               "fine_structure": "Fine structure", "high_curvature": "High curvature", "low_visibility": "Low visibility"}


def chunks(items, n):
    """Balanced pages of at most n items, in order (26 -> 6, 5, 5, 5, 5)."""
    pages = -(-len(items) // n)
    base, extra = divmod(len(items), max(pages, 1))
    out, i = [], 0
    for k in range(pages):
        size = base + (k < extra)
        out.append(items[i:i + size])
        i += size
    return out


def taxo_pages(axis, zone, objs, spec, metrics, views=None, per_page=6, h="1.62cm"):
    """One axis of the taxonomy: one row per method, one column per object that has the property (at most `per_page`
    per page). Accuracy map restricted to the zone, precision / recall / F of the zone below, photos as reference row.
    `objs`: {id: object}; `views`: {(axis, id): view} where it differs from the reference view."""
    tier, views = spec["cells"]["views_tier"], views or {}
    pages, tex = chunks(zone["ids"], per_page), ""
    for k, ids in enumerate(pages):
        w = rf"{0.93 / max(len(ids), 4):.3f}\textwidth"
        view = {o: views.get((axis, o), objs[o]["view"]) for o in ids}
        tex += head(TAXO_LABELS[axis] + (f" ({k + 1}/{len(pages)})" if len(pages) > 1 else ""))
        tex += tabular(len(ids), w)
        tex += " & " + " & ".join(rf"\footnotesize\textsc{{{objs[o]['name'].replace('_', ' ')}}}" for o in ids) + r" \\" + "\n"
        for method in spec["methods"]:
            pics, nums = [], []
            for o in ids:
                cell = f"{method}_{tier}"
                prf, row = zone["object"][o].get(method), metrics.get((o, cell))
                if prf is None:
                    pics.append(rf"\failbox{{{w}}}{{{h}}}{{{fail_label(row['status']) if row else 'not run'}}}"), nums.append("")
                    continue
                pics.append(graphic(f"vizz/{axis}/{o}/{cell}/visualizations/accuracy/view_{view[o]:03d}.png", w, h))
                nums.append(r"\scriptsize " + " / ".join(f"{v:.1f}" for v in prf))
            tex += label(method, spec, h) + " & " + " & ".join(pics) + r" \\[-1pt]" + "\n"
            tex += " & " + " & ".join(nums) + r" \\[1pt]" + "\n"
        tex += r"\midrule" + "\n" + side("Reference", h) + " & "
        tex += " & ".join(graphic(f"photos/{objs[o]['name']}_{view[o]:03d}.jpg", w, h) for o in ids) + r" \\" + "\n"
        tex += r"\end{tabular}" + CBAR + "\n\\clearpage\n"
    return tex


def pooled_pages(axes, spec, per_page=3):
    """Per axis: precision / recall / F curves pooled over the zone, without a table, 3 axes per page."""
    tex = ""
    for k, axis in enumerate(axes):
        tex += head(TAXO_LABELS[axis]) + rf"\noindent\includegraphics[width=\textwidth]{{taxo/prf_{axis}.pdf}}\par" + "\n"
        tex += "\\clearpage\n" if k % per_page == per_page - 1 or k == len(axes) - 1 else "\\bigskip\n"
    return tex


def grid_page(obj, spec, metrics, methods, columns, w, h):
    """columns: list of (title, function method -> cell name, or None when the cell is not declared)."""
    oid, view = obj["obj_id"], obj["view"]
    tex = head(obj["name"]) + tabular(len(columns), w)
    tex += " & " + " & ".join(rf"\footnotesize {{{t}}}" for t, _ in columns) + r" \\" + "\n"
    for method in methods:
        pics, nums = [], []
        for _, cell_of in columns:
            row = metrics.get((oid, cell_of(method) or ""))
            if row is None:
                pics.append(""), nums.append("")
            elif row["status"] != "OK":
                pics.append(rf"\failbox{{{w}}}{{{h}}}{{{fail_label(row['status'])}}}"), nums.append("")
            else:
                pics.append(graphic(f"montage/{oid}/{cell_of(method)}/view_{view:03d}.jpg", w, h)), nums.append(rf"\scriptsize {scores(row, stacked=True)}")
        tex += label(method, spec, h) + " & " + " & ".join(pics) + r" \\[2pt]" + "\n"
        tex += " & " + " & ".join(nums) + r" \\[5pt]" + "\n"
    return tex + r"\end{tabular}" + CBAR + "\n\\clearpage\n"


def tiers_page(obj, spec, metrics, w=r"0.225\textwidth", h="3.0cm"):
    dec, full = spec["cells"]["declared"], spec["cells"]["full_views"]
    names = {t: text.split("\n")[-1].replace("x", r"$\times$") for t, text in spec["resolution_views"].get("tier_labels", {}).items()}
    cols = [(names.get(t, t), lambda m, t=t: cell_name(m, t, full, spec) if t in dec[m]["tiers"] else None)
            for t in spec["resolution_views"]["tiers"]]
    return grid_page(obj, spec, metrics, spec["resolution_views"]["methods"], cols, w, h)


def views_page(obj, spec, metrics, w=r"0.18\textwidth", h="3.0cm"):
    dec, full, tier = spec["cells"]["declared"], spec["cells"]["full_views"], spec["cells"]["views_tier"]
    cols = [(f"{v} views", lambda m, v=v: cell_name(m, tier, v, spec) if v == full or v in dec[m].get("views", []) else None)
            for v in sorted(spec["resolution_views"]["views"], reverse=True)]
    return grid_page(obj, spec, metrics, spec["resolution_views"]["methods"], cols, w, h)


def prf_pages(names, per_page=3):
    """PRF curves per object, then the "mean" curve: all objects pooled, as in the paper."""

    def fig(title, f):
        return rf"\noindent{title}\par\noindent\includegraphics[width=\textwidth]{{prf/{f}.pdf}}\par\medskip" + "\n"

    tex = ""
    for i, n in enumerate(names):
        tex += fig(rf"\obj{{{n.replace('_', ' ')}}}", f"prf_{n}")
        if i % per_page == per_page - 1 and i != len(names) - 1:
            tex += "\\clearpage\n"
    if len(names) % per_page == 0:
        tex += "\\clearpage\n"
    tex += r"\noindent\rule{\textwidth}{0.4pt}\par\smallskip" + "\n"
    return tex + fig(r" \textsc{mean} (point distances of all objects pooled, as in the paper)", "prf_mean") + "\\clearpage\n"


def curves_pages(names, per_page=2):
    """Precision, recall and F-score against resolution and number of views: one full-width figure per object."""
    tex = ""
    for i, n in enumerate(names):
        tex += rf"\noindent\obj{{{n.replace('_', ' ')}}}\par\noindent\includegraphics[width=\textwidth]{{curves/curves_{n}.pdf}}\par\bigskip" + "\n"
        if i % per_page == per_page - 1 or i == len(names) - 1:
            tex += "\\clearpage\n"
    return tex


def guide(spec):
    """Front page: what the document contains and how to read its pages."""
    ms = " \\quad ".join(rf"\{'mthh' if m.get('hollow') else 'mth'}{{{m['display']}}}{{{m['color'].lstrip('#')}}}" for m in spec["methods"].values())
    return r"""
\noindent This document gives, for each of the 26 objects of \martine{}, the reconstructions of all evaluated methods,
rendered from one reference view, with their error maps and scores. It complements the aggregated results of the paper.

\paragraph{Methods.} {\small """ + ms + r"""}\par\smallskip
\noindent One hue per family; a hollow marker is the second member of its family. The same order and colours are used
in every table and figure.

\paragraph{Sections.}
\begin{enumerate}\setlength{\itemsep}{1pt}
\item \textbf{Per-object results at quarter resolution}: one page per object, one row per method, all 84 views.
\item \textbf{Image resolution}: accuracy maps of every method at each image resolution it was run at.
\item \textbf{Number of input views}: accuracy maps at quarter resolution with 84, 40, 20, 10 and 5 input views.
\item \textbf{Precision, recall and F-score against resolution and number of views}, per object.
\item \textbf{Precision, recall and F-score curves} against the distance threshold, per object, then pooled.
\end{enumerate}

\paragraph{Maps.} \textbf{Reconstruction}: the evaluated mesh, uniformly shaded. \textbf{Accuracy}: the reconstruction
coloured by the distance of each of its points to the ground truth. \textbf{Completeness}: the ground truth coloured by
the distance of each of its points to the reconstruction. Distances are clipped at 4\,mm.
""" + CBAR + r"""

\noindent The reference row shows the ground-truth mesh, the input image of the reference view and the normal map
predicted by Uni-MS-PS at the corresponding photometric-stereo pose (input of RNb-NeuS2 and SuperNormal).

\paragraph{Scores.} F@0.5: F-score (\%) at a distance threshold of 0.5\,mm. CD: Chamfer distance (mm).
MAE: mean angular error between the normal maps of the reconstruction and of the ground truth, at the displayed view.

\paragraph{Failures and empty cells.} A grey box is a failed reconstruction: \textit{SfM failed} (structure from motion
on the images of the configuration gave no usable sparse model), \textit{empty mesh} or \textit{run failed} (no mesh
produced). An empty cell is a configuration that was not run for this method.
\clearpage
"""


def referenced_files(tex):
    return re.findall(r"\{([^{}#\s]+\.(?:png|jpg|pdf))\}", tex)


def jpeg_paths(tex):
    """Replace each PNG render (.../view_NNN.png) by jpg/<same path>.jpg; return the text and the (png, jpg) pairs."""
    pngs = sorted(set(re.findall(r"\{([^{}#\s]+/view_\d+\.png)\}", tex)))
    pairs = [(p, "jpg/" + p[:-4] + ".jpg") for p in pngs]
    for src, dst in pairs:
        tex = tex.replace("{" + src + "}", "{" + dst + "}")
    return tex, pairs


def load_metrics(path):
    with open(path, newline="") as f:
        return {(r["obj_id"], r["cell"]): r for r in csv.DictReader(f, delimiter="\t")}


def load_mae(root, objs, sub):
    """{(object, cell): MAE in degrees at the displayed view}, from <root>/<object>/<cell>/<sub>/angular_error/mae.json."""
    out = {}
    for o in objs:
        for f in sorted(Path(root).glob(f"{o['obj_id']}/*/{sub}/angular_error/mae.json")):
            views = json.loads(f.read_text())["views"]
            if str(o["view"]) in views:
                out[(o["obj_id"], f.parts[-4])] = views[str(o["view"])]["mae_deg"]
    return out


def load_objects(views_tsv, metrics):
    """One object per row of views.tsv (first view listed), ordered by object id."""
    ids = {r["object"]: r["obj_id"] for r in metrics.values()}
    with open(views_tsv, newline="") as f:
        rows = [r for r in csv.DictReader(f, delimiter="\t") if not r["object"].startswith("#")]
    objs = [dict(obj_id=ids[r["object"]], name=r["object"], view=int(r["view"].split(",")[0])) for r in rows]
    return sorted(objs, key=lambda o: o["obj_id"])


CURVES_THRESHOLD = 0.5   # mm, threshold of the curves of part 4; pass the same value to the `curves` tool
SECTIONS = ["object", "tiers", "views", "curves", "prf"]
TAXO_SECTIONS = ["taxo", "pooled"]   # added when the zone scores are given
TITLES = {"object": "Per-object results at quarter resolution", "tiers": "Image resolution", "views": "Number of input views",
          "curves": "Precision, recall and F-score against image resolution and number of views",
          "prf": "Precision, recall and F-score against the distance threshold",
          "taxo": "Surface properties", "pooled": "Surface properties, pooled over objects"}


def intros(spec, normals):
    n, t = len(spec["methods"]), CURVES_THRESHOLD
    return {
        "object": rf"The {n} methods at quarter resolution with all 84 views, from the reference view of the object; scores below the maps."
                  + (r" MAE: mean angular error to the ground-truth normals." if normals else "") ,
        "tiers": r"One method per family at each image resolution it was run at, with all 84 views. Each cell shows the reconstructed mesh (left) and its accuracy map (right); scores below.",
        "views": r"The same methods run on subsets of the input views, at quarter resolution. Each cell shows the reconstructed mesh (left) and its accuracy map (right); scores below.",
        "curves": rf"Per object, precision, recall and F-score at a distance threshold of {t:g}\,mm against image resolution (top) and number of input views (bottom). A cross at 0 is a failed reconstruction.",
        "taxo": rf"For each surface property of the taxonomy, the objects that carry it. Each cell shows the accuracy map of a method restricted to the zone (grey: outside the zone), with precision / recall / F-score of the zone at {t:g}\,mm below.",
        "pooled": rf"For each surface property, precision, recall and F-score as a function of the distance threshold $\tau$, pooling the point distances of the zone over its objects; the table gives the values at {t:g}\,mm.",
        "prf": r"Per object, precision, recall and F-score of every method at quarter resolution as a function of the distance threshold $\tau$. The last figure pools the point distances of all objects, as in the paper.",
    }


def document(objs, spec, metrics, mae=None, sections=None, zones=None, views=None):
    """Part title, short text, then the pages (each one starts with the object name alone)."""
    names = [o["name"] for o in objs]
    body = {"object": lambda: "".join(object_page(o, spec, metrics, mae=mae) for o in objs),
            "tiers": lambda: "".join(tiers_page(o, spec, metrics) for o in objs),
            "views": lambda: "".join(views_page(o, spec, metrics) for o in objs),
            "curves": lambda: curves_pages(names), "prf": lambda: prf_pages(names)}
    if zones is not None:
        by_id = {o["obj_id"]: o for o in objs}
        body["taxo"] = lambda: "".join(taxo_pages(a, z, by_id, spec, metrics, views) for a, z in zones["axes"].items())
        body["pooled"] = lambda: pooled_pages(zones["axes"], spec)
    text = intros(spec, mae is not None)
    tex = PRE + guide(spec)
    for key in sections or SECTIONS + (TAXO_SECTIONS if zones is not None else []):
        tex += rf"\section{{{TITLES[key]}}}" + "\n" + rf"\noindent {text[key]}\par\medskip" + "\n" + body[key]()
    return tex + r"\end{document}" + "\n"


def load_zone_views(path, objs):
    """{(axis, object id): view} from a TSV `axis<TAB>published object<TAB>view`: views other than the reference one."""
    names = {o["name"]: o["obj_id"] for o in objs}
    views = {}
    for line in Path(path).read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            a, name, v = line.split("\t")[:3]
            views[(a, names[name])] = int(v)
    return views


def main(argv=None):
    import yaml
    argv = sys.argv[1:] if argv is None else argv
    spec_path, metrics_path, views_path, out = argv[:4]
    with open(spec_path) as f:
        spec = yaml.safe_load(f)
    metrics = load_metrics(metrics_path)
    objs = load_objects(views_path, metrics)
    out = Path(out)
    mae = load_mae(out / "vizn", objs, "visualizations") if len(argv) > 4 and argv[4] == "normals" else None
    zones = json.loads(Path(os.environ["ZONES"]).read_text()) if os.environ.get("ZONES") else None
    views = load_zone_views(os.environ["TAXO_VIEWS"], objs) if os.environ.get("TAXO_VIEWS") else {}
    tex = document(objs, spec, metrics, mae, os.environ["SECTIONS"].split("+") if os.environ.get("SECTIONS") else None,
                   zones=zones, views=views)
    tex, pairs = jpeg_paths(tex)
    (out / "supp.tex").write_text(tex)
    (out / "jpeg_pairs.tsv").write_text("".join(f"{a}\t{b}\n" for a, b in pairs))
    files = set(referenced_files(tex)) - {b for _, b in pairs} | {a for a, _ in pairs}
    missing = sorted(p for p in files if not (out / p).exists())
    print("objects", len(objs), "referenced files", len(files), "of which renders to convert", len(pairs),
          "missing", len(missing))
    print("\n".join(missing[:30]))


if __name__ == "__main__":
    main()
