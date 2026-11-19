"""Montage of a cell: reconstructed mesh on the left, accuracy map on the right, separated by an oblique cut.

Usage: python -m eval3d.figures.supplementary.montage <renders_root> <subdir> <out_dir>
Reads <renders_root>/<object>/<cell>/<subdir>/{uniform,accuracy}/view_NNN.png and writes
<out_dir>/<object>/<cell>/view_NNN.jpg. A montage newer than its two renders is kept.
"""

import sys
from pathlib import Path

import numpy as np


def split(mesh, acc, lean=0.12, line=3, color=(70, 70, 70)):
    """Image whose pixels come from `mesh` left of the cut and from `acc` right of it. The cut goes through the centre
    of the object's support (pixels of `mesh` that differ from the background, read at the top-left corner) and leans
    by ±lean x width."""
    h, w = mesh.shape[:2]
    cols = np.nonzero((mesh != mesh[0, 0]).any(-1).any(0))[0]
    cx = (cols.min() + cols.max()) / 2 if len(cols) else (w - 1) / 2
    y = np.arange(h)[:, None]
    cut = cx + lean * w * (1 - 2 * y / max(h - 1, 1))
    d = np.arange(w)[None, :] - cut
    out = np.where((d < 0)[..., None], mesh, acc)
    out[np.abs(d) <= line / 2] = color
    return out


def main(argv=None):
    import cv2
    argv = sys.argv[1:] if argv is None else argv
    root, sub, out = Path(argv[0]), argv[1], Path(argv[2])
    n = skipped = 0
    for a in sorted(root.glob(f"*/*/{sub}/accuracy/view_*.png")):
        u = a.parent.parent / "uniform" / a.name
        dst = out / a.parts[-5] / a.parts[-4] / a.with_suffix(".jpg").name
        if dst.exists() and dst.stat().st_mtime > max(a.stat().st_mtime, u.stat().st_mtime):
            skipped += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst), split(cv2.imread(str(u)), cv2.imread(str(a))), [cv2.IMWRITE_JPEG_QUALITY, 92])
        n += 1
    print("montages", n, "already up to date", skipped)


if __name__ == "__main__":
    main()
