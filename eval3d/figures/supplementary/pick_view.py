"""Rank the rendered views of a cell by visible zone area (coloured pixels: neither grey nor white).

Helps choosing the view of a zone render when the zone is hidden from the reference view of the object.
Usage: python -m eval3d.figures.supplementary.pick_view <.../visualizations/accuracy>
Prints the ten best views.
"""

import sys
from pathlib import Path


def colored_fraction(img):
    """Share of coloured pixels among the object's pixels (white background excluded), and number of coloured pixels."""
    rgb = img[..., :3].astype(int)
    spread = rgb.max(-1) - rgb.min(-1)
    obj = rgb.min(-1) < 250
    col = obj & (spread > 30)
    return (col.sum() / max(obj.sum(), 1)), int(col.sum())


def main(argv=None):
    import cv2
    argv = sys.argv[1:] if argv is None else argv
    rows = []
    for p in sorted(Path(argv[0]).glob("view_*.png")):
        frac, n = colored_fraction(cv2.imread(str(p)))
        rows.append((frac, n, p.stem))
    for frac, n, name in sorted(rows, reverse=True)[:10]:
        print(f"{name} zone={100 * frac:.1f}% pixels={n}")


if __name__ == "__main__":
    main()
