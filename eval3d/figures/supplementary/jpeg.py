"""Convert to JPEG the PNG renders referenced by supp.tex (jpeg_pairs.tsv written by `document`), to lighten the PDF.

Usage: python -m eval3d.figures.supplementary.jpeg <build_dir>
A JPEG newer than its PNG is kept.
"""

import sys
from pathlib import Path


def convert(build_dir, quality=90):
    """Write the JPEG of each (png, jpg) pair of <build_dir>/jpeg_pairs.tsv; return (converted, already up to date)."""
    import cv2
    out = Path(build_dir)
    n = skipped = 0
    for line in (out / "jpeg_pairs.tsv").read_text().splitlines():
        src, dst = (out / p for p in line.split("\t"))
        if dst.exists() and dst.stat().st_mtime > src.stat().st_mtime:
            skipped += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst), cv2.imread(str(src)), [cv2.IMWRITE_JPEG_QUALITY, quality])
        n += 1
    return n, skipped


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    n, skipped = convert(argv[0])
    print("jpeg", n, "already up to date", skipped)


if __name__ == "__main__":
    main()
