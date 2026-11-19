"""Reference photos of each object, cropped to the frame of the renders, and the colorbars of the document.

Usage: python -m eval3d.figures.supplementary.photos <published_root> <renders_root> <views.tsv> <out_dir>
           --tasks <tasks.tsv> [--subdir visualizations] [--normals-root <dir>]
           [--zone-views <tsv> --zone-root <dir>]

- ``published_root``: the dataset, laid out as Martine: ``<object>/mvs/d4/{images/NNN.exr,masks/NNN.png}`` and
  ``<object>/mvps/d4/{images/pose_NN/lighting_00.exr,masks/pose_NN.png}``.
- ``renders_root``: ``<object id>/Groundtruth/<subdir>/bbox.json`` (written by `eval-pipeline visualize`) gives the
  crop of each view.
- ``views.tsv``: header ``object<TAB>view``; ``view`` is one view or several separated by commas.
- ``--tasks``: TSV whose first two columns are the eval name and the published name of each object (see `tasks`).
- ``--normals-root``: ``<object>/unimsps_normals/d4/pose_NN.png``, the 16-bit normal maps predicted by Uni-MS-PS;
  without it the ``_unimsps`` images are not written.
- ``--zone-views`` / ``--zone-root``: TSV ``axis<TAB>published object<TAB>view`` of the zone renders shown from another
  view than the reference one, and the root of the zone renders (``<axis>/<object id>/Groundtruth/...``) that gives
  their crop.

Writes in <out_dir>: ``<object>_<view>.jpg`` (MVS image), ``<object>_<view>_mvps.jpg`` (photometric-stereo image
under one light), ``<object>_<view>_unimsps.jpg``, ``colorbar.png`` (viridis) and ``colorbar_angular.png`` (magma).
Reading the EXR images needs OpenCV built with OpenEXR (``OPENCV_IO_ENABLE_OPENEXR=1``, set here before the import).
"""

import argparse
import csv
import json
import os
from pathlib import Path

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
JPEG_QUALITY = 92


def crop_box(bbox, factor, shape):
    """Frame [y0, y1, x0, x1] in full-resolution pixels -> slices in an image reduced by `factor`."""
    y0, y1, x0, x1 = (int(round(v / factor)) for v in bbox)
    return slice(max(y0, 0), min(y1, shape[0])), slice(max(x0, 0), min(x1, shape[1]))


def support_box(mask, margin):
    """Slices of the bounding box of a boolean mask, grown by `margin` pixels and clamped to the image."""
    import numpy as np
    ys, xs = np.nonzero(mask)
    return (slice(max(int(ys.min()) - margin, 0), min(int(ys.max()) + 1 + margin, mask.shape[0])),
            slice(max(int(xs.min()) - margin, 0), min(int(xs.max()) + 1 + margin, mask.shape[1])))


def unimsps(normals_png, mask_png):
    """Uni-MS-PS normal map (16-bit PNG) on a white background, cropped to its mask (the photometric-stereo pose with
    the same index as the view)."""
    import cv2
    n = (cv2.imread(str(normals_png), cv2.IMREAD_UNCHANGED)[..., :3] / 257).astype("uint8")
    mask = cv2.imread(str(mask_png), cv2.IMREAD_GRAYSCALE) > 127
    n[~mask] = 255
    ys, xs = support_box(mask, 4)
    n = n[ys, xs]
    return cv2.resize(n, (1000, max(int(round(n.shape[0] * 1000 / n.shape[1])), 1)), interpolation=cv2.INTER_AREA)


def mvps_photo(exr, mask_png):
    """Photometric-stereo image under a single light, masked, cropped to its mask like the Uni-MS-PS map."""
    import cv2
    import numpy as np
    img = cv2.imread(str(exr), cv2.IMREAD_UNCHANGED)[..., :3].astype(np.float32)
    mask = cv2.imread(str(mask_png), cv2.IMREAD_GRAYSCALE) > 127
    out = np.clip(img / max(np.percentile(img[mask], 99), 1e-6), 0, 1) ** (1 / 2.2)
    out[~mask] = 1.0
    ys, xs = support_box(mask, 4)
    out = out[ys, xs]
    out = cv2.resize(out, (1000, max(int(round(out.shape[0] * 1000 / out.shape[1])), 1)), interpolation=cv2.INTER_AREA)
    return (out * 255).astype(np.uint8)


def photo(obj_dir, view, bbox, tier=4):
    """MVS image of a view, masked, cropped to the frame `bbox` of the renders."""
    import cv2
    import numpy as np
    img = cv2.imread(str(obj_dir / f"mvs/d{tier}/images/{view:03d}.exr"), cv2.IMREAD_UNCHANGED)[..., :3].astype(np.float32)
    mask = cv2.imread(str(obj_dir / f"mvs/d{tier}/masks/{view:03d}.png"), cv2.IMREAD_GRAYSCALE) > 127
    scale = np.percentile(img[mask], 99) if mask.any() else 1.0
    out = np.clip(img / max(scale, 1e-6), 0, 1) ** (1 / 2.2)
    out[~mask] = 1.0
    ys, xs = crop_box(bbox, tier, out.shape)
    out = out[ys, xs]
    out = cv2.resize(out, (1000, max(int(round(out.shape[0] * 1000 / out.shape[1])), 1)), interpolation=cv2.INTER_AREA)
    return (out * 255).astype(np.uint8)


def main(argv=None):
    import cv2
    import numpy as np
    parser = argparse.ArgumentParser(description="Reference photos and colorbars of the supplementary report.")
    parser.add_argument("published_root", type=Path)
    parser.add_argument("renders_root", type=Path)
    parser.add_argument("views_tsv", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--tasks", type=Path, required=True, help="TSV: eval name, published name, ... (see tasks)")
    parser.add_argument("--subdir", default="visualizations", help="render directory holding bbox.json")
    parser.add_argument("--normals-root", type=Path, help="root of the Uni-MS-PS normal maps")
    parser.add_argument("--zone-views", type=Path, help="TSV: axis, published object, view")
    parser.add_argument("--zone-root", type=Path, help="root of the zone renders (with --zone-views)")
    args = parser.parse_args(argv)
    if (args.zone_views is None) != (args.zone_root is None):
        parser.error("--zone-views and --zone-root go together")
    pub, viz, out, quality = args.published_root, args.renders_root, args.out_dir, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    out.mkdir(parents=True, exist_ok=True)
    names = {line.split("\t")[1]: line.split("\t")[0] for line in args.tasks.read_text().splitlines() if line.strip()}
    n = 0
    with open(args.views_tsv, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    for r in rows:
        name = r["object"]
        boxes = json.loads((viz / names[name] / f"Groundtruth/{args.subdir}/bbox.json").read_text())
        for v in (int(v) for v in r["view"].split(",")):
            mask = pub / name / f"mvps/d4/masks/pose_{v:02d}.png"
            cv2.imwrite(str(out / f"{name}_{v:03d}.jpg"), photo(pub / name, v, boxes[str(v)]), quality)
            if args.normals_root is not None:
                normals = args.normals_root / name / f"unimsps_normals/d4/pose_{v:02d}.png"
                cv2.imwrite(str(out / f"{name}_{v:03d}_unimsps.jpg"), unimsps(normals, mask), quality)
            exr = pub / name / f"mvps/d4/images/pose_{v:02d}/lighting_00.exr"
            cv2.imwrite(str(out / f"{name}_{v:03d}_mvps.jpg"), mvps_photo(exr, mask), quality)
            n += 1
    if args.zone_views is not None:                        # zone renders shown from another view: crop of that render
        for line in args.zone_views.read_text().splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            axis, name, v = line.split("\t")[:3]
            boxes = json.loads((args.zone_root / axis / names[name] / "Groundtruth/visualizations/bbox.json").read_text())
            cv2.imwrite(str(out / f"{name}_{int(v):03d}.jpg"), photo(pub / name, int(v), boxes[str(int(v))]), quality)
            n += 1
    ramp = np.tile(np.linspace(0, 255, 800).astype(np.uint8), (36, 1))
    cv2.imwrite(str(out / "colorbar.png"), cv2.applyColorMap(ramp, cv2.COLORMAP_VIRIDIS))
    cv2.imwrite(str(out / "colorbar_angular.png"), cv2.applyColorMap(ramp, cv2.COLORMAP_MAGMA))
    print("photos", n)


if __name__ == "__main__":
    main()
