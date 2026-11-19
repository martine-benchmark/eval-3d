"""Pooled PRF curves per zone, drawn like the PRF figure of the paper (eval3d.figures.prf.plot).

Usage: python -m eval3d.figures.supplementary.plot_zones <spec.yaml> <zone_scores.json> <out_dir>
Writes <out_dir>/prf_<axis>.pdf for each axis of zone_scores.json (written by `zones scores`).
"""

import json
import sys
from pathlib import Path

from .. import prf
from ..spec import load_spec


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    spec, data, out = load_spec(argv[0]), json.loads(Path(argv[1]).read_text()), Path(argv[2])
    out.mkdir(parents=True, exist_ok=True)
    for key, entry in data["axes"].items():
        one = spec.model_copy(update={"prf": spec.prf.model_copy(update={"output": f"prf_{key}.pdf"})})
        numbers = {"thresholds": data["thresholds"], "methods": {
            m: {k: [v / 100 for v in c[k]] for k in ("precision", "recall", "fscore")}
            for m, c in entry["curves"].items()}}
        prf.plot(one, numbers, out)
        print("OK", key)


if __name__ == "__main__":
    main()
