"""PRF curves per object: the PRF figure of the paper (same code, same specification) restricted to one object.

Usage: python -m eval3d.figures.supplementary.prf_objects <eval_config.yaml> <figures_spec.yaml> <out_dir> [mean]
Writes <out_dir>/prf_<published object>.pdf (+ .png) and prf_objects.json (numbers). With `mean`, writes instead
prf_mean.pdf and prf_mean.json: all objects pooled, i.e. the PRF figure of the paper.
"""

import json
import sys
from pathlib import Path

from ...config import load_config
from .. import prf
from ..cells import Results
from ..spec import load_spec


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    config, spec, out = load_config(argv[0]), load_spec(argv[1]), Path(argv[2])
    out.mkdir(parents=True, exist_ok=True)
    alias = config.dataset.object_aliases
    if len(argv) > 3 and argv[3] == "mean":
        mean = spec.model_copy(update={"prf": spec.prf.model_copy(update={"output": "prf_mean.pdf"})})
        pooled = prf.compute(config, mean, Results(config, mean))
        prf.plot(mean, pooled, out)
        (out / "prf_mean.json").write_text(json.dumps(pooled))
        print("OK mean")
        return
    numbers = {}
    for obj in spec.object_list(config):
        pub = alias.get(obj, obj)
        one = spec.model_copy(update={"objects": [obj], "prf": spec.prf.model_copy(update={"output": f"prf_{pub}.pdf"})})
        print(f"== {pub}", flush=True)
        numbers[pub] = prf.compute(config, one, Results(config, one))
        prf.plot(one, numbers[pub], out)
    (out / "prf_objects.json").write_text(json.dumps(numbers))
    print("OK", len(numbers))


if __name__ == "__main__":
    main()
