"""Run the 2D (cold-plate) studies and cache the results.

    python paper/run_2d_study.py          # ~10 min on 10 cores; writes paper/results_2d.json

``paper/make_figures_2d.py`` turns the cache into figures and a text summary.
"""

import json
import time
from dataclasses import asdict, replace
from pathlib import Path

from runaway.candidates import CANDIDATES
from runaway.composites import COMPOSITIONS
from runaway.scenarios import containment_scan2d
from runaway.solver2d import Params2D

OUT = Path(__file__).with_name("results_2d.json")

SPECS = (
    [("design", d) for d in ("conductor", "insulator", "switchable")]
    + [("candidate", k) for k in CANDIDATES]
    + [("composition", c) for c in COMPOSITIONS]
)

# Cold-plate sensitivity for the leading options of each mechanism.
H_PLATE = (50.0, 100.0, 200.0, 500.0)
SENS_SPECS = [
    ("candidate", "graphene_switch"), ("candidate", "aerogel"), ("candidate", "mica"),
    ("composition", "ATH 40%"), ("composition", "bubbles 40% + ATH 20%"),
    ("composition", "glass bubbles 50%"),
]


def pack(res):
    return {
        " | ".join(map(str, spec)): {
            "thickness_mm": r.thickness_mm.tolist(),
            "n_propagated": r.n_propagated.tolist(),
            "normal_peak_C": r.normal_peak_C.tolist(),
            "min_containing_mm": r.min_containing_mm,
            "peak_at_min_C": r.peak_at_min_C,
        }
        for spec, r in res.items()
    }


def main():
    t0 = time.time()
    base = Params2D()
    data = {"params2d": asdict(base), "main": pack(containment_scan2d(SPECS, base))}
    print(f"main study done in {time.time() - t0:.0f} s", flush=True)
    data["h_plate"] = {}
    for h in H_PLATE:
        data["h_plate"][str(h)] = pack(containment_scan2d(SENS_SPECS, replace(base, h_plate=h)))
        print(f"h_plate = {h} done at {time.time() - t0:.0f} s", flush=True)
    OUT.write_text(json.dumps(data, indent=1))
    print(f"wrote {OUT} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
