"""Refinement of the formulation screen around its Pareto front.

    python paper/run_formulation_refine.py    # ~3 min; writes paper/results_formulation_refine.json

Every formulation on the screen's thickness/temperature Pareto front used the
largest microsphere fraction screened (40 vol%). This stage extends the
microsphere fraction to 50 vol% and runs the full robustness set on the
front's candidates, so the final pick is made on worst-case performance.
"""

import json
import time
from dataclasses import replace
from pathlib import Path

from run_formulation_study import WIRES, _pmap, describe, evaluate, variants

from runaway.formulation import Formulation

OUT = Path(__file__).with_name("results_formulation_refine.json")


def F(pw, pms, ath):
    parts = (("ath", ath),) if ath else ()
    name = " + ".join(x for x in (f"AgNW 40x40 {pw * 100:g}%" if pw else "unfilled", f"EMS {pms * 100:g}%",
                                  f"ATH {ath * 100:g}%" if ath else "") if x)
    return Formulation(name, particles=parts, wire=WIRES["AgNW 40x40"] if pw else None, phi_w=pw,
                       spheres="ems_051", phi_ms=pms)


PICK = F(0.0, 0.40, 0.20)


def main():
    t0 = time.time()
    ext = [F(pw, pms, ath) for pms in (0.45, 0.50) for pw in (0.0, 0.01, 0.02, 0.03) for ath in (0.0, 0.10)
           if pw + pms + ath <= 0.60 + 1e-9]
    res = _pmap(evaluate, [("formulation", f) for f in ext], None)
    data = {"extend": [{**describe(f), **r} for f, r in zip(ext, res)]}
    print(f"extension: {len(ext)} formulations, {time.time() - t0:.0f} s", flush=True)

    cands = [F(0.0, 0.40, 0.0), F(0.0, 0.40, 0.20), F(0.01, 0.40, 0.10), F(0.02, 0.40, 0.0),
             F(0.02, 0.40, 0.10), F(0.03, 0.40, 0.10)]
    specs, owners = [], []
    for f in cands:
        for label, spec in variants(f):
            specs.append(spec)
            owners.append((f.name, label))
    vres = _pmap(evaluate, specs, None)
    data["robust"] = [{"formulation": o[0], "case": o[1], **{k: r[k] for k in ("L_min", "peak", "k_on", "k_off")}}
                      for o, r in zip(owners, vres)]
    data["cands"] = [describe(f) for f in cands]
    print(f"robustness: {len(specs)} runs, {time.time() - t0:.0f} s", flush=True)

    # The pick: smallest worst-case normal peak among candidates whose
    # worst-case containing thickness is <= 4 mm (see the paper). Maps around it.
    data["pick"] = describe(PICK)
    grid = [replace(PICK, phi_ms=pms, E=E) for pms in (0.2, 0.3, 0.4, 0.5) for E in (2.0, 3.0, 5.0, 10.0, 20.0)]
    data["map_ms_E"] = [{"phi_ms": f.phi_ms, "E": f.E, **r}
                        for f, r in zip(grid, _pmap(evaluate, [("formulation", f) for f in grid], None))]
    w = WIRES["AgNW 40x40"]
    wgrid = [replace(PICK, wire=replace(w, g=replace(w.g, value=g)), phi_w=pw, network=net)
             for pw in (0.005, 0.01, 0.02, 0.03) for g in (0.0, 0.05, 0.5) for net in ("dilute", "segregated")]
    data["map_wire"] = [{"phi_w": f.phi_w, "g": f.wire.g.value, "network": f.network, **r}
                        for f, r in zip(wgrid, _pmap(evaluate, [("formulation", f) for f in wgrid], None))]
    OUT.write_text(json.dumps(data, indent=1))
    print(f"wrote {OUT} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
