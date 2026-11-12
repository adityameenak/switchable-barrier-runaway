"""Formulation study: screen silicone barrier formulations and narrow down to one.

    python paper/run_formulation_study.py     # ~15 min on 10 cores; writes paper/results_formulation.json
    python paper/make_figures_formulation.py  # figures + paper/formulation_summary.txt

Every formulation is scored by two numbers from the 1D model: the thinnest
barrier that confines runaway to the trigger cell (bisection on a 0.125 mm
grid between 0.25 and 10 mm) and the normal-operation peak at that thickness.

Stages
  1. screen   - conductive system x microsphere fraction x endothermic filler
  2. robust   - the leading formulations under every model input the data leave open
  3. refs     - idealised switch and the sourced materials, scored the same way
"""

import itertools
import json
import time
from dataclasses import replace
from pathlib import Path

from runaway.formulation import R_K_POOR, Formulation, agnw
from runaway.params import StackParams
from runaway.scenarios import SWEEP_SIM, _pmap, barrier_from_spec
from runaway.solver import simulate_runaway, solve_normal_operation

OUT = Path(__file__).with_name("results_formulation.json")
STEP = 0.125  # mm
I_LO, I_HI = 2, 80  # 0.25 .. 10 mm


def _n(spec, L_mm):
    return simulate_runaway(StackParams(barrier=barrier_from_spec(spec, L_mm * 1e-3)), SWEEP_SIM).n_propagated


def _peak(spec, L_mm):
    return solve_normal_operation(StackParams(barrier=barrier_from_spec(spec, L_mm * 1e-3))).peak_C


def evaluate(spec):
    """Thinnest containing barrier [mm] (None = not up to 10 mm) and normal peak there."""
    out = {"peak_2mm": _peak(spec, 2.0)}
    if spec[0] == "formulation":
        out.update(spec[1].properties())
    if _n(spec, I_HI * STEP) != 1:
        return {**out, "L_min": None, "peak": None}
    lo, hi = I_LO, I_HI
    if _n(spec, lo * STEP) == 1:
        hi = lo
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if _n(spec, mid * STEP) == 1:
            hi = mid
        else:
            lo = mid
    L = hi * STEP
    return {**out, "L_min": L, "peak": _peak(spec, L)}


def describe(f: Formulation) -> dict:
    return {"name": f.name, "particles": list(f.particles),
            "wire": None if f.wire is None else f.wire.key, "phi_w": f.phi_w,
            "spheres": f.spheres, "phi_ms": f.phi_ms, "E": f.E, "network": f.network,
            "sinter": f.sinter, "k_gas": f.k_gas,
            "wire_g": None if f.wire is None else f.wire.g.value,
            "wire_RK": None if f.wire is None else f.wire.R_K.value}


# ---------------------------------------------------------------------------
# Stage 1: screening grid
# ---------------------------------------------------------------------------

WIRES = {"AgNW 40x40": agnw(40, 40), "AgNW 60x20": agnw(60, 20), "AgNW 100x15": agnw(100, 15)}

# (label, particles, wire label, phi_w)
CONDUCTIVE = [("unfilled", (), None, 0.0)]
for wl in WIRES:
    for pw in (0.005, 0.01, 0.02, 0.03):
        CONDUCTIVE.append((f"{wl} {pw * 100:g}%", (), wl, pw))
for phi in (0.10, 0.20):
    CONDUCTIVE.append((f"h-BN {phi * 100:g}%", (("hbn", phi),), None, 0.0))
for phi in (0.20, 0.30):
    CONDUCTIVE.append((f"Al2O3 5um {phi * 100:g}%", (("al2o3_micro", phi),), None, 0.0))
for pw in (0.01, 0.02):
    CONDUCTIVE.append((f"AgNW 40x40 {pw * 100:g}% + h-BN 10%", (("hbn", 0.10),), "AgNW 40x40", pw))
    CONDUCTIVE.append((f"AgNW 40x40 {pw * 100:g}% + Al2O3 50nm 10%", (("al2o3_nano", 0.10),), "AgNW 40x40", pw))

PHI_MS = (0.0, 0.10, 0.20, 0.30, 0.40)
ENDO = [("", ()), ("ATH 10%", (("ath", 0.10),)), ("ATH 20%", (("ath", 0.20),)),
        ("MDH 10%", (("mdh", 0.10),)), ("MDH 20%", (("mdh", 0.20),))]
PHI_MAX_TOTAL = 0.60  # processability limit assumed for a castable silicone


def screen_formulations():
    out = []
    for (cl, parts, wl, pw), pms, (el, endo) in itertools.product(CONDUCTIVE, PHI_MS, ENDO):
        total = sum(p for _, p in parts) + pw + pms + sum(p for _, p in endo)
        if total > PHI_MAX_TOTAL + 1e-9:
            continue
        name = " + ".join(x for x in (cl, f"EMS {pms * 100:g}%" if pms else "", el) if x)
        out.append(Formulation(name, particles=parts + endo, wire=WIRES.get(wl), phi_w=pw,
                               spheres="ems_051" if pms else None, phi_ms=pms))
    return out


# ---------------------------------------------------------------------------
# Stage 2: robustness variants
# ---------------------------------------------------------------------------

def variants(f: Formulation):
    """(label, spec) for each model input the data do not pin down."""
    v = [("baseline", ("formulation", f))]
    if f.wire is not None:
        w = f.wire
        v += [
            ("no junction conduction (g = 0)", ("formulation", replace(f, wire=replace(w, g=replace(w.g, value=0.0))))),
            ("strong junctions (g = 0.5)", ("formulation", replace(f, wire=replace(w, g=replace(w.g, value=0.5))))),
            ("sintered junctions (x10)", ("formulation", replace(f, sinter=10.0))),
            ("poorly dispersed wires (R_K fit)", ("formulation", replace(f, wire=replace(w, R_K=replace(w.R_K, value=R_K_POOR))))),
            ("wire k x 0.5", ("formulation", replace(f, wire=replace(w, k_w=replace(w.k_w, value=0.5 * w.k_w.value))))),
        ]
    if f.spheres:
        v += [
            ("expansion E = 3", ("formulation", replace(f, E=3.0))),
            ("expansion E = 10", ("formulation", replace(f, E=10.0))),
            ("segregated network", ("formulation", replace(f, network="segregated"))),
            ("gas-cell k = 0.06", ("formulation", replace(f, k_gas=0.06))),
            ("Expancel 920 (T_sw 163 C)", ("formulation", replace(f, spheres="ems_920"))),
            ("slow switch (tau = 30 s)", ("formulation", f, {"switch_tau": 30.0})),
            ("instant switch (tau = 0)", ("formulation", f, {"switch_tau": 0.0})),
        ]
    v += [("rho cp x 0.7", ("formulation", f, {"cp": 0.7 * f.properties()["cp"]}))]
    return v


def main():
    t0 = time.time()
    data = {}

    forms = screen_formulations()
    res = _pmap(evaluate, [("formulation", f) for f in forms], None)
    data["screen"] = [{**describe(f), **r} for f, r in zip(forms, res)]
    print(f"screen: {len(forms)} formulations, {time.time() - t0:.0f} s", flush=True)

    ok = [(f, r) for f, r in zip(forms, res) if r["L_min"] is not None and r["L_min"] <= 6.0]
    ok.sort(key=lambda fr: fr[1]["peak"])
    # Leading formulations: best few overall, plus the best of each family so
    # the robustness stage compares mechanisms, not near-duplicates.
    lead, seen = [], set()
    for f, r in ok:
        fam = ("wire" if f.wire else "particles" if f.particles else "none", bool(f.spheres))
        if len(lead) < 4 or fam not in seen:
            lead.append(f)
            seen.add(fam)
        if len(lead) >= 8:
            break
    specs, owners = [], []
    for f in lead:
        for label, spec in variants(f):
            specs.append(spec)
            owners.append((f.name, label))
    vres = _pmap(evaluate, specs, None)
    data["robust"] = [{"formulation": o[0], "case": o[1], **{k: r[k] for k in ("L_min", "peak", "k_on", "k_off")}}
                      for o, r in zip(owners, vres)]
    data["lead"] = [describe(f) for f in lead]
    print(f"robustness: {len(specs)} runs, {time.time() - t0:.0f} s", flush=True)

    # The final pick and the maps around it are made in run_formulation_refine.py,
    # after extending the microsphere range and testing the Pareto front.

    # Stage 4: references scored identically
    refs = [("design", "switchable"), ("candidate", "graphene_switch"), ("candidate", "aerogel"),
            ("candidate", "mica"), ("candidate", "ceramic_fibre")]
    rres = _pmap(evaluate, refs, None)
    data["refs"] = {f"{s[0]} | {s[1]}": r for s, r in zip(refs, rres)}

    OUT.write_text(json.dumps(data, indent=1))
    print(f"wrote {OUT} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
