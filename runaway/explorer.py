"""Build the interactive results explorer (a single self-contained HTML page).

    python -m runaway.explorer            # writes docs/index.html

Runs the same studies as ``runaway.demo``, packs the results into compact
JSON and injects them into ``explorer_template.html``. The output needs no
server or Python: open it in a browser, or serve ``docs/`` with GitHub Pages.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from .params import KELVIN, KineticsParams, SimParams, StackParams, barrier_switchable
from .scenarios import compare_designs, design_map, normal_operation, stack_with
from .scenarios import switch_temperature_sweep
from .solver import simulate_runaway

TEMPLATE = Path(__file__).with_name("explorer_template.html")


def _r(a, nd=0):
    """Round for compact JSON; integers when nd == 0."""
    a = np.round(np.asarray(a, dtype=float), nd)
    return a.astype(int).tolist() if nd == 0 else a.tolist()


def _node_subset(grid) -> np.ndarray:
    """Every 2nd node inside cells, every barrier node (steep gradients there)."""
    keep = np.zeros(grid.n, dtype=bool)
    for c in range(grid.stack.n_cells):
        idx = np.flatnonzero(grid.cell_id == c)
        keep[idx[::2]] = True
        keep[idx[-1]] = True  # keep the node next to each interface
    keep[~grid.is_cell] = True
    return np.flatnonzero(keep)


def _layers(grid):
    return [{"x0": round(a * 1e3, 3), "x1": round(b * 1e3, 3), "kind": k}
            for a, b, k in grid.layer_edges]


def export_runaway(results, n_times: int = 260) -> dict:
    t_targets = np.concatenate([[0.0], np.geomspace(1e-2, 3600.0, n_times - 1)])
    out = {"t": _r(t_targets, 3), "designs": {}}
    for name, r in results.items():
        g = r.grid
        idx_t = np.clip(np.searchsorted(r.t, t_targets), 0, r.t.size - 1)
        nodes = _node_subset(g)
        b = g.stack.barrier
        out["designs"][name] = {
            "x": _r(g.x[nodes] * 1e3, 3),
            "layers": _layers(g),
            "T": [_r(row) for row in r.T_C[idx_t][:, nodes]],
            "Tcell": [_r(row, 1) for row in r.T_cell_mean_C[idx_t]],
            "alpha": [_r(row, 3) for row in g.cell_mean(r.alpha[idx_t])],
            "t_prop": [None if np.isnan(v) else round(float(v), 1) for v in r.t_prop],
            "n_prop": r.n_propagated,
            "energy_err": float(f"{r.max_relative_energy_error:.1e}"),
            "barrier": {"thickness_mm": b.thickness * 1e3, "k_on": b.k_on, "k_off": b.k_off,
                        "T_sw": b.T_sw_C if b.is_switchable else None},
        }
    return out


def export_convergence() -> dict:
    """Cell-2 propagation time vs mesh, with and without the rate ceiling."""
    base = SimParams(t_end=60.0, n_log_saves=2, save_every=60.0)
    ns = [10, 20, 40, 80, 160]
    rows = {}
    for label, kin in [("capped", KineticsParams()), ("uncapped", KineticsParams(k_max=np.inf))]:
        stack = replace(stack_with("conductor"), kinetics=kin)
        rows[label] = []
        for n in ns:
            if label == "uncapped" and n > 80:
                rows[label].append(None)  # unresolvable front: skip the costly runs
                continue
            sim = replace(base, n_nodes_cell=n, n_nodes_barrier=max(3, 3 * n // 10))
            rows[label].append(round(float(simulate_runaway(stack, sim).t_prop[1]), 2))
    return {"n": ns, **rows}


def build(
    out: Path, quick: bool = False, workers: int | None = None, fragment: bool = False
) -> dict:
    base = StackParams()
    runaway = compare_designs(base, SimParams())
    normal = normal_operation(base, SimParams())

    if quick:
        thick = np.linspace(0.5, 5.0, 7) * 1e-3
        koff = np.logspace(np.log10(0.02), 0, 7)
        tsw = np.concatenate([np.arange(25.0, 60.0, 5.0), np.arange(60.0, 251.0, 20.0)])
    else:
        thick = np.linspace(0.5, 5.0, 19) * 1e-3
        koff = np.logspace(np.log10(0.02), 0, 19)
        tsw = np.concatenate([np.arange(25.0, 60.0, 1.0), np.arange(60.0, 251.0, 5.0)])
    dm = design_map(thick, koff, base, workers=workers)
    sw = switch_temperature_sweep(tsw, StackParams(barrier=barrier_switchable()), workers=workers)

    normal_out = {}
    for name, s in normal.items():
        nodes = _node_subset(s.grid)
        normal_out[name] = {"x": _r(s.grid.x[nodes] * 1e3, 3), "T": _r(s.T_C[nodes], 2),
                            "peak": round(s.peak_C, 2), "layers": _layers(s.grid)}

    kin, cell = base.kinetics, base.cell
    data = {
        "params": {
            "n_cells": base.n_cells, "cell_mm": cell.thickness * 1e3, "k_cell": cell.k,
            "rho": cell.rho, "cp": cell.cp, "Ea_kJ": kin.Ea / 1e3, "A": kin.A,
            "dT_ad": kin.dT_ad, "k_max": kin.k_max, "h": base.h, "T_amb": base.T_amb_C,
            "q_normal": base.q_normal, "T_trigger": SimParams().T_trigger_C,
            "kelvin": KELVIN,
        },
        "runaway": export_runaway(runaway),
        "normal": normal_out,
        "design_map": {"thickness_mm": _r(dm.thickness * 1e3, 3), "k_off": _r(dm.k_off, 4),
                       "n": dm.n_propagated.astype(int).tolist(),
                       "default": {"thickness_mm": base.barrier.thickness * 1e3,
                                   "k_off": base.barrier.k_off}},
        "tsw": {"T_sw": _r(sw.T_sw_C, 1), "peak": _r(sw.normal_peak_C, 2),
                "n": sw.n_propagated.astype(int).tolist()},
        "convergence": export_convergence(),
    }

    html = TEMPLATE.read_text().replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    if not fragment:
        # A standalone page (e.g. GitHub Pages) needs a doctype for standards mode;
        # html/head/body tags are optional in HTML5.
        html = '<!doctype html>\n<html lang="en">\n<meta charset="utf-8">\n' + html
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    return data


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="docs/index.html")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--fragment", action="store_true",
                    help="omit the doctype wrapper (for hosts that add their own)")
    args = ap.parse_args(argv)
    out = Path(args.out)
    build(out, quick=args.quick, workers=args.workers, fragment=args.fragment)
    print(f"wrote {out} ({out.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
