"""Named studies built on the solver: design comparison and parameter sweeps."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace

import numpy as np

from .params import DESIGNS, SimParams, StackParams, barrier_switchable
from .solver import RunawayResult, SteadyResult, simulate_runaway, solve_normal_operation

# Sweeps run many short simulations; let each one stop as soon as the outcome
# is decided (all cells gone, or the whole stack back below 60 C), and give
# slow ignitions a generous 2 h window.
SWEEP_SIM = SimParams(stop_when_settled=True, t_end=7200.0, save_every=30.0, n_log_saves=20)


def stack_with(design: str, base: StackParams | None = None, **barrier_kw) -> StackParams:
    """Stack using one of the named barrier designs (see ``params.DESIGNS``)."""
    base = base or StackParams()
    return replace(base, barrier=DESIGNS[design](**barrier_kw))


def compare_designs(
    base: StackParams | None = None, sim: SimParams | None = None
) -> dict[str, RunawayResult]:
    """Runaway transient for every named barrier design."""
    return {name: simulate_runaway(stack_with(name, base), sim) for name in DESIGNS}


def normal_operation(
    base: StackParams | None = None, sim: SimParams | None = None
) -> dict[str, SteadyResult]:
    """Normal-operation steady state for every named barrier design."""
    return {name: solve_normal_operation(stack_with(name, base), sim) for name in DESIGNS}


# ---------------------------------------------------------------------------
# Sweeps (parallelised over processes; each task is an independent run)
# ---------------------------------------------------------------------------


def _n_propagated(stack: StackParams) -> int:
    return simulate_runaway(stack, SWEEP_SIM).n_propagated


def _pmap(func, items, workers: int | None):
    workers = workers or os.cpu_count() or 1
    if workers == 1:
        return [func(i) for i in items]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(func, items, chunksize=4))


@dataclass
class DesignMap:
    thickness: np.ndarray  # [m]
    k_off: np.ndarray  # [W/(m K)]
    n_propagated: np.ndarray  # (n_k_off, n_thickness), includes trigger cell


def design_map(
    thickness: np.ndarray,
    k_off: np.ndarray,
    base: StackParams | None = None,
    workers: int | None = None,
) -> DesignMap:
    """Cells in runaway for a switchable barrier over (thickness, k_off)."""
    base = base or StackParams()
    b0 = base.barrier
    stacks = [
        replace(base, barrier=replace(b0, thickness=float(L), k_off=float(ko)))
        for ko in k_off
        for L in thickness
    ]
    n = np.array(_pmap(_n_propagated, stacks, workers)).reshape(len(k_off), len(thickness))
    return DesignMap(np.asarray(thickness), np.asarray(k_off), n)


@dataclass
class SwitchSweep:
    T_sw_C: np.ndarray
    normal_peak_C: np.ndarray  # steady peak temperature in normal operation
    n_propagated: np.ndarray  # cells in runaway after triggering cell 1


def _tsw_point(stack: StackParams) -> tuple[float, int]:
    return solve_normal_operation(stack).peak_C, _n_propagated(stack)


def switch_temperature_sweep(
    T_sw_C: np.ndarray, base: StackParams | None = None, workers: int | None = None
) -> SwitchSweep:
    """Normal-operation peak and propagation outcome vs. switch temperature."""
    base = base or StackParams(barrier=barrier_switchable())
    stacks = [replace(base, barrier=replace(base.barrier, T_sw_C=float(T))) for T in T_sw_C]
    out = _pmap(_tsw_point, stacks, workers)
    peak, n = (np.array(v) for v in zip(*out))
    return SwitchSweep(np.asarray(T_sw_C), peak, n.astype(int))


# ---------------------------------------------------------------------------
# Real-material comparison (sourced data in ``candidates.py``)
# ---------------------------------------------------------------------------

# Thicknesses scanned for each candidate [mm].
MATERIAL_THICKNESS_MM = np.round(np.arange(0.5, 10.0 + 1e-9, 0.25), 2)


@dataclass
class MaterialSweep:
    key: str
    thickness_mm: np.ndarray
    n_propagated: np.ndarray
    normal_peak_C: np.ndarray

    @property
    def min_containing_mm(self) -> float | None:
        """Thinnest barrier that keeps runaway to the trigger cell (None = never)."""
        ok = self.n_propagated == 1
        return float(self.thickness_mm[ok][0]) if ok.any() else None

    @property
    def peak_at_min_C(self) -> float | None:
        ok = self.n_propagated == 1
        return float(self.normal_peak_C[ok][0]) if ok.any() else None


def _material_point(args) -> tuple[int, float]:
    key, L_mm, overrides, base = args
    from .candidates import to_barrier

    stack = replace(base, barrier=to_barrier(key, L_mm * 1e-3, **overrides))
    return _n_propagated(stack), solve_normal_operation(stack).peak_C


def material_sweep(
    key: str,
    thickness_mm: np.ndarray = MATERIAL_THICKNESS_MM,
    base: StackParams | None = None,
    workers: int | None = None,
    **overrides,
) -> MaterialSweep:
    """Containment and normal-operation peak vs. thickness for one candidate.

    ``overrides`` are passed to ``candidates.to_barrier`` (sensitivity runs).
    """
    base = base or StackParams()
    out = _pmap(_material_point, [(key, float(L), overrides, base) for L in thickness_mm], workers)
    n, peak = (np.array(v) for v in zip(*out))
    return MaterialSweep(key, np.asarray(thickness_mm), n.astype(int), peak)


# Sensitivity cases: (label, candidate, overrides). Each changes one input the
# sourced data leave open, by a plausible amount.
MATERIAL_SENSITIVITY = [
    ("switch: T_sw = 70 C", "graphene_switch", {"T_sw_C": 70.0}),
    ("switch: T_sw = 100 C", "graphene_switch", {"T_sw_C": 100.0}),
    ("switch: instant (tau = 0)", "graphene_switch", {"switch_tau": 0.0}),
    ("switch: slow (tau = 30 s)", "graphene_switch", {"switch_tau": 30.0}),
    ("switch: reversible", "graphene_switch", {"switch_irreversible": False}),
    ("switch: rho cp x2", "graphene_switch", {"rho": 2000.0}),
    ("switch: k_off x1.5", "graphene_switch", {"k_off": 0.15}),
    ("aerogel: rho cp x0.7", "aerogel", {"cp": 700.0}),
    ("PCM: no latent heat", "pcm_eg", {"latent_heat": 0.0}),
]


# ---------------------------------------------------------------------------
# 2D studies (cold plate): real materials, ideal designs and composites
# ---------------------------------------------------------------------------

# A barrier is named by a picklable spec so sweeps can run in worker processes:
#   ("design", name)                  idealised designs in params.DESIGNS
#   ("candidate", key, overrides)     sourced materials in candidates.py
#   ("composition", name)             filled silicones in composites.py


def barrier_from_spec(spec, thickness: float):
    kind = spec[0]
    if kind == "design":
        b = DESIGNS[spec[1]]()
        return replace(b, thickness=thickness)
    if kind == "candidate":
        from .candidates import to_barrier

        return to_barrier(spec[1], thickness, **(spec[2] if len(spec) > 2 else {}))
    if kind == "composition":
        from .composites import COMPOSITIONS

        return COMPOSITIONS[spec[1]].to_barrier(thickness)
    if kind == "formulation":  # ("formulation", Formulation[, overrides])
        return spec[1].to_barrier(thickness, **(spec[2] if len(spec) > 2 else {}))
    raise ValueError(spec)


def _point2d(args):
    from .solver2d import simulate_runaway2d, solve_normal_operation2d

    spec, L_mm, p2 = args
    stack = StackParams(barrier=barrier_from_spec(spec, L_mm * 1e-3))
    return simulate_runaway2d(stack, p2).n_propagated, solve_normal_operation2d(stack, p2).peak_C


@dataclass
class Containment2D:
    spec: tuple
    thickness_mm: np.ndarray  # scanned (and refined) thicknesses, sorted
    n_propagated: np.ndarray
    normal_peak_C: np.ndarray

    @property
    def min_containing_mm(self) -> float | None:
        ok = self.n_propagated == 1
        return float(self.thickness_mm[ok][0]) if ok.any() else None

    @property
    def peak_at_min_C(self) -> float | None:
        ok = self.n_propagated == 1
        return float(self.normal_peak_C[ok][0]) if ok.any() else None


SCAN2D_MM = np.arange(0.5, 10.0 + 1e-9, 0.5)


def containment_scan2d(specs, p2=None, scan_mm=SCAN2D_MM, refine_to_mm=0.125, workers=None):
    """Scan thickness for every spec, then bisect each containment boundary.

    All points of one round run in parallel. Returns {spec: Containment2D}.
    """
    from .solver2d import Params2D

    p2 = p2 or Params2D()
    pts = {spec: {} for spec in specs}
    todo = [(spec, float(L), p2) for spec in specs for L in scan_mm]
    while todo:
        for (spec, L, _), out in zip(todo, _pmap(_point2d, todo, workers)):
            pts[spec][L] = out
        todo = []
        for spec, d in pts.items():
            Ls = sorted(d)
            ok = [L for L in Ls if d[L][0] == 1]
            if not ok:
                continue
            first = ok[0]
            below = [L for L in Ls if L < first]
            if below and first - below[-1] > refine_to_mm + 1e-9:
                todo.append((spec, round(0.5 * (first + below[-1]), 6), p2))
    out = {}
    for spec, d in pts.items():
        Ls = np.array(sorted(d))
        out[spec] = Containment2D(spec, Ls, np.array([d[L][0] for L in Ls]),
                                  np.array([d[L][1] for L in Ls]))
    return out
