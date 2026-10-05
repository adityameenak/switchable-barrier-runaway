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
