"""Single-node chemistry: adiabatic limit and time history."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from runaway.materials import build_grid
from runaway.params import KELVIN, KineticsParams, SimParams, StackParams, barrier_none
from runaway.solver import run_transient


def to_K(T_C):
    return np.asarray(T_C, dtype=float) + KELVIN


def _single_node(kinetics=None):
    stack = StackParams(n_cells=1, barrier=barrier_none(), kinetics=kinetics or KineticsParams())
    sim = SimParams(n_nodes_cell=1, t_end=600.0, n_log_saves=400, save_every=1.0)
    return stack, sim, build_grid(stack, sim)


@pytest.mark.parametrize("T0_C", [180.0, 250.0, 400.0])
def test_adiabatic_node_reaches_T0_plus_dT_ad(T0_C):
    stack, sim, grid = _single_node()
    res = run_transient(grid, to_K([T0_C]), np.zeros(1), sim, h=0.0)
    assert res.alpha[-1, 0] == pytest.approx(1.0, abs=1e-9)
    assert res.T_C[-1, 0] == pytest.approx(T0_C + stack.kinetics.dT_ad, abs=1e-6)


@pytest.mark.parametrize("k_max", [1.0, np.inf])
def test_adiabatic_history_matches_ode_solver(k_max):
    """The split exponential update tracks a tight-tolerance ODE solution.

    For one adiabatic node T = T0 + dT_ad * alpha, so the system reduces to
    a single ODE for alpha. Compare the time to 50 % conversion (ignition).
    """
    kin = KineticsParams(k_max=k_max)
    _, sim, grid = _single_node(kin)
    T0 = to_K(200.0)
    res = run_transient(grid, np.array([T0]), np.zeros(1), sim, h=0.0)

    def rhs(t, a):
        return kin.rate_constant(T0 + kin.dT_ad * a) * (1.0 - a)

    def half(t, a):  # event: 50 % conversion
        return a[0] - 0.5

    half.terminal = True
    ref = solve_ivp(rhs, (0, sim.t_end), [0.0], method="Radau", rtol=1e-10, atol=1e-12,
                    events=half)
    t_half_ref = ref.t_events[0][0]

    t_half = np.interp(0.5, res.alpha[:, 0], res.t)
    assert t_half == pytest.approx(t_half_ref, rel=0.01)


def test_exponential_update_is_stable_for_huge_steps():
    """alpha stays in [0, 1] and energy is still exact with an absurd dt."""
    _, sim, grid = _single_node(KineticsParams(k_max=np.inf))
    sim = replace(sim, dt_max=100.0, dT_rxn_max=1e9, t_end=1000.0)
    res = run_transient(grid, to_K([300.0]), np.zeros(1), sim, h=0.0)
    assert np.all((res.alpha >= 0) & (res.alpha <= 1))
    assert res.T_C[-1, 0] == pytest.approx(900.0, abs=1e-6)
