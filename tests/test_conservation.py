"""Energy conservation of the discrete scheme."""

from dataclasses import replace

import numpy as np

from runaway.materials import build_grid
from runaway.params import KELVIN
from runaway.solver import run_transient


def to_K(T_C):
    return np.asarray(T_C, dtype=float) + KELVIN


def _hot_trigger_state(grid):
    T0 = np.full(grid.n, to_K(25.0))
    T0[grid.cell_id == 0] = to_K(400.0)
    return T0, np.zeros(grid.n)


def test_insulated_no_reaction_conserves_thermal_energy(switch_stack, coarse_sim):
    """h = 0, chemistry off: sum(rho cp T dx) must stay constant.

    Uses the switchable barrier so the conductivity is strongly nonlinear and
    spans a 100x jump; conservation must still hold to round-off because the
    scheme is in flux form.
    """
    sim = replace(coarse_sim, t_end=2000.0, dt_max=1.0)
    grid = build_grid(switch_stack, sim)
    T0, a0 = _hot_trigger_state(grid)

    res = run_transient(grid, T0, a0, sim, h=0.0, react=False)

    E = (res.T_C + 273.15) @ (grid.rho_cp * grid.dx)
    assert np.allclose(E, E[0], rtol=1e-12, atol=0.0)
    # ...and heat really did move: the trigger cell cooled, its neighbour warmed
    Tm = res.T_cell_mean_C
    assert Tm[-1, 0] < 300.0 and Tm[-1, 1] > 40.0


def test_insulated_with_reaction_conserves_total_energy(switch_stack, coarse_sim):
    """h = 0, chemistry on: thermal + unreleased chemical energy is constant."""
    sim = replace(coarse_sim, t_end=600.0)
    grid = build_grid(switch_stack, sim)
    T0, a0 = _hot_trigger_state(grid)

    res = run_transient(grid, T0, a0, sim, h=0.0, react=True)

    assert res.alpha[-1][grid.cell_id == 0].min() > 0.99  # the trigger did react
    assert res.max_relative_energy_error < 1e-10


def test_energy_budget_closes_with_convective_losses(switch_stack, coarse_sim):
    """With h > 0 the tracked boundary loss balances the change in energy."""
    sim = replace(coarse_sim, t_end=1200.0)
    grid = build_grid(switch_stack, sim)
    T0, a0 = _hot_trigger_state(grid)

    res = run_transient(grid, T0, a0, sim, react=True)

    assert res.max_relative_energy_error < 1e-10
