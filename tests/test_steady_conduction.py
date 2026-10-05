"""Steady conduction through layers in series vs. the analytical solution."""

import numpy as np
import pytest

from runaway.materials import build_grid
from runaway.params import SimParams, StackParams, barrier_conductor, barrier_insulator
from runaway.solver import diffusion_step, solve_normal_operation


@pytest.mark.parametrize("barrier", [barrier_insulator(), barrier_conductor()])
@pytest.mark.parametrize("h", [50.0, 1e12])  # finite convection and ~Dirichlet ends
def test_series_resistance(barrier, h):
    """cell | barrier | cell between two different ambients.

    Exact solution: one heat flux q = dT / R_total with
        R_total = 1/h + L_c/k_c + L_b/k_b + L_c/k_c + 1/h
    and temperature linear inside each layer. Harmonic-mean face
    conductances make the finite-volume solution exact (to round-off)
    for piecewise-linear profiles, even across a 27x conductivity jump.
    """
    stack = StackParams(n_cells=2, barrier=barrier, h=h)
    sim = SimParams(n_nodes_cell=7, n_nodes_barrier=3)  # deliberately coarse
    grid = build_grid(stack, sim)
    T_left, T_right = 400.0, 300.0
    k = grid.conductivity(np.full(grid.n, 350.0))

    T, _ = diffusion_step(grid, np.zeros(grid.n), np.inf, k, h, (T_left, T_right))

    kc, Lc = stack.cell.k, stack.cell.thickness
    kb, Lb = barrier.k_on, barrier.thickness
    R = 2.0 / h + 2.0 * Lc / kc + Lb / kb
    q = (T_left - T_right) / R

    # Analytical profile: integrate the resistance from the left ambient
    def T_exact(x):
        r = 1.0 / h
        out = np.empty_like(x)
        for i, xi in enumerate(x):
            if xi <= Lc:
                ri = r + xi / kc
            elif xi <= Lc + Lb:
                ri = r + Lc / kc + (xi - Lc) / kb
            else:
                ri = r + Lc / kc + Lb / kb + (xi - Lc - Lb) / kc
            out[i] = T_left - q * ri
        return out

    np.testing.assert_allclose(T, T_exact(grid.x), rtol=0, atol=1e-8)


def test_normal_operation_matches_conductor_closed_form():
    """Uniform generation, conductive barrier: peak equals the closed form.

    Treating the conductive barrier as part of the conduction path, total heat
    N q L_c leaves symmetrically through the two ends. Checked here for a
    single cell, where the exact peak is
        T_amb + (q L / 2) / h + q L^2 / (8 k).
    """
    stack = StackParams(n_cells=1)
    ss = solve_normal_operation(stack, SimParams(n_nodes_cell=41))
    q, L, k, h = stack.q_normal, stack.cell.thickness, stack.cell.k, stack.h
    exact = stack.T_amb_C + 0.5 * q * L / h + q * L**2 / (8 * k)
    assert ss.converged
    # FV peak is at the centre node; profile is quadratic -> O(dx^2) error.
    assert ss.peak_C == pytest.approx(exact, abs=0.01)
