"""Verification of the 2D solver."""

from dataclasses import replace

import numpy as np
import pytest

from runaway.params import KELVIN, SimParams, StackParams, barrier_conductor, barrier_switchable
from runaway.solver import simulate_runaway, solve_normal_operation
from runaway.solver2d import (
    SIM2D, Params2D, build_grid2d, reduce_to_1d, run_transient2d, simulate_runaway2d,
    solve_normal_operation2d,
)


@pytest.mark.parametrize("barrier", [barrier_conductor(), barrier_switchable()], ids=["cond", "switch"])
def test_reduces_to_1d_without_plate(barrier):
    stack = StackParams(barrier=barrier)
    p2 = reduce_to_1d(Params2D(ny=3))
    r1, r2 = simulate_runaway(stack, SIM2D), simulate_runaway2d(stack, p2, SIM2D)
    np.testing.assert_allclose(r2.t_prop, r1.t_prop, rtol=1e-9, equal_nan=True)
    assert solve_normal_operation2d(stack, p2, SIM2D).peak_C == pytest.approx(
        solve_normal_operation(stack, SIM2D).peak_C, abs=1e-8)


def test_energy_closes_with_plate_and_latent_heat():
    b = replace(barrier_switchable(), latent_heat=2e5, T_melt_C=74.0)
    stack = StackParams(barrier=b)
    r = simulate_runaway2d(stack, Params2D(ny=6), replace(SIM2D, t_end=300.0))
    assert r.max_relative_energy_error < 1e-9


def test_steady_plate_only_matches_fin_free_slab():
    """One cell, no ends/top, heated uniformly, cooled by the plate only:
    T(y) = T_p + q H / h + q (H y - y^2 / 2) / k_y. The FV solution matches at
    every node up to a uniform shift q dy^2 / (8 k_y), the error of the linear
    half-cell at the cooled boundary."""
    stack = StackParams(n_cells=1, h=0.0)
    p2 = Params2D(ny=10, h_top=0.0, h_plate=200.0)
    s = solve_normal_operation2d(stack, p2, SIM2D)
    g = s.grid
    y = (np.arange(p2.ny) + 0.5) * g.dy
    q, H, k = stack.q_normal, p2.height, p2.k_cell_y
    exact = p2.T_plate_C + q * H / p2.h_plate + q * (H * y - y**2 / 2) / k + q * g.dy**2 / (8 * k)
    np.testing.assert_allclose(s.T_C.reshape(p2.ny, g.nx)[:, 0], exact, atol=1e-8)


def test_cold_plate_cools_normal_operation():
    stack = StackParams(barrier=barrier_conductor())
    assert solve_normal_operation2d(stack).peak_C < solve_normal_operation(stack).peak_C - 3.0
