"""Verification of the real-material extensions: tabulated k(T), latent heat, switch state."""

import numpy as np
import pytest
from dataclasses import replace

from runaway.materials import build_grid, table_conductivity
from runaway.params import KELVIN, SimParams, StackParams, barrier_switchable
from runaway.solver import run_transient, simulate_runaway


def test_table_conductivity_hits_points_and_clamps():
    table = ((0.0, 0.02), (100.0, 0.04), (200.0, 0.08))
    T = np.array([-50.0, 0.0, 100.0, 150.0, 200.0, 900.0]) + KELVIN
    k = table_conductivity(T, table)
    assert k == pytest.approx([0.02, 0.02, 0.04, np.sqrt(0.04 * 0.08), 0.08, 0.08])


def _pcm_stack():
    b = replace(barrier_switchable(), k_on=2.0, k_off=2.0, rho=1000.0, cp=2000.0,
                latent_heat=200e3, T_melt_C=74.0, melt_width=3.0)
    return StackParams(n_cells=2, barrier=b)


def test_latent_heat_conserves_enthalpy_and_sets_final_temperature():
    """Insulated 2-cell stack, one cell hot, no chemistry: the final uniform
    temperature must satisfy the enthalpy balance including melting."""
    stack = _pcm_stack()
    sim = SimParams(n_nodes_cell=10, n_nodes_barrier=6, t_end=4.0e4, save_every=1000.0,
                    n_log_saves=10, dt_max=20.0)
    grid = build_grid(stack, sim)
    T0 = np.full(grid.n, 25.0 + KELVIN)
    T0[grid.cell_id == 0] = 200.0 + KELVIN
    res = run_transient(grid, T0, np.zeros(grid.n), sim, h=0.0, react=False)
    assert res.max_relative_energy_error < 1e-9

    T_end = res.T_C[-1] + KELVIN
    assert np.ptp(T_end) < 0.01  # equilibrated
    H0 = np.sum(grid.enthalpy(T0, KELVIN) * grid.dx)
    H1 = np.sum(grid.enthalpy(T_end, KELVIN) * grid.dx)
    assert H1 == pytest.approx(H0, rel=1e-9)
    # Without latent heat the stack would end hotter
    no_pcm = np.sum(grid.rho_cp * (T0 - KELVIN) * grid.dx) / np.sum(grid.rho_cp * grid.dx)
    assert T_end.mean() - KELVIN < no_pcm - 1.0


def test_one_way_switch_stays_insulating_after_cooling():
    """Heat the barrier past T_sw, let it cool: a one-way switch stays off."""
    b = barrier_switchable(T_sw_C=85.0)
    sim = SimParams(n_nodes_cell=10, n_nodes_barrier=4, t_end=3000.0, save_every=100.0,
                    n_log_saves=10, T_trigger_C=200.0)
    no_chem = replace(StackParams().kinetics, dT_ad=0.0)
    for irreversible in (True, False):
        stack = StackParams(kinetics=no_chem,
                            barrier=replace(b, switch_irreversible=irreversible, switch_tau=1.0))
        res = simulate_runaway(stack, sim)
        assert res.T_C[-1].max() < 60.0  # cooled well below T_sw again
        s_first = res.meta["s_switch"][:sim.n_nodes_barrier]  # barrier next to trigger cell
        if irreversible:
            # Only the hot face crosses T_sw: once it switches it shields the rest.
            assert s_first[0] < 0.05
        else:
            assert s_first.min() > 0.95
        assert res.max_relative_energy_error < 1e-9


def test_switch_time_constant_delays_switching():
    b = barrier_switchable(T_sw_C=85.0)
    sim = SimParams(n_nodes_cell=10, n_nodes_barrier=4, t_end=5.0, save_every=1.0, n_log_saves=5,
                    T_trigger_C=300.0)
    no_chem = replace(StackParams().kinetics, dT_ad=0.0)
    fast = simulate_runaway(StackParams(kinetics=no_chem, barrier=replace(b, switch_tau=0.0,
                                        switch_irreversible=True)), sim)
    slow = simulate_runaway(StackParams(kinetics=no_chem, barrier=replace(b, switch_tau=100.0,
                                        switch_irreversible=True)), sim)
    n = sim.n_nodes_barrier
    assert slow.meta["s_switch"][:n].min() > fast.meta["s_switch"][:n].min() + 0.5


def test_material_ranking_headline():
    """Paper result: at 2 mm nothing contains; at 4 mm the switch contains with a cool
    pack, aerogel contains but runs hot, and the gap pad still propagates."""
    from runaway.candidates import to_barrier
    from runaway.scenarios import SWEEP_SIM
    from runaway.solver import solve_normal_operation

    def run(key, L):
        st = StackParams(barrier=to_barrier(key, L))
        return simulate_runaway(st, SWEEP_SIM).n_propagated, solve_normal_operation(st).peak_C

    assert run("aerogel", 2e-3)[0] == 5
    assert run("graphene_switch", 2e-3)[0] == 5
    n_sw, peak_sw = run("graphene_switch", 4e-3)
    n_ae, peak_ae = run("aerogel", 4e-3)
    assert n_sw == 1 and n_ae == 1
    assert peak_sw < 40.0 < 100.0 < peak_ae
    assert run("gap_pad", 4e-3)[0] == 5
