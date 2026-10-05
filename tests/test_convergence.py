"""Grid and time-step convergence of the cell-2 propagation time."""

from dataclasses import replace

import numpy as np
import pytest

from runaway.params import SimParams
from runaway.scenarios import stack_with
from runaway.solver import simulate_runaway

# Static conductor: the design where propagation actually happens.
STACK = stack_with("conductor")
BASE = SimParams(t_end=40.0, n_log_saves=2, save_every=40.0)


def t_cell2(**kw) -> float:
    r = simulate_runaway(STACK, replace(BASE, **kw))
    assert np.isfinite(r.t_prop[1]), "cell 2 should propagate with a conductor"
    return float(r.t_prop[1])


def test_grid_convergence():
    """Refining the mesh 2x at a time: errors shrink and the default is converged."""
    t = {n: t_cell2(n_nodes_cell=n, n_nodes_barrier=max(3, 3 * n // 10)) for n in (20, 40, 80)}
    d1 = abs(t[40] - t[20])
    d2 = abs(t[80] - t[40])
    assert d2 < 0.5 * d1, f"not converging: {t}"
    # Observed order p from three grids should be about 2 (second-order FV)
    p = np.log2(d1 / d2)
    assert p > 1.5, f"observed order {p:.2f}"
    # The default mesh (40 per cell) is within 3 % of the finer one
    assert d2 / t[80] < 0.03


def test_timestep_convergence():
    """The adaptive step is controlled by dT_rxn_max (and capped by dt_max)."""
    t = {d: t_cell2(dT_rxn_max=d) for d in (8.0, 4.0, 2.0, 1.0)}
    diffs = [abs(t[8.0] - t[4.0]), abs(t[4.0] - t[2.0]), abs(t[2.0] - t[1.0])]
    assert diffs[0] > diffs[1] > diffs[2], f"not converging: {t}"
    assert diffs[2] / t[1.0] < 0.01  # default (2 K) is within 1 % of 1 K
    # dt_max barely matters because the reaction limit controls the step
    assert t_cell2(dt_max=0.1) == pytest.approx(t[2.0], rel=0.01)
