"""Regression tests for the headline result (with the illustrative defaults)."""

import numpy as np
import pytest

from runaway.materials import switch_conductivity
from runaway.params import KELVIN, SimParams, barrier_switchable
from runaway.scenarios import SWEEP_SIM, compare_designs, normal_operation


def test_switch_conductivity_limits():
    b = barrier_switchable(k_on=5.0, k_off=0.05, T_sw_C=120.0, sw_width=5.0)
    T = np.array([20.0, 120.0, 220.0]) + KELVIN
    k = switch_conductivity(T, b)
    assert k[0] == pytest.approx(5.0, rel=1e-6)
    assert k[1] == pytest.approx(np.sqrt(5.0 * 0.05))  # geometric mean at T_sw
    assert k[2] == pytest.approx(0.05, rel=1e-6)


def test_tradeoff_story():
    """Conductor: cool but propagates. Insulator: contains but hot. Switch: both."""
    runaway = compare_designs(sim=SWEEP_SIM)
    normal = normal_operation(sim=SimParams())
    assert runaway["none"].n_propagated == 5
    assert runaway["conductor"].n_propagated == 5
    assert runaway["insulator"].n_propagated == 1
    assert runaway["switchable"].n_propagated == 1

    assert normal["insulator"].peak_C > normal["conductor"].peak_C + 10.0
    assert normal["switchable"].peak_C == pytest.approx(normal["conductor"].peak_C, abs=0.5)
