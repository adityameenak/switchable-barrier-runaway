"""Shared fixtures for the validation tests."""

import pytest

from runaway.params import SimParams, StackParams, barrier_switchable


@pytest.fixture
def switch_stack() -> StackParams:
    """Default stack with the (nonlinear) switchable barrier."""
    return StackParams(barrier=barrier_switchable())


@pytest.fixture
def coarse_sim() -> SimParams:
    """Cheap numerical settings for tests that do not probe resolution."""
    return SimParams(n_nodes_cell=10, n_nodes_barrier=4, n_log_saves=10, save_every=10.0)
