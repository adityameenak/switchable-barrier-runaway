"""Checks on the effective-medium composite model against known limits."""

from dataclasses import replace

import numpy as np
import pytest

from runaway.candidates import Value
from runaway.composites import (
    COMPOSITIONS, FILLERS, MATRICES, Composition, depolarization, effective_k,
)

SILICONE = MATRICES["silicone"]


def _sphere(k_p, a, R_K):
    return replace(FILLERS["caco3_micro"], k_11=Value(k_p, "W/(m K)", "assumed"),
                   k_33=Value(k_p, "W/(m K)", "assumed"), a1=a,
                   R_K=Value(R_K, "m^2 K/W", "assumed"), decomposition=None)


def _with(filler):
    FILLERS["_test"] = filler
    return "_test"


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    FILLERS.pop("_test", None)


def test_depolarization_limits():
    assert depolarization(1.0) == pytest.approx((1 / 3, 1 / 3))
    L11, L33 = depolarization(1e4)  # long rod
    assert L11 == pytest.approx(0.5, abs=1e-3) and L33 == pytest.approx(0.0, abs=1e-3)
    L11, L33 = depolarization(1e-4)  # thin disc
    assert L11 == pytest.approx(0.0, abs=1e-3) and L33 == pytest.approx(1.0, abs=1e-3)
    for p in (0.05, 0.5, 0.999, 1.001, 3.0, 300.0):
        L11, L33 = depolarization(p)
        assert 2 * L11 + L33 == pytest.approx(1.0)
    # continuous across p = 1
    assert depolarization(0.999)[0] == pytest.approx(depolarization(1.001)[0], abs=1e-3)


@pytest.mark.parametrize("phi", [0.05, 0.2, 0.4])
def test_spheres_without_interface_resistance_match_maxwell(phi):
    k_m, k_p = SILICONE.k.value, 3.5
    key = _with(_sphere(k_p, 1e-6, 0.0))
    maxwell = k_m * (k_p + 2 * k_m + 2 * phi * (k_p - k_m)) / (k_p + 2 * k_m - phi * (k_p - k_m))
    assert effective_k(SILICONE, {key: phi}) == pytest.approx(maxwell, rel=1e-12)


@pytest.mark.parametrize("a", [25e-9, 2.5e-6])
def test_spheres_with_interface_resistance_match_hasselman_johnson(a):
    k_m, k_p, R_K, phi = SILICONE.k.value, 3.5, 1e-8, 0.3
    key = _with(_sphere(k_p, a, R_K))
    al = R_K * k_m / a
    hj = k_m * (k_p * (1 + 2 * al) + 2 * k_m + 2 * phi * (k_p * (1 - al) - k_m)) / (
        k_p * (1 + 2 * al) + 2 * k_m - phi * (k_p * (1 - al) - k_m))
    assert effective_k(SILICONE, {key: phi}) == pytest.approx(hj, rel=1e-12)


def test_interface_resistance_makes_small_particles_worse_conductors():
    nano = effective_k(SILICONE, {"caco3_nano": 0.3})
    micro = effective_k(SILICONE, {"caco3_micro": 0.3})
    assert SILICONE.k.value < nano < micro
    cnt_ideal = replace(FILLERS["cnt"], R_K=Value(0.0, "m^2 K/W", "assumed"))
    assert effective_k(SILICONE, {_with(cnt_ideal): 0.01}) > 1.5 * effective_k(SILICONE, {"cnt": 0.01})


def test_insulating_filler_lowers_k_and_zero_loading_is_matrix():
    assert effective_k(SILICONE, {}) == pytest.approx(SILICONE.k.value)
    assert effective_k(SILICONE, {"glass_bubbles": 0.5}) < SILICONE.k.value


def test_loading_outside_model_range_is_rejected():
    with pytest.raises(ValueError):
        effective_k(SILICONE, {"cnt": 0.2})


def test_composition_mixture_rules():
    c = Composition("t", (("ath", 0.4),))
    p = c.properties()
    rho_f, rho_m = FILLERS["ath"].rho.value, SILICONE.rho.value
    assert p["rho"] == pytest.approx(0.4 * rho_f + 0.6 * rho_m)
    w = 0.4 * rho_f / p["rho"]
    assert p["latent_heat"] == pytest.approx(w * 1.17e6)
    assert p["T_melt_C"] == 260.0
    b = c.to_barrier(3e-3)
    assert b.thickness == 3e-3 and b.has_latent_heat and not b.is_switchable


def test_all_compositions_evaluate():
    for c in COMPOSITIONS.values():
        p = c.properties()
        assert np.isfinite(p["k"]) and p["k"] > 0
