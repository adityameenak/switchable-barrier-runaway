"""Checks on the formulation model: limits, calibration points and switch mapping."""

from dataclasses import replace

import numpy as np
import pytest

from runaway.composites import MATRICES, effective_k
from runaway.formulation import (
    CAL_TARGET,
    PARTICLES,
    R_K_POOR,
    Formulation,
    ag_wire_k,
    agnw,
    emt,
    epoxy,
    fit_R_K,
    formulation_k,
    network_k,
)

SILICONE = MATRICES["silicone"]


def test_particles_only_matches_composites_module():
    # Without wires or spheres the two-step model is the Nan model of composites.py.
    k = formulation_k(SILICONE, [(PARTICLES["ath"], 0.3), (PARTICLES["hbn"], 0.1)])["k"]
    assert k == pytest.approx(effective_k(SILICONE, {"ath": 0.3, "hbn": 0.1}), rel=1e-12)


def test_wire_size_scaling_anchor():
    assert ag_wire_k(84e-9) == pytest.approx(332.0)
    assert ag_wire_k(40e-9) < ag_wire_k(84e-9) < ag_wire_k(1e-6) < 429.0


def test_percolation_threshold_excluded_volume():
    w = agnw(50, 20)
    assert w.phi_c == pytest.approx(50e-9 / (2 * 20e-6))
    assert network_k(w, 0.99 * w.phi_c, 0.2) == 0.0
    # quadratic above threshold
    d = 0.01
    assert network_k(w, w.phi_c + 2 * d, 0.2) == pytest.approx(4 * network_k(w, w.phi_c + d, 0.2))


def test_fin_efficiency_limits():
    w = agnw(100, 15)
    eta = w.fin_efficiency(0.2)
    assert 0.0 < eta < 1.0
    short = replace(w, L=200e-9)  # short rod: nearly isothermal
    assert short.fin_efficiency(0.2) > 0.95
    assert w.fin_efficiency(2.0) < eta  # better host -> more exchange -> shorter fin length


def test_calibration_points():
    # Chen 2014 coated wires (no fit): within 10 %.
    k = formulation_k(epoxy(0.19), [], agnw(100, 15, True), 0.04)["k"]
    assert k == pytest.approx(CAL_TARGET["coated"], rel=0.10)
    # Zhang 2019 (no fit): within 10 %.
    k = formulation_k(epoxy(0.22), [], agnw(120, 15, False), 0.00455)["k"]
    assert k == pytest.approx(0.35, rel=0.10)
    # Chen 2014 uncoated reproduced by the poor-dispersion R_K.
    assert fit_R_K(CAL_TARGET["uncoated"]) == pytest.approx(R_K_POOR, rel=0.02)


def test_wires_far_below_nan_bound():
    from runaway.composites import COMPOSITIONS

    k_nan = COMPOSITIONS["Ag NW 5%"].properties()["k"]
    k_fin = formulation_k(SILICONE, [], agnw(84, 20), 0.05)["k"]
    assert k_fin < 0.5 * k_nan


def test_expansion_lowers_k_and_maps_to_one_way_switch():
    f = Formulation("t", (("ath", 0.1),), agnw(40, 40), 0.01, "ems_051", 0.3)
    kc, kh = f.k_cold()["k"], f.k_hot()["k"]
    assert kh < kc
    assert replace(f, E=10.0).k_hot()["k"] < kh
    # segregated network is the conservative bound: insulates less
    assert replace(f, network="segregated").k_hot()["k"] >= kh
    b = f.to_barrier(3e-3)
    assert b.k_on == pytest.approx(kc) and b.k_off == pytest.approx(kh)
    assert b.switch_irreversible and b.is_switchable
    assert 107.5 < b.T_sw_C < 146.5
    assert b.latent_heat > 0 and b.T_melt_C == 260.0


def test_no_spheres_is_static_and_mass_balance():
    f = Formulation("t", (("al2o3_micro", 0.2),), agnw(60, 20), 0.02)
    p = f.properties()
    assert p["k_on"] == p["k_off"]
    rho = 0.78 * SILICONE.rho.value + 0.2 * 3950.0 + 0.02 * 10490.0
    assert p["rho"] == pytest.approx(rho)


def test_emt_dilute_sphere_limit():
    # Maxwell, first order in phi: k/k_m = 1 + 3 phi (k_p - k_m)/(k_p + 2 k_m)
    f, phi = PARTICLES["al2o3_micro"], 1e-5
    f0 = replace(f, R_K=replace(f.R_K, value=0.0))
    k = emt(0.2, [(f0, phi)])
    kp = f.k_11.value
    assert (k / 0.2 - 1) / phi == pytest.approx(3 * (kp - 0.2) / (kp + 0.4), rel=1e-4)
    assert np.isfinite(k)
