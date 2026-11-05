"""Sanity checks on the sourced barrier-material data."""

import pytest

from runaway.candidates import CANDIDATES, SOURCES, Value, to_barrier, to_markdown


def test_every_sourced_value_cites_a_known_source():
    for c in CANDIDATES.values():
        if c.k_provenance != "assumed":
            assert c.k_source in SOURCES, c.key
        for v in (c.rho, c.cp, c.T_max_C, c.latent_heat, c.T_melt_C):
            if v is not None and v.provenance != "assumed":
                assert v.source in SOURCES, c.key


def test_value_rejects_missing_source():
    with pytest.raises(ValueError):
        Value(1.0, "W/(m K)", "datasheet", None)
    with pytest.raises(ValueError):
        Value(1.0, "W/(m K)", "made-up")


@pytest.mark.parametrize("c", CANDIDATES.values(), ids=lambda c: c.key)
def test_physical_ranges(c):
    temps = [T for T, _ in c.k_table]
    assert temps == sorted(temps)
    assert all(0.005 < k < 50.0 for _, k in c.k_table)  # W/(m K), solids
    assert 50.0 < c.rho.value < 5000.0  # kg/m^3
    assert 300.0 < c.cp.value < 3000.0  # J/(kg K)
    if c.mechanism == "pcm":
        assert c.latent_heat is not None and c.T_melt_C is not None
    if c.mechanism == "switch":
        assert c.k_table[0][1] > c.k_table[-1][1]  # conducts cold, insulates hot


def test_markdown_lists_all_candidates_and_sources():
    md = to_markdown()
    for c in CANDIDATES.values():
        assert c.name in md
    for s in SOURCES.values():
        assert s.key in md


@pytest.mark.parametrize("key", CANDIDATES)
def test_to_barrier_reproduces_published_k(key):
    import numpy as np

    from runaway.materials import switch_conductivity
    from runaway.params import KELVIN

    b = to_barrier(key)
    c = CANDIDATES[key]
    T_C, k = c.k_table[0]
    got = switch_conductivity(np.array([T_C + KELVIN]), b)[0]
    assert got == pytest.approx(k, rel=0.05)
    assert b.rho_cp == pytest.approx(c.rho.value * c.cp.value)
    if c.mechanism == "switch":
        hot = switch_conductivity(np.array([c.k_table[-1][0] + KELVIN]), b)[0]
        assert hot == pytest.approx(c.k_table[-1][1], rel=0.15)
