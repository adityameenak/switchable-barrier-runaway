"""Filled-polymer barrier compositions: effective properties from constituents.

A composite barrier is a matrix (here a silicone elastomer) loaded with one or
more fillers. Its through-plane conductivity is estimated with the
effective-medium model of Nan et al. (J. Appl. Phys. 81, 6692, 1997) for
randomly oriented ellipsoidal inclusions with an interfacial (Kapitza)
resistance R_K:

    k / k_m = [3 + sum_i f_i (2 b11 (1 - L11) + b33 (1 - L33))]
              / [3 - sum_i f_i (2 b11 L11 + b33 L33)]

    b_jj   = (kc_jj - k_m) / (k_m + L_jj (kc_jj - k_m))
    kc_jj  = k_p,jj / (1 + gamma L_jj k_p,jj / k_m)
    gamma  = (2 + 1/p) a_K / a1   (p >= 1, rods)
           = (1 + 2p)  a_K / a3   (p <= 1, platelets)
    a_K    = R_K k_m              (Kapitza radius)

with p = a3/a1 the aspect ratio of a spheroid with semi-axes a1 = a2 and a3
(axis 3 is the symmetry axis) and L the depolarisation factors. For spheres
this is exactly the Hasselman-Johnson result, and for R_K = 0 the Maxwell
result. The model ignores percolation, so it is used only at loadings below
the percolation threshold of each filler shape (``Filler.phi_max``).

The size of a particle matters only through a_K / a: when the Kapitza radius
is comparable to the particle size (nanoparticles, nanotubes), the interface
dominates and the filler conducts far less than its bulk value suggests.

Endothermic fillers (Al(OH)3, Mg(OH)2, CaCO3) decompose on heating and absorb
heat. The decomposition is represented with the barrier's latent-heat
mechanism (a Gaussian release over the decomposition range), carrying the
filler's mass fraction of its decomposition enthalpy. The reaction is
irreversible and releases water vapour or CO2; neither is modelled.

Provenance follows ``candidates.py``: each value is "datasheet", "paper",
"secondary" or "assumed".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .candidates import Value
from .params import BarrierParams

@dataclass(frozen=True)
class Matrix:
    name: str
    k: Value
    rho: Value
    cp: Value
    T_max_C: float


@dataclass(frozen=True)
class Filler:
    """A filler phase. Spheroid with semi-axes a1 = a2 and a3; p = a3 / a1.

    ``k_11`` is the particle conductivity along axis 1 (and 2), ``k_33`` along
    the symmetry axis: for a rod (p > 1) k_33 is the axial value, for a
    platelet (p < 1) k_11 is the in-plane value.
    """

    key: str
    name: str
    k_11: Value
    k_33: Value
    rho: Value
    cp: Value
    a1: float  # [m] semi-axis 1
    p: float  # aspect ratio a3 / a1
    R_K: Value  # [m^2 K/W] interfacial resistance with the matrix
    phi_max: float  # largest volume fraction the (non-percolating) model is used at
    decomposition: tuple[Value, float, float] | None = None  # (dH [J/kg], T_c [C], sigma [K])
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def a3(self) -> float:
        return self.a1 * self.p


def depolarization(p: float) -> tuple[float, float]:
    """Depolarisation factors (L11, L33) of a spheroid with aspect ratio p = a3/a1."""
    if abs(p - 1.0) < 1e-9:
        return 1.0 / 3.0, 1.0 / 3.0
    if p > 1.0:
        L11 = p**2 / (2 * (p**2 - 1)) - p / (2 * (p**2 - 1) ** 1.5) * np.arccosh(p)
    else:
        L11 = p**2 / (2 * (p**2 - 1)) + p / (2 * (1 - p**2) ** 1.5) * np.arccos(p)
    return float(L11), float(1.0 - 2.0 * L11)


def _filler_terms(f: Filler, k_m: float) -> tuple[float, float]:
    """Numerator and denominator contributions per unit volume fraction (Nan 1997)."""
    L11, L33 = depolarization(f.p)
    a_K = f.R_K.value * k_m
    gamma = (2.0 + 1.0 / f.p) * a_K / f.a1 if f.p >= 1.0 else (1.0 + 2.0 * f.p) * a_K / f.a3
    out = []
    for kp, L in ((f.k_11.value, L11), (f.k_33.value, L33)):
        kc = kp / (1.0 + gamma * L * kp / k_m)
        out.append(((kc - k_m) / (k_m + L * (kc - k_m)), L))
    (b11, L11), (b33, L33) = out
    num = 2 * b11 * (1 - L11) + b33 * (1 - L33)
    den = 2 * b11 * L11 + b33 * L33
    return num, den


def effective_k(matrix: Matrix, loading: dict[str, float]) -> float:
    """Effective conductivity [W/(m K)] for {filler key: volume fraction}."""
    k_m = matrix.k.value
    num = den = 0.0
    for key, phi in loading.items():
        f = FILLERS[key]
        if not 0.0 <= phi <= f.phi_max + 1e-12:
            raise ValueError(f"{key}: volume fraction {phi} outside model range [0, {f.phi_max}]")
        n, d = _filler_terms(f, k_m)
        num += phi * n
        den += phi * d
    return k_m * (3.0 + num) / (3.0 - den)


@dataclass(frozen=True)
class Composition:
    """A matrix plus fillers at given volume fractions."""

    name: str
    loading: tuple[tuple[str, float], ...]
    matrix: str = "silicone"

    @property
    def phi_total(self) -> float:
        return sum(phi for _, phi in self.loading)

    def properties(self) -> dict[str, float]:
        """k, rho, cp, and latent-heat parameters of the composite."""
        m = MATRICES[self.matrix]
        phi_m = 1.0 - self.phi_total
        if phi_m < 0.3:
            raise ValueError("total filler fraction above 70 vol%")
        rho = phi_m * m.rho.value + sum(phi * FILLERS[k].rho.value for k, phi in self.loading)
        mass = {k: phi * FILLERS[k].rho.value / rho for k, phi in self.loading}
        w_m = phi_m * m.rho.value / rho
        cp = w_m * m.cp.value + sum(w * FILLERS[k].cp.value for k, w in mass.items())
        # Endothermic decomposition: one filler at most carries latent heat.
        latent, T_c, sigma = 0.0, 0.0, 3.0
        decomposing = [k for k in mass if FILLERS[k].decomposition is not None]
        if len(decomposing) > 1:
            raise ValueError("only one decomposing filler per composition is supported")
        if decomposing:
            k = decomposing[0]
            dH, T_c, sigma = FILLERS[k].decomposition
            latent = mass[k] * dH.value
        return {
            "k": effective_k(m, dict(self.loading)), "rho": rho, "cp": cp,
            "latent_heat": latent, "T_melt_C": T_c, "melt_width": sigma,
        }

    def to_barrier(self, thickness: float = 2e-3) -> BarrierParams:
        p = self.properties()
        return BarrierParams(
            name=self.name, thickness=thickness, k_on=p["k"], k_off=p["k"],
            rho=p["rho"], cp=p["cp"], latent_heat=p["latent_heat"],
            T_melt_C=p["T_melt_C"], melt_width=p["melt_width"],
        )


MATRICES = {
    "silicone": Matrix(
        "silicone elastomer (PDMS)",
        k=Value(0.20, "W/(m K)", "datasheet", "sylgard184", "datasheet copies give 0.16-0.27"),
        rho=Value(1030.0, "kg/m^3", "datasheet", "sylgard184", "specific gravity 1.03"),
        cp=Value(1460.0, "J/(kg K)", "assumed", note="typical PDMS value"),
        T_max_C=200.0,
    ),
}

# Interfacial resistance used where no filler-specific value was found. The
# CNT value is ~8x larger; the sensitivity study varies R_K over 1e-9..1e-7.
R_K_DEFAULT = Value(1e-8, "m^2 K/W", "assumed", note="typical ceramic/polymer order of magnitude")


def _v(x, unit, source=None, note="", prov="secondary"):
    """Sourced value (secondary unless stated) or, with no source, an assumed one."""
    return Value(x, unit, "assumed", None, note) if source is None else Value(x, unit, prov, source, note)


FILLERS: dict[str, Filler] = {
    f.key: f
    for f in [
        Filler(
            "caco3_nano", "CaCO3 nanoparticles (50 nm)",
            k_11=_v(3.5, "W/(m K)", "horai1971"), k_33=_v(3.5, "W/(m K)", "horai1971"),
            rho=_v(2710.0, "kg/m^3", "minerals"),
            cp=_v(830.0, "J/(kg K)", note="typical carbonate"),
            a1=25e-9, p=1.0, R_K=R_K_DEFAULT, phi_max=0.40,
            decomposition=(_v(1.778e6, "J/kg", "calcite_decomp", "177.8 kJ/mol / 0.1001 kg/mol"), 700.0, 40.0),
            notes=("Decomposes to CaO + CO2 above ~600 C (range taken as 700 +/- 40 C).",),
        ),
        Filler(
            "caco3_micro", "CaCO3 microparticles (5 um)",
            k_11=_v(3.5, "W/(m K)", "horai1971"), k_33=_v(3.5, "W/(m K)", "horai1971"),
            rho=_v(2710.0, "kg/m^3", "minerals"), cp=_v(830.0, "J/(kg K)"),
            a1=2.5e-6, p=1.0, R_K=R_K_DEFAULT, phi_max=0.40,
            decomposition=(_v(1.778e6, "J/kg", "calcite_decomp"), 700.0, 40.0),
        ),
        Filler(
            "ath", "Al(OH)3 (ATH, 1 um)",
            k_11=_v(3.0, "W/(m K)", note="not found; composite k insensitive for k_p >> k_m"),
            k_33=_v(3.0, "W/(m K)"),
            rho=_v(2340.0, "kg/m^3", "minerals", "gibbsite"),
            cp=_v(1000.0, "J/(kg K)"),
            a1=0.5e-6, p=1.0, R_K=R_K_DEFAULT, phi_max=0.40,
            decomposition=(_v(1.17e6, "J/kg", "hydroxides"), 260.0, 20.0),
            notes=("Releases water from ~220 C; range taken as 260 +/- 20 C.",),
        ),
        Filler(
            "mdh", "Mg(OH)2 (MDH, 1 um)",
            k_11=_v(3.0, "W/(m K)"), k_33=_v(3.0, "W/(m K)"),
            rho=_v(2390.0, "kg/m^3", "minerals", "brucite"),
            cp=_v(1300.0, "J/(kg K)"),
            a1=0.5e-6, p=1.0, R_K=R_K_DEFAULT, phi_max=0.40,
            decomposition=(_v(1.356e6, "J/kg", "hydroxides"), 365.0, 20.0),
            notes=("Releases water from ~330 C; range taken as 365 +/- 20 C.",),
        ),
        Filler(
            "hbn", "h-BN platelets (5 um x 100 nm)",
            k_11=_v(300.0, "W/(m K)", "huang2018", "in-plane 300-600; low end used"),
            k_33=_v(3.0, "W/(m K)", "huang2018", "through-plane 2-30; low end used"),
            rho=_v(2100.0, "kg/m^3"), cp=_v(800.0, "J/(kg K)"),
            a1=2.5e-6, p=0.02, R_K=R_K_DEFAULT, phi_max=0.30,
        ),
        Filler(
            "agnw", "Ag nanowires (84 nm x 20 um)",
            k_11=_v(332.0, "W/(m K)", note="no transverse value found; axial value used"),
            k_33=_v(332.0, "W/(m K)", "agnw_nsf"),
            rho=_v(10490.0, "kg/m^3", note="bulk silver"), cp=_v(235.0, "J/(kg K)", note="bulk silver"),
            a1=42e-9, p=20e-6 / 84e-9, R_K=R_K_DEFAULT, phi_max=0.05,
        ),
        Filler(
            "cnt", "MWCNT (20 nm x 10 um)",
            k_11=_v(10.0, "W/(m K)", note="radial value not found; small"),
            k_33=_v(3000.0, "W/(m K)", "kim2001", ">3000 measured", prov="paper"),
            rho=_v(1800.0, "kg/m^3"), cp=_v(700.0, "J/(kg K)"),
            a1=10e-9, p=500.0,
            R_K=_v(8.3e-8, "m^2 K/W", "huxtable2003", "1 / (12 MW m^-2 K^-1)", prov="paper"),
            phi_max=0.05,
        ),
        Filler(
            "glass_bubbles", "hollow glass microspheres (3M K1)",
            k_11=_v(0.047, "W/(m K)", "3m_k1", prov="datasheet"), k_33=_v(0.047, "W/(m K)", "3m_k1", prov="datasheet"),
            rho=_v(125.0, "kg/m^3", "3m_k1", prov="datasheet"),
            cp=_v(840.0, "J/(kg K)"),
            a1=32e-6, p=1.0, R_K=R_K_DEFAULT, phi_max=0.50,
            notes=("Bubble k is an effective value (glass shell + gas).",),
        ),
    ]
}


# Compositions compared in the paper: single fillers at a few loadings, plus
# two hybrids that pair an insulating or conducting filler with an
# endothermic one.
COMPOSITIONS: dict[str, Composition] = {
    c.name: c
    for c in [
        Composition("silicone", ()),
        Composition("CaCO3 nano 30%", (("caco3_nano", 0.30),)),
        Composition("CaCO3 micro 30%", (("caco3_micro", 0.30),)),
        Composition("ATH 40%", (("ath", 0.40),)),
        Composition("MDH 40%", (("mdh", 0.40),)),
        Composition("h-BN 30%", (("hbn", 0.30),)),
        Composition("Ag NW 5%", (("agnw", 0.05),)),
        Composition("CNT 5%", (("cnt", 0.05),)),
        Composition("glass bubbles 50%", (("glass_bubbles", 0.50),)),
        Composition("bubbles 40% + ATH 20%", (("glass_bubbles", 0.40), ("ath", 0.20))),
        Composition("h-BN 20% + ATH 20%", (("hbn", 0.20), ("ath", 0.20))),
    ]
}
