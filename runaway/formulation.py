"""Barrier formulation design: nanowires, nanoparticles, hybrids and a thermal switch.

``composites.py`` estimates the conductivity of a silicone loaded with fillers
at fixed recipes, using the Nan et al. effective-medium model. This module
extends it in four ways, so that a formulation can be *designed* rather than
picked from a list:

1. **Nanowires, fin-limited.** In Nan's model a long rod conducts along its
   whole length, so for nanowires (aspect ratio 100-1000) it predicts
   conductivities one to two orders of magnitude above measurements (5 vol%
   Ag nanowires: ~5 W/(m K) predicted, ~0.6-1 measured). Heat must enter and
   leave a wire through its surface, against the interface resistance R_K and
   the spreading resistance of the polymer around it. Treating the wire as a
   fin with exchange coefficient

       h_eff = 1 / (R_K + D / (2 k_host))

   gives the effective axial conductivity

       k_ax,eff = eta k_w,   eta = tanh(m l) / (m l),   m = sqrt(4 h_eff / (k_w D)),

   with l = L/2 the half length. This value replaces k_33 in Nan's model.

2. **Percolating wire networks.** Above the excluded-volume threshold of
   randomly oriented slender rods (N_c V_ex = 1, V_ex = (pi/2) L^2 D, so
   phi_c = D / (2 L)), wires touch and form a network. Its conductivity is
   estimated with an affine mean-field argument (each wire isothermal, heat
   crossing junctions of conductance G_c; see ``network_k``):

       k_net = (pi/72) n^2 L^4 D G_c  ->  (2 / (9 pi)) (L/D)^2 g k_host (phi - phi_c)^2

   with number density n = 4 phi / (pi D^2 L) and G_c = g k_host D, i.e. a
   junction bridged by the polymer (and anything dispersed in it). The
   (phi - phi_c)^2 form makes k_net vanish at threshold with the 3D
   conductivity exponent t ~ 2 and reduces to the affine n^2 scaling well above
   it. g is calibrated to measured Ag-nanowire/epoxy composites.

3. **Hybrids (two-step homogenisation).** Spherical or platelet particles are
   first homogenised with the matrix (Nan's model, fractions relative to the
   non-wire volume) to give a host conductivity k_host; the wires are then
   embedded in that host. Particles therefore help wires in two ways: a better
   host carries heat into and out of each wire (higher h_eff) and bridges the
   junctions (higher G_c). This is the "synergy" reported for nanowire +
   particle hybrids, in a form the model can quantify.

4. **Expandable-microsphere switch.** Thermoplastic microspheres (Expancel)
   expand when heated through their T_start-T_max range. Expansion by a
   volume ratio E dilutes everything else by X = 1 + phi_ms (E - 1) and turns
   the spheres into gas-filled cells. If the wire network is dispersed
   (``network="dilute"``), the dilution can drop it below phi_c, collapsing
   k_net: a percolation-amplified switch. The conservative alternative
   (``network="segregated"``) keeps the network intact in the polymer struts
   and only adds the gas cells. Expansion is one-way. Barrier thickness and
   heat capacity are held at their unexpanded values (expansion is assumed
   to be taken up by a compliant allowance or laterally); a barrier that
   also grew in thickness would insulate better.

Provenance follows ``candidates.py``. Values marked "assumed" are design
choices or generic class values; the formulation study varies them.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .candidates import Value
from .composites import FILLERS, MATRICES, Filler, Matrix, _filler_terms
from .params import BarrierParams


def _v(x, unit, source=None, note="", prov="secondary"):
    return Value(x, unit, "assumed", None, note) if source is None else Value(x, unit, prov, source, note)


# ---------------------------------------------------------------------------
# Constituents
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Wire:
    """A nanowire population: diameter D, length L, junction coefficient g."""

    key: str
    name: str
    k_w: Value  # [W/(m K)] intrinsic axial conductivity of one wire
    rho: Value
    cp: Value
    D: float  # [m]
    L: float  # [m]
    R_K: Value  # [m^2 K/W] wire-polymer interface resistance
    g: Value  # [-] junction conductance G_c = g k_host D
    electrically_insulating: bool = False
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def aspect(self) -> float:
        return self.L / self.D

    @property
    def phi_c(self) -> float:
        """Excluded-volume percolation threshold of isotropic slender rods."""
        return self.D / (2.0 * self.L)

    def fin_efficiency(self, k_host: float) -> float:
        h_eff = 1.0 / (self.R_K.value + self.D / (2.0 * k_host))
        ml = np.sqrt(4.0 * h_eff / (self.k_w.value * self.D)) * self.L / 2.0
        return float(np.tanh(ml) / ml)

    def as_filler(self, k_host: float) -> Filler:
        """Nan-model inclusion with the fin-limited axial conductivity."""
        k_ax = self.fin_efficiency(k_host) * self.k_w.value
        return Filler(
            self.key, self.name, k_11=self.k_w, k_33=Value(k_ax, "W/(m K)", "assumed", note="fin-limited"),
            rho=self.rho, cp=self.cp, a1=self.D / 2.0, p=self.aspect, R_K=self.R_K, phi_max=1.0,
        )


@dataclass(frozen=True)
class Microspheres:
    """Expandable thermoplastic microspheres (unexpanded in the cold state)."""

    key: str
    name: str
    T_start_C: Value
    T_max_C: Value
    k_cold: Value  # unexpanded sphere (polymer shell + liquid hydrocarbon)
    k_hot: Value  # expanded, gas-filled sphere
    rho: Value  # unexpanded true density
    cp: Value
    a1: float  # [m] unexpanded radius

    def as_filler(self, k: float) -> Filler:
        return Filler(self.key, self.name, k_11=Value(k, "W/(m K)", "assumed"),
                      k_33=Value(k, "W/(m K)", "assumed"), rho=self.rho, cp=self.cp,
                      a1=self.a1, p=1.0, R_K=Value(0.0, "m^2 K/W", "assumed"), phi_max=1.0)


# Calibration against measured Ag-nanowire/epoxy composites. Chen et al.
# (2014), 4 vol% in epoxy (k_m = 0.19 W/(m K)): 1.03 W/(m K) with silica-coated
# wires, 0.57 uncoated; wire size not in the abstract, 100 nm x 15 um assumed.
# Zhang et al. (2019): 120 nm x 15 um uncoated wires, 4 wt% (0.455 vol%) in
# epoxy (k_m = 0.22): 0.35 W/(m K). With the default interface resistance the
# fin-limited model gives 1.11 and 0.33, without fitting. Chen's uncoated value
# needs a far larger effective interface resistance (R_K_POOR, fitted by
# ``fit_R_K``), read here as poor dispersion; it is used as a sensitivity case.
CAL_D, CAL_L, CAL_PHI, CAL_KM = 100e-9, 15e-6, 0.04, 0.19
CAL_TARGET = {"coated": 1.03, "uncoated": 0.57}
R_K_COATED = 1e-8
R_K_UNCOATED = 1e-8
R_K_POOR = 3.5e-6  # recomputed by tests/test_formulation.py
# Junction coefficient: weak, polymer-bridged junctions by default, consistent
# with the coated calibration point (it changes the prediction there by <1 %)
# and with the weak thermal percolation usually seen in polymer composites.
# The study varies it from 0 to 0.5 and models sintered junctions with x10.
G_DEFAULT = 0.05

# Size-dependent wire conductivity: boundary scattering, anchored to the
# measured 332 W/(m K) at 84 nm (agnw_nsf) and bulk silver (429 W/(m K)):
# k_w = 429 / (1 + D0 / D), D0 = 24.5 nm.
AG_BULK_K, AG_D0 = 429.0, 84e-9 * (429.0 / 332.0 - 1.0)


def ag_wire_k(D: float) -> float:
    return AG_BULK_K / (1.0 + AG_D0 / D)


def agnw(D_nm: float, L_um: float, coated: bool = True, g: float | None = None,
         R_K: float | None = None) -> Wire:
    """Silver nanowire of given size; silica-coated (electrically insulating) by default."""
    D = D_nm * 1e-9
    rk = (R_K_COATED if coated else R_K_UNCOATED) if R_K is None else R_K
    return Wire(
        key=f"agnw{'@SiO2' if coated else ''}_{D_nm:g}x{L_um:g}",
        name=f"Ag nanowire{' @SiO2' if coated else ''} {D_nm:g} nm x {L_um:g} um",
        k_w=Value(ag_wire_k(D), "W/(m K)", "assumed",
                  note="size scaling anchored to agnw_nsf (332 at 84 nm) and bulk Ag"),
        rho=_v(10490.0, "kg/m^3", note="bulk silver"), cp=_v(235.0, "J/(kg K)", note="bulk silver"),
        D=D, L=L_um * 1e-6,
        R_K=Value(rk, "m^2 K/W", "assumed",
                  note="default ceramic/polymer value; reproduces chen2014agnw (coated) and zhang2019agnw"),
        g=Value(G_DEFAULT if g is None else g, "-", "assumed", note="junction coefficient"),
        electrically_insulating=coated,
    )


PARTICLES: dict[str, Filler] = {
    "al2o3_nano": Filler(
        "al2o3_nano", "Al2O3 nanoparticles (50 nm)",
        k_11=_v(30.0, "W/(m K)", "kyocera_al2o3", "34.9 for dense F99.7; 30 used for particles"),
        k_33=_v(30.0, "W/(m K)", "kyocera_al2o3"),
        rho=_v(3950.0, "kg/m^3", note="alpha-alumina"), cp=_v(880.0, "J/(kg K)", note="alpha-alumina"),
        a1=25e-9, p=1.0, R_K=FILLERS["ath"].R_K, phi_max=0.40,
    ),
    "al2o3_micro": Filler(
        "al2o3_micro", "Al2O3 microparticles (5 um)",
        k_11=_v(30.0, "W/(m K)", "kyocera_al2o3"), k_33=_v(30.0, "W/(m K)", "kyocera_al2o3"),
        rho=_v(3950.0, "kg/m^3"), cp=_v(880.0, "J/(kg K)"),
        a1=2.5e-6, p=1.0, R_K=FILLERS["ath"].R_K, phi_max=0.40,
    ),
    "hbn": FILLERS["hbn"],
    "ath": FILLERS["ath"],
    "mdh": FILLERS["mdh"],
    "glass_bubbles": FILLERS["glass_bubbles"],
}

MICROSPHERES: dict[str, Microspheres] = {
    "ems_051": Microspheres(
        "ems_051", "Expancel 051 DU 40",
        T_start_C=_v(107.5, "C", "expancel051", "105-110 C", prov="datasheet"),
        T_max_C=_v(146.5, "C", "expancel051", "142-151 C", prov="datasheet"),
        k_cold=_v(0.12, "W/(m K)", note="polymer shell + liquid isobutane; not published"),
        k_hot=_v(0.04, "W/(m K)", note="gas-filled cell, air-like at 100-300 C"),
        rho=_v(1100.0, "kg/m^3", note="unexpanded true density; not on datasheet"),
        cp=_v(1500.0, "J/(kg K)", note="generic thermoplastic"),
        a1=6e-6,
    ),
    "ems_920": Microspheres(
        "ems_920", "Expancel 920 DU 120",
        T_start_C=_v(127.0, "C", "expancel920", "122-132 C"),
        T_max_C=_v(200.0, "C", "expancel920", "194-206 C"),
        k_cold=_v(0.12, "W/(m K)"), k_hot=_v(0.04, "W/(m K)"),
        rho=_v(1100.0, "kg/m^3"), cp=_v(1500.0, "J/(kg K)"), a1=16e-6,
    ),
}

# Volume expansion ratio of a microsphere inside the cured silicone. Free
# expansion is >= 40x (datasheet TMA density <= 25 kg/m^3 from ~1100); a
# crosslinked matrix resists, so a much smaller ratio is assumed and varied.
E_DEFAULT = 5.0
SWITCH_TAU = 10.0  # [s] as for the graphene switch (candidates.MODEL_ASSUMPTIONS)


# ---------------------------------------------------------------------------
# Conductivity model
# ---------------------------------------------------------------------------

def emt(k_m: float, phases) -> float:
    """Nan et al. effective conductivity for [(Filler, volume fraction), ...]."""
    num = den = 0.0
    for f, phi in phases:
        if phi <= 0.0:
            continue
        n, d = _filler_terms(f, k_m)
        num += phi * n
        den += phi * d
    return k_m * (3.0 + num) / (3.0 - den)


def network_k(wire: Wire, phi_w: float, k_host: float, sinter: float = 1.0) -> float:
    """Conductivity of the percolating wire network (0 below phi_c).

    Affine mean-field estimate: imposing a uniform gradient on isothermal
    wires, the dissipation sum over junctions gives k = (n z / 2) G_c <dx^2>
    with z = n (pi/2) L^2 D contacts per wire and <dx^2> = L^2 / 18 between
    the centres of touching wires, i.e. k = (pi/72) n^2 L^4 D G_c.
    ``sinter`` multiplies G_c (soldered or sintered junctions).
    """
    if phi_w <= wire.phi_c:
        return 0.0
    p = wire.aspect
    return 2.0 / (9.0 * np.pi) * p**2 * wire.g.value * sinter * k_host * (phi_w - wire.phi_c) ** 2


def formulation_k(matrix: Matrix, particles, wire: Wire | None = None, phi_w: float = 0.0,
                  sinter: float = 1.0, wire_network: bool = True) -> dict[str, float]:
    """Two-step k: particles in matrix -> host; wires (fin-limited + network) in host.

    ``particles`` is [(Filler, phi), ...] with fractions of the *total* volume.
    """
    phi_p = sum(phi for _, phi in particles)
    if phi_p + phi_w >= 1.0:
        raise ValueError("filler fractions sum to >= 1")
    rel = 1.0 / (1.0 - phi_w)
    k_host = emt(matrix.k.value, [(f, phi * rel) for f, phi in particles])
    if wire is None or phi_w <= 0.0:
        return {"k": k_host, "k_host": k_host, "k_net": 0.0, "eta": 0.0}
    k_dis = emt(k_host, [(wire.as_filler(k_host), phi_w)])
    k_net = network_k(wire, phi_w, k_host, sinter) if wire_network else 0.0
    return {"k": k_dis + k_net, "k_host": k_host, "k_net": k_net, "eta": wire.fin_efficiency(k_host)}


# ---------------------------------------------------------------------------
# Formulations
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Formulation:
    """Silicone + particles + optional nanowires + optional expandable microspheres.

    All fractions are volume fractions of the cold (unexpanded) composite.
    """

    name: str
    particles: tuple[tuple[str, float], ...] = ()
    wire: Wire | None = None
    phi_w: float = 0.0
    spheres: str | None = None
    phi_ms: float = 0.0
    E: float = E_DEFAULT
    network: str = "dilute"  # or "segregated"
    sinter: float = 1.0
    k_gas: float | None = None  # override the expanded-sphere conductivity
    matrix: str = "silicone"

    @property
    def phi_total(self) -> float:
        return sum(p for _, p in self.particles) + self.phi_w + self.phi_ms

    def _particles(self, scale=1.0, sphere_state=None):
        out = [(PARTICLES[k], phi * scale) for k, phi in self.particles]
        if self.spheres and self.phi_ms > 0:
            ms = MICROSPHERES[self.spheres]
            if sphere_state == "cold":
                out.append((ms.as_filler(ms.k_cold.value), self.phi_ms * scale))
            elif sphere_state == "hot":
                X = 1.0 + self.phi_ms * (self.E - 1.0)
                out.append((replace(ms.as_filler(self._k_gas()), a1=ms.a1 * self.E ** (1 / 3)),
                            self.phi_ms * self.E / X))
        return out

    def _k_gas(self) -> float:
        return MICROSPHERES[self.spheres].k_hot.value if self.k_gas is None else self.k_gas

    def k_cold(self) -> dict[str, float]:
        m = MATRICES[self.matrix]
        return formulation_k(m, self._particles(1.0, "cold"), self.wire, self.phi_w, self.sinter)

    def k_hot(self) -> dict[str, float]:
        """Conductivity after full microsphere expansion (== cold without spheres)."""
        if not self.spheres or self.phi_ms <= 0:
            return self.k_cold()
        m = MATRICES[self.matrix]
        X = 1.0 + self.phi_ms * (self.E - 1.0)
        if self.network == "dilute":
            return formulation_k(m, self._particles(1.0 / X, "hot"), self.wire, self.phi_w / X,
                                 self.sinter)
        # Segregated: wires stay connected in the polymer struts; the struts
        # (everything but the spheres) are homogenised first, then gas cells.
        s = 1.0 / (1.0 - self.phi_ms)
        strut = formulation_k(m, [(PARTICLES[k], phi * s) for k, phi in self.particles],
                              self.wire, self.phi_w * s, self.sinter)
        ms = MICROSPHERES[self.spheres]
        gas = replace(ms.as_filler(self._k_gas()), a1=ms.a1 * self.E ** (1 / 3))
        k = emt(strut["k"], [(gas, self.phi_ms * self.E / X)])
        return {**strut, "k": k}

    def properties(self) -> dict[str, float]:
        m = MATRICES[self.matrix]
        phi_m = 1.0 - self.phi_total
        if phi_m < 0.3:
            raise ValueError("total filler fraction above 70 vol%")
        parts = [(PARTICLES[k].rho.value, PARTICLES[k].cp.value, phi, PARTICLES[k]) for k, phi in self.particles]
        if self.wire is not None and self.phi_w > 0:
            parts.append((self.wire.rho.value, self.wire.cp.value, self.phi_w, None))
        if self.spheres and self.phi_ms > 0:
            ms = MICROSPHERES[self.spheres]
            parts.append((ms.rho.value, ms.cp.value, self.phi_ms, None))
        rho = phi_m * m.rho.value + sum(r * phi for r, _, phi, _ in parts)
        cp = (phi_m * m.rho.value * m.cp.value + sum(r * c * phi for r, c, phi, _ in parts)) / rho
        latent, T_c, sigma = 0.0, 0.0, 3.0
        dec = [(f, r * phi / rho) for r, _, phi, f in parts if f is not None and f.decomposition]
        if len(dec) > 1:
            raise ValueError("only one decomposing filler per formulation is supported")
        if dec:
            f, w = dec[0]
            dH, T_c, sigma = f.decomposition
            latent = w * dH.value
        return {"k_on": self.k_cold()["k"], "k_off": self.k_hot()["k"], "rho": rho, "cp": cp,
                "latent_heat": latent, "T_melt_C": T_c, "melt_width": sigma}

    def to_barrier(self, thickness: float = 2e-3, **overrides) -> BarrierParams:
        p = self.properties()
        kw = dict(name=self.name, thickness=thickness, k_on=p["k_on"], k_off=p["k_off"],
                  rho=p["rho"], cp=p["cp"], latent_heat=p["latent_heat"],
                  T_melt_C=p["T_melt_C"], melt_width=p["melt_width"])
        if self.spheres and self.phi_ms > 0:
            ms = MICROSPHERES[self.spheres]
            T0, T1 = ms.T_start_C.value, ms.T_max_C.value
            # logistic centred mid-range; 10-90 % of the change spans T_start..T_max
            kw.update(T_sw_C=0.5 * (T0 + T1), sw_width=(T1 - T0) / 4.39,
                      switch_tau=SWITCH_TAU, switch_irreversible=True)
        kw.update(overrides)
        return BarrierParams(**kw)


def epoxy(k_m: float) -> Matrix:
    return replace(MATRICES["silicone"], name="epoxy", k=Value(k_m, "W/(m K)", "assumed"))


def fit_R_K(target: float, D=CAL_D, L=CAL_L, phi=CAL_PHI, k_m=CAL_KM, coated=False) -> float:
    """Wire interface resistance that reproduces a measured composite k."""
    from scipy.optimize import brentq

    def f(lg):
        return formulation_k(epoxy(k_m), [], agnw(D * 1e9, L * 1e6, coated, R_K=10**lg), phi)["k"] - target

    return 10 ** brentq(f, -10.0, -4.0)
