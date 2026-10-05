"""Model parameters.

!!! ILLUSTRATIVE VALUES ONLY !!!
Every number in this module is an order-of-magnitude placeholder chosen to make
the physics of the barrier tradeoff visible. None of them is taken from, or
calibrated against, a specific publication or measurement. Each carries a TODO
marking where a literature-sourced or measured value should go before the model
is used for anything quantitative.

Conventions
-----------
* SI units throughout (m, s, W, J, kg, K).
* Temperatures that a person would set by hand (ambient, trigger, switch point)
  are given in degrees Celsius and named ``*_C``; the solver converts to kelvin.
* Dataclasses are frozen; build variants with ``dataclasses.replace``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Universal gas constant [J/(mol K)]
R_GAS = 8.314462618

KELVIN = 273.15


@dataclass(frozen=True)
class CellParams:
    """Homogenised lithium-ion cell, treated as one through-plane slab."""

    thickness: float = 10e-3  # [m]  TODO: replace with the actual cell format
    k: float = 0.8  # [W/(m K)] through-plane conductivity. TODO: measured value
    rho: float = 2500.0  # [kg/m^3]  TODO: measured cell density
    cp: float = 1000.0  # [J/(kg K)] TODO: measured (ARC/DSC) heat capacity

    @property
    def rho_cp(self) -> float:
        """Volumetric heat capacity [J/(m^3 K)]."""
        return self.rho * self.cp


@dataclass(frozen=True)
class KineticsParams:
    """Lumped single-step Arrhenius self-heating of the cell.

    d(alpha)/dt = A exp(-Ea / (R T)) (1 - alpha)
    q_rxn       = dH_v d(alpha)/dt,    dH_v = rho cp dT_ad

    With these values the Arrhenius rate is ~1e-4 1/s at 150 C (slow onset)
    and ~1 1/s at 250 C (fast runaway).

    Rate ceiling (a deliberate modelling choice, see README "Numerics"):
    extrapolated to the ~900 C burnt-cell temperature, the pure Arrhenius law
    gives k ~ 1e9 1/s, i.e. a reaction front ~0.1 um thick that no practical
    grid resolves; propagation times then scale with the mesh size instead
    of converging. Real high-temperature decomposition is limited by
    transport and multi-step chemistry, so the rate is capped smoothly:

        k_eff = k_arr k_max / (k_arr + k_max)

    k_max = 1 1/s leaves onset (k_arr << k_max) untouched and makes a single
    cell burn over a few seconds. Set k_max = inf to recover pure Arrhenius.
    """

    Ea: float = 169e3  # [J/mol]  TODO: fit to ARC / accelerating-rate data
    A: float = 7.3e16  # [1/s]    TODO: fit to ARC / accelerating-rate data
    dT_ad: float = 600.0  # [K] adiabatic temperature rise. TODO: from ARC data
    k_max: float = 1.0  # [1/s] rate ceiling. TODO: fit to measured runaway duration

    def rate_constant(self, T_K):
        """Effective rate constant k(T) [1/s]; T in kelvin."""
        k_arr = self.A * np.exp(-self.Ea / (R_GAS * T_K))
        if not np.isfinite(self.k_max):
            return k_arr
        return k_arr * self.k_max / (k_arr + self.k_max)

    def dH_v(self, cell: CellParams) -> float:
        """Volumetric heat of reaction [J/m^3] implied by dT_ad."""
        return cell.rho_cp * self.dT_ad


@dataclass(frozen=True)
class BarrierParams:
    """Inter-cell thermal barrier with (optionally) temperature-switchable k.

    ln k(T) = ln k_off + ln(k_on / k_off) / (1 + exp((T - T_sw) / sw_width))

    so k -> k_on well below T_sw and k -> k_off well above it. A static
    barrier is simply k_on == k_off. The switch is reversible: k depends only
    on the current local temperature, with no hysteresis.

    The volumetric heat capacity is deliberately the same for every design so
    that the comparison isolates the effect of conductivity.
    """

    name: str
    thickness: float = 2e-3  # [m]; 0 means direct cell-to-cell contact
    k_on: float = 5.0  # [W/(m K)] conductivity in the cold / "on" state
    k_off: float = 5.0  # [W/(m K)] conductivity in the hot / "off" state
    T_sw_C: float = 120.0  # [C] switch temperature (irrelevant if static)
    sw_width: float = 5.0  # [K] logistic width of the switch
    rho: float = 1000.0  # [kg/m^3]  TODO: measured barrier density
    cp: float = 1000.0  # [J/(kg K)] TODO: measured barrier heat capacity

    @property
    def rho_cp(self) -> float:
        return self.rho * self.cp

    @property
    def is_switchable(self) -> bool:
        return self.k_on != self.k_off


@dataclass(frozen=True)
class StackParams:
    """The 1D stack: cell | barrier | cell | ... | cell, cooled at both ends."""

    n_cells: int = 5
    cell: CellParams = field(default_factory=CellParams)
    kinetics: KineticsParams = field(default_factory=KineticsParams)
    barrier: BarrierParams = field(default_factory=lambda: barrier_switchable())
    # Effective heat-transfer coefficient at both stack ends [W/(m^2 K)].
    # 100 W/(m^2 K) represents an end plate tied to a cooling system, well
    # above natural convection (~5-10) and below a direct liquid cold plate.
    # TODO: replace with the heat-transfer coefficient of the real pack design.
    h: float = 100.0
    T_amb_C: float = 25.0  # [C] coolant / ambient temperature
    # Uniform volumetric heat generation in each cell during normal operation
    # (Joule + reversible heat at a high C-rate) [W/m^3].
    # TODO: replace with I^2 R + entropic heat from a cell model at the duty cycle.
    q_normal: float = 2e4


@dataclass(frozen=True)
class SimParams:
    """Numerical settings for the transient runaway simulation."""

    n_nodes_cell: int = 40  # finite volumes per cell (0.25 mm; see convergence test)
    n_nodes_barrier: int = 12  # finite volumes per barrier (if thickness > 0)
    dt_max: float = 0.5  # [s] largest allowed time step
    # Adaptive step: cap the reaction temperature rise per step [K]. This only
    # bites while a cell is actively running away; elsewhere dt = dt_max.
    dT_rxn_max: float = 2.0
    picard_iters: int = 1  # 1 = conductivity lagged from previous step
    t_end: float = 3600.0  # [s] simulated time
    save_every: float = 2.0  # [s] snapshot spacing for the output history...
    n_log_saves: int = 200  # ...plus this many log-spaced snapshots from 1 ms
    T_trigger_C: float = 300.0  # [C] initial temperature of the trigger cell
    alpha_propagated: float = 0.9  # mean conversion that counts as "runaway"
    # Optional early exit (used by parameter sweeps): stop when every cell has
    # run away, or when the whole stack has cooled below this temperature.
    stop_when_settled: bool = False
    T_settled_C: float = 60.0


# ---------------------------------------------------------------------------
# The four barrier designs compared throughout the project.
# All values ILLUSTRATIVE (see module docstring).
# ---------------------------------------------------------------------------


def barrier_none() -> BarrierParams:
    """Cells in direct (perfect) thermal contact."""
    return BarrierParams(name="none", thickness=0.0)


def barrier_conductor(thickness: float = 2e-3) -> BarrierParams:
    """Static conductive spacer (e.g. a filled polymer / graphite sheet)."""
    # TODO: measured conductivity of a candidate conductive spacer
    return BarrierParams(name="conductor", thickness=thickness, k_on=5.0, k_off=5.0)


def barrier_insulator(thickness: float = 2e-3) -> BarrierParams:
    """Static aerogel-like insulator."""
    # TODO: measured conductivity of a candidate aerogel / mica sheet
    return BarrierParams(name="insulator", thickness=thickness, k_on=0.03, k_off=0.03)


def barrier_switchable(
    thickness: float = 2e-3,
    k_on: float = 5.0,
    k_off: float = 0.05,
    T_sw_C: float = 120.0,
    sw_width: float = 5.0,
) -> BarrierParams:
    """Thermal switch: conducts below T_sw, insulates above it."""
    # TODO: replace the sigmoid with k(T) fitted to measured switch data
    return BarrierParams(
        name="switchable",
        thickness=thickness,
        k_on=k_on,
        k_off=k_off,
        T_sw_C=T_sw_C,
        sw_width=sw_width,
    )


DESIGNS = {
    "none": barrier_none,
    "conductor": barrier_conductor,
    "insulator": barrier_insulator,
    "switchable": barrier_switchable,
}
