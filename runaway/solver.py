"""Finite-volume solver for 1D conduction with Arrhenius self-heating.

Governing equation (per unit stack area, x through the stack thickness):

    rho cp dT/dt = d/dx( k(T) dT/dx ) + q_rxn + q_gen
    d(alpha)/dt  = A exp(-Ea / R T) (1 - alpha)          (cell nodes only)
    q_rxn        = dH_v d(alpha)/dt

    -k dT/dx = h (T_amb - T)   at both ends (Robin / convective)

Time integration uses Lie operator splitting. Each step does:

1. Reaction: with the rate constant frozen, the alpha ODE is linear and has
   the exact solution  alpha_new = 1 - (1 - alpha) exp(-k_r dt). k_r is
   evaluated at a predicted mid-step temperature (exponential midpoint). The
   released heat is added as dT = dT_ad * (alpha_new - alpha). This update is
   unconditionally stable and keeps 0 <= alpha <= 1 for any dt, which is what
   makes the very stiff Arrhenius kinetics tractable.
2. Diffusion: backward (implicit) Euler, with k(T) lagged from the start of
   the step (optionally refined by Picard iterations). The resulting
   tridiagonal system is solved with ``scipy.linalg.solve_banded``.

Because the diffusion update is written in flux (conservative) form, the
discrete energy balance closes to round-off: stored + chemical energy changes
only through the boundary fluxes and the volumetric source. The solver
tracks this balance and reports the residual.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import solve_banded

from .materials import Grid, build_grid
from .params import KELVIN, SimParams, StackParams

# ---------------------------------------------------------------------------
# Spatial operator
# ---------------------------------------------------------------------------


def face_conductances(grid: Grid, k: np.ndarray, h: float) -> tuple[np.ndarray, float, float]:
    """Thermal conductances [W/(m^2 K)] of interior faces and both boundaries.

    Interior face between nodes i and i+1: two half-volumes in series,

        G = 1 / (dx_i / (2 k_i) + dx_{i+1} / (2 k_{i+1}))

    which is the distance-weighted harmonic mean of k divided by the
    centre-to-centre distance. At a material interface this gives the exact
    series resistance, so property jumps need no special treatment.

    Boundary faces: convection (1/h) in series with a half-volume. h = 0
    gives an adiabatic (insulated) end.
    """
    dx = grid.dx
    G = 1.0 / (0.5 * dx[:-1] / k[:-1] + 0.5 * dx[1:] / k[1:])
    if h > 0.0:
        G_left = 1.0 / (1.0 / h + 0.5 * dx[0] / k[0])
        G_right = 1.0 / (1.0 / h + 0.5 * dx[-1] / k[-1])
    else:
        G_left = G_right = 0.0
    return G, G_left, G_right


def diffusion_step(
    grid: Grid,
    T: np.ndarray,
    dt: float,
    k: np.ndarray,
    h: float,
    T_amb: float | tuple[float, float],
    q_vol: np.ndarray | float = 0.0,
) -> tuple[np.ndarray, float]:
    """One backward-Euler conduction step with fixed conductivity ``k``.

    Solves  (m + K) T_new = m T + q dx + boundary terms,  m = rho cp dx / dt.

    ``T_amb`` may be a (left, right) pair. ``dt = inf`` drops the storage
    term and returns the steady-state solution for the given k.

    Returns the new temperature field and the energy that left through the
    two boundaries during the step [J/m^2] (0 for the steady case).
    """
    T_left, T_right = (T_amb, T_amb) if np.isscalar(T_amb) else T_amb
    G, G_left, G_right = face_conductances(grid, k, h)
    m = grid.rho_cp * grid.dx / dt

    # Banded storage for solve_banded with one sub- and one super-diagonal:
    # row 0 = super-diagonal, row 1 = diagonal, row 2 = sub-diagonal.
    ab = np.zeros((3, grid.n))
    ab[0, 1:] = -G
    ab[2, :-1] = -G
    diag = m.copy()
    diag[:-1] += G
    diag[1:] += G
    diag[0] += G_left
    diag[-1] += G_right
    ab[1] = diag

    rhs = m * T + q_vol * grid.dx
    rhs[0] += G_left * T_left
    rhs[-1] += G_right * T_right

    T_new = solve_banded((1, 1), ab, rhs, check_finite=False)
    # Boundary flux evaluated at the new time level, consistent with the
    # implicit scheme, so the discrete energy budget closes exactly.
    if not np.isfinite(dt):
        return T_new, 0.0
    loss = dt * (G_left * (T_new[0] - T_left) + G_right * (T_new[-1] - T_right))
    return T_new, loss


def reaction_step(
    grid: Grid, T: np.ndarray, alpha: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """Exponential update of the Arrhenius progress (exponential midpoint).

    For a fixed rate constant the update alpha_new = 1 - (1 - alpha) e^{-k dt}
    is exact. Evaluating k at the start-of-step temperature is only first
    order accurate while a node accelerates toward ignition, so k is taken at
    a predicted mid-step temperature instead (one extra exponential). The
    update keeps 0 <= alpha <= 1 and is stable for any dt, and the heat added
    is exactly dT_ad * (alpha_new - alpha), so energy is conserved exactly.
    """
    kin = grid.stack.kinetics
    cells = grid.is_cell
    T = T.copy()
    alpha = alpha.copy()
    a_old = alpha[cells]
    T_old = T[cells]
    # Predictor: half step with the start-of-step rate
    a_half = 1.0 - (1.0 - a_old) * np.exp(-kin.rate_constant(T_old) * 0.5 * dt)
    T_half = T_old + kin.dT_ad * (a_half - a_old)
    # Corrector: full step with the mid-step rate
    kr = kin.rate_constant(T_half)
    a_new = 1.0 - (1.0 - a_old) * np.exp(-kr * dt)
    alpha[cells] = a_new
    # dH_v / (rho cp) == dT_ad on cell nodes by construction.
    T[cells] += kin.dT_ad * (a_new - a_old)
    return T, alpha


def stable_reaction_dt(grid: Grid, T: np.ndarray, alpha: np.ndarray, sim: SimParams) -> float:
    """Largest dt keeping the per-step reaction temperature rise <= dT_rxn_max.

    This is an accuracy limit, not a stability limit: the exponential update
    is stable for any dt, but resolving the ignition transient needs small
    steps while a node is actively running away.
    """
    kin = grid.stack.kinetics
    cells = grid.is_cell
    heating_rate = kin.dT_ad * kin.rate_constant(T[cells]) * (1.0 - alpha[cells])  # K/s
    peak = heating_rate.max()
    if peak <= 0.0:
        return sim.dt_max
    return float(np.clip(sim.dT_rxn_max / peak, 1e-9, sim.dt_max))


# ---------------------------------------------------------------------------
# Transient runaway simulation
# ---------------------------------------------------------------------------


@dataclass
class RunawayResult:
    """History of a transient run. Temperatures are stored in Celsius."""

    grid: Grid
    t: np.ndarray  # [s] snapshot times
    T_C: np.ndarray  # [C] (n_times, n_nodes)
    alpha: np.ndarray  # [-] (n_times, n_nodes)
    t_prop: np.ndarray  # [s] time each cell's mean alpha passed threshold (nan = never)
    energy_residual: np.ndarray  # [J/m^2] discrete energy-balance error at snapshots
    energy_scale: float  # [J/m^2] total chemical energy, for a relative residual
    n_steps: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def T_cell_mean_C(self) -> np.ndarray:
        return self.grid.cell_mean(self.T_C)

    @property
    def T_cell_max_C(self) -> np.ndarray:
        out = np.empty((self.t.size, self.grid.stack.n_cells))
        for c in range(self.grid.stack.n_cells):
            out[:, c] = self.T_C[:, self.grid.cell_id == c].max(axis=1)
        return out

    @property
    def propagated(self) -> np.ndarray:
        return np.isfinite(self.t_prop)

    @property
    def n_propagated(self) -> int:
        """Number of cells in runaway, including the trigger cell."""
        return int(self.propagated.sum())

    @property
    def max_relative_energy_error(self) -> float:
        return float(np.abs(self.energy_residual).max() / self.energy_scale)


def run_transient(
    grid: Grid,
    T0_K: np.ndarray,
    alpha0: np.ndarray,
    sim: SimParams,
    *,
    h: float | None = None,
    q_vol: np.ndarray | float = 0.0,
    react: bool = True,
) -> RunawayResult:
    """March the coupled conduction / reaction system from an initial state.

    ``h`` overrides the stack heat-transfer coefficient (h=0: insulated ends);
    ``react=False`` switches the chemistry off (used by the validation tests).
    """
    stack = grid.stack
    h = stack.h if h is None else h
    T_amb = stack.T_amb_C + KELVIN
    kin = stack.kinetics
    dH = kin.dH_v(stack.cell)
    q_vol = np.broadcast_to(np.asarray(q_vol, dtype=float), (grid.n,))

    # Weight matrix: cell-mean of a node field is W @ field.
    W = np.zeros((stack.n_cells, grid.n))
    for c in range(stack.n_cells):
        m = grid.cell_id == c
        W[c, m] = grid.dx[m] / grid.dx[m].sum()

    T = np.asarray(T0_K, dtype=float).copy()
    alpha = np.asarray(alpha0, dtype=float).copy()
    cells = grid.is_cell

    # Energy bookkeeping (per unit area, relative to ambient):
    #   E = sum rho cp (T - T_amb) dx  +  sum dH (1 - alpha) dx
    # dE/dt = generation - boundary loss  =>  residual should stay ~0.
    def energy(T, alpha):
        thermal = np.sum(grid.rho_cp * (T - T_amb) * grid.dx)
        chemical = np.sum(dH * (1.0 - alpha[cells]) * grid.dx[cells]) if react else 0.0
        return thermal + chemical

    E0 = energy(T, alpha)
    lost = 0.0
    generated = 0.0
    energy_scale = max(np.sum(dH * grid.dx[cells]), abs(E0), 1e-30)

    t = 0.0
    t_prop = np.full(stack.n_cells, np.nan)
    times, T_hist, a_hist, res_hist = [0.0], [T - KELVIN], [alpha.copy()], [0.0]
    # Snapshot times: log-spaced early (ignition happens in < 1 s) merged
    # with an even spacing later on.
    save_times = np.unique(np.concatenate([
        np.geomspace(1e-3, sim.t_end, sim.n_log_saves),
        np.arange(sim.save_every, sim.t_end + 1e-9, sim.save_every),
        [sim.t_end],
    ]))
    i_save = 0
    T_settled = sim.T_settled_C + KELVIN
    n_steps = 0

    def check_propagation(t_now):
        newly = np.isnan(t_prop) & (W @ alpha > sim.alpha_propagated)
        t_prop[newly] = t_now

    check_propagation(0.0)

    while t < sim.t_end - 1e-12:
        dt = stable_reaction_dt(grid, T, alpha, sim) if react else sim.dt_max
        dt = min(dt, save_times[i_save] - t)

        # 1) reaction (exact exponential update at frozen T)
        if react:
            T, alpha = reaction_step(grid, T, alpha, dt)

        # 2) conduction (backward Euler, k lagged; optional Picard refinement)
        T_iter = T
        for _ in range(max(1, sim.picard_iters)):
            k = grid.conductivity(T_iter)
            T_iter, loss = diffusion_step(grid, T, dt, k, h, T_amb, q_vol)
        T = T_iter
        lost += loss
        generated += dt * np.sum(q_vol * grid.dx)

        t += dt
        n_steps += 1
        if react:
            check_propagation(t)

        if t >= save_times[i_save] - 1e-12:
            times.append(t)
            T_hist.append(T - KELVIN)
            a_hist.append(alpha.copy())
            res_hist.append(energy(T, alpha) + lost - generated - E0)
            i_save = min(i_save + 1, save_times.size - 1)

            if sim.stop_when_settled and (
                np.all(np.isfinite(t_prop)) or T.max() < T_settled
            ):
                break

    return RunawayResult(
        grid=grid,
        t=np.array(times),
        T_C=np.array(T_hist),
        alpha=np.array(a_hist),
        t_prop=t_prop,
        energy_residual=np.array(res_hist),
        energy_scale=energy_scale,
        n_steps=n_steps,
    )


def simulate_runaway(stack: StackParams, sim: SimParams | None = None) -> RunawayResult:
    """Trigger cell 1 at ``sim.T_trigger_C``; everything else starts at ambient."""
    sim = sim or SimParams()
    grid = build_grid(stack, sim)
    T0 = np.full(grid.n, stack.T_amb_C + KELVIN)
    T0[grid.cell_id == 0] = sim.T_trigger_C + KELVIN
    alpha0 = np.zeros(grid.n)
    res = run_transient(grid, T0, alpha0, sim)
    res.meta["barrier"] = stack.barrier.name
    return res


# ---------------------------------------------------------------------------
# Normal operation: steady state with uniform heat generation in the cells
# ---------------------------------------------------------------------------


@dataclass
class SteadyResult:
    grid: Grid
    T_C: np.ndarray  # [C] steady temperature profile
    iterations: int
    converged: bool

    @property
    def peak_C(self) -> float:
        return float(self.T_C.max())

    @property
    def T_cell_mean_C(self) -> np.ndarray:
        return self.grid.cell_mean(self.T_C)


def solve_normal_operation(
    stack: StackParams,
    sim: SimParams | None = None,
    *,
    tol: float = 1e-7,
    max_iter: int = 5000,
) -> SteadyResult:
    """Steady temperature under uniform cell heat generation ``stack.q_normal``.

    Solved by pseudo-transient continuation: backward-Euler steps from a pack
    at ambient temperature with a geometrically growing time step. For a
    constant-k barrier this converges to the linear steady state in a few
    dozen steps. For a switchable barrier it follows the physical start-up
    path, so if the heating trips the switch the solution lands on the
    (hotter) insulating branch, as a real pack would. Self-heating chemistry
    is negligible at these temperatures and is left out.
    """
    sim = sim or SimParams()
    grid = build_grid(stack, sim)
    T_amb = stack.T_amb_C + KELVIN
    q = np.where(grid.is_cell, stack.q_normal, 0.0)

    T = np.full(grid.n, T_amb)
    dt = 1.0
    converged = False
    for it in range(1, max_iter + 1):
        k = grid.conductivity(T)
        T_new, _ = diffusion_step(grid, T, dt, k, stack.h, T_amb, q)
        change = np.abs(T_new - T).max()
        T = T_new
        if dt >= 1e8 and change < tol:
            converged = True
            break
        dt = min(dt * 1.3, 1e9)
    return SteadyResult(grid=grid, T_C=T - KELVIN, iterations=it, converged=converged)
