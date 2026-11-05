"""Two-dimensional extension: stack thickness (x) by cell height (y).

The 1D model's only heat sink is the two stack ends, which exaggerates how
much an insulating barrier heats the pack in normal use. Real modules sit on
a cold plate. This module adds the cell height as a second dimension:

    rho cp dT/dt = d/dx(k_x dT/dx) + d/dy(k_y dT/dy) + q_rxn + q_gen

    x = 0, L : convection to T_amb with h_end   (as in 1D)
    y = 0    : cold plate, convection to T_plate with h_plate
    y = H    : top surface, convection to T_amb with h_top

Cells are anisotropic: k_x is the through-plane value used in 1D and k_y the
much larger in-plane value (heat runs along the electrode sheets). Barriers
are isotropic. Per unit depth (z), everything is two-dimensional.

The numerics mirror ``solver.py``: finite volumes with interfaces on faces
(harmonic-mean face conductances), exact exponential update of the Arrhenius
progress, backward-Euler conduction with a sparse direct solve, the chord
heat capacity for latent heat, and an exact energy budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from .materials import build_grid, switch_conductivity, switch_fraction
from .params import KELVIN, SimParams, StackParams
from .solver import LATENT_TOL, MAX_LATENT_ITERS


@dataclass(frozen=True)
class Params2D:
    """Second-dimension geometry and boundary conditions. ILLUSTRATIVE values."""

    height: float = 0.10  # [m] cell height (y extent)
    ny: int = 20  # finite volumes over the height
    k_cell_y: float = 20.0  # [W/(m K)] in-plane cell conductivity. TODO: measured value
    h_plate: float = 200.0  # [W/(m^2 K)] cold plate incl. interface material. TODO
    T_plate_C: float = 25.0  # [C] coolant temperature
    h_top: float = 5.0  # [W/(m^2 K)] top surface, natural convection
    plate_under_barrier: bool = True  # barriers also touch the cold plate


# Coarser in x than the 1D default: the 2D runs are ~20x larger per step.
SIM2D = SimParams(n_nodes_cell=10, n_nodes_barrier=4, stop_when_settled=True,
                  t_end=7200.0, save_every=30.0, n_log_saves=20)


@dataclass
class Grid2D:
    stack: StackParams
    p2: Params2D
    dx: np.ndarray  # (nx,)
    dy: float
    cell_id: np.ndarray  # (nx,), -1 = barrier
    rho_cp: np.ndarray  # (n,) flattened, row-major (y, x)
    rho_L: np.ndarray  # (n,)
    area: np.ndarray  # (n,) control-volume area dx*dy [m^2 per m depth]

    @property
    def nx(self) -> int:
        return self.dx.size

    @property
    def ny(self) -> int:
        return self.p2.ny

    @property
    def n(self) -> int:
        return self.nx * self.ny

    @property
    def is_cell(self) -> np.ndarray:
        return np.tile(self.cell_id >= 0, self.ny)

    @property
    def node_cell(self) -> np.ndarray:
        return np.tile(self.cell_id, self.ny)

    def conductivity(self, T: np.ndarray, s_barrier: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        cells = self.is_cell
        kx = np.full(self.n, self.stack.cell.k)
        ky = np.full(self.n, self.p2.k_cell_y)
        kb = switch_conductivity(T[~cells], self.stack.barrier, s_barrier)
        kx[~cells] = kb
        ky[~cells] = kb
        return kx, ky

    # enthalpy with latent heat (same formulation as the 1D grid)
    def melt_fraction(self, T):
        from scipy.special import erf
        b = self.stack.barrier
        return 0.5 * (1.0 + erf((T - (b.T_melt_C + KELVIN)) / (np.sqrt(2.0) * b.melt_width)))

    def enthalpy(self, T, T_ref):
        H = self.rho_cp * (T - T_ref)
        if self.stack.barrier.has_latent_heat:
            H = H + self.rho_L * (self.melt_fraction(T) - self.melt_fraction(T_ref))
        return H

    def secant_capacity(self, T_old, T_new):
        b = self.stack.barrier
        dT = T_new - T_old
        dH = self.enthalpy(T_new, 0.0) - self.enthalpy(T_old, 0.0)
        z = (T_old - (b.T_melt_C + KELVIN)) / b.melt_width
        apparent = self.rho_cp + self.rho_L * np.exp(-0.5 * z**2) / (np.sqrt(2 * np.pi) * b.melt_width)
        small = np.abs(dT) < 1e-6
        return np.where(small, apparent, dH / np.where(small, 1.0, dT))


def build_grid2d(stack: StackParams, p2: Params2D, sim: SimParams) -> Grid2D:
    g1 = build_grid(stack, sim)
    b = stack.barrier
    rho_L_1d = np.where(g1.is_cell, 0.0, b.rho * b.latent_heat)
    dy = p2.height / p2.ny
    return Grid2D(
        stack=stack, p2=p2, dx=g1.dx, dy=dy, cell_id=g1.cell_id,
        rho_cp=np.tile(g1.rho_cp, p2.ny), rho_L=np.tile(rho_L_1d, p2.ny),
        area=np.tile(g1.dx * dy, p2.ny),
    )


def _assemble(g: Grid2D, kx: np.ndarray, ky: np.ndarray, h_end: float):
    """Face conductances [W/(m K)] per unit depth and boundary terms.

    Returns (rows, cols, G) for interior faces, plus per-node boundary
    conductance G_b and the weighted ambient sum G_b * T_b (for the RHS).
    """
    nx, ny, dy = g.nx, g.ny, g.dy
    p2 = g.p2
    KX = kx.reshape(ny, nx)
    KY = ky.reshape(ny, nx)
    DX = np.broadcast_to(g.dx, (ny, nx))
    idx = np.arange(g.n).reshape(ny, nx)

    # x-faces
    Gx = dy / (0.5 * DX[:, :-1] / KX[:, :-1] + 0.5 * DX[:, 1:] / KX[:, 1:])
    # y-faces
    Gy = DX[:-1, :] / (0.5 * dy / KY[:-1, :] + 0.5 * dy / KY[1:, :])
    rows = np.concatenate([idx[:, :-1].ravel(), idx[:-1, :].ravel()])
    cols = np.concatenate([idx[:, 1:].ravel(), idx[1:, :].ravel()])
    G = np.concatenate([Gx.ravel(), Gy.ravel()])

    T_amb = g.stack.T_amb_C + KELVIN
    T_plate = p2.T_plate_C + KELVIN
    Gb = np.zeros(g.n)
    GbT = np.zeros(g.n)

    def add(nodes, cond, T_b):
        Gb[nodes] += cond
        GbT[nodes] += cond * T_b

    if h_end > 0:
        add(idx[:, 0], dy / (1 / h_end + 0.5 * DX[:, 0] / KX[:, 0]), T_amb)
        add(idx[:, -1], dy / (1 / h_end + 0.5 * DX[:, -1] / KX[:, -1]), T_amb)
    bottom = idx[0, :]
    plate_mask = np.ones(nx, bool) if p2.plate_under_barrier else (g.cell_id >= 0)
    if p2.h_plate > 0:
        cond = g.dx / (1 / p2.h_plate + 0.5 * dy / KY[0, :])
        add(bottom[plate_mask], cond[plate_mask], T_plate)
    if p2.h_top > 0:
        add(idx[-1, :], g.dx / (1 / p2.h_top + 0.5 * dy / KY[-1, :]), T_amb)
    return rows, cols, G, Gb, GbT


def diffusion_step2d(g: Grid2D, T, dt, kx, ky, h_end, q_vol, cap=None):
    rows, cols, G, Gb, GbT = _assemble(g, kx, ky, h_end)
    C = g.rho_cp if cap is None else cap
    m = C * g.area / dt if np.isfinite(dt) else np.zeros(g.n)
    diag = m + Gb
    np.add.at(diag, rows, G)
    np.add.at(diag, cols, G)
    A = sp.csr_matrix(
        (np.concatenate([diag, -G, -G]),
         (np.concatenate([np.arange(g.n), rows, cols]), np.concatenate([np.arange(g.n), cols, rows]))),
        shape=(g.n, g.n),
    )
    rhs = m * T + q_vol * g.area + GbT
    T_new = spsolve(A, rhs)
    loss = 0.0 if not np.isfinite(dt) else dt * float(np.sum(Gb * T_new - GbT))
    return T_new, loss


@dataclass
class Result2D:
    grid: Grid2D
    t: np.ndarray
    T_C: np.ndarray  # (n_saves, n)
    t_prop: np.ndarray
    energy_residual: np.ndarray
    energy_scale: float
    n_steps: int
    meta: dict = field(default_factory=dict)

    @property
    def n_propagated(self) -> int:
        return int(np.isfinite(self.t_prop).sum())

    @property
    def max_relative_energy_error(self) -> float:
        return float(np.abs(self.energy_residual).max() / self.energy_scale)

    def field(self, i: int = -1) -> np.ndarray:
        return self.T_C[i].reshape(self.grid.ny, self.grid.nx)


def run_transient2d(g: Grid2D, T0, alpha0, sim: SimParams, *, h_end=None, q_vol=0.0, react=True) -> Result2D:
    stack = g.stack
    kin = stack.kinetics
    h_end = stack.h if h_end is None else h_end
    cells = g.is_cell
    node_cell = g.node_cell
    barrier = stack.barrier
    latent = barrier.has_latent_heat
    q_vol = np.broadcast_to(np.asarray(q_vol, float), (g.n,))
    dH_v = kin.dH_v(stack.cell)
    T_ref = stack.T_amb_C + KELVIN

    W = np.zeros((stack.n_cells, g.n))
    for c in range(stack.n_cells):
        m = node_cell == c
        W[c, m] = g.area[m] / g.area[m].sum()

    T = np.asarray(T0, float).copy()
    alpha = np.asarray(alpha0, float).copy()
    s_sw = switch_fraction(T[~cells], barrier) if barrier.has_switch_state else None

    def energy(T, alpha):
        th = np.sum(g.enthalpy(T, T_ref) * g.area)
        ch = np.sum(dH_v * (1 - alpha[cells]) * g.area[cells]) if react else 0.0
        return th + ch

    E0 = energy(T, alpha)
    scale = max(np.sum(dH_v * g.area[cells]), abs(E0), 1e-30)
    lost = generated = 0.0
    t = 0.0
    t_prop = np.full(stack.n_cells, np.nan)
    times, hist, res = [0.0], [T - KELVIN], [0.0]
    save_times = np.unique(np.concatenate([
        np.geomspace(1e-3, sim.t_end, sim.n_log_saves),
        np.arange(sim.save_every, sim.t_end + 1e-9, sim.save_every), [sim.t_end]]))
    i_save = 0
    n_steps = 0
    T_settled = sim.T_settled_C + KELVIN

    def check(t_now):
        newly = np.isnan(t_prop) & (W @ alpha > sim.alpha_propagated)
        t_prop[newly] = t_now

    check(0.0)
    while t < sim.t_end - 1e-12:
        if react:
            rate = kin.dT_ad * kin.rate_constant(T[cells]) * (1 - alpha[cells])
            peak = rate.max()
            dt = sim.dt_max if peak <= 0 else float(np.clip(sim.dT_rxn_max / peak, 1e-9, sim.dt_max))
        else:
            dt = sim.dt_max
        dt = min(dt, save_times[i_save] - t)

        if react:  # exponential-midpoint update, as in 1D
            a0, T0c = alpha[cells], T[cells]
            a_half = 1 - (1 - a0) * np.exp(-kin.rate_constant(T0c) * 0.5 * dt)
            T_half = T0c + kin.dT_ad * (a_half - a0)
            a_new = 1 - (1 - a0) * np.exp(-kin.rate_constant(T_half) * dt)
            alpha[cells] = a_new
            T[cells] += kin.dT_ad * (a_new - a0)

        T_it = T
        for _ in range(MAX_LATENT_ITERS if latent else max(1, sim.picard_iters)):
            kx, ky = g.conductivity(T_it, s_sw)
            cap = g.secant_capacity(T, T_it) if latent else None
            T_next, loss = diffusion_step2d(g, T, dt, kx, ky, h_end, q_vol, cap)
            done = latent and np.abs(T_next - T_it).max() < LATENT_TOL
            T_it = T_next
            if done:
                break
        T = T_it
        lost += loss
        generated += dt * np.sum(q_vol * g.area)

        if s_sw is not None:
            s_eq = switch_fraction(T[~cells], barrier)
            s_new = s_sw + (s_eq - s_sw) * (1 - np.exp(-dt / barrier.switch_tau)) if barrier.switch_tau > 0 else s_eq
            s_sw = np.minimum(s_sw, s_new) if barrier.switch_irreversible else s_new

        t += dt
        n_steps += 1
        if react:
            check(t)
        if t >= save_times[i_save] - 1e-12:
            times.append(t)
            hist.append(T - KELVIN)
            res.append(energy(T, alpha) + lost - generated - E0)
            i_save = min(i_save + 1, save_times.size - 1)
            if sim.stop_when_settled and (np.all(np.isfinite(t_prop)) or T.max() < T_settled):
                break

    return Result2D(g, np.array(times), np.array(hist), t_prop, np.array(res), scale, n_steps,
                    meta={} if s_sw is None else {"s_switch": s_sw})


def simulate_runaway2d(stack: StackParams, p2: Params2D | None = None, sim: SimParams | None = None) -> Result2D:
    """Trigger cell 1 (whole height) at ``sim.T_trigger_C``."""
    p2 = p2 or Params2D()
    sim = sim or SIM2D
    g = build_grid2d(stack, p2, sim)
    T0 = np.full(g.n, stack.T_amb_C + KELVIN)
    T0[g.node_cell == 0] = sim.T_trigger_C + KELVIN
    return run_transient2d(g, T0, np.zeros(g.n), sim)


@dataclass
class Steady2D:
    grid: Grid2D
    T_C: np.ndarray
    converged: bool

    @property
    def peak_C(self) -> float:
        return float(self.T_C.max())


def solve_normal_operation2d(stack: StackParams, p2: Params2D | None = None, sim: SimParams | None = None,
                             tol=1e-7, max_iter=5000) -> Steady2D:
    """Steady state with uniform cell heating, by pseudo-transient continuation."""
    p2 = p2 or Params2D()
    sim = sim or SIM2D
    g = build_grid2d(stack, p2, sim)
    q = np.where(g.is_cell, stack.q_normal, 0.0)
    T = np.full(g.n, stack.T_amb_C + KELVIN)
    dt, converged = 1.0, False
    for _ in range(max_iter):
        kx, ky = g.conductivity(T, None)
        T_new, _ = diffusion_step2d(g, T, dt, kx, ky, stack.h, q)
        change = np.abs(T_new - T).max()
        T = T_new
        if dt >= 1e8 and change < tol:
            converged = True
            break
        dt = min(dt * 1.3, 1e9)
    return Steady2D(g, T - KELVIN, converged)


def reduce_to_1d(p2: Params2D) -> Params2D:
    """2D settings that reproduce the 1D model (no plate, adiabatic top)."""
    return replace(p2, h_plate=0.0, h_top=0.0)
