"""Finite-volume grid and temperature-dependent material properties.

The stack is discretised layer by layer: each cell gets ``n_nodes_cell``
equal control volumes and each barrier ``n_nodes_barrier``. Control-volume
faces coincide with every material interface, so a property jump never falls
inside a control volume. The face conductances (see ``solver.py``) then use
the harmonic mean of the neighbouring conductivities, which is exact for
steady conduction through layers in series.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .params import KELVIN, BarrierParams, SimParams, StackParams


def switch_conductivity(T_K: np.ndarray, barrier: BarrierParams) -> np.ndarray:
    """Smooth, reversible conductivity of a (possibly switchable) barrier.

    ln k(T) = ln k_off + (ln k_on - ln k_off) * s(T),
    s(T)    = 1 / (1 + exp((T - T_sw) / w))

    ``s`` is 1 well below the switch temperature and 0 well above it, and
    k(T_sw) is the geometric mean of k_on and k_off. Blending ln k (rather
    than k) matters for a ~100x switch ratio: a linear blend leaves k several
    times k_off until ~20 K past T_sw, so the barrier would barely insulate
    near its nominal switch point. In log space the width ``w`` means the
    same thing on both sides of the switch.
    """
    if not barrier.is_switchable:
        return np.full_like(T_K, barrier.k_on, dtype=float)
    z = (T_K - (barrier.T_sw_C + KELVIN)) / barrier.sw_width
    # Clip the exponent so exp() never overflows far from the switch.
    s = 1.0 / (1.0 + np.exp(np.clip(z, -60.0, 60.0)))
    return barrier.k_off * (barrier.k_on / barrier.k_off) ** s


@dataclass
class Grid:
    """1D finite-volume grid and the per-node static properties."""

    x: np.ndarray  # control-volume centres [m]
    dx: np.ndarray  # control-volume widths [m]
    rho_cp: np.ndarray  # volumetric heat capacity [J/(m^3 K)]
    cell_id: np.ndarray  # index of the cell a node belongs to; -1 = barrier
    layer_edges: list[tuple[float, float, str]]  # (x0, x1, "cell"|"barrier")
    stack: StackParams

    @property
    def n(self) -> int:
        return self.x.size

    @property
    def is_cell(self) -> np.ndarray:
        return self.cell_id >= 0

    @property
    def length(self) -> float:
        return float(self.dx.sum())

    def conductivity(self, T_K: np.ndarray) -> np.ndarray:
        """Node conductivity [W/(m K)] at the given temperature field."""
        k = np.full(self.n, self.stack.cell.k)
        barrier_nodes = ~self.is_cell
        if barrier_nodes.any():
            k[barrier_nodes] = switch_conductivity(T_K[barrier_nodes], self.stack.barrier)
        return k

    def cell_mean(self, field: np.ndarray) -> np.ndarray:
        """Volume-weighted mean of a node field over each cell.

        Works on a 1D field (n,) or a history (n_times, n).
        """
        field = np.asarray(field)
        out = np.empty(field.shape[:-1] + (self.stack.n_cells,))
        for c in range(self.stack.n_cells):
            m = self.cell_id == c
            w = self.dx[m] / self.dx[m].sum()
            out[..., c] = field[..., m] @ w
        return out


def build_grid(stack: StackParams, sim: SimParams) -> Grid:
    """Lay out cells and barriers left to right and mesh each layer."""
    widths, rho_cp, cell_id, layers = [], [], [], []
    x0 = 0.0
    for c in range(stack.n_cells):
        L = stack.cell.thickness
        n = sim.n_nodes_cell
        widths.append(np.full(n, L / n))
        rho_cp.append(np.full(n, stack.cell.rho_cp))
        cell_id.append(np.full(n, c))
        layers.append((x0, x0 + L, "cell"))
        x0 += L

        is_last = c == stack.n_cells - 1
        if not is_last and stack.barrier.thickness > 0.0:
            L = stack.barrier.thickness
            n = sim.n_nodes_barrier
            widths.append(np.full(n, L / n))
            rho_cp.append(np.full(n, stack.barrier.rho_cp))
            cell_id.append(np.full(n, -1))
            layers.append((x0, x0 + L, "barrier"))
            x0 += L

    dx = np.concatenate(widths)
    x = np.cumsum(dx) - 0.5 * dx
    return Grid(
        x=x,
        dx=dx,
        rho_cp=np.concatenate(rho_cp),
        cell_id=np.concatenate(cell_id).astype(int),
        layer_edges=layers,
        stack=stack,
    )
