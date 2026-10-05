"""Regenerate every figure in ``figures/``.

    python -m runaway.demo            # full resolution (~1-2 min on a laptop)
    python -m runaway.demo --quick    # coarse sweeps, for a fast check
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402

from .params import SimParams, StackParams, barrier_switchable  # noqa: E402
from .scenarios import (  # noqa: E402
    compare_designs,
    design_map,
    normal_operation,
    switch_temperature_sweep,
)
from .solver import RunawayResult  # noqa: E402

# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------

INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"

# One fixed colour per barrier design, used in every figure.
DESIGN_COLORS = {
    "none": "#2a78d6",
    "conductor": "#eb6834",
    "insulator": "#1baf7a",
    "switchable": "#4a3aa7",
}
DESIGN_LABELS = {
    "none": "No barrier (direct contact)",
    "conductor": "Static conductor",
    "insulator": "Static insulator",
    "switchable": "Switchable barrier",
}
CELL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelcolor": INK_2,
            "axes.edgecolor": INK_2,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "text.color": INK,
            "lines.linewidth": 2.0,
            "legend.frameon": False,
        }
    )


def barrier_desc(stack: StackParams) -> str:
    b = stack.barrier
    if b.thickness == 0:
        return "cells in direct contact"
    if b.is_switchable:
        return (
            f"{b.thickness * 1e3:g} mm, k = {b.k_on:g} → {b.k_off:g} W/m·K "
            f"at {b.T_sw_C:g} °C"
        )
    return f"{b.thickness * 1e3:g} mm, k = {b.k_on:g} W/m·K"


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def fig_cell_temperatures(results: dict[str, RunawayResult], out: Path) -> None:
    """(a) Mean temperature of every cell vs time, one panel per design."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    for ax, (name, r) in zip(axes.flat, results.items()):
        Tm = r.T_cell_mean_C
        t = np.maximum(r.t, 1e-3)  # log axis
        for c in range(Tm.shape[1]):
            ax.plot(t, Tm[:, c], color=CELL_COLORS[c], label=f"cell {c + 1}")
            if np.isfinite(r.t_prop[c]) and c > 0:
                j = np.searchsorted(r.t, r.t_prop[c])
                ax.plot(r.t_prop[c], Tm[min(j, len(t) - 1), c], "o",
                        ms=6, color=CELL_COLORS[c], mec="white", mew=1.2, zorder=5)
        if r.grid.stack.barrier.is_switchable:
            T_sw = r.grid.stack.barrier.T_sw_C
            ax.axhline(T_sw, color=INK_2, lw=1, ls="--")
            ax.text(0.012, T_sw + 15, f"T_sw = {T_sw:g} °C", color=INK_2, fontsize=8)
        n = r.n_propagated
        verdict = "contained" if n == 1 else f"{n}/{r.grid.stack.n_cells} cells in runaway"
        ax.set_title(f"{DESIGN_LABELS[name]} — {verdict}\n", loc="left")
        ax.text(0.0, 1.02, barrier_desc(r.grid.stack), transform=ax.transAxes,
                fontsize=8.5, color=INK_2, va="bottom")
        ax.set_xscale("log")
        ax.set_xlim(1e-2, r.t[-1])
        ax.set_ylim(0, 950)
    for ax in axes[1]:
        ax.set_xlabel("Time after trigger [s]")
    for ax in axes[:, 0]:
        ax.set_ylabel("Cell-mean temperature [°C]")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Thermal runaway propagation: cell temperatures by barrier design",
                 x=0.01, ha="left", fontweight="bold", y=1.0)
    fig.text(0.01, 0.955, "Cell 1 triggered at 300 °C. Dots mark the time each cell "
             "reaches 90 % mean conversion.", color=INK_2, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(out)
    plt.close(fig)


def fig_heatmap(results: dict[str, RunawayResult], out: Path) -> None:
    """(b) Space-time temperature map, conductor vs switchable."""
    names = ["conductor", "switchable"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=False)
    for ax, name in zip(axes, names):
        r = results[name]
        g = r.grid
        t = np.maximum(r.t, 1e-3)
        x_edges = np.concatenate([[0.0], np.cumsum(g.dx)]) * 1e3
        t_edges = np.concatenate([[t[0] * 0.8], 0.5 * (t[1:] + t[:-1]), [t[-1]]])
        pc = ax.pcolormesh(t_edges, x_edges, r.T_C.T, cmap="inferno", vmin=25, vmax=900,
                           shading="flat", rasterized=True)
        for x0, x1, kind in g.layer_edges:
            if kind == "barrier":
                ax.axhspan(x0 * 1e3, x1 * 1e3, color="white", alpha=0.18, lw=0)
                for xe in (x0, x1):
                    ax.axhline(xe * 1e3, color="white", lw=0.6, alpha=0.7)
        centers = [0.5 * (x0 + x1) * 1e3 for x0, x1, k in g.layer_edges if k == "cell"]
        ax.set_yticks(centers, [f"cell {i + 1}" for i in range(len(centers))])
        ax.tick_params(axis="y", length=0)
        ax.set_xscale("log")
        ax.set_xlim(1e-2, r.t[-1])
        ax.set_ylim(x_edges[-1], 0)
        ax.grid(False)
        ax.set_xlabel("Time after trigger [s]")
        ax.set_title(f"{DESIGN_LABELS[name]} ({barrier_desc(g.stack)})", loc="left",
                     fontsize=10)
    cb = fig.colorbar(pc, ax=axes, shrink=0.9, pad=0.02)
    cb.set_label("Temperature [°C]")
    cb.outline.set_visible(False)
    fig.suptitle("Where the heat goes: temperature through the stack over time "
                 "(barriers shaded)", x=0.01, ha="left", fontweight="bold")
    fig.savefig(out)
    plt.close(fig)


def fig_tradeoff(normal: dict, runaway: dict[str, RunawayResult], out: Path) -> None:
    """(c) Normal-operation peak temperature and runaway outcome, by design."""
    names = list(normal)
    colors = [DESIGN_COLORS[n] for n in names]
    labels = [DESIGN_LABELS[n].replace(" (direct contact)", "").replace(" ", "\n", 1)
              for n in names]
    x = np.arange(len(names))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))

    peaks = [normal[n].peak_C for n in names]
    T_amb = next(iter(normal.values())).grid.stack.T_amb_C
    a1.bar(x, np.array(peaks) - T_amb, bottom=T_amb, color=colors, width=0.62)
    for xi, p in zip(x, peaks):
        a1.text(xi, p + 1, f"{p:.1f} °C", ha="center", va="bottom", fontsize=9)
    a1.set_ylim(T_amb, max(peaks) * 1.15)
    a1.set_ylabel("Peak cell temperature [°C]")
    a1.set_title("Normal operation: steady-state peak\n(lower is better)", loc="left")

    n_prop = [runaway[n].n_propagated for n in names]
    n_cells = next(iter(runaway.values())).grid.stack.n_cells
    a2.bar(x, n_prop, color=colors, width=0.62)
    for xi, n in zip(x, n_prop):
        a2.text(xi, n + 0.08, f"{n}/{n_cells}", ha="center", va="bottom", fontsize=9)
    a2.set_ylim(0, n_cells + 0.8)
    a2.set_yticks(range(n_cells + 1))
    a2.set_ylabel("Cells in runaway (incl. trigger)")
    a2.set_title("Abuse: cell 1 triggered at 300 °C\n(lower is better)", loc="left")

    for ax in (a1, a2):
        ax.set_xticks(x, labels)
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="x", visible=False)
    fig.suptitle("The barrier tradeoff: only the switchable design wins on both",
                 x=0.01, ha="left", fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def boundary_conductance(dm) -> float | None:
    """Hot-state conductance k_off/L at the containment boundary [W/(m^2 K)].

    For each thickness, the boundary lies between the largest contained and
    the smallest propagating k_off; take the geometric mean of the two,
    divide by L, and average (geometrically) over thicknesses.
    """
    G = []
    for j, L in enumerate(dm.thickness):
        col = dm.n_propagated[:, j]
        ok, bad = dm.k_off[col == 1], dm.k_off[col > 1]
        if ok.size and bad.size:
            G.append(np.sqrt(ok.max() * bad.min()) / L)
    return float(np.exp(np.mean(np.log(G)))) if G else None


def fig_design_map(dm, default_stack: StackParams, out: Path) -> None:
    """(d) Switchable barrier: cells in runaway over thickness x k_off."""
    n_cells = default_stack.n_cells
    # Sequential single-hue ramp: 1 cell (contained) light -> all cells dark.
    ramp = ["#e7f4ee", "#f6c9b4", "#f0a07c", "#e3713f", "#b9471b"][:n_cells]
    cmap = ListedColormap(ramp)
    norm = BoundaryNorm(np.arange(0.5, n_cells + 1.5), cmap.N)

    def edges(v, log=False):
        v = np.log10(v) if log else v
        mid = 0.5 * (v[1:] + v[:-1])
        e = np.concatenate([[v[0] - (mid[0] - v[0])], mid, [v[-1] + (v[-1] - mid[-1])]])
        return 10**e if log else e

    fig, ax = plt.subplots(figsize=(7.2, 5))
    xe, ye = edges(dm.thickness * 1e3), edges(dm.k_off, log=True)
    pc = ax.pcolormesh(xe, ye, dm.n_propagated, cmap=cmap, norm=norm, shading="flat",
                       edgecolors="white", linewidth=0.4)
    # Contour of the containment boundary (n = 1 vs n > 1)
    ax.contour(dm.thickness * 1e3, dm.k_off, (dm.n_propagated == 1).astype(float),
               levels=[0.5], colors=INK, linewidths=1.5)
    # Guide: constant hot-state barrier conductance G = k_off / L through the
    # boundary. Containment needs G below a threshold set by the end cooling.
    G_star = boundary_conductance(dm)
    if G_star is not None:
        L = np.linspace(dm.thickness[0], dm.thickness[-1], 50)
        ax.plot(L * 1e3, G_star * L, color=INK, lw=1, ls=":")
        Lt = dm.thickness[int(0.75 * len(dm.thickness))]
        ax.annotate(f"k_off / L ≈ {G_star:.0f} W/m²K", (Lt * 1e3, G_star * Lt),
                    xytext=(-6, 8), textcoords="offset points", ha="right", fontsize=8.5,
                    color="white")
    b = default_stack.barrier
    ax.plot(b.thickness * 1e3, b.k_off, marker="*", ms=14, color=INK, mec="white", mew=1)
    ax.annotate("default design", (b.thickness * 1e3, b.k_off), xytext=(10, -14),
                textcoords="offset points", fontsize=9, color=INK)
    ax.text(0.97, 0.06, "contained", transform=ax.transAxes, ha="right", fontsize=10,
            fontweight="bold", color="#1b6b4f")
    ax.text(0.03, 0.94, "propagates through stack", transform=ax.transAxes, fontsize=10,
            fontweight="bold", color="white", va="top")
    ax.set_yscale("log")
    ax.set_xlim(xe[0], xe[-1])
    ax.set_ylim(ye[0], ye[-1])
    ax.set_xlabel("Barrier thickness [mm]")
    ax.set_ylabel("Hot-state conductivity k_off [W/m·K]")
    ax.grid(False)
    cb = fig.colorbar(pc, ax=ax, ticks=range(1, n_cells + 1), pad=0.02)
    cb.set_label("Cells in runaway (incl. trigger)")
    cb.outline.set_visible(False)
    ax.set_title(f"Switchable barrier design map (k_on = {b.k_on:g} W/m·K, "
                 f"T_sw = {b.T_sw_C:g} °C)", loc="left")
    fig.savefig(out)
    plt.close(fig)


def fig_tsw_sweep(
    sw, conductor_peak: float, out: Path, insulator_peak: float | None = None
) -> tuple[float, float] | None:
    """(e) Switch-temperature window: cools in operation AND stops propagation."""
    cools = sw.normal_peak_C <= conductor_peak + 1.0  # switch not tripped in use
    protects = sw.n_propagated == 1
    ok = cools & protects
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.5, 6), sharex=True,
                                 gridspec_kw={"height_ratios": [1.2, 1]})
    window = None
    if ok.any():
        lo, hi = sw.T_sw_C[ok].min(), sw.T_sw_C[ok].max()
        window = (lo, hi)
        for ax in (a1, a2):
            ax.axvspan(lo, hi, color="#1baf7a", alpha=0.12, lw=0)
        a2.text(0.5 * (lo + hi), 0.92, "works window", transform=a2.get_xaxis_transform(),
                ha="center", va="top", color="#12704e", fontweight="bold", fontsize=9)

    a1.plot(sw.T_sw_C, sw.normal_peak_C, color=DESIGN_COLORS["switchable"], marker="o", ms=4)
    a1.axhline(conductor_peak, color=DESIGN_COLORS["conductor"], lw=1, ls="--")
    a1.annotate("static conductor", (1.0, conductor_peak), xycoords=("axes fraction", "data"),
                xytext=(0, 4), textcoords="offset points", ha="right", fontsize=8,
                color=INK_2)
    if insulator_peak is not None:
        a1.axhline(insulator_peak, color=DESIGN_COLORS["insulator"], lw=1, ls="--")
        a1.annotate("static insulator", (1.0, insulator_peak),
                    xycoords=("axes fraction", "data"), xytext=(0, -11),
                    textcoords="offset points", ha="right", fontsize=8, color=INK_2)
    a1.set_ylabel("Normal-op peak [°C]")
    a1.set_title("Too low: the switch trips in normal use and traps heat", loc="left",
                 fontsize=10)

    a2.step(sw.T_sw_C, sw.n_propagated, where="mid", color=DESIGN_COLORS["switchable"])
    a2.plot(sw.T_sw_C, sw.n_propagated, "o", ms=4, color=DESIGN_COLORS["switchable"])
    a2.set_ylim(0.5, 5.5)
    a2.set_yticks(range(1, 6))
    a2.set_ylabel("Cells in runaway")
    a2.set_xlabel("Switch temperature T_sw [°C]")
    a2.set_title("Too high: the barrier switches too late to stop propagation",
                 loc="left", fontsize=10)
    fig.suptitle("Choosing the switch temperature", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return window


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="figures", help="output directory")
    ap.add_argument("--quick", action="store_true", help="coarse sweeps")
    ap.add_argument("--workers", type=int, default=None, help="processes for sweeps")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    set_style()
    base = StackParams()
    sim = SimParams()
    lines: list[str] = []

    def log(msg=""):
        print(msg, flush=True)
        lines.append(msg)

    t0 = time.time()
    log("ILLUSTRATIVE parameters -- see runaway/params.py\n")

    runaway = compare_designs(base, sim)
    normal = normal_operation(base, sim)
    log(f"{'design':<11} {'normal peak':>12} {'cells':>6}  propagation times [s]"
        f"      max |energy residual|")
    for name, r in runaway.items():
        tp = "  ".join("   -  " if np.isnan(v) else f"{v:6.1f}" for v in r.t_prop)
        log(f"{name:<11} {normal[name].peak_C:9.1f} C {r.n_propagated:>4}/5  {tp}"
            f"   {r.max_relative_energy_error:.1e} (relative)")
    log()

    fig_cell_temperatures(runaway, out / "a_cell_temperatures.png")
    fig_heatmap(runaway, out / "b_spacetime_heatmap.png")
    fig_tradeoff(normal, runaway, out / "c_tradeoff.png")

    if args.quick:
        thick = np.linspace(0.5, 5.0, 7) * 1e-3
        koff = np.logspace(np.log10(0.02), 0, 7)
        tsw = np.concatenate([np.arange(25.0, 60.0, 5.0), np.arange(60.0, 251.0, 20.0)])
    else:
        thick = np.linspace(0.5, 5.0, 19) * 1e-3
        koff = np.logspace(np.log10(0.02), 0, 19)
        tsw = np.concatenate([np.arange(25.0, 60.0, 1.0), np.arange(60.0, 251.0, 5.0)])

    dm = design_map(thick, koff, base, workers=args.workers)
    fig_design_map(dm, base, out / "d_design_map.png")
    G_star = boundary_conductance(dm)
    log(f"design map: {dm.n_propagated.size} runs, contained in "
        f"{(dm.n_propagated == 1).sum()}; outcomes seen: {sorted({int(v) for v in dm.n_propagated.flat})}")
    if G_star:
        log(f"containment boundary at hot-state conductance k_off/L ~ {G_star:.0f} W/m^2K"
            f" (end cooling h = {base.h:g} W/m^2K)")

    sw = switch_temperature_sweep(tsw, StackParams(barrier=barrier_switchable()),
                                  workers=args.workers)
    window = fig_tsw_sweep(sw, normal["conductor"].peak_C, out / "e_switch_temperature.png",
                           insulator_peak=normal["insulator"].peak_C)
    if window:
        log(f"T_sw window that both cools and protects: {window[0]:g} - {window[1]:g} C")
    else:
        log("no T_sw in the sweep both cools and protects")

    log(f"\nfigures written to {out}/ in {time.time() - t0:.0f} s")
    (out / "summary.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
