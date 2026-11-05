"""Paper-only figures: mesh convergence and the real-material comparison.

    python paper/make_figures.py      # writes paper/fig_convergence.png,
                                      # paper/fig_materials.png and
                                      # paper/materials_summary.txt

The other figures in the paper come from ``python -m runaway.demo``.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from runaway.candidates import CANDIDATES  # noqa: E402
from runaway.demo import DESIGN_COLORS, INK_2, set_style  # noqa: E402
from runaway.explorer import export_convergence  # noqa: E402
from runaway.params import StackParams, barrier_conductor  # noqa: E402
from runaway.scenarios import MATERIAL_SENSITIVITY, material_sweep  # noqa: E402
from runaway.solver import solve_normal_operation  # noqa: E402

HERE = Path(__file__).parent

# Categorical slots 1-6 in fixed order (validated: CVD and normal-vision pass;
# three hues are below 3:1 contrast, so every line is directly labelled and the
# numbers are also tabulated in the paper).
MATERIAL_STYLE = {
    "graphene_switch": ("#2a78d6", "o", "graphene switch"),
    "aerogel": ("#eb6834", "s", "silica aerogel"),
    "ceramic_fibre": ("#1baf7a", "D", "ceramic fibre"),
    "mica": ("#eda100", "^", "mica paper"),
    "pcm_eg": ("#e87ba4", "v", "paraffin/EG PCM"),
    "gap_pad": ("#008300", "P", "silicone gap pad"),
}


def convergence() -> None:
    c = export_convergence()
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for key, color, label in [
        ("capped", DESIGN_COLORS["switchable"], r"with rate ceiling $k_\mathrm{max}=1\ \mathrm{s^{-1}}$"),
        ("uncapped", DESIGN_COLORS["conductor"], "pure Arrhenius"),
    ]:
        pts = [(n, v) for n, v in zip(c["n"], c[key]) if v is not None]
        ax.plot(*zip(*pts), "o-", color=color, label=label, ms=6, mec="white", mew=1.2)
    ax.set_xscale("log", base=2)
    ax.set_xticks(c["n"], [str(n) for n in c["n"]])
    ax.set_ylim(0, None)
    ax.set_xlabel("Finite volumes per cell")
    ax.set_ylabel("Cell-2 propagation time [s]")
    ax.legend(loc="lower left")
    ax.text(0.99, 0.97, "static conductor barrier", transform=ax.transAxes, ha="right",
            va="top", fontsize=8, color=INK_2)
    out = HERE / "fig_convergence.png"
    fig.savefig(out)
    print(f"wrote {out}", c)


def materials() -> None:
    sweeps = {key: material_sweep(key) for key in MATERIAL_STYLE}
    ref_peak = solve_normal_operation(StackParams(barrier=barrier_conductor())).peak_C

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.0, 4.0), gridspec_kw={"width_ratios": [1.35, 1]})
    y_max = 120.0
    end_labels = []
    for key, (color, marker, label) in MATERIAL_STYLE.items():
        s = sweeps[key]
        L, peak, ok = s.thickness_mm, s.normal_peak_C, s.n_propagated == 1
        a1.plot(L, peak, color=color, lw=1.2, ls=(0, (3, 2)), alpha=0.9)  # whole curve, dashed
        a1.plot(np.where(ok, L, np.nan), np.where(ok, peak, np.nan), color=color, lw=2.4)
        if s.min_containing_mm is not None:
            a1.plot(s.min_containing_mm, s.peak_at_min_C, marker, color=color, ms=8, mec="white",
                    mew=1.2, zorder=5)
        # direct label at the right edge (or where the curve leaves the axes)
        i = np.nonzero(peak <= y_max)[0][-1]
        end_labels.append([peak[i], L[i], label])
    # Dodge right-edge labels so close curves do not overprint (min 3.5 C apart).
    end_labels.sort()
    for j in range(1, len(end_labels)):
        end_labels[j][0] = max(end_labels[j][0], end_labels[j - 1][0] + 3.5)
    for y, x, label in end_labels:
        a1.text(x + 0.15, y, label, fontsize=8, color=INK_2, va="center")
    a1.hlines(ref_peak, 0.5, 10.0, color=INK_2, lw=0.8, ls=":")
    a1.text(0.6, ref_peak - 1.0, f"ideal conductor, {ref_peak:.0f} °C", fontsize=7.5, color=INK_2, va="top")
    a1.set_xlim(0.5, 12.6)
    a1.set_ylim(30, y_max)
    a1.set_xlabel("Barrier thickness [mm]")
    a1.set_ylabel("Normal-operation peak temperature [°C]")
    a1.set_title("(a) Cooling cost vs. thickness", loc="left")
    a1.text(0.42, 0.97, "solid: contains runaway   dashed: propagates\nmarker: thinnest containing barrier",
            transform=a1.transAxes, fontsize=7.5, color=INK_2, va="top")

    never = []
    for key, (color, marker, label) in MATERIAL_STYLE.items():
        s = sweeps[key]
        if s.min_containing_mm is None:
            never.append(label)
            continue
        a2.plot(s.min_containing_mm, s.peak_at_min_C, marker, color=color, ms=10, mec="white", mew=1.5)
        a2.annotate(f"{label}\n{s.min_containing_mm:g} mm, {s.peak_at_min_C:.0f} °C",
                    (s.min_containing_mm, s.peak_at_min_C), xytext=(8, -2), textcoords="offset points",
                    fontsize=8, color=INK_2, va="center")
    a2.axhline(ref_peak, color=INK_2, lw=0.8, ls=":")
    a2.set_xlim(0, 9.5)
    a2.set_ylim(30, 90)
    a2.set_xlabel("Thinnest barrier that contains runaway [mm]")
    a2.set_ylabel("Normal-operation peak at that thickness [°C]")
    a2.set_title("(b) Trade-off at the containment limit", loc="left")
    a2.text(0.98, 0.97, "never contains up to 10 mm:\n" + "\n".join(never), transform=a2.transAxes,
            fontsize=7.5, color=INK_2, ha="right", va="top")
    a2.text(0.03, 0.03, "better", transform=a2.transAxes, fontsize=8, color=INK_2, style="italic")
    fig.tight_layout()
    out = HERE / "fig_materials.png"
    fig.savefig(out)
    print(f"wrote {out}")

    lines = ["Real-material comparison (python paper/make_figures.py)", "",
             f"ideal-conductor normal peak: {ref_peak:.1f} C", "",
             f"{'material':18s} {'min L [mm]':>10s} {'peak [C]':>9s}   cells at 2 mm   peak at 2 mm"]
    for key in MATERIAL_STYLE:
        s = sweeps[key]
        i2 = int(np.argmin(np.abs(s.thickness_mm - 2.0)))
        mn = "never" if s.min_containing_mm is None else f"{s.min_containing_mm:g}"
        pk = "-" if s.peak_at_min_C is None else f"{s.peak_at_min_C:.1f}"
        lines.append(f"{key:18s} {mn:>10s} {pk:>9s}   {s.n_propagated[i2]:>13d}   {s.normal_peak_C[i2]:>10.1f}")
        if s.min_containing_mm is None:
            lines.append(f"{'':18s} cells in runaway vs thickness: "
                         + ", ".join(f"{L:g} mm: {n}" for L, n in zip(s.thickness_mm[::4], s.n_propagated[::4])))
    lines += ["", "sensitivity (thinnest containing barrier; normal peak there)"]
    for label, key, ov in MATERIAL_SENSITIVITY:
        s = material_sweep(key, **ov)
        mn = "never" if s.min_containing_mm is None else f"{s.min_containing_mm:g} mm"
        pk = "" if s.peak_at_min_C is None else f"{s.peak_at_min_C:.1f} C"
        lines.append(f"  {label:28s} {mn:>9s}  {pk}")
    # How much does the PCM's latent heat buy? Cell-2 propagation time at 4 mm.
    from runaway.candidates import to_barrier
    from runaway.scenarios import SWEEP_SIM
    from runaway.solver import simulate_runaway

    lines += ["", "PCM latent-heat effect, 4 mm barrier, cell propagation times [s]"]
    for lab, ov in [("with latent heat", {}), ("latent heat removed", {"latent_heat": 0.0})]:
        r = simulate_runaway(StackParams(barrier=to_barrier("pcm_eg", 4e-3, **ov)), SWEEP_SIM)
        lines.append(f"  {lab:22s} " + "  ".join(f"{t:.0f}" for t in r.t_prop))
    (HERE / "materials_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    set_style()
    convergence()
    materials()
