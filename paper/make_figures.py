"""Paper-only figure: mesh convergence of the cell-2 propagation time.

    python paper/make_figures.py      # writes paper/fig_convergence.png

The other figures in the paper come from ``python -m runaway.demo``.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from runaway.demo import DESIGN_COLORS, INK_2, set_style  # noqa: E402
from runaway.explorer import export_convergence  # noqa: E402


def main() -> None:
    set_style()
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
    out = Path(__file__).with_name("fig_convergence.png")
    fig.savefig(out)
    print(f"wrote {out}", c)


if __name__ == "__main__":
    main()
