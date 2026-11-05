"""Figures and summary for the 2D (cold-plate) study and the composites.

    python paper/run_2d_study.py      # once; caches paper/results_2d.json
    python paper/make_figures_2d.py   # fig_2d_field, fig_composites_k,
                                      # fig_2d_ranking, fig_hplate, results_2d_summary.txt
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from runaway.composites import FILLERS, MATRICES, effective_k  # noqa: E402
from runaway.demo import INK, INK_2, set_style  # noqa: E402
from runaway.params import StackParams  # noqa: E402
from runaway.scenarios import barrier_from_spec  # noqa: E402
from runaway.solver2d import Params2D, simulate_runaway2d  # noqa: E402

HERE = Path(__file__).parent
DATA = json.loads((HERE / "results_2d.json").read_text())

# Categorical slots 1-8 (validated palette; direct labels everywhere).
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

LABELS = {
    "design | conductor": "ideal conductor (5 W/m·K)",
    "design | insulator": "ideal insulator (0.03 W/m·K)",
    "design | switchable": "ideal switch (5 → 0.05)",
    "candidate | aerogel": "silica aerogel",
    "candidate | mica": "mica paper",
    "candidate | ceramic_fibre": "ceramic fibre",
    "candidate | gap_pad": "silicone gap pad",
    "candidate | pcm_eg": "paraffin/EG PCM",
    "candidate | graphene_switch": "graphene switch",
}
GROUP = {"design": ("ideal design", INK_2), "candidate": ("real material", SLOTS[0]),
         "composition": ("filled silicone", SLOTS[1])}


def label(key):
    return LABELS.get(key, "silicone + " + key.split(" | ")[1] if "| silicone" not in key else "neat silicone")


def field_figure():
    """Temperature field 60 s after triggering: conductor vs graphene switch."""
    p2 = Params2D()
    specs = [(("design", "conductor"), 3.5, "ideal conductor, 3.5 mm"),
             (("candidate", "graphene_switch"), 3.5, "graphene switch, 3.5 mm")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3), sharey=True)
    for ax, (spec, L, title) in zip(axes, specs):
        r = simulate_runaway2d(StackParams(barrier=barrier_from_spec(spec, L * 1e-3)), p2)
        i = int(np.argmin(np.abs(r.t - 60.0)))
        g = r.grid
        x_edges = np.concatenate([[0], np.cumsum(g.dx)]) * 1e3
        y_edges = np.linspace(0, p2.height, p2.ny + 1) * 1e3
        im = ax.pcolormesh(x_edges, y_edges, r.field(i), cmap="inferno", vmin=25, vmax=900,
                           shading="flat", rasterized=True)
        for x0, x1, kind in _layers(g):
            if kind == "barrier":
                ax.axvspan(x0, x1, ymin=0, ymax=1, fill=False, ec="white", lw=0.6, ls=":")
        ax.set_title(f"{title}  (t = {r.t[i]:.0f} s)", loc="left")
        ax.set_xlabel("x through stack [mm]")
        ax.grid(False)
    axes[0].set_ylabel("height above cold plate [mm]")
    cb = fig.colorbar(im, ax=axes, shrink=0.9, pad=0.015)
    cb.set_label("T [°C]")
    out = HERE / "fig_2d_field.png"
    fig.savefig(out)
    print("wrote", out)


def _layers(g):
    x, out = 0.0, []
    cid = g.cell_id
    start = 0
    for i in range(1, cid.size + 1):
        if i == cid.size or (cid[i] >= 0) != (cid[start] >= 0) or (cid[i] >= 0 and cid[i] != cid[start]):
            w = g.dx[start:i].sum() * 1e3
            out.append((x, x + w, "cell" if cid[start] >= 0 else "barrier"))
            x += w
            start = i
    return out


def composites_k_figure():
    m = MATRICES["silicone"]
    order = ["cnt", "agnw", "hbn", "caco3_micro", "caco3_nano", "ath", "glass_bubbles"]
    names = {"cnt": "MWCNT", "agnw": "Ag nanowires", "hbn": "h-BN platelets",
             "caco3_micro": "CaCO3, 5 µm", "caco3_nano": "CaCO3, 50 nm", "ath": "ATH / MDH, 1 µm",
             "glass_bubbles": "hollow glass bubbles"}
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for i, key in enumerate(order):
        f = FILLERS[key]
        phi = np.linspace(0, f.phi_max, 60)
        k = [effective_k(m, {key: x}) for x in phi]
        ax.plot(phi * 100, k, color=SLOTS[i], lw=2)
        ax.annotate(names[key], (phi[-1] * 100, k[-1]), xytext=(4, 0), textcoords="offset points",
                    fontsize=8, color=INK_2, va="center")
    ax.axhline(m.k.value, color=INK_2, lw=0.8, ls=":")
    ax.text(41, m.k.value * 1.05, "neat silicone", fontsize=7.5, color=INK_2)
    ax.set_yscale("log")
    ax.set_xlim(0, 62)
    ax.set_xlabel("Filler volume fraction [%]")
    ax.set_ylabel("Effective k [W/(m·K)]")
    ax.set_title("Filled silicone, Nan et al. effective-medium model", loc="left")
    out = HERE / "fig_composites_k.png"
    fig.savefig(out)
    print("wrote", out)


def ranking_figure(main):
    rows = sorted(main.items(), key=lambda kv: (kv[1]["min_containing_mm"] is None,
                                                kv[1]["min_containing_mm"] or 0))
    n = len(rows)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 0.32 * n + 1.2), sharey=True)
    y = np.arange(n)[::-1]
    for yi, (key, r) in zip(y, rows):
        grp, color = GROUP[key.split(" | ")[0]]
        if r["min_containing_mm"] is None:
            a1.text(0.2, yi, "does not contain up to 10 mm", fontsize=7.5, color=INK_2, va="center")
            continue
        a1.plot(r["min_containing_mm"], yi, "o", color=color, ms=7, mec="white", mew=1)
        a2.plot(r["peak_at_min_C"], yi, "o", color=color, ms=7, mec="white", mew=1)
        a1.text(r["min_containing_mm"] + 0.15, yi, f"{r['min_containing_mm']:g}", fontsize=7.5,
                color=INK_2, va="center")
        a2.text(r["peak_at_min_C"] + 0.3, yi, f"{r['peak_at_min_C']:.1f}", fontsize=7.5,
                color=INK_2, va="center")
    a1.set_yticks(y, [label(k) for k, _ in rows], fontsize=8)
    a1.set_xlim(0, 11)
    a1.set_xlabel("Thinnest barrier that contains runaway [mm]")
    a2.set_xlabel("Normal-operation peak at that thickness [°C]")
    lo = min(r["peak_at_min_C"] for _, r in rows if r["peak_at_min_C"] is not None)
    hi = max(r["peak_at_min_C"] for _, r in rows if r["peak_at_min_C"] is not None)
    a2.set_xlim(lo - 1, hi + 3)
    a1.set_title("(a) Containment", loc="left")
    a2.set_title("(b) Normal-use cost", loc="left")
    for g, (name, color) in GROUP.items():
        a2.plot([], [], "o", color=color, label=name)
    a2.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out = HERE / "fig_2d_ranking.png"
    fig.savefig(out)
    print("wrote", out)


def hplate_figure(hp):
    hs = sorted(hp, key=float)
    keys = list(hp[hs[0]])
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for i, key in enumerate(keys):
        xs, ys = [], []
        for h in hs:
            v = hp[h][key]["min_containing_mm"]
            xs.append(float(h))
            ys.append(np.nan if v is None else v)
        ax.plot(xs, ys, "o-", color=SLOTS[i], ms=6, mec="white", mew=1)
        last = next((j for j in range(len(ys) - 1, -1, -1) if np.isfinite(ys[j])), None)
        if last is not None:
            ax.annotate(label(key), (xs[last], ys[last]), xytext=(5, 0), textcoords="offset points",
                        fontsize=8, color=INK_2, va="center")
    ax.set_xscale("log")
    ax.set_xticks([float(h) for h in hs], [f"{float(h):g}" for h in hs])
    ax.set_xlim(40, 1500)
    ax.set_ylim(0, None)
    ax.set_xlabel("Cold-plate heat-transfer coefficient h_plate [W/(m²·K)]")
    ax.set_ylabel("Thinnest containing barrier [mm]")
    ax.set_title("Sensitivity to cold-plate cooling", loc="left")
    out = HERE / "fig_hplate.png"
    fig.savefig(out)
    print("wrote", out)


def summary(main, hp):
    lines = ["2D study (python paper/run_2d_study.py; python paper/make_figures_2d.py)", "",
             f"Params2D: {DATA['params2d']}", "",
             f"{'barrier':36s} {'min L [mm]':>10s} {'peak [C]':>9s}"]
    rows = sorted(main.items(), key=lambda kv: (kv[1]["min_containing_mm"] is None,
                                                kv[1]["min_containing_mm"] or 0))
    for key, r in rows:
        mn = "never" if r["min_containing_mm"] is None else f"{r['min_containing_mm']:g}"
        pk = "-" if r["peak_at_min_C"] is None else f"{r['peak_at_min_C']:.1f}"
        lines.append(f"{label(key):36s} {mn:>10s} {pk:>9s}")
    lines += ["", "cold-plate sensitivity: thinnest containing barrier [mm] (peak there [C])"]
    for key in hp[next(iter(hp))]:
        parts = []
        for h in sorted(hp, key=float):
            r = hp[h][key]
            parts.append(f"h={float(h):g}: " + ("never" if r["min_containing_mm"] is None else
                                                 f"{r['min_containing_mm']:g} ({r['peak_at_min_C']:.1f})"))
        lines.append(f"  {label(key):34s} " + "; ".join(parts))
    (HERE / "results_2d_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    set_style()
    composites_k_figure()
    ranking_figure(DATA["main"])
    hplate_figure(DATA["h_plate"])
    field_figure()
    summary(DATA["main"], DATA["h_plate"])
