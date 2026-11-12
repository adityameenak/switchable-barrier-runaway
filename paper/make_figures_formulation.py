"""Figures and summary for the formulation study.

    python paper/run_formulation_study.py      # once; caches paper/results_formulation.json
    python paper/make_figures_formulation.py   # fig_form_k, fig_composites_k, fig_form_screen,
                                               # fig_form_switch, fig_form_robust, formulation_summary.txt
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from runaway.composites import MATRICES, effective_k
from runaway.demo import INK, INK_2, set_style
from runaway.formulation import (
    CAL_TARGET,
    PARTICLES,
    R_K_POOR,
    agnw,
    epoxy,
    formulation_k,
)

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SIL = MATRICES["silicone"]


def k_figure():
    """(a) calibration against measured AgNW/epoxy; (b) fillers in silicone."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 4.0))
    phi = np.linspace(0, 0.05, 80)
    curves = [
        ("100 nm x 15 µm, default $R_K$", lambda x: formulation_k(epoxy(0.19), [], agnw(100, 15), x)["k"], SLOTS[0], "-"),
        ("same, no junction conduction", lambda x: formulation_k(epoxy(0.19), [], agnw(100, 15, g=0.0), x)["k"], SLOTS[0], ":"),
        ("same, poorly dispersed ($R_K$ fit)", lambda x: formulation_k(epoxy(0.19), [], agnw(100, 15, R_K=R_K_POOR), x)["k"], SLOTS[1], "-"),
        ("120 nm x 15 µm in $k_m$ = 0.22", lambda x: formulation_k(epoxy(0.22), [], agnw(120, 15, False), x)["k"], SLOTS[2], "-"),
    ]
    for lab, fn, c, ls in curves:
        a.plot(phi * 100, [fn(x) for x in phi], color=c, ls=ls, lw=1.8, label=lab)
    from runaway.composites import FILLERS, Composition  # noqa: F401

    nan = [effective_k(SIL, {"agnw": x}) for x in phi]
    a.plot(phi * 100, nan, color=INK_2, ls="--", lw=1.2, label="Nan model, no fin limit (Sec. 2.5)")
    a.plot([4], [CAL_TARGET["coated"]], "o", color=SLOTS[0], mec=INK, ms=7)
    a.plot([4], [CAL_TARGET["uncoated"]], "s", color=SLOTS[1], mec=INK, ms=7)
    a.plot([0.455], [0.35], "D", color=SLOTS[2], mec=INK, ms=7)
    a.annotate("Chen 2014, coated", (4, 1.03), xytext=(-8, 8), textcoords="offset points", ha="right", fontsize=7.5, color=INK_2)
    a.annotate("Chen 2014, uncoated", (4, 0.57), xytext=(-8, -12), textcoords="offset points", ha="right", fontsize=7.5, color=INK_2)
    a.annotate("Zhang 2019", (0.455, 0.35), xytext=(6, -12), textcoords="offset points", fontsize=7.5, color=INK_2)
    a.set_ylim(0.1, 8)
    a.set_yscale("log")
    a.set_xlabel("Ag nanowire volume fraction [%]")
    a.set_ylabel("k [W/(m·K)]")
    a.set_title("(a) Nanowire model vs. measured epoxy composites", loc="left")
    a.legend(fontsize=7, loc="upper left", frameon=False)

    rows = [
        ("AgNW 40 nm x 40 µm", lambda x: formulation_k(SIL, [], agnw(40, 40), x)["k"], 0.05, SLOTS[0], "-"),
        ("AgNW 60 nm x 20 µm", lambda x: formulation_k(SIL, [], agnw(60, 20), x)["k"], 0.05, SLOTS[6], "-"),
        ("AgNW 100 nm x 15 µm", lambda x: formulation_k(SIL, [], agnw(100, 15), x)["k"], 0.05, SLOTS[7], "-"),
        ("h-BN platelets", lambda x: formulation_k(SIL, [(PARTICLES["hbn"], x)])["k"], 0.30, SLOTS[2], "-"),
        ("Al2O3, 5 µm", lambda x: formulation_k(SIL, [(PARTICLES["al2o3_micro"], x)])["k"], 0.40, SLOTS[3], "-"),
        ("Al2O3, 50 nm", lambda x: formulation_k(SIL, [(PARTICLES["al2o3_nano"], x)])["k"], 0.40, SLOTS[3], ":"),
        ("ATH, 1 µm", lambda x: formulation_k(SIL, [(PARTICLES["ath"], x)])["k"], 0.40, SLOTS[1], "-"),
    ]
    dy = {"Al2O3, 50 nm": -7, "ATH, 1 µm": 3, "Al2O3, 5 µm": 4}
    for lab, fn, pmax, c, ls in rows:
        x = np.geomspace(1e-3, pmax, 80)
        y = [fn(v) for v in x]
        b.plot(x * 100, y, color=c, ls=ls, lw=1.8)
        b.annotate(lab, (x[-1] * 100, y[-1]), xytext=(4, dy.get(lab, 0)), textcoords="offset points",
                   fontsize=7.5, color=INK_2, va="center")
    b.set_xscale("log")
    b.set_xticks([0.1, 0.3, 1, 3, 10, 30], ["0.1", "0.3", "1", "3", "10", "30"])
    b.set_yscale("log")
    b.set_xlim(0.1, 200)
    b.set_xlabel("Filler volume fraction [%]")
    b.set_ylabel("k [W/(m·K)]")
    b.set_title("(b) Single fillers in silicone", loc="left")
    from matplotlib.ticker import FormatStrFormatter, NullFormatter
    for ax in (a, b):
        ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
        ax.yaxis.set_minor_formatter(NullFormatter())
    a.set_yticks([0.1, 0.2, 0.5, 1, 2, 5])
    b.set_yticks([0.2, 0.5, 1, 2, 4])
    fig.tight_layout()
    fig.savefig(HERE / "fig_form_k.png")
    print("wrote fig_form_k.png")


FAMILY = [
    ("unfilled / endotherm only", lambda r: r["wire"] is None and not any(k in ("hbn", "al2o3_micro", "al2o3_nano") for k, _ in r["particles"]), INK_2),
    ("particles (h-BN, Al2O3)", lambda r: r["wire"] is None, SLOTS[2]),
    ("Ag nanowires", lambda r: r["wire"] is not None and not any(k in ("hbn", "al2o3_nano") for k, _ in r["particles"]), SLOTS[0]),
    ("nanowire + nanoparticle hybrid", lambda r: r["wire"] is not None, SLOTS[6]),
]


def family(r):
    for name, test, color in FAMILY:
        if test(r):
            return name, color
    raise ValueError(r)


def screen_figure(d):
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [1.15, 1]})
    for r in d["screen"]:
        if r["L_min"] is None:
            continue
        name, c = family(r)
        mk = "o" if r["phi_ms"] > 0 else "^"
        a.plot(r["L_min"], r["peak"], mk, color=c, ms=4.5, alpha=0.75, mec="none")
    for name, _, c in FAMILY:
        a.plot([], [], "o", color=c, label=name)
    a.plot([], [], "o", color="white", mec=INK_2, label="with microspheres (switch)")
    a.plot([], [], "^", color="white", mec=INK_2, label="static (no microspheres)")
    refs = d["refs"]
    lab = {"design | switchable": "ideal switch", "candidate | graphene_switch": "graphene switch",
           "candidate | aerogel": "aerogel", "candidate | mica": "mica", "candidate | ceramic_fibre": "ceramic fibre"}
    for k, r in refs.items():
        if r["L_min"] is None:
            continue
        a.plot(r["L_min"], r["peak"], "x", color=INK, ms=7, mew=1.6)
        a.annotate(lab[k], (r["L_min"], r["peak"]), xytext=(5, 3), textcoords="offset points", fontsize=7.5, color=INK)
    p = pick_row(d)
    a.plot(p["L_min"], p["peak"], "*", color=SLOTS[7], ms=16, mec=INK, mew=0.8, zorder=5)
    a.annotate("selected", (p["L_min"], p["peak"]), xytext=(-6, 12), textcoords="offset points", fontsize=8, ha="right",
               color=SLOTS[7], weight="bold")
    a.set_xlabel("Thinnest barrier that contains runaway [mm]")
    a.set_ylabel("Normal-operation peak there [°C]")
    a.set_title("(a) All screened formulations (1D)", loc="left")
    a.legend(fontsize=7, loc="upper right", frameon=False)
    a.set_ylim(37.5, None)

    # (b) best achievable per family: peak at L_min vs microsphere fraction
    for name, _, c in FAMILY:
        xs, ys = [], []
        for pms in sorted({r["phi_ms"] for r in d["screen"]}):
            rows = [r for r in d["screen"] if r["phi_ms"] == pms and r["L_min"] is not None
                    and family(r)[0] == name and r["L_min"] <= 6.0]
            if rows:
                xs.append(pms * 100)
                ys.append(min(r["peak"] for r in rows))
        b.plot(xs, ys, "o-", color=c, lw=1.6, ms=5, label=name)
    b.set_xlabel("Expandable-microsphere fraction [vol%]")
    b.set_ylabel("Best normal peak, containing at ≤ 6 mm [°C]")
    b.set_title("(b) Best per family", loc="left")
    b.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(HERE / "fig_form_screen.png")
    print("wrote fig_form_screen.png")


def pick_row(d):
    pk = d["pick"]
    for r in d["screen"]:
        if r["name"] == pk["name"]:
            return r
    raise KeyError(pk["name"])


def switch_figure(d):
    """Around the pick: (a) k_off and (b) containment vs microsphere fraction and
    expansion ratio; (c) effect of adding nanowires, by junction strength."""
    rows = d["map_ms_E"]
    Es = sorted({r["E"] for r in rows})
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(12.5, 3.8))
    for i, E in enumerate(Es):
        rr = sorted([r for r in rows if r["E"] == E], key=lambda r: r["phi_ms"])
        a.plot([r["phi_ms"] * 100 for r in rr], [r["k_off"] for r in rr], "o-", color=SLOTS[i], lw=1.6, ms=5,
               label=f"E = {E:g}")
        ok = [r for r in rr if r["L_min"] is not None]
        b.plot([r["phi_ms"] * 100 for r in ok], [r["L_min"] for r in ok], "o-", color=SLOTS[i], lw=1.6, ms=5,
               label=f"E = {E:g}")
    rr = sorted([r for r in rows if r["E"] == Es[0]], key=lambda r: r["phi_ms"])
    a.plot([r["phi_ms"] * 100 for r in rr], [r["k_on"] for r in rr], "s--", color=INK_2, lw=1.2, ms=4,
           label="$k_\\mathrm{on}$ (cold)")
    a.set_yscale("log")
    from matplotlib.ticker import FormatStrFormatter, NullFormatter
    a.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    a.yaxis.set_minor_formatter(NullFormatter())
    a.set_yticks([0.05, 0.1, 0.2, 0.3])
    a.set_xlabel("Microsphere fraction [vol%]")
    a.set_ylabel("k [W/(m·K)]")
    a.set_title("(a) Hot-state $k_\\mathrm{off}$", loc="left")
    a.legend(fontsize=7, frameon=False, ncol=2)
    b.set_xlabel("Microsphere fraction [vol%]")
    b.set_ylabel("Thinnest containing barrier [mm]")
    b.set_title("(b) Containment", loc="left")
    b.axvline(40, color=INK_2, lw=0.8, ls=":")
    b.legend(fontsize=7, frameon=False)
    wm = d["map_wire"]
    styles = {(0.0, "dilute"): (SLOTS[2], "-"), (0.05, "dilute"): (SLOTS[0], "-"), (0.5, "dilute"): (SLOTS[7], "-"),
              (0.0, "segregated"): (SLOTS[2], "--"), (0.05, "segregated"): (SLOTS[0], "--"),
              (0.5, "segregated"): (SLOTS[7], "--")}
    base = [r for r in d["robust"] if r["formulation"] == d["pick"]["name"] and r["case"] == "baseline"][0]
    for (g, net), (col, ls) in styles.items():
        rr = sorted([r for r in wm if r["g"] == g and r["network"] == net], key=lambda r: r["phi_w"])
        xs = [0.0] + [r["phi_w"] * 100 for r in rr]
        ys = [base["L_min"]] + [np.nan if r["L_min"] is None else r["L_min"] for r in rr]
        c.plot(xs, ys, "o" + ls, color=col, lw=1.5, ms=4, label=f"g = {g:g}, {net}")
    c.set_ylim(0, 10.5)
    c.set_xlabel("Ag nanowire (40 nm x 40 µm) fraction [vol%]")
    c.set_ylabel("Thinnest containing barrier [mm]")
    c.set_title("(c) Adding nanowires to the pick", loc="left")
    c.legend(fontsize=6.5, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(HERE / "fig_form_switch.png")
    print("wrote fig_form_switch.png")


def robust_figure(d):
    pk = d["pick"]["name"]
    rows = [r for r in d["robust"] if r["formulation"] == pk]
    nice = {"baseline": "baseline (E = 5, Expancel 051, tau = 10 s)"}
    fig, ax = plt.subplots(figsize=(7.5, 0.3 * len(rows) + 1.2))
    y = np.arange(len(rows))[::-1]
    for yi, r in zip(y, rows):
        if r["L_min"] is None:
            ax.text(0.3, yi, "does not contain up to 10 mm", fontsize=7.5, color=INK_2, va="center")
            continue
        c = SLOTS[7] if r["case"] == "baseline" else SLOTS[0]
        ax.barh(yi, r["L_min"], color=c, height=0.6)
        ax.text(r["L_min"] + 0.08, yi, f"{r['L_min']:g} mm, {r['peak']:.1f} °C", fontsize=7.5, color=INK_2, va="center")
    ax.set_yticks(y, [nice.get(r["case"], r["case"]) for r in rows], fontsize=8)
    ax.set_xlabel("Thinnest containing barrier [mm] (label: normal peak there)")
    ax.set_title("Selected formulation under uncertain inputs", loc="left")
    ax.set_xlim(0, max(r["L_min"] or 0 for r in rows) * 1.45 + 0.5)
    fig.tight_layout()
    fig.savefig(HERE / "fig_form_robust.png")
    print("wrote fig_form_robust.png")


def composites_k_figure():
    """Nan-model conductivity of the eleven fixed compositions' fillers (Sec. 2.5).

    Same figure as make_figures_2d.composites_k_figure, which needs the 2D cache.
    """
    from runaway.composites import FILLERS

    order = ["cnt", "agnw", "hbn", "caco3_micro", "caco3_nano", "ath", "glass_bubbles"]
    names = {"cnt": "MWCNT", "agnw": "Ag nanowires", "hbn": "h-BN platelets",
             "caco3_micro": "CaCO3, 5 µm", "caco3_nano": "CaCO3, 50 nm", "ath": "ATH / MDH, 1 µm",
             "glass_bubbles": "hollow glass bubbles"}
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for i, key in enumerate(order):
        f = FILLERS[key]
        phi = np.linspace(0, f.phi_max, 60)
        k = [effective_k(SIL, {key: x}) for x in phi]
        ax.plot(phi * 100, k, color=SLOTS[i], lw=2)
        ax.annotate(names[key], (phi[-1] * 100, k[-1]), xytext=(4, 0), textcoords="offset points",
                    fontsize=8, color=INK_2, va="center")
    ax.axhline(SIL.k.value, color=INK_2, lw=0.8, ls=":")
    ax.text(41, SIL.k.value * 1.05, "neat silicone", fontsize=7.5, color=INK_2)
    ax.set_yscale("log")
    ax.set_xlim(0, 62)
    ax.set_xlabel("Filler volume fraction [%]")
    ax.set_ylabel("Effective k [W/(m·K)]")
    ax.set_title("Filled silicone, Nan et al. effective-medium model", loc="left")
    fig.savefig(HERE / "fig_composites_k.png")
    print("wrote fig_composites_k.png")


def summary(d):
    L = ["Formulation study (python paper/run_formulation_study.py; python paper/make_figures_formulation.py)", ""]
    ok = sorted([r for r in d["screen"] if r["L_min"] is not None], key=lambda r: r["peak"])
    L.append(f"screened: {len(d['screen'])}; containing at <= 10 mm: {len(ok)}; at <= 6 mm: "
             f"{sum(r['L_min'] <= 6 for r in ok)}")
    L += ["", f"{'formulation':58s} {'k_on':>6s} {'k_off':>6s} {'L_min':>6s} {'peak':>6s}"]
    for r in ok[:25]:
        L.append(f"{r['name']:58s} {r['k_on']:6.3f} {r['k_off']:6.3f} {r['L_min']:6g} {r['peak']:6.1f}")
    L += ["", "best per family (containing at <= 6 mm):"]
    for name, _, _ in FAMILY:
        for sw in (False, True):
            rows = [r for r in ok if family(r)[0] == name and (r["phi_ms"] > 0) == sw and r["L_min"] <= 6]
            if rows:
                r = rows[0]
                L.append(f"  {name:32s} {'switch' if sw else 'static':6s} {r['name']:52s} "
                         f"k {r['k_on']:.3f}/{r['k_off']:.3f}  L {r['L_min']:g}  peak {r['peak']:.1f}")
    L += ["", "references:"]
    for k, r in d["refs"].items():
        L.append(f"  {k:30s} L_min {r['L_min']}  peak {r['peak'] if r['peak'] is None else round(r['peak'], 1)}")
    L += ["", f"lead formulations: {[x['name'] for x in d['lead']]}", "", "robustness (L_min mm, peak C):"]
    for x in d["lead"]:
        rows = [r for r in d["robust"] if r["formulation"] == x["name"]]
        L.append(f"  {x['name']}")
        for r in rows:
            pk = "-" if r["peak"] is None else f"{r['peak']:.1f}"
            L.append(f"      {r['case']:36s} k {r['k_on']:.3f}/{r['k_off']:.3f}  L {r['L_min']}  peak {pk}")
    L += ["", "extension to 45-50 vol% microspheres:"]
    for r in sorted(d.get("extend", []), key=lambda r: r["L_min"] or 99):
        L.append(f"  {r['name']:44s} k {r['k_on']:.3f}/{r['k_off']:.3f}  L {r['L_min']}  peak {r['peak']:.1f}")
    L += ["", f"PICK: {d['pick']}", "", "map phi_ms x E (L_min, peak, k_off):"]
    for r in d["map_ms_E"]:
        L.append(f"  ms {r['phi_ms']:.2f} E {r['E']:4g}  L {r['L_min']}  peak "
                 f"{'-' if r['peak'] is None else round(r['peak'], 1)}  k {r['k_on']:.3f}/{r['k_off']:.3f}")
    if "map_wire" in d:
        L += ["", "map phi_w x g:"]
        for r in d["map_wire"]:
            L.append(f"  w {r['phi_w']:.3f} g {r['g']:4g} {r.get('network', '')}  L {r['L_min']}  peak "
                     f"{'-' if r['peak'] is None else round(r['peak'], 1)}  k {r['k_on']:.3f}/{r['k_off']:.3f}")
    (HERE / "formulation_summary.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    set_style()
    k_figure()
    composites_k_figure()
    d = json.loads((HERE / "results_formulation.json").read_text())
    ref = json.loads((HERE / "results_formulation_refine.json").read_text())
    d.update(pick=ref["pick"], robust=d["robust"] + ref["robust"], lead=d["lead"] + ref["cands"],
             map_ms_E=ref["map_ms_E"], map_wire=ref["map_wire"], extend=ref["extend"])
    screen_figure(d)
    switch_figure(d)
    robust_figure(d)
    summary(d)
