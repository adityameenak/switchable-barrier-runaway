"""Real barrier-material candidates, with the source of every number.

Unlike ``params.py`` (illustrative placeholders), each value here is tagged
with where it came from:

* ``"datasheet"``  -- a manufacturer's published typical value
* ``"paper"``      -- a peer-reviewed measurement
* ``"secondary"``  -- reported via a summary of the primary source that has not
                      yet been checked against the original document
* ``"assumed"``    -- not found in any source; a generic value for the material
                      class, to be replaced. Results that depend on an assumed
                      value must say so.

``to_barrier`` maps a candidate onto ``BarrierParams`` for the solver. Where
the data do not pin down a model input (the switch temperature and time
constant of the graphene switch, for instance) the choice is made there and
recorded in ``MODEL_ASSUMPTIONS``.

Run ``python -m runaway.candidates`` to regenerate ``data/barrier_candidates.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from dataclasses import replace
from pathlib import Path

from .params import BarrierParams

PROVENANCE = ("datasheet", "paper", "secondary", "assumed")


@dataclass(frozen=True)
class Source:
    key: str
    citation: str
    url: str


SOURCES: dict[str, Source] = {
    s.key: s
    for s in [
        Source(
            "aspen_pyrogel_xte",
            "Aspen Aerogels, Pyrogel XTE product data sheet, Rev 4.1 (2022). "
            "k per ASTM C177 at 2 psi compressive load.",
            "https://www.aerogel.com/wp-content/uploads/2021/06/Pyrogel_XTE_Datasheet_English_v4.1.pdf",
        ),
        Source(
            "gund_mica_p",
            "The Gund Company, Mica P - Flexible silicone-bonded phlogopite mica paper "
            "laminate, technical data sheet TGCR 04/2026. k per ASTM E1461.",
            "https://thegundcompany.com/wp-content/uploads/2025/04/Mica-P-Flexible-Silicone-Bonded-Mica-Paper-Laminate-by-The-Gund-Co-TDS.pdf",
        ),
        Source(
            "morgan_cerablanket",
            "Morgan Advanced Materials, Refractory Ceramic Fiber Blanket datasheet "
            "5-14-205 (02/2016), Cerablanket column. k per ASTM C201.",
            "https://www.pacorinc.com/pdf/refractory_ceramic_fiber_blanket-datasheet.pdf",
        ),
        Source(
            "henkel_tgp3000",
            "Henkel, BERGQUIST GAP PAD TGP 3000 product page (values taken from a "
            "search summary; the page itself could not be retrieved).",
            "https://www.henkel-adhesives.com/ca/en/product/thermal-gap-pad-materials/bergquist_gap_padtgp3000.html",
        ),
        Source(
            "zhao2025pweg",
            "Y. Zhao, S. Huang, Z. Jin, Z. Xie, H. Guo, H. Xie, Thermally conductive "
            "shape-stabilized phase change materials enabled by paraffin wax and "
            "nanoporous structural expanded graphite, Nanomaterials 15(2), 110 (2025). "
            "doi:10.3390/nano15020110. Sample SH5 (10 wt% EG, pressure-induced).",
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC11767520/",
        ),
        Source(
            "wang2024regulator",
            "J. Wang, X. Feng, Y. Yu, H. Huang, M. Zheng, Y. Xu, J. Wu, Y. Yang, J. Lu, "
            "Rapid temperature-responsive thermal regulator for safety management of "
            "battery modules, Nature Energy 9(8), 939-946 (2024). "
            "doi:10.1038/s41560-024-01535-5. Values from the abstract only.",
            "https://www.nature.com/articles/s41560-024-01535-5",
        ),
        # --- composite constituents (see composites.py) ---
        Source("nan1997", "C.-W. Nan, R. Birringer, D. R. Clarke, H. Gleiter, Effective thermal "
               "conductivity of particulate composites with interfacial thermal resistance, "
               "J. Appl. Phys. 81, 6692-6699 (1997). doi:10.1063/1.365209.",
               "https://doi.org/10.1063/1.365209"),
        Source("huxtable2003", "S. T. Huxtable et al., Interfacial heat flow in carbon nanotube "
               "suspensions, Nature Materials 2, 731-734 (2003). doi:10.1038/nmat996. "
               "Interface conductance ~12 MW/(m^2 K).", "https://doi.org/10.1038/nmat996"),
        Source("kim2001", "P. Kim, L. Shi, A. Majumdar, P. L. McEuen, Thermal transport measurements "
               "of individual multiwalled nanotubes, Phys. Rev. Lett. 87, 215502 (2001). "
               "doi:10.1103/PhysRevLett.87.215502.", "https://arxiv.org/abs/cond-mat/0106578"),
        Source("agnw_nsf", "Single silver nanowire, 84 nm hydraulic diameter: 332 W/(m K) at 300 K "
               "(NSF Public Access Repository 10280473; values from a search summary).",
               "https://par.nsf.gov/servlets/purl/10280473"),
        Source("huang2018", "C. Huang, X. Qian, R. Yang, Thermal conductivity of polymers and polymer "
               "nanocomposites, arXiv:1805.05561 (2018); h-BN values from a search summary.",
               "https://arxiv.org/abs/1805.05561"),
        Source("hydroxides", "Elsevier ScienceDirect topic overviews, Aluminum hydroxide / Magnesium "
               "hydroxide (decomposition enthalpy and onset; values from a search summary).",
               "https://www.sciencedirect.com/topics/chemistry/magnesium-hydroxide"),
        Source("calcite_decomp", "C. Rodriguez-Navarro et al., Thermal decomposition of calcite: "
               "mechanisms of formation and textural evolution of CaO nanocrystals, American "
               "Mineralogist 94, 578-593 (2009); dH ~ 177.8 kJ/mol (search summary).",
               "https://www.ugr.es/~grupo179/pdf/CRNavarro%2009a.pdf"),
        Source("horai1971", "K. Horai, Thermal conductivity of rock-forming minerals, J. Geophys. Res. "
               "76, 1278-1308 (1971); calcite 3.4-3.6 W/(m K) as quoted by secondary sources.",
               "https://doi.org/10.1029/JB076i005p01278"),
        Source("minerals", "Mineral density tables: webmineral.com (gibbsite 2.34, brucite 2.37 g/cm^3), "
               "SSHADE (brucite 2.39), Engineering ToolBox (calcite 2.71).",
               "https://engineeringtoolbox.com/mineral-density-d_1555.html"),
        Source("3m_k1", "3M Glass Bubbles K1 data sheet: true density 0.125 g/cc, calculated thermal "
               "conductivity 0.047 W/(m K) at 21 C.",
               "https://Aircraftspruce.Com/catalog/pdf/3MGLASS%20BUBBLES.pdf"),
        Source("sylgard184", "Dow SYLGARD 184 silicone elastomer data sheet: specific gravity 1.03; "
               "thermal conductivity 0.16-0.27 W/(m K) depending on the copy of the sheet.",
               "https://neyco.fr/uploads/media/product/0001/05/Sylgard-184-dow-corning-neyco.pdf"),
    ]
}


@dataclass(frozen=True)
class Value:
    """A number, its unit, how we know it, and (unless assumed) its source."""

    value: float
    unit: str
    provenance: str
    source: str | None = None
    note: str = ""

    def __post_init__(self):
        if self.provenance not in PROVENANCE:
            raise ValueError(f"unknown provenance {self.provenance!r}")
        if self.provenance != "assumed" and self.source not in SOURCES:
            raise ValueError(f"sourced value needs a key from SOURCES, got {self.source!r}")


@dataclass(frozen=True)
class Candidate:
    """One barrier material and everything the comparison needs to know about it.

    ``k_table`` holds (temperature [C], through-plane k [W/(m K)]) points as
    published; it is not extrapolated here. ``mechanism`` is one of
    ``static``, ``pcm`` (latent heat) or ``switch`` (k drops with temperature).
    """

    key: str
    name: str
    product: str
    mechanism: str
    k_table: tuple[tuple[float, float], ...]
    k_provenance: str
    k_source: str | None
    rho: Value  # [kg/m^3]
    cp: Value  # [J/(kg K)]
    T_max_C: Value  # highest rated temperature
    latent_heat: Value | None = None  # [J/kg]
    T_melt_C: Value | None = None
    thickness_range_mm: tuple[float, float] | None = None
    caveats: tuple[str, ...] = field(default_factory=tuple)


CANDIDATES: dict[str, Candidate] = {
    c.key: c
    for c in [
        Candidate(
            key="aerogel",
            name="Silica aerogel blanket",
            product="Aspen Pyrogel XTE",
            mechanism="static",
            k_table=(
                (0, 0.020), (100, 0.023), (200, 0.028), (300, 0.035),
                (400, 0.046), (500, 0.064), (600, 0.089),
            ),
            k_provenance="datasheet",
            k_source="aspen_pyrogel_xte",
            rho=Value(200.0, "kg/m^3", "datasheet", "aspen_pyrogel_xte", "nominal 0.20 g/cc"),
            cp=Value(1000.0, "J/(kg K)", "assumed", note="not on datasheet; generic silica/glass-fibre value"),
            T_max_C=Value(650.0, "C", "datasheet", "aspen_pyrogel_xte", "ASTM C447 max use temperature"),
            thickness_range_mm=(5.0, 10.0),
            caveats=(
                "Industrial pipe insulation grade; battery-specific aerogel sheets are thinner "
                "(~1-3 mm) and may differ in k and density.",
                "k measured at 2 psi; cell swelling compresses barriers further.",
            ),
        ),
        Candidate(
            key="mica",
            name="Phlogopite mica paper laminate",
            product="Gund Mica P (flexible, silicone bonded)",
            mechanism="static",
            k_table=((50, 0.23),),
            k_provenance="datasheet",
            k_source="gund_mica_p",
            rho=Value(1900.0, "kg/m^3", "datasheet", "gund_mica_p", "1.9 g/cm^3"),
            cp=Value(850.0, "J/(kg K)", "assumed", note="not on datasheet; generic silicate-mineral value"),
            T_max_C=Value(700.0, "C", "datasheet", "gund_mica_p", "continuous; 900 C intermittent"),
            thickness_range_mm=(0.013, 3.175),
            caveats=(
                "Through-plane k given at 50 C only; datasheet's 0.21 W/(m K) at 250 C is "
                "in-plane, so the high-temperature through-plane value is unknown.",
            ),
        ),
        Candidate(
            key="ceramic_fibre",
            name="Alumina-silica ceramic fibre",
            product="Morgan Cerablanket, 128 kg/m^3",
            mechanism="static",
            k_table=((260, 0.06), (538, 0.13), (816, 0.23), (1093, 0.34)),
            k_provenance="datasheet",
            k_source="morgan_cerablanket",
            rho=Value(128.0, "kg/m^3", "datasheet", "morgan_cerablanket"),
            cp=Value(1000.0, "J/(kg K)", "assumed", note="not on datasheet; generic alumina-silica value"),
            T_max_C=Value(1177.0, "C", "datasheet", "morgan_cerablanket", "continuous use"),
            thickness_range_mm=(6.0, 50.0),
            caveats=(
                "Blanket data, not the thin papers used between cells.",
                "No k below 260 C, so normal-operation k must be extrapolated.",
            ),
        ),
        Candidate(
            key="gap_pad",
            name="Conductive silicone gap pad",
            product="Bergquist GAP PAD TGP 3000",
            mechanism="static",
            k_table=((25, 3.0),),
            k_provenance="secondary",
            k_source="henkel_tgp3000",
            rho=Value(3200.0, "kg/m^3", "secondary", "henkel_tgp3000", "3.2 g/cm^3"),
            cp=Value(1000.0, "J/(kg K)", "secondary", "henkel_tgp3000", "1.0 J/(g K), ASTM E1269"),
            T_max_C=Value(200.0, "C", "secondary", "henkel_tgp3000", "operating range -60 to 200 C"),
            caveats=(
                "Rated only to 200 C: the silicone degrades well below cell runaway "
                "temperatures, so its high-temperature behaviour is not modelled by a constant k.",
                "Values not yet checked against the Henkel TDS.",
            ),
        ),
        Candidate(
            key="pcm_eg",
            name="Paraffin / expanded-graphite composite PCM",
            product="PW/EG, 10 wt% EG (Zhao et al. 2025, sample SH5)",
            mechanism="pcm",
            k_table=((25, 9.99),),
            k_provenance="paper",
            k_source="zhao2025pweg",
            rho=Value(1103.0, "kg/m^3", "paper", "zhao2025pweg", "Table 2, SH5"),
            cp=Value(2000.0, "J/(kg K)", "assumed", note="not reported; generic solid-paraffin value"),
            T_max_C=Value(85.0, "C", "paper", "zhao2025pweg",
                          "highest cycling temperature tested (25-85 C, 50 cycles); not a rating"),
            latent_heat=Value(240.06e3, "J/kg", "paper", "zhao2025pweg", "240.06 J/g"),
            T_melt_C=Value(74.0, "C", "paper", "zhao2025pweg", "paraffin melting point 'about 74 C'"),
            caveats=(
                "Developed for spacecraft electronics, not battery barriers.",
                "The 9.99 W/(m K) measurement temperature is not given in the text; 25 C is assumed.",
                "Paper reports k decreasing with temperature but the k(T) curve is only in a figure.",
                "Pressure-aligned EG: the reported k may be the in-plane value, not through-plane.",
                "Paraffin is flammable and boils/decomposes far below runaway temperatures; "
                "behaviour above ~85 C is unmeasured.",
            ),
        ),
        Candidate(
            key="graphene_switch",
            name="Graphene / expandable-microsphere thermal switch",
            product="Thermal regulator of Wang et al. (Nature Energy 2024)",
            mechanism="switch",
            k_table=((25, 1.33), (100, 0.10)),
            k_provenance="paper",
            k_source="wang2024regulator",
            rho=Value(1000.0, "kg/m^3", "assumed", note="not in abstract"),
            cp=Value(1000.0, "J/(kg K)", "assumed", note="not in abstract"),
            T_max_C=Value(100.0, "C", "paper", "wang2024regulator",
                          "temperature of the 0.1 W/(m K) state; not a rating"),
            caveats=(
                "Switch ratio ~13x, far below the idealised 100x in params.py.",
                "Switches 'within 30 s'; the model currently assumes instant switching.",
                "Expanding microspheres suggest a one-way (irreversible) switch; "
                "reversibility is not stated in the abstract.",
                "Only the abstract was available: density, cp and thickness are unknown.",
            ),
        ),
    ]
}


# Modelling choices that the sourced data do not determine. Each is a knob the
# material comparison varies or flags.
MODEL_ASSUMPTIONS = {
    "graphene_switch": (
        "Logistic switch k_on = 1.33, k_off = 0.10 W/(m K) with T_sw = 85 C, w = 5 K, so k "
        "reaches ~0.11 W/(m K) at 100 C as reported. 'Within 30 s' is modelled as a first-order "
        "lag with tau = 10 s (95% of the change in 30 s). One-way (irreversible) by default; "
        "the reversible variant is also run."
    ),
    "pcm_eg": "Constant k = 9.99 W/(m K); melting spread as a Gaussian with 3 K standard deviation.",
    "gap_pad": "Constant k = 3.0 W/(m K) at all temperatures, although it is rated only to 200 C.",
    "mica": "Constant k = 0.23 W/(m K) (only through-plane value available).",
    "aerogel": "Tabulated k(T); held at 0.089 W/(m K) above 600 C.",
    "ceramic_fibre": "Tabulated k(T); held at 0.06 W/(m K) below 260 C (no low-temperature data).",
}


def to_barrier(key: str, thickness: float = 2e-3, **overrides) -> BarrierParams:
    """Solver input for candidate ``key`` at the given thickness [m]."""
    c = CANDIDATES[key]
    k0 = c.k_table[0][1]
    b = BarrierParams(
        name=key, thickness=thickness, k_on=k0, k_off=k0, rho=c.rho.value, cp=c.cp.value
    )
    if len(c.k_table) > 1 and c.mechanism == "static":
        b = replace(b, k_table=c.k_table)
    if c.mechanism == "pcm":
        b = replace(b, latent_heat=c.latent_heat.value, T_melt_C=c.T_melt_C.value, melt_width=3.0)
    if c.mechanism == "switch":
        b = replace(
            b, k_on=c.k_table[0][1], k_off=c.k_table[-1][1], T_sw_C=85.0, sw_width=5.0,
            switch_tau=10.0, switch_irreversible=True,
        )
    return replace(b, **overrides)


def to_markdown() -> str:
    """Render the candidate table with provenance flags and the reference list."""
    def fmt(v: Value | None, scale: float = 1.0, digits: int = 0) -> str:
        if v is None:
            return "-"
        flag = "" if v.provenance in ("datasheet", "paper") else f" ({v.provenance})"
        return f"{v.value * scale:.{digits}f}{flag}"

    lines = [
        "# Barrier material candidates",
        "",
        "Generated by `python -m runaway.candidates` from `runaway/candidates.py`; do not edit by hand.",
        "",
        "Values marked *(assumed)* were not found in any source; *(secondary)* values come from a",
        "summary of the source and still need checking against the original document.",
        "",
        "| Material | Mechanism | k [W/(m K)] at T [C] | rho [kg/m3] | cp [J/(kg K)] | Latent heat [kJ/kg] | T_melt [C] | T_max [C] |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in CANDIDATES.values():
        k = ", ".join(f"{k:g} @ {T:g}" for T, k in c.k_table)
        if c.k_provenance not in ("datasheet", "paper"):
            k += f" ({c.k_provenance})"
        lines.append(
            f"| **{c.name}**<br>{c.product} | {c.mechanism} | {k} | {fmt(c.rho)} | {fmt(c.cp)} "
            f"| {fmt(c.latent_heat, 1e-3, 1)} | {fmt(c.T_melt_C)} | {fmt(c.T_max_C)} |"
        )
    lines += ["", "## Caveats", ""]
    for c in CANDIDATES.values():
        lines.append(f"**{c.name}**")
        lines += [f"- {txt}" for txt in c.caveats]
        lines.append("")
    lines += ["## Sources", ""]
    lines += [f"- `{s.key}`: {s.citation} <{s.url}>" for s in SOURCES.values()]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "data" / "barrier_candidates.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(to_markdown())
    print(f"wrote {out}")
