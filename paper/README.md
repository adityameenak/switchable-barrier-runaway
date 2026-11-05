# Technical report

`paper.tex` is a short technical report on the model and results (not peer
reviewed); `explainer.tex` is a plain-English companion guide. Compiled copies:
[paper.pdf](paper.pdf) and [explainer.pdf](explainer.pdf). Every number in it comes from `python -m runaway.demo` and
`python paper/make_figures.py`. The real-material comparison (Section 5.5) uses the sourced
barrier data in [`runaway/candidates.py`](../runaway/candidates.py) (readable table:
[`data/barrier_candidates.md`](../data/barrier_candidates.md)); its numbers are written to
[`materials_summary.txt`](materials_summary.txt).

## Build the PDF

**Overleaf (no install):** create a new project, upload `paper.tex`, `refs.bib`,
`fig_convergence.png`, `fig_materials.png` and the five PNGs from `../figures/` (the `\graphicspath`
also searches the project root), then compile with pdfLaTeX.

**Locally** with MacTeX / TeX Live (e.g. VS Code + LaTeX Workshop, or the command line):

```bash
cd paper
pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper
```

Regenerate the convergence and material figures with `python paper/make_figures.py` (run from the
repository root).
