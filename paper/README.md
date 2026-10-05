# Technical report

`paper.tex` is a short technical report on the model and results (not peer
reviewed); `explainer.tex` is a plain-English companion guide. Compiled copies:
[paper.pdf](paper.pdf) and [explainer.pdf](explainer.pdf). Every number in it comes from `python -m runaway.demo` and
`python paper/make_figures.py`.

## Build the PDF

**Overleaf (no install):** create a new project, upload `paper.tex`, `refs.bib`,
`fig_convergence.png` and the five PNGs from `../figures/` (the `\graphicspath`
also searches the project root), then compile with pdfLaTeX.

**Locally** with MacTeX / TeX Live (e.g. VS Code + LaTeX Workshop, or the command line):

```bash
cd paper
pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper
```

Regenerate the convergence figure with `python paper/make_figures.py` (run from the
repository root).
