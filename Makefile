# Makefile for the Audit-the-Scaffold paper + companion website.
#
#   make check         run every quality gate; non-zero exit on the first failure
#   make paper         build paper.pdf (Tectonic if available, else TeX Live)
#   make supplement    build formula_supplement.pdf (the colour-coded formula decoder)
#   make arxiv         build the arXiv source bundle in arxiv_submit/ (see scripts/prepare_arxiv.sh)
#   make site-assets   sync the companion-site assets in docs/assets/
#   make serve         preview the companion site locally at http://localhost:8000/
#   make clean-assets  remove generated docs/assets/ contents

# The 10 paper-referenced figures shown on the companion site (see docs/index.html).
# llm_fig_overlap joined the list when the overlap/diversity result became the lead
# contribution -- it had been paper-referenced (Fig. 6) but never synced to the site.
SITE_FIGS := \
	swe_fig1_training_error \
	swe_fig2_generalization \
	swe_fig3_refinement_game \
	swe_fig4_margin_distribution \
	llm_fig_edge_collapse \
	llm_fig_overlap \
	emp_fig1_churn_trajectories \
	emp_fig2_survivor_by_tier \
	emp_fig3_empirical_vs_simulated \
	emp_fig4_session_length_dist

DOCS_FIG_DIR := docs/assets/figures
SERVE_PORT ?= 8000

# Everything the complexity gate measures: the five top-level modules, the eight test
# modules, and scripts/. `wildcard` does not match dotfiles, so .vulture_allowlist.py
# stays out (bare names, no blocks to measure), and lean_proofs/ is never named --
# its .lake/packages/mathlib/scripts/ holds 29 vendored third-party Python files that
# are not ours to grade.
PY_SOURCES := $(wildcard *.py) scripts

.PHONY: check paper supplement arxiv site-assets serve clean-assets

# The single executable definition of the quality gates. README used to list these as
# seven loose commands a human had to remember and retype; two of them were not gates
# at all. Make stops at the first non-zero exit, so a failure names itself.
#
# `--extra dev` on every line is load-bearing: these tools live in the dev
# optional-dependency group, and plain `uv run <tool>` silently falls through to
# whatever is on PATH (this repo has been bitten by an Anaconda mypy 0.991 that way).
#
# On xenon: `radon cc -n C` cannot fail -- `-n` filters what is *displayed* and radon
# exits 0 no matter what it prints, so the "gate" it replaced was a report. Thresholds
# are letter ranks: absolute C = no single block over CC 20; modules C = no file whose
# *average* exceeds C; average B = the whole tree stays at B or better. Use the long
# flags -- the short forms are -b absolute, -m modules, -a average, which is not the
# mnemonic anyone guesses.
#
# vulture, ruff and xenon all run over $(PY_SOURCES) rather than a hand-listed subset.
# vulture used to see only llm_simulations.py; widening it to the tree found one real
# unused import in the legacy simulations.py -- which ruff could not have caught,
# because ruff excludes that file.
check:
	uv run --extra dev ruff check .
	uv run --extra dev ruff format --check .
	uv run --extra dev mypy
	uv run --extra dev xenon --max-absolute C --max-modules C --max-average B $(PY_SOURCES)
	uv run --extra dev vulture $(PY_SOURCES) .vulture_allowlist.py --min-confidence 80
	uv run --extra dev tach check
	uv run --extra dev pytest

# Four pdflatex passes, not three: with the appendix in place, the third pass still
# emits "Label(s) may have changed. Rerun to get cross-references right." A PDF whose
# whole job is proving a 9-page limit must not be one pass short of stable.
# Tectonic reruns to convergence itself.
paper:
	@if command -v tectonic >/dev/null 2>&1; then \
		tectonic paper.tex; \
	else \
		pdflatex paper.tex && bibtex paper && pdflatex paper.tex && pdflatex paper.tex && pdflatex paper.tex; \
	fi

# The standalone formula supplement (colour-coded decoder; shares stats_macros.tex, and
# never \inputs the paper). Two passes because hyperref needs a second one to settle its
# anchors; no bibtex, since the file has no citations.
supplement:
	@if command -v tectonic >/dev/null 2>&1; then \
		tectonic formula_supplement.tex; \
	else \
		pdflatex formula_supplement.tex && pdflatex formula_supplement.tex; \
	fi

# Stripped sources + paper.bbl + the figures paper.tex includes, verified by compiling
# the unpacked tarball. Independent of `make paper`: it runs its own clean build.
arxiv:
	./scripts/prepare_arxiv.sh

# Copy the compiled PDFs and the site figures into docs/assets/.
# The PNGs are gitignored under figures/ but tracked under docs/assets/ (see .gitignore).
# formula_supplement.pdf is synced here rather than left to a human step: the site links
# it, and an unsynced companion artifact is exactly how the supplement's cross-references
# drifted 11-of-12 stale in the first place.
site-assets:
	@mkdir -p $(DOCS_FIG_DIR)
	@if [ ! -f paper.pdf ]; then echo "paper.pdf not found — run 'make paper' first"; exit 1; fi
	@cp paper.pdf docs/assets/paper.pdf
	@echo "synced docs/assets/paper.pdf"
	@if [ ! -f formula_supplement.pdf ]; then echo "formula_supplement.pdf not found — run 'make supplement' first"; exit 1; fi
	@cp formula_supplement.pdf docs/assets/formula_supplement.pdf
	@echo "synced docs/assets/formula_supplement.pdf"
	@for f in $(SITE_FIGS); do \
		if [ -f figures/$$f.png ]; then \
			cp figures/$$f.png $(DOCS_FIG_DIR)/$$f.png; \
			echo "synced $(DOCS_FIG_DIR)/$$f.png"; \
		else \
			echo "MISSING figures/$$f.png — regenerate figures first"; exit 1; \
		fi; \
	done
	@echo "site-assets: done ($(words $(SITE_FIGS)) figures + paper.pdf + formula_supplement.pdf)"

# Preview the site exactly as GitHub Pages will serve it.
serve:
	@echo "Serving docs/ at http://localhost:$(SERVE_PORT)/  (Ctrl+C to stop)"
	@cd docs && python3 -m http.server $(SERVE_PORT)

clean-assets:
	@rm -f docs/assets/paper.pdf docs/assets/formula_supplement.pdf
	@rm -f $(DOCS_FIG_DIR)/*.png
	@echo "removed generated docs/assets/ contents"
