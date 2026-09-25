# AGENTS.md — Agentic Coding Paper Repository Standards

Applies to all AI agents working on this specific research project (Agentic Coding as Boosting).

## Operating Model & Paper Writing

- **No Hallucinations:** Never invent citations, theorems, or data. If you are missing a reference, add a `TODO` or ask the user to provide it.
- **Incremental LaTeX Edits:** When editing `paper.tex` or `paper_appendix.tex`, make focused, small changes. Do not completely rewrite sections unless explicitly asked.
- **Compilation is Required:** If you change the LaTeX code, you must verify it compiles. Use `tectonic paper.tex` or standard `pdflatex`/`bibtex` to compile the PDF. Do not assume your LaTeX compiles without testing it.
- **Math Formatting:** Ensure all equations and proofs are properly formatted in LaTeX. Keep mathematical notations consistent with existing definitions.
- **Bibliography Management:** Always use `references.bib` for citations. Do not hardcode references in the text. Ensure BibTeX entries are complete (author, title, year, venue).

## Simulations & Data

- **Python Scripts:** The simulation code (`simulations.py`, `swe_bench_simulations.py`) must be reproducible. Ensure any dependencies are noted.
- **Updating Figures:** If you modify simulation logic, you must re-run the script (`python3 simulations.py`) to generate updated figures in the `figures/` directory and updated JSON in `data/`.
- **Do not commit large data files:** Only small simulation outputs (like JSON) should be saved in `data/`.
- **Match Paper Claims:** Any simulation output or graph must accurately reflect the theoretical claims in the paper (e.g., AdaBoost bounds, VC-dimension generalization). Do not manipulate data to fit the theory.

## Change Discipline

- **Spec-driven changes:** State your intent before making any non-trivial changes to the proofs or the Python code.
- **Proof Validation:** If editing `paper_appendix.tex`, explain the logical steps of your proof mathematically before implementing it in LaTeX.
- **Simplicity first:** If a simpler mathematical proof or a simpler simulation script works, use it. Do not overcomplicate the logic.
- **Match existing patterns:** Follow the visual style and structural layout already established in `paper.tex` and NeurIPS formatting (`neurips_2026.sty`).

## Anti-Patterns — Never Do

- Do not refactor the entire paper structure without permission.
- Do not add new LaTeX packages to `paper.tex` unless absolutely necessary (and verify they are compatible with NeurIPS style).
- Do not fake simulation data. All graphs and data points must be reproducible by running the Python scripts.
- Never finalize changes with a broken LaTeX build.
