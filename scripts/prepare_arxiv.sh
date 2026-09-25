#!/usr/bin/env bash
# Build the arXiv source bundle for paper.tex into arxiv_submit/ax.tar.gz.
#
# Adapted from bobaseb/coupling_gated_cortical_coherence's prepare_arxiv.sh, keeping
# its four rules and dropping its restyling and supplement merge (paper.tex is
# already the [preprint] build and \inputs its own appendix):
#
#   1. The file set is read from the sources on every run, never listed here, so a
#      figure added to or dropped from the paper needs no edit to this script. It is
#      read *after* comments are stripped: a commented-out \includegraphics or
#      \input must not be packed.
#   2. arXiv gets paper.bbl, not references.bib, so its BibTeX version cannot
#      reformat or break the references; \pdfoutput=1 tells it to run pdflatex.
#   3. Comments are stripped, because arXiv serves the source tarball publicly.
#      A trailing comment keeps its bare %, so line-joining is unchanged.
#   4. The bundle is compiled *from the unpacked tarball*, so a file missing from
#      the archive fails here rather than on arXiv's servers.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/arxiv_submit"
STAGE="$OUT/src"
cd "$ROOT"

die() { echo "prepare_arxiv: $*" >&2; exit 1; }
for tool in pdflatex bibtex pdfinfo perl shasum; do
  command -v "$tool" >/dev/null || die "$tool not found"
done

rm -rf "$OUT"
mkdir -p "$STAGE"

# Delete comment-only lines; cut trailing comments back to a bare %. An escaped \%
# is kept: the % must follow an even number of backslashes to start a comment.
strip_comments() {  # strip_comments <file>
  perl -i -ne 'next if /^\s*%/; s/((?:^|[^\\])(?:\\\\)*)%.*$/$1%/; print' "$1"
}

# 1. The TeX sources: paper.tex and whatever it \inputs, stripped as they are copied.
sources=(paper.tex neurips_2026.sty)
cp paper.tex neurips_2026.sty "$STAGE/"
strip_comments "$STAGE/paper.tex"
while IFS= read -r inp; do
  [ -f "$inp.tex" ] || die "\\input{$inp} has no source file"
  mkdir -p "$STAGE/$(dirname "$inp")"
  cp "$inp.tex" "$STAGE/$inp.tex"
  strip_comments "$STAGE/$inp.tex"
  sources+=("$inp.tex")
done < <(grep -ho '\\input{[^}]*}' "$STAGE/paper.tex" | sed 's/.*{\(.*\)}/\1/' | sort -u)

# 2. Figures, read from the stripped sources. A reference with no file is an error.
missing=0
while IFS= read -r fig; do
  if [ ! -f "$fig" ]; then
    echo "prepare_arxiv: no file on disk for \\includegraphics{$fig}" >&2
    missing=1
    continue
  fi
  mkdir -p "$STAGE/$(dirname "$fig")"
  cp "$fig" "$STAGE/$fig"
  sources+=("$fig")
done < <(grep -ho '\\includegraphics\(\[[^]]*\]\)\?{[^}]*}' "$STAGE"/*.tex |
         sed 's/.*{//; s/}$//' | sort -u)
[ "$missing" -eq 0 ] || die "figure references unresolved"

# 3. Build once in the stage to produce paper.bbl, then drop everything arXiv should
#    not rebuild from. Four pdflatex passes, as in `make paper`.
{ head -1 "$STAGE/paper.tex" | grep -q '^\\pdfoutput=1'; } ||
  perl -i -pe 'print "\\pdfoutput=1\n" if $. == 1' "$STAGE/paper.tex"
cp references.bib "$STAGE/"
sources+=(references.bib)
(
  cd "$STAGE"
  pdflatex -interaction=nonstopmode paper.tex >build.out 2>&1 || true
  bibtex paper >>build.out 2>&1 || die "bibtex failed; see $STAGE/build.out"
  for _ in 1 2 3; do pdflatex -interaction=nonstopmode paper.tex >>build.out 2>&1 || true; done
)
[ -s "$STAGE/paper.bbl" ] || die "no paper.bbl produced; see $STAGE/build.out"
stage_pages="$(pdfinfo "$STAGE/paper.pdf" | awk '/^Pages:/{print $2}')"

packed=(paper.tex paper.bbl)
for src in "${sources[@]}"; do
  case "$src" in paper.tex|references.bib) ;; *) packed+=("$src") ;; esac
done
(cd "$STAGE" && tar -czf "$OUT/ax.tar.gz" "${packed[@]}")

# 4. Compile what is about to be shipped, from the unpacked archive and nothing else.
verify="$OUT/verify"
mkdir -p "$verify"
tar -xzf "$OUT/ax.tar.gz" -C "$verify"
(
  cd "$verify"
  for _ in 1 2 3 4; do pdflatex -interaction=nonstopmode paper.tex >build.out 2>&1 || true; done
)
fail=0
check() {  # check <description> <extended regex>
  if grep -qE "$2" "$verify/paper.log"; then
    echo "prepare_arxiv: $1" >&2
    grep -E "$2" "$verify/paper.log" | head -5 >&2
    fail=1
  fi
}
check "LaTeX errors in the bundle"  '^! '
check "undefined references"        'LaTeX Warning: Reference .* undefined'
check "undefined citations"         'LaTeX Warning: Citation .* undefined'
check "missing files"               'File .* not found'
[ -f "$verify/paper.pdf" ] || { echo "prepare_arxiv: no paper.pdf produced" >&2; fail=1; }
[ "$fail" -eq 0 ] || die "NOT shipping a broken submission"

pages="$(pdfinfo "$verify/paper.pdf" | awk '/^Pages:/{print $2}')"
[ "$pages" = "$stage_pages" ] || die "bundle builds $pages pages, the stage built $stage_pages"
cp "$verify/paper.pdf" "$OUT/paper.pdf"
rm -rf "$STAGE" "$verify"

# 5. Record what this was built from; arxiv_submit/ is gitignored, so nothing else can.
{
  printf '# arXiv bundle built %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '# from commit %s%s\n' "$(git rev-parse --short HEAD)" \
    "$(git diff --quiet HEAD -- "${sources[@]}" || echo ' (sources modified)')"
  printf '# regenerate with make arxiv\n'
  for src in "${sources[@]}"; do shasum -a 256 "$src"; done
} >"$OUT/BUILD_MANIFEST"

echo "prepare_arxiv: compiled cleanly from the tarball, $pages pages"
echo "prepare_arxiv: $OUT/ax.tar.gz is ready"
tar -tzf "$OUT/ax.tar.gz"
