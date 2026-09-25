"""Report stats-macro usage across the paper sources.

There is no automated gate ensuring every ``\\stat*`` macro used in ``paper.tex`` or
``paper_appendix.tex`` is defined in ``stats_macros.tex``, nor one flagging macros that
fall out of use when prose is cut. This script provides both, so a prose revision can
be checked rather than assumed.

Usage::

    python scripts/check_macro_usage.py            # report
    python scripts/check_macro_usage.py --strict   # exit 1 if any macro is used but undefined
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

MACRO_RE = re.compile(r"\\(stat[A-Za-z]+)")
DEF_RE = re.compile(r"\\newcommand\{\\(stat[A-Za-z]+)\}")

SOURCES = ("paper.tex", "paper_appendix.tex")
MACRO_FILE = "stats_macros.tex"


def defined_macros(root: Path) -> set[str]:
    return set(DEF_RE.findall((root / MACRO_FILE).read_text()))


def used_macros(root: Path) -> dict[str, set[str]]:
    return {name: set(MACRO_RE.findall((root / name).read_text())) for name in SOURCES}


def abstract_only(root: Path, appendix_used: set[str]) -> set[str]:
    """Macros used solely inside the abstract, which a rewrite would orphan."""
    text = (root / "paper.tex").read_text()
    start = text.index(r"\begin{abstract}")
    end = text.index(r"\end{abstract}")
    abstract = text[start:end]
    body = text[:start] + text[end:]
    return set(MACRO_RE.findall(abstract)) - set(MACRO_RE.findall(body)) - appendix_used


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true", help="exit 1 on used-but-undefined")
    parser.add_argument("--root", default=".", help="repository root")
    args = parser.parse_args()

    root = Path(args.root)
    defined = defined_macros(root)
    used = used_macros(root)
    all_used = used[SOURCES[0]] | used[SOURCES[1]]

    undefined = sorted(all_used - defined)
    unused = sorted(defined - all_used)
    orphan_risk = sorted(abstract_only(root, used["paper_appendix.tex"]))

    print(f"defined: {len(defined)}   used: {len(all_used)}")
    print(f"used-but-undefined ({len(undefined)}): {undefined}")
    print(f"defined-but-unused ({len(unused)}): {unused}")
    print(f"abstract-only, would orphan if dropped ({len(orphan_risk)}): {orphan_risk}")

    if args.strict and undefined:
        print("FAIL: macros used in the paper are not defined in stats_macros.tex")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
