"""Guard the bibliography, the one unguarded surface in an AI-drafted paper.

The Lean layer checks the algebra against hallucinated derivations; nothing checked
the references. These tests are offline and deterministic -- no network in the suite
-- and they pin three invariants:

1. every ``\\cite*`` key in the paper resolves to an entry in ``references.bib``;
2. every entry carries the fields ``AGENTS.md`` requires (title, author, year);
3. the arXiv entries added for the test-time-scaling discussion still match the
   titles and IDs that were verified against ``arxiv.org/abs`` and the arXiv API;
4. the DOI-verified journal entries still match the metadata Crossref returns for
   their DOI (title, journal, volume, issue, pages, year).

Invariants 3 and 4 are the anti-hallucination ones. Neither re-verifies over the
network -- they freeze what verification established, so that a later edit to a
title, an author list, an eprint ID or a page range fails loudly instead of silently
drifting away from the record a reviewer would load. To add an entry, verify it
first (arXiv API for preprints, api.crossref.org for anything with a DOI) and paste
the confirmed values. Crossref is preferred over aggregators for journal articles:
Semantic Scholar reports the wrong year for at least one entry here, having merged a
conference version with its journal version.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parent
BIB = ROOT / "references.bib"
SOURCES = ("paper.tex", "paper_appendix.tex")

# arXiv id -> (cite key, verified title). Verified 2026-08-14 against
# arxiv.org/abs/<id> and export.arxiv.org/api/query, with integrity controls: a
# nonexistent id returns totalResults=0 from the API (the /abs/ page 404s), and a
# neighbouring id returns an unrelated real paper.
VERIFIED_ARXIV = {
    "1909.13231": (
        "sun2019testtime",
        "Test-Time Training with Self-Supervision for Generalization under Distribution Shifts",
    ),
    "2407.04620": (
        "sun2024learning",
        "Learning to (Learn at Test Time): {RNNs} with Expressive Hidden States",
    ),
    # v1 was titled "...for Abstract Reasoning"; v2 retitled to "...for Few-Shot
    # Learning". The v2 string is what a reviewer loading 2411.07279 now sees.
    "2411.07279": (
        "akyurek2024testtime",
        "The Surprising Effectiveness of Test-Time Training for Few-Shot Learning",
    ),
    # Added 2026-08-18 for the continual-adaptation discussion (§2, app:rsi).
    # Verified against export.arxiv.org/api/query?id_list=2506.10943: totalResults=1,
    # six authors in this order, published 2025-06-12 (v2 2025-09-18), primary
    # category cs.LG, no journal_ref. The NeurIPS 2025 booktitle comes from a separate
    # check against api2.openreview.net (forum JsNUE84Hxi, venueid
    # NeurIPS.cc/2025/Conference), whose record omits one arXiv author -- see the
    # note in references.bib for which list is kept and why.
    "2506.10943": (
        "zweiger2025seal",
        "Self-Adapting Language Models",
    ),
    "2305.18466": (
        "hardt2023nearest",
        "Test-Time Training on Nearest Neighbors for Large Language Models",
    ),
    "2410.08020": (
        "hubotter2024efficiently",
        "Efficiently Learning at Test-Time: Active Fine-Tuning of {LLMs}",
    ),
    "2503.24235": (
        "zhang2025testtime",
        "A Survey on Test-Time Scaling in Large Language Models: What, How, Where, and How Well?",
    ),
    "2508.10024": (
        "munoz2025rttc",
        "{RTTC}: Reward-Guided Collaborative Test-Time Compute",
    ),
    "2507.19457": (
        "agrawal2025gepa",
        "{GEPA}: Reflective Prompt Evolution Can Outperform Reinforcement Learning",
    ),
    "2608.09629": (
        "xue2026rethinking",
        "Rethinking Self-Evolving Agents: Do We Still Need Prescribed Optimization Pipelines?",
    ),
    # The two cross-family aggregation systems the diversity result is positioned
    # against (S2). Verified 2026-08-14 via export.arxiv.org/api/query.
    "2306.02561": (
        "jiang2023llmblender",
        "{LLM}-Blender: Ensembling Large Language Models with Pairwise Ranking "
        "and Generative Fusion",
    ),
    "2406.04692": (
        "wang2024mixtureofagents",
        "Mixture-of-Agents Enhances Large Language Model Capabilities",
    ),
}

# DOI -> (cite key, verified title, journal, volume, number, pages, year). Verified
# 2026-08-14 against api.crossref.org/works/<doi>, which is authoritative for DOI
# metadata. Integrity note: Semantic Scholar's record for this same DOI reports
# year=1998 and venue="COLT' 98", conflating the COLT 1998 conference version with the
# Machine Learning journal version; Crossref's published-print date (1999, vol 37,
# no 3) is what a reviewer resolving the DOI sees, and is what is frozen here.
VERIFIED_DOI = {
    "10.1023/A:1007614523901": (
        "schapire1999improved",
        "Improved boosting algorithms using confidence-rated predictions",
        {"journal": "Machine Learning", "volume": "37", "number": "3", "pages": "297--336"},
        "1999",
    ),
}

ENTRY_RE = re.compile(r"^@(\w+)\s*\{\s*([^,\s]+)\s*,", re.MULTILINE)
CITE_RE = re.compile(r"\\cite[a-zA-Z]*\**(?:\[[^\]]*\])*\{([^}]*)\}")


def _bib_text() -> str:
    return BIB.read_text()


def _entries() -> dict[str, str]:
    """Map cite key -> raw entry body, splitting on top-level @entry starts."""
    text = _bib_text()
    starts = [(m.start(), m.group(2)) for m in ENTRY_RE.finditer(text)]
    out: dict[str, str] = {}
    for i, (pos, key) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(text)
        out[key] = text[pos:end]
    return out


def _cited_keys() -> set[str]:
    keys: set[str] = set()
    for name in SOURCES:
        for group in CITE_RE.findall((ROOT / name).read_text()):
            keys.update(k.strip() for k in group.split(",") if k.strip())
    return keys


def test_every_cited_key_resolves_to_a_bib_entry() -> None:
    missing = sorted(_cited_keys() - set(_entries()))
    assert not missing, f"cited but absent from references.bib: {missing}"


@pytest.mark.parametrize("field", ["title", "author", "year"])
def test_every_entry_has_the_required_field(field: str) -> None:
    """AGENTS.md requires complete entries; an entry missing these is unverifiable."""
    bad = sorted(
        key for key, body in _entries().items() if f"{field}=" not in body.replace(" ", "")
    )
    assert not bad, f"entries missing {field}: {bad}"


def test_arxiv_entries_declare_eprint_and_archive_prefix() -> None:
    """House style for preprints: eprint + archivePrefix, so the ID is checkable."""
    bad = []
    for key, body in _entries().items():
        if "eprint=" in body.replace(" ", "") and "archivePrefix" not in body:
            bad.append(key)
    assert not bad, f"entries with eprint but no archivePrefix: {bad}"


def test_eprint_ids_are_well_formed_arxiv_identifiers() -> None:
    """Both arXiv schemes are valid: post-2007 ``YYMM.NNNNN`` and the older
    ``archive/YYMMNNN`` (this bibliography contains one of the latter,
    ``cs/0309048`` for the Goedel machine)."""
    modern = r"\d{4}\.\d{4,5}(v\d+)?"
    legacy = r"[a-z-]+(\.[A-Z]{2})?/\d{7}(v\d+)?"
    ids = re.findall(r"eprint\s*=\s*\{([^}]*)\}", _bib_text())
    malformed = [i for i in ids if not re.fullmatch(f"{modern}|{legacy}", i)]
    assert not malformed, f"not well-formed arXiv ids: {malformed}"


@pytest.mark.parametrize(("arxiv_id", "expected"), sorted(VERIFIED_ARXIV.items()))
def test_verified_arxiv_entry_still_matches_the_record(
    arxiv_id: str, expected: tuple[str, str]
) -> None:
    """Freeze the verified (id, key, title) triples against silent drift."""
    key, title = expected
    entries = _entries()
    assert key in entries, f"{key} disappeared from references.bib"
    body = entries[key]
    assert f"eprint={{{arxiv_id}}}" in body.replace(" ", ""), (
        f"{key} no longer carries eprint {arxiv_id}"
    )
    # Compare titles insensitive to LaTeX line wrapping only.
    got = re.search(r"title\s*=\s*\{(.*?)\}\s*,\s*\n", body, re.DOTALL)
    assert got, f"{key} has no parseable title"
    normalised = " ".join(got.group(1).split())
    assert normalised == title, (
        f"{key} title drifted from the verified record.\n  bib:      {normalised}\n"
        f"  verified: {title}"
    )


@pytest.mark.parametrize(("doi", "expected"), sorted(VERIFIED_DOI.items()))
def test_verified_doi_entry_still_matches_the_record(doi: str, expected: tuple) -> None:
    """Same freeze as the arXiv triples, for entries verified through Crossref.

    A journal entry has more drift surface than a preprint -- volume, issue and page
    range are all quotable and all wrong-able -- so every field that was checked is
    pinned, not just the title.
    """
    key, title, fields, year = expected
    entries = _entries()
    assert key in entries, f"{key} disappeared from references.bib"
    body = entries[key]
    squashed = body.replace(" ", "")

    assert f"doi={{{doi}}}" in squashed, f"{key} no longer carries doi {doi}"
    assert f"year={{{year}}}" in squashed, f"{key} year drifted from the verified {year}"
    for field, value in fields.items():
        expected_frag = f"{field}={{{value}}}".replace(" ", "")
        assert expected_frag in squashed, (
            f"{key} {field} drifted from the verified record (expected {value})"
        )

    got = re.search(r"title\s*=\s*\{(.*?)\}\s*,\s*\n", body, re.DOTALL)
    assert got, f"{key} has no parseable title"
    normalised = " ".join(got.group(1).split())
    assert normalised == title, (
        f"{key} title drifted from the verified record.\n  bib:      {normalised}\n"
        f"  verified: {title}"
    )


def test_every_bib_entry_is_cited() -> None:
    """Widened from verified-only to every entry: bibtex silently drops an uncited entry,
    so it never reaches the PDF and never gets reviewed, while still looking like part of
    the bibliography to anyone reading the .bib."""
    uncited = sorted(set(_entries()) - _cited_keys())
    assert not uncited, (
        f"bib entries cited by neither {' nor '.join(SOURCES)}: {uncited}. "
        "Cite them or delete them."
    )
