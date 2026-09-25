"""
resolve_author_aliases.py
--------------------------
Builds a ``(name, email) -> canonical identity`` alias map for control-mode
extraction, so ``git_session_extractor.py`` can hash one developer identity
per real person instead of once per email/handle they happened to commit
under.

Hashing developer identity by email overcounts distinct developers: the same
person's email varies across machines/accounts (personal vs. corporate,
GitHub-noreply vs. direct, LAN-IP/hostname-based) far more than their git
``user.name`` does. Hashing by display name alone helps but is structurally
incomplete: it can't merge cases where the *name itself* differs across
commits by the same person (a GitHub handle used as the author name on some
commits, a real "First Last" name on others) — a confirmed pattern in this
org's history. This module closes that gap with three progressively
stronger, auditable layers (see ``cluster_authors``).

The pure clustering logic (``cluster_authors``) takes no git dependency and
is fully unit-testable. Only ``main()`` touches git/the filesystem.

The heuristic is auditable but still under-merges, so the layers alone are not a
validated headcount. For a curated roster, a human reviews the clustering:
``--emit-candidates`` writes a TSV of every raw identity with the evidence behind
its cluster, the reviewer fills in ``decision`` and ``person``, and
``--from-reviewed`` turns that back into a *strict* map, which
``load_alias_resolver`` treats as an allowlist — unlisted identities are excluded
from the cohort, not merely left unmerged.

The candidate table deliberately carries no survivor-ratio or churn column.
Curation must be blind to outcomes; a roster curated while looking at
per-developer results would stop being a control.

Usage
-----
Heuristic map only (no human in the loop)::

    COMPANY_REPOS=/path/a,/path/b \\
    COMPANY_CONTROL_SINCE=2021-11-08 \\
    COMPANY_CONTROL_UNTIL=2022-09-19 \\
    COMPANY_CONTROL_ALIAS_MAP=/tmp/.../alias_map.json \\
        python3 scripts/resolve_author_aliases.py

Curated roster, in three steps::

    # 1. emit the reviewable table
    COMPANY_REPOS=... COMPANY_CONTROL_SINCE=... COMPANY_CONTROL_UNTIL=... \\
    COMPANY_CONTROL_IDENTITY_AUDIT=/tmp/.../candidates.tsv \\
        python3 scripts/resolve_author_aliases.py --emit-candidates

    # 2. a human edits candidates.tsv: decision=keep|drop, person=<label>

    # 3. build the strict roster map (no git access needed)
    COMPANY_CONTROL_ALIAS_MAP=/tmp/.../roster.json \\
        python3 scripts/resolve_author_aliases.py --from-reviewed /tmp/.../candidates.tsv

Both output files contain real names and emails and must never be committed or
written inside the repo — same rule as ``RAW_SESSIONS_PATH`` in
``anonymize_sessions.py``, and now enforced rather than merely documented.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Final

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import git_session_extractor as gse  # noqa: E402

GH_NOREPLY_ID_RE = re.compile(r"^(\d+)\+([^@]+)@users\.noreply\.github\.com$", re.IGNORECASE)
HANDLE_LIKE_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")

FUZZY_SIM_THRESHOLD = 0.72
FUZZY_AMBIGUITY_MARGIN = 0.05


def _name_key(name: str) -> str:
    return name.strip().casefold()


def _email_local(email: str) -> str:
    return email.split("@", 1)[0]


class _UnionFind:
    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def add(self, x: str) -> None:
        self._parent.setdefault(x, x)

    def find(self, x: str) -> str:
        self.add(x)
        while self._parent[x] != x:
            x = self._parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[ra] = rb


@dataclass
class AuthorStats:
    """Identity-only evidence for one raw ``(name, email)`` pair.

    Deliberately carries no survivor-ratio or churn field. Curation decisions
    must be made on identity evidence alone -- a roster curated while looking at
    per-developer outcomes would stop being a control.
    """

    n_commits: int = 0
    first_commit: str = ""
    last_commit: str = ""
    repos: set[str] = field(default_factory=set)

    def observe(self, iso_date: str, repo: str) -> None:
        self.n_commits += 1
        self.repos.add(repo)
        if not self.first_commit or iso_date < self.first_commit:
            self.first_commit = iso_date
        if not self.last_commit or iso_date > self.last_commit:
            self.last_commit = iso_date


AUDIT_COLUMNS: Final = (
    "cluster",
    "name",
    "email",
    "n_commits",
    "first_commit",
    "last_commit",
    "n_repos",
    "evidence",
    "fuzzy_score",
    "second_best",
    "decision",
    "person",
)


@dataclass
class AliasMerge:
    handle_name: str
    handle_email: str
    matched_name: str
    score: float
    second_best_score: float


@dataclass
class AliasResolution:
    canonical: dict[tuple[str, str], str]
    accepted_merges: list[AliasMerge] = field(default_factory=list)
    unresolved: list[tuple[str, str, str]] = field(default_factory=list)
    layer_cluster_counts: dict[str, int] = field(default_factory=dict)


def cluster_authors(pairs: set[tuple[str, str]]) -> AliasResolution:
    """Cluster raw ``(name, email)`` pairs into per-person canonical identities.

    Layer 1: group by ``casefold(name)`` — the same person keeping one
    display name across different emails is the common case and needs no
    fuzzy logic at all.

    Layer 2: union any two name-clusters whose emails share a GitHub-noreply
    *numeric* ID (``N+username@users.noreply.github.com`` — the numeric ID
    is a stable per-account identifier; the bare ``username@...`` form has no
    such guarantee and is left alone). Deterministic, no false-positive risk.

    Layer 3: for commits whose author *name* is itself a handle (no space,
    and either equal to the email local-part or matching a bare-username
    pattern), fuzzy-match it against every "First Last"-style name in the
    pool using ``rapidfuzz`` (already a project dependency — see
    ``llm_simulations.py`` for the same lazy-import convention). A merge is
    only accepted when the best match clears ``FUZZY_SIM_THRESHOLD`` *and*
    beats the second-best candidate by ``FUZZY_AMBIGUITY_MARGIN`` — every
    accepted and rejected candidate is recorded so the merge list can be
    reviewed, not just trusted blindly.
    """
    uf = _UnionFind()

    _layer1_seed_name_clusters(uf, pairs)
    layer1_clusters = _count_clusters(uf, pairs)

    _layer2_union_github_ids(uf, pairs)
    layer2_clusters = _count_clusters(uf, pairs)

    accepted, unresolved = _layer3_merge_handles(uf, pairs)
    layer3_clusters = _count_clusters(uf, pairs)

    return AliasResolution(
        canonical=_canonical_labels(uf, pairs),
        accepted_merges=accepted,
        unresolved=unresolved,
        layer_cluster_counts={
            "layer1_casefold_name": layer1_clusters,
            "layer2_github_id_union": layer2_clusters,
            "layer3_fuzzy_handle_merge": layer3_clusters,
        },
    )


def _count_clusters(uf: _UnionFind, pairs: set[tuple[str, str]]) -> int:
    """Distinct union-find roots across the pool, i.e. the person count so far."""
    return len({uf.find(_name_key(n)) for n, _ in pairs})


def _layer1_seed_name_clusters(uf: _UnionFind, pairs: set[tuple[str, str]]) -> None:
    """Layer 1: one cluster per casefolded display name."""
    for name, _email in pairs:
        uf.add(_name_key(name))


def _layer2_union_github_ids(uf: _UnionFind, pairs: set[tuple[str, str]]) -> None:
    """Layer 2: union names sharing a GitHub-noreply *numeric* ID.

    Only the ``N+username@users.noreply.github.com`` form counts -- the numeric ID is a
    stable per-account identifier. The bare ``username@...`` form has no such guarantee
    and is deliberately left alone.
    """
    id_to_names: dict[str, set[str]] = {}
    for name, email in pairs:
        m = GH_NOREPLY_ID_RE.match(email.strip())
        if m:
            id_to_names.setdefault(m.group(1), set()).add(_name_key(name))
    for names in id_to_names.values():
        ordered = sorted(names)
        for other in ordered[1:]:
            uf.union(ordered[0], other)


def _is_handle_like(stripped_name: str, email: str) -> bool:
    """True when a space-free author name looks like an account handle."""
    if _name_key(stripped_name) == _email_local(email).casefold():
        return True
    return HANDLE_LIKE_RE.match(stripped_name) is not None


def _score_handle(handle: str, real_names: list[str]) -> list[tuple[float, str]]:
    """Score one handle against every real name, best first.

    Plain ``fuzz.ratio`` only -- ``partial_ratio`` (best local alignment, not true
    substring containment) was tried and rejected: it inflates scores for short handles
    against long names by chance (e.g. a 4-letter handle can align well against *some*
    4-letter run inside an unrelated 17-letter name), which ``fuzz.ratio``'s
    whole-string comparison does not.
    """
    from rapidfuzz import fuzz

    handle_key = _name_key(handle).replace("-", "").replace("_", "").replace(".", "")
    scored = [
        (fuzz.ratio(handle_key, _name_key(real).replace(" ", "")) / 100.0, real)
        for real in real_names
    ]
    scored.sort(reverse=True)
    return scored


def _layer3_merge_handles(
    uf: _UnionFind, pairs: set[tuple[str, str]]
) -> tuple[list[AliasMerge], list[tuple[str, str, str]]]:
    """Layer 3: fuzzy-match handle-style names onto "First Last"-style ones.

    A merge is only accepted when the best match clears ``FUZZY_SIM_THRESHOLD`` *and*
    beats the second-best candidate by ``FUZZY_AMBIGUITY_MARGIN``. Every accepted and
    rejected candidate is recorded so the merge list can be reviewed, not just trusted.
    """
    real_names = sorted({name for name, _ in pairs if " " in name.strip()})
    accepted: list[AliasMerge] = []
    unresolved: list[tuple[str, str, str]] = []

    for name, email in pairs:
        stripped = name.strip()
        if " " in stripped or not _is_handle_like(stripped, email):
            continue

        scored = _score_handle(stripped, real_names)
        if not scored:
            unresolved.append((name, email, "no real-name candidates in pool"))
            continue

        best_score, best_real = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else 0.0
        margin_ok = (best_score - second_score) >= FUZZY_AMBIGUITY_MARGIN
        if best_score >= FUZZY_SIM_THRESHOLD and margin_ok:
            uf.union(_name_key(stripped), _name_key(best_real))
            accepted.append(AliasMerge(name, email, best_real, best_score, second_score))
        else:
            unresolved.append(
                (name, email, f"best={best_real!r} score={best_score:.3f} 2nd={second_score:.3f}")
            )

    return accepted, unresolved


def _canonical_labels(uf: _UnionFind, pairs: set[tuple[str, str]]) -> dict[tuple[str, str], str]:
    """Map every pair to a stable, readable label for its cluster.

    Prefers a "First Last"-style member if the cluster has one, else falls back to the
    lexically smallest name key (deterministic, not meaningful beyond that).
    """
    cluster_members: dict[str, set[str]] = {}
    for name, _email in pairs:
        cluster_members.setdefault(uf.find(_name_key(name)), set()).add(name.strip())

    canonical_label: dict[str, str] = {}
    for root, members in cluster_members.items():
        full_names = sorted(m for m in members if " " in m)
        canonical_label[root] = _name_key(full_names[0]) if full_names else root

    return {(name, email): canonical_label[uf.find(_name_key(name))] for name, email in pairs}


def _tab_safe(value: str) -> str:
    """Neutralize characters that would corrupt the TSV round-trip."""
    return value.replace("\t", " ").replace("\r", " ").replace("\n", " ")


def emit_candidates(stats: dict[tuple[str, str], AuthorStats], resolution: AliasResolution) -> str:
    """Render the reviewable candidate table as TSV text.

    Rows are grouped by heuristic cluster, clusters ordered by total commits
    descending and rows within a cluster likewise, so the highest-volume people
    -- the ones whose merge decisions actually move the headcount -- are at the
    top of the file. ``decision`` and ``person`` are left blank for the human:
    ``decision`` takes ``keep``/``drop``, ``person`` overrides the heuristic
    cluster to force a merge or a split.
    """
    merged_by_pair = {(m.handle_name, m.handle_email): m for m in resolution.accepted_merges}
    unresolved_pairs = {(name, email) for name, email, _reason in resolution.unresolved}
    shared_gh_ids = _shared_github_id_pairs(stats)

    cluster_totals: dict[str, int] = {}
    for pair, st in stats.items():
        cluster = resolution.canonical.get(pair, "")
        cluster_totals[cluster] = cluster_totals.get(cluster, 0) + st.n_commits

    def sort_key(item: tuple[tuple[str, str], AuthorStats]) -> tuple[int, str, int, str]:
        pair, st = item
        cluster = resolution.canonical.get(pair, "")
        return (-cluster_totals.get(cluster, 0), cluster, -st.n_commits, pair[0])

    buf = io.StringIO()
    writer = csv.writer(buf, delimiter="\t", lineterminator="\n")
    writer.writerow(AUDIT_COLUMNS)
    for pair, st in sorted(stats.items(), key=sort_key):
        name, email = pair
        merge = merged_by_pair.get(pair)
        if merge is not None:
            evidence = "fuzzy-merge"
            score, second = f"{merge.score:.3f}", f"{merge.second_best_score:.3f}"
        elif pair in unresolved_pairs:
            evidence, score, second = "fuzzy-unresolved", "", ""
        elif pair in shared_gh_ids:
            evidence, score, second = "gh-id-union", "", ""
        else:
            evidence, score, second = "casefold-name", "", ""
        writer.writerow(
            [
                _tab_safe(resolution.canonical.get(pair, "")),
                _tab_safe(name),
                _tab_safe(email),
                st.n_commits,
                st.first_commit,
                st.last_commit,
                len(st.repos),
                evidence,
                score,
                second,
                "",
                "",
            ]
        )
    return buf.getvalue()


def _shared_github_id_pairs(
    stats: dict[tuple[str, str], AuthorStats],
) -> set[tuple[str, str]]:
    """Pairs whose GitHub-noreply numeric ID is shared with a differently-named
    pair — i.e. the ones layer 2 actually unioned, as opposed to merely having a
    noreply address."""
    by_id: dict[str, set[tuple[str, str]]] = {}
    for name, email in stats:
        m = GH_NOREPLY_ID_RE.match(email.strip())
        if m:
            by_id.setdefault(m.group(1), set()).add((name, email))
    shared: set[tuple[str, str]] = set()
    for pairs in by_id.values():
        if len({_name_key(n) for n, _ in pairs}) > 1:
            shared |= pairs
    return shared


def build_reviewed_map(tsv_text: str) -> dict:
    """Turn a human-reviewed candidate table into a strict roster alias map.

    ``decision`` of ``drop`` omits the row, so the identity is excluded from the
    cohort entirely. A non-blank ``person`` overrides the heuristic ``cluster``,
    which is how a reviewer forces a merge the heuristic missed or splits one it
    made wrongly. Blank ``decision`` is treated as ``keep`` so a reviewer only
    has to mark the exceptions.

    Emits ``strict: true``, which is what tells ``load_alias_resolver`` to treat
    the map as an allowlist rather than a merge-only hint.
    """
    reader = csv.DictReader(io.StringIO(tsv_text), delimiter="\t")
    if reader.fieldnames is None:
        raise ValueError("reviewed table is empty — expected a TSV header row")
    missing = {"name", "email", "cluster", "decision", "person"} - set(reader.fieldnames)
    if missing:
        raise ValueError(f"reviewed table missing required column(s): {sorted(missing)}")

    lookup: dict[str, str] = {}
    raw_rows = 0
    dropped = 0
    for row in reader:
        raw_rows += 1
        decision = (row.get("decision") or "").strip().casefold()
        if decision in {"drop", "exclude", "no"}:
            dropped += 1
            continue
        if decision not in {"", "keep", "yes"}:
            raise ValueError(
                f"unrecognized decision {decision!r} for {row.get('name')!r} — "
                "expected 'keep', 'drop', or blank"
            )
        person = (row.get("person") or "").strip()
        canonical = person or (row.get("cluster") or "").strip()
        if not canonical:
            raise ValueError(
                f"row for {row.get('name')!r} has neither a cluster nor a person label"
            )
        lookup[f"{row['name']}\x1e{row['email']}"] = canonical.casefold()

    return {
        "lookup": lookup,
        "strict": True,
        "audit": {
            "raw_identities": raw_rows,
            "after_drop": len(lookup),
            "dropped": dropped,
            "after_merge": len(set(lookup.values())),
        },
    }


def _collect_author_stats(
    repo: str,
    since: str,
    until: str,
    into: dict[tuple[str, str], AuthorStats],
    *,
    claude_only: bool = False,
) -> None:
    """Accumulate per-``(name, email)`` commit counts and date ranges from one repo.

    Also yields the pair set for ``cluster_authors`` via ``set(into)``; the extra
    columns exist so a reviewer can judge an identity on volume and activity span
    rather than on the name alone.

    ``claude_only`` restricts counting to commits carrying a Claude co-author
    trailer, which is what the AI-tier extraction keeps. Without it, an AI-tier
    roster collects every author who ever touched those repositories -- on this
    org's 23 AI-tier repos that is 269 identities against the ~36 the extraction
    actually uses, so the reviewer would spend most of the pass on rows that can
    never match a commit. The control cohort wants the opposite (it keeps all
    non-bot commits), hence a flag rather than a change of default.
    """
    # The body (%B) is only needed when the Claude trailer has to be checked.
    fields = ["%an", "%ae", "%ad"] + (["%B"] if claude_only else [])
    fmt = "--format=" + "\x1f".join(fields) + "\x1e"
    result = subprocess.run(
        [
            "git",
            "-C",
            repo,
            "log",
            # `--all` walks every ref, which includes refs/notes/*. Note commits are
            # authored by whatever wrote the note -- review bots, CI annotators, or
            # local tooling -- so without this exclusion those identities enter the
            # roster as if they were developers who had authored code. `--exclude`
            # must precede `--all` to apply to it.
            "--exclude=refs/notes/*",
            "--all",
            f"--since={since}",
            f"--until={until}",
            fmt,
            "--date=short",
            # Narrow with git, then let the extractor's own regex decide, so the
            # roster cannot drift from what extraction actually keeps.
            *(["--grep=Co-Authored-By:", "-i"] if claude_only else []),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return
    n_fields = 4 if claude_only else 3
    for block in result.stdout.split("\x1e"):
        block = block.strip("\n")
        if not block:
            continue
        parts = block.split("\x1f", n_fields - 1)
        if len(parts) != n_fields:
            continue
        name, email, date = parts[0], parts[1], parts[2]
        if claude_only and not gse.CLAUDE_CO_AUTHOR_RE.search(parts[3]):
            continue
        if gse.BOT_AUTHOR_RE.search(name) or gse.BOT_AUTHOR_RE.search(email):
            continue
        into.setdefault((name, email), AuthorStats()).observe(date, repo)


def _report_resolution(resolution: AliasResolution) -> None:
    for layer, count in resolution.layer_cluster_counts.items():
        print(f"  {layer}: {count} clusters", file=sys.stderr)
    print(f"  {len(resolution.accepted_merges)} accepted fuzzy merges:", file=sys.stderr)
    for merge in resolution.accepted_merges:
        print(
            f"    {merge.handle_name!r} -> {merge.matched_name!r}  "
            f"score={merge.score:.3f} (2nd-best={merge.second_best_score:.3f})",
            file=sys.stderr,
        )
    print(f"  {len(resolution.unresolved)} handle-like rows left unresolved.", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__ or "")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--emit-candidates",
        action="store_true",
        help="Write the reviewable candidate table (TSV) to "
        "$COMPANY_CONTROL_IDENTITY_AUDIT instead of an alias map.",
    )
    mode.add_argument(
        "--from-reviewed",
        metavar="TSV",
        help="Build a strict roster alias map from a human-reviewed candidate "
        "table. Needs no git access.",
    )
    parser.add_argument(
        "--claude-only",
        action="store_true",
        help="Count only Claude-co-authored commits. Use for the AI-tier roster, "
        "whose extraction keeps exactly those; omit for the control cohort, "
        "which keeps all non-bot commits.",
    )
    args = parser.parse_args()

    if args.from_reviewed:
        out_path = os.environ.get("COMPANY_CONTROL_ALIAS_MAP", "")
        if not out_path:
            print("Error: COMPANY_CONTROL_ALIAS_MAP must be set.", file=sys.stderr)
            sys.exit(1)
        gse.require_outside_repo(out_path, "the alias map")
        with open(args.from_reviewed) as fh:
            payload = build_reviewed_map(fh.read())
        with open(out_path, "w") as fh:
            json.dump(payload, fh, indent=2)
        audit = payload["audit"]
        print(
            f"Reviewed {audit['raw_identities']} identities: dropped {audit['dropped']}, "
            f"kept {audit['after_drop']} across {audit['after_merge']} people.\n"
            f"Wrote strict roster map -> {out_path}",
            file=sys.stderr,
        )
        return

    repos_env = os.environ.get("COMPANY_REPOS", "")
    since = os.environ.get("COMPANY_CONTROL_SINCE", "")
    until = os.environ.get("COMPANY_CONTROL_UNTIL", "")
    out_env = (
        "COMPANY_CONTROL_IDENTITY_AUDIT" if args.emit_candidates else "COMPANY_CONTROL_ALIAS_MAP"
    )
    out_path = os.environ.get(out_env, "")
    if not (repos_env and since and until and out_path):
        print(
            f"Error: COMPANY_REPOS, COMPANY_CONTROL_SINCE, COMPANY_CONTROL_UNTIL, "
            f"and {out_env} must all be set.",
            file=sys.stderr,
        )
        sys.exit(1)
    gse.require_outside_repo(
        out_path, "the candidate table" if args.emit_candidates else "the alias map"
    )

    repos = [p.strip() for p in repos_env.split(",") if p.strip()]
    stats: dict[tuple[str, str], AuthorStats] = {}
    for repo in repos:
        _collect_author_stats(repo, since, until, stats, claude_only=args.claude_only)
    scope = "Claude-co-authored" if args.claude_only else "non-bot"
    print(
        f"Collected {len(stats)} distinct (name, email) pairs from {scope} commits "
        f"across {len(repos)} repos.",
        file=sys.stderr,
    )

    resolution = cluster_authors(set(stats))
    _report_resolution(resolution)

    if args.emit_candidates:
        with open(out_path, "w") as fh:
            fh.write(emit_candidates(stats, resolution))
        print(
            f"\nWrote candidate table ({len(stats)} rows, "
            f"{len(set(resolution.canonical.values()))} heuristic clusters) -> {out_path}\n"
            f"Fill in 'decision' (keep/drop) and 'person' (to force a merge or split), "
            f"then re-run with --from-reviewed.\n"
            f"Judge identities on name/email/volume/dates only — the table carries no "
            f"outcome column on purpose.",
            file=sys.stderr,
        )
        return

    lookup = {
        f"{name}\x1e{email}": canonical for (name, email), canonical in resolution.canonical.items()
    }
    with open(out_path, "w") as fh:
        json.dump({"lookup": lookup}, fh, indent=2)
    print(f"\nWrote alias map ({len(lookup)} entries) -> {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
