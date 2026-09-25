"""
anonymize_sessions.py
---------------------
Turns a raw (internal) company sessions JSON into the sanitized artifact that
is safe to publish as ``data/company_sessions.json``.

The script reads the raw file from ``RAW_SESSIONS_PATH`` and writes sanitized
JSON to stdout. It performs four independent sanitization passes:

1. **Repo-name anonymization.** Real repository names are replaced with opaque
   ``repo_A`` / ``repo_B`` / ... labels. The real->anonymous mapping is *not*
   hardcoded here — supply it out of band via ``COMPANY_REPO_MAP`` (a JSON
   object) or ``COMPANY_REPO_MAP_FILE`` (path to such a JSON file). With no
   mapping provided the pass is a no-op (e.g. when re-running on already
   anonymized data).
2. **Explicit session linkage.** Each commit is matched to its session using
   the same timestamp-window logic the simulation uses, then stamped with its
   session's ``session_id`` and a ``seq`` order index. This makes the
   session<->commit relationship survive step 4. Inputs carrying
   ``same_developer_only: true`` additionally require the commit's ``developer``
   to match the session's, since developer-split sessions produce overlapping
   timestamp windows that a developer-blind match resolves arbitrarily.
3. **Commit-hash synthesis.** Real commit SHAs are replaced with synthetic,
   order-stable identifiers (``c0000``, ``c0001``, ...). SHAs are never read by
   the simulation; only the linkage from step 2 is.
4. **Timestamp coarsening.** All timestamps (``timestamp``, ``start_ts``,
   ``end_ts``) are coarsened to date granularity (``YYYY-MM-DD``), dropping
   time-of-day and timezone offset.

Only structural metrics (diff sizes, commit counts, model tier, coarse dates)
survive. No code, file paths, author identities, or commit messages are emitted.

Usage
-----
    RAW_SESSIONS_PATH=/path/to/raw_sessions.json \
    COMPANY_REPO_MAP='{"real-repo-1":"repo_A","real-repo-2":"repo_B"}' \
        python3 scripts/anonymize_sessions.py > data/company_sessions.json
"""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime


def load_mapping() -> dict[str, str]:
    """Load the real->anonymous repo-name mapping from the environment.

    Precedence: ``COMPANY_REPO_MAP`` (inline JSON) then ``COMPANY_REPO_MAP_FILE``
    (path to a JSON file). Returns an empty mapping if neither is set.
    """
    inline = os.environ.get("COMPANY_REPO_MAP")
    if inline:
        return json.loads(inline)
    path = os.environ.get("COMPANY_REPO_MAP_FILE")
    if path:
        with open(path) as fh:
            return json.load(fh)
    return {}


def apply_name_mapping(data: dict, mapping: dict[str, str]) -> None:
    """Replace internal repo names in-place (top-level, sessions, commits)."""
    if not mapping:
        return
    data["repos"] = [mapping.get(r, r) for r in data.get("repos", [])]
    for session in data.get("sessions", []):
        if session.get("repo") in mapping:
            session["repo"] = mapping[session["repo"]]
    for commit in data.get("commits", []):
        if commit.get("repo") in mapping:
            commit["repo"] = mapping[commit["repo"]]


def _parse_ts(value: str) -> datetime:
    """Parse an offset-aware ISO-8601 timestamp, tolerating naive input.

    Naive timestamps (test fixtures, and any extraction that lost its offset) are
    treated as UTC so they remain comparable with offset-aware ones instead of
    raising on the mixed comparison.
    """
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def link_commits_to_sessions(data: dict) -> None:
    """Stamp each commit with its session's ``session_id`` and a ``seq`` index.

    Matching replicates the simulation's timestamp-window logic, using the *full*
    timestamps (before coarsening) so the ordering is faithful. Commits that fall
    outside every session window are left unlinked (``session_id: null``) and are
    unused downstream.

    When the input carries ``same_developer_only: true`` (control-mode
    extraction), a commit must additionally match its session's ``developer``.
    Those sessions were split on developer change, so the timestamp window alone
    is ambiguous: two developers active in the same repository inside the same
    window produce overlapping session windows, and a developer-blind match
    assigns commits to whichever session it tries last. On the pre-fix control
    dataset that mis-assigned 147 commits and left 124 unlinked, yielding 93
    sessions holding commits from more than one developer -- impossible under the
    grouping rule that built them.

    The condition is deliberately *not* applied unconditionally: AI-tier sessions
    are grouped with ``same_developer_only=False``, so one session may legitimately
    span several developers and its ``developer`` is the plurality. Requiring
    equality there would silently drop the minority-developer commits.
    """
    sessions = data.get("sessions", [])
    commits = data.get("commits", [])
    same_developer_only = bool(data.get("same_developer_only", False))
    for c in commits:
        c["session_id"] = None
        c["seq"] = None
    # Compare instants, not strings. Raw timestamps are offset-aware ISO-8601 and
    # this org's contributors span 11 distinct UTC offsets, so lexical comparison
    # mis-orders across them: "...T10:00:00+05:30" sorts after "...T09:00:00-04:00"
    # while occurring four hours earlier. Sorting and window tests both need real
    # instants or commits fall outside their own session's window.
    instant = {id(c): _parse_ts(c["timestamp"]) for c in commits}
    for s in sessions:
        start, end = _parse_ts(s["start_ts"]), _parse_ts(s["end_ts"])
        matched = [
            c
            for c in commits
            if c["repo"] == s["repo"]
            and start <= instant[id(c)] <= end
            and (not same_developer_only or c.get("developer") == s.get("developer"))
        ]
        matched.sort(key=lambda c: instant[id(c)])
        for seq, c in enumerate(matched):
            c["session_id"] = s["session_id"]
            c["seq"] = seq


def synthesize_shas(data: dict) -> None:
    """Replace real commit SHAs with order-stable synthetic identifiers."""
    for i, c in enumerate(data.get("commits", [])):
        c["sha"] = f"c{i:04d}"


def _to_date(value: str) -> str:
    """Coarsen an ISO datetime string to date granularity (first 10 chars)."""
    return value[:10]


def coarsen_timestamps(data: dict) -> None:
    """Coarsen all timestamps to ``YYYY-MM-DD`` (drops time-of-day + offset)."""
    for s in data.get("sessions", []):
        if "start_ts" in s:
            s["start_ts"] = _to_date(s["start_ts"])
        if "end_ts" in s:
            s["end_ts"] = _to_date(s["end_ts"])
    for c in data.get("commits", []):
        if "timestamp" in c:
            c["timestamp"] = _to_date(c["timestamp"])


def sanitize(data: dict, mapping: dict[str, str]) -> dict:
    """Run all sanitization passes in order and return the mutated dict."""
    apply_name_mapping(data, mapping)
    link_commits_to_sessions(data)  # before coarsening — needs full timestamps
    synthesize_shas(data)
    coarsen_timestamps(data)
    return data


def main() -> None:
    raw_path = os.environ.get("RAW_SESSIONS_PATH")
    if not raw_path:
        print(
            "Error: set RAW_SESSIONS_PATH to the path of the raw sessions JSON.",
            file=sys.stderr,
        )
        sys.exit(1)

    mapping = load_mapping()
    with open(raw_path) as fh:
        data = json.load(fh)

    sanitized = sanitize(data, mapping)

    # Sanity checks: no real repo names, and no un-synthesized SHAs survive.
    serialized = json.dumps(sanitized)
    for original in mapping:
        if original in serialized:
            print(
                f"Warning: '{original}' still present after anonymization.",
                file=sys.stderr,
            )
    for c in sanitized.get("commits", []):
        if not str(c.get("sha", "")).startswith("c"):
            print("Warning: a commit SHA was not synthesized.", file=sys.stderr)
            break

    json.dump(sanitized, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
