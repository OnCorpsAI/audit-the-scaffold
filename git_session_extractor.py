"""
Extract IP-safe session metrics from repos with Claude Co-Authored-By commits.

Outputs only structural statistics — no code, no file paths, no commit messages.
A "session" = run of consecutive Claude commits where inter-commit gap < SESSION_GAP_H hours.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Final

SESSION_GAP_H = 4  # hours; gap that splits one session from the next


class _ExcludeIdentity:
    """Type of the :data:`EXCLUDE_IDENTITY` sentinel."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover -- debugging aid only
        return "EXCLUDE_IDENTITY"


EXCLUDE_IDENTITY: Final = _ExcludeIdentity()
"""Resolver verdict: this author is not on the curated roster -- drop the commit.

An identity resolver has three possible answers, and they are distinct:
``str`` (use this canonical identity), ``None`` (no opinion, fall back to the
default casefolded author name), and ``EXCLUDE_IDENTITY`` (this author is not a
member of the cohort at all). A sentinel object rather than a magic string, so
no real canonical label can ever collide with it.
"""

IdentityResolver = Callable[[str, str], "str | _ExcludeIdentity | None"]


def _get_repos() -> list[str]:
    """Return repo list from COMPANY_REPOS env var; raise clearly if unset."""
    env = os.environ.get("COMPANY_REPOS", "")
    if not env:
        raise OSError(
            "COMPANY_REPOS env var is required (comma-separated repo paths). See .env.example."
        )
    return [p.strip() for p in env.split(",") if p.strip()]


def _get_salt() -> str:
    """Return the HMAC salt from COMPANY_HASH_SALT env var; raise clearly if unset."""
    salt = os.environ.get("COMPANY_HASH_SALT", "")
    if not salt:
        raise OSError(
            "COMPANY_HASH_SALT env var is required (arbitrary secret string, never committed)."
        )
    return salt


def _hash_id(salt: str, value: str, *, prefix: str) -> str:
    digest = hmac.new(salt.encode(), value.encode(), hashlib.sha256).hexdigest()
    return f"{prefix}_{digest[:8]}"


def require_outside_repo(path: str, what: str) -> None:
    """Exit non-zero if ``path`` is inside this repository.

    Several artifacts in this pipeline carry real names, emails, or repository
    paths and must never land in the published tree: the raw sessions file, the
    identity audit table, the roster alias map, the cohort manifest. That rule
    was documented in docstrings but unenforced, and ``.gitignore``'s
    ``data/*.json`` pattern only covers one of the directories they could land
    in.
    """
    repo_root = Path(__file__).resolve().parent
    resolved = Path(path).expanduser().resolve()
    if resolved == repo_root or repo_root in resolved.parents:
        print(
            f"Error: refusing to write {what} to {resolved}.\n"
            f"It contains real identities or repository paths and must live outside "
            f"the repository ({repo_root}). Use a path under $TMPDIR or another "
            f"private location.",
            file=sys.stderr,
        )
        sys.exit(1)


def load_alias_resolver(path: str) -> IdentityResolver:
    """Load a ``(name, email) -> canonical identity`` map from a JSON file
    produced by ``scripts/resolve_author_aliases.py`` and return a resolver
    closure for ``extract_claude_commits(identity_resolver=...)``.

    Two modes, selected by the map's own ``strict`` key so the behaviour travels
    with the artifact rather than depending on a separate flag at the call site:

    * **non-strict** (default, and the behaviour of every map written before
      curation existed): a listed author gets their canonical identity, an
      unlisted one returns ``None`` and falls back to the default casefolded
      author name. The map only ever *merges*.
    * **strict**: the map is a curated roster, so it is the sole authority on
      cohort membership. An unlisted author returns :data:`EXCLUDE_IDENTITY` and
      their commits are dropped. The map both merges and filters.

    The file must never live inside the repo (it contains real names/emails,
    same rule as ``RAW_SESSIONS_PATH``) — this function only reads it.
    """
    raw = json.loads(Path(path).read_text())
    lookup: dict[str, str] = raw["lookup"]
    strict = bool(raw.get("strict", False))

    def resolve(name: str, email: str) -> str | _ExcludeIdentity | None:
        canonical = lookup.get(f"{name}\x1e{email}")
        if canonical is not None:
            return canonical
        return EXCLUDE_IDENTITY if strict else None

    return resolve


CLAUDE_CO_AUTHOR_RE = re.compile(
    r"Co-Authored-By:\s+(Claude[^<]+)<noreply@anthropic\.com>", re.IGNORECASE
)
# Broader net used only when require_ai_trailer=False (control-mode extraction):
# a commit is excluded if it carries a co-author trailer for *any* known AI
# coding tool, not just Claude. Defense in depth on top of the date cutoff.
AI_TRAILER_RE = re.compile(
    r"Co-Authored-By:\s*(Claude|GitHub Copilot|Copilot|Cursor|Codex|ChatGPT|Devin|OpenAI)"
    r"|Generated[- ]by[: ]",
    re.IGNORECASE,
)
COMMIT_TYPE_RE = re.compile(
    r"^(feat|fix|refactor|docs|test|chore|style|perf|build|ci|revert)[\(:]",
    re.IGNORECASE,
)
# CI/service-account authors are not developers — counting them inflates the
# "developer" count and, for control-mode extraction, would misclassify
# automated commits as part of the "fully human" baseline. Matches GitHub's
# own bot-account convention (trailing "[bot]") plus well-known automation
# account names; deliberately does NOT match generic "noreply@github.com"
# (used by both bots and real humans editing via the web UI, and by GitHub's
# own email-privacy feature for real users) to avoid excluding real people.
BOT_AUTHOR_RE = re.compile(
    r"\[bot\]|dependabot|renovate|github-actions|snyk-bot|greenkeeper|"
    r"semantic-release-bot|allcontributors|imgbot|circleci|jenkins-bot|"
    r"terraform-bot|automation-bot|ci-bot|codecov-commenter",
    re.IGNORECASE,
)
DIR_CATEGORY_RE = {
    "test": re.compile(r"^(tests?|spec|__tests__)(/|$)", re.IGNORECASE),
    "infra": re.compile(r"^(terraform|k8s|kube|helm|infra|ansible|deploy)(/|$)", re.IGNORECASE),
    "config": re.compile(r"^(config|\.github|ci)(/|$)", re.IGNORECASE),
    "docs": re.compile(r"^(docs?)(/|$)", re.IGNORECASE),
    "code": re.compile(r"^(src|lib|app|packages)(/|$)", re.IGNORECASE),
}


def _dir_category(top_dirs: list[str]) -> str:
    """Coarse task-type proxy: category of the dominant top-level touched dir.

    Never surfaces the raw directory name — only one of a fixed small set of
    coarse labels. ``other`` covers repo-root files and anything unmatched.
    """
    if not top_dirs:
        return "other"
    dominant = max(set(top_dirs), key=top_dirs.count)
    for category, pattern in DIR_CATEGORY_RE.items():
        if pattern.match(dominant):
            return category
    return "other"


@dataclass
class CommitMetrics:
    sha: str
    timestamp: datetime
    model: str | None  # e.g. "Claude Sonnet 4.6 (1M context)"; None for non-AI (control) commits
    commit_type: str  # feat/fix/refactor/docs/other — from conventional commit prefix
    files_changed: int
    insertions: int
    deletions: int
    churn: int  # insertions + deletions
    dir_category: str  # code/test/docs/infra/config/other — from dominant touched top-level dir
    developer: str  # salt-hashed author id ("dev_" + 8 hex); never the raw email


@dataclass
class SessionMetrics:
    repo: str
    session_id: int
    start_ts: str  # ISO
    end_ts: str  # ISO
    duration_minutes: float
    n_commits: int
    models: list[str | None]  # distinct models used; [None] for non-AI (control) sessions
    commit_types: dict  # {feat: N, fix: N, ...}
    total_files_changed: int
    total_insertions: int
    total_deletions: int
    total_churn: int
    mean_churn_per_commit: float
    survivor_ratio: float | None  # churn that "survived" = net lines / total churn; None if churn=0
    task_type: str  # plurality dir_category across the session's commits
    developer: str  # plurality developer id across the session's commits


def _git(repo: str, *args: str) -> str:
    result = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True, check=False)
    # Only strip the trailing newline subprocess output carries — a bare
    # .strip() also eats \x1c-\x1f (Python's str.isspace() treats the ASCII
    # separator control chars as whitespace), which corrupts the \x1e/\x1f
    # -delimited log format extract_claude_commits() parses whenever a
    # commit's body is empty and lands at the very end of the captured output.
    return result.stdout.strip("\n")


def _parse_shortstat(stat: str) -> tuple[int, int, int]:
    """Return (files_changed, insertions, deletions) from git diff --shortstat output."""
    files = int(m.group(1)) if (m := re.search(r"(\d+) file", stat)) else 0
    ins = int(m.group(1)) if (m := re.search(r"(\d+) insertion", stat)) else 0
    dels = int(m.group(1)) if (m := re.search(r"(\d+) deletion", stat)) else 0
    return files, ins, dels


def _remote_branches(repo: str) -> list[str]:
    """List locally-known remote branch names (requires the caller to have already
    fetched all branches — this script itself never runs ``git fetch``)."""
    raw = _git(repo, "branch", "-r")
    branches = []
    for line in raw.splitlines():
        name = line.strip().removeprefix("* ").strip()
        if not name or "->" in name:
            continue
        branches.append(name.removeprefix("origin/"))
    return branches


def _dominant_branch(repo: str) -> str:
    """Resolve the branch with the most Claude Co-Authored-By commits.

    Branch-naming conventions vary across repos (``main``, personal version
    branches like ``1.x``/``2.x``, ad-hoc feature branches), so ``origin/HEAD``
    is not a reliable signal for "where the Claude-assisted work happened."
    The commit count itself is naming-agnostic and is the same criterion the
    live repo survey uses to select repos in the first place.
    """
    branches = _remote_branches(repo)
    if not branches:
        return _git(repo, "symbolic-ref", "--short", "HEAD")

    best_branch, best_count = branches[0], -1
    for branch in branches:
        raw = _git(repo, "log", f"origin/{branch}", "--format=%b\x1e")
        count = sum(1 for block in raw.split("\x1e") if CLAUDE_CO_AUTHOR_RE.search(block))
        if count > best_count:
            best_branch, best_count = branch, count
    return best_branch


def _top_level_dir(path: str) -> str:
    return path.split("/", 1)[0] if "/" in path else ""


def extract_claude_commits(
    repo: str,
    salt: str,
    *,
    since: str | None = None,
    until: str | None = None,
    require_ai_trailer: bool = True,
    exclude_ai_trailer: bool = True,
    use_all_branches: bool = False,
    identity_resolver: IdentityResolver | None = None,
) -> list[CommitMetrics]:
    """Pull commits from the repo's dominant branch only (default), or from
    every branch (``use_all_branches=True``).

    ``identity_resolver``, if given, is called with ``(author_name,
    author_email)`` for every commit and may answer three ways: a ``str`` is
    hashed as the developer identity instead of ``casefold(author_name)``;
    ``None`` means no opinion and keeps the default; and
    :data:`EXCLUDE_IDENTITY` drops the commit entirely, before any ``dev_``
    label is minted. The first two merge aliases, the third also filters the
    cohort to a curated roster (see ``scripts/resolve_author_aliases.py``).
    Passing no resolver leaves behaviour exactly as it was.

    ``--all`` (the prior behaviour) double-counts commits reachable from
    squash-merged feature branches as well as the dominant branch; restricting
    to a single branch avoids that. ``_dominant_branch`` picks the branch with
    the most Claude-trailer commits, which is meaningless for a repo with none
    at all — exactly the case control-mode extraction (``require_ai_trailer=
    False``) hits on repos with zero AI-assisted history: the "dominant"
    branch there is an arbitrary tie-break, not where the real history lives.
    ``use_all_branches=True`` sidesteps that by scanning every branch instead
    of picking one; the resulting minor squash-merge double-counting risk is
    a much smaller problem than silently missing most of a repo's commits.

    By default (``require_ai_trailer=True``) this keeps only commits carrying
    a Claude Co-Authored-By trailer — the original behaviour, unchanged.
    Passing ``require_ai_trailer=False`` switches to control-mode extraction:
    all commits are kept *except* those matching ``AI_TRAILER_RE`` (any known
    AI coding tool, not just Claude) when ``exclude_ai_trailer=True`` — a
    defense-in-depth filter on top of whatever ``until`` date cutoff the
    caller supplies. ``since``/``until`` bound the ``git log`` window
    (``YYYY-MM-DD`` or anything ``git log --since/--until`` accepts).
    """
    if use_all_branches:
        # `--all` walks every ref, refs/notes/* included. A note commit is authored by
        # whatever wrote the note -- a review bot, a CI annotator, or local git tooling
        # -- so without this exclusion those commits are ingested as developer commits.
        # It bites hardest in the modes that reach this branch: control_mode and the
        # non-AI within-era side pass require_ai_trailer=False, so the trailer filter
        # that protects the AI tier does not drop them, and BOT_AUTHOR_RE does not match
        # typical note-writer identities. `--exclude` must precede `--all` to apply.
        ref_args = ["--exclude=refs/notes/*", "--all"]
    else:
        ref_args = [f"origin/{_dominant_branch(repo)}"]
    log_args = ["log", *ref_args]
    if since is not None:
        log_args.append(f"--since={since}")
    if until is not None:
        log_args.append(f"--until={until}")
    # format: sha | iso-timestamp | author-name | author-email | subject | body (NUL-delimited)
    log_args.append("--format=%H\x1f%aI\x1f%an\x1f%ae\x1f%s\x1f%b\x1e")
    raw = _git(repo, *log_args)
    commits: list[CommitMetrics] = []

    for block in raw.split("\x1e"):
        # NOTE: bare .strip() would also eat a trailing/leading "\x1f" field
        # separator (Python's str.isspace() treats the ASCII separator
        # control chars 0x1c-0x1f as whitespace) — fatal when a commit's body
        # is empty, since that \x1f is the only thing marking the empty body
        # field. Only strip the newline git inserts after each record.
        block = block.strip("\n")
        if not block:
            continue
        parts = block.split("\x1f", 5)
        if len(parts) < 6:
            continue
        sha, iso, author_name, author_email, subject, body = parts

        if BOT_AUTHOR_RE.search(author_email) or BOT_AUTHOR_RE.search(author_name):
            continue

        match = CLAUDE_CO_AUTHOR_RE.search(body)
        model: str | None
        if require_ai_trailer:
            if not match:
                continue
            model = match.group(1).strip()
        else:
            if exclude_ai_trailer and (match or AI_TRAILER_RE.search(body)):
                continue
            model = None

        ts = datetime.fromisoformat(iso)
        # Identity key is the author's display name, not email, by default:
        # the same person's email varies across machines/accounts (personal
        # vs corporate, GitHub-noreply vs direct) far more than their git
        # user.name does — hashing on email measurably overcounts distinct
        # developers (confirmed against this org's own history).
        # `identity_resolver` can override this per-commit for cases the
        # display name itself can't disambiguate (same person, different
        # name *and* email — e.g. a GitHub handle vs. a real name).
        resolved = identity_resolver(author_name, author_email) if identity_resolver else None
        if resolved is EXCLUDE_IDENTITY:
            # Not on the curated roster: drop before hashing, so no dev_ label is
            # ever minted for an identity that is not part of the cohort.
            del author_email, author_name
            continue
        identity = resolved if isinstance(resolved, str) else author_name.strip().casefold()
        developer = _hash_id(salt, identity, prefix="dev")
        del author_email, author_name  # never retained past hashing

        m = COMMIT_TYPE_RE.match(subject)
        commit_type = m.group(1).lower() if m else "other"

        stat = _git(repo, "diff", "--shortstat", f"{sha}^..{sha}")
        files, ins, dels = _parse_shortstat(stat)

        names = _git(repo, "diff", "--name-only", f"{sha}^..{sha}")
        top_dirs = [_top_level_dir(p) for p in names.splitlines() if p.strip()]
        dir_category = _dir_category(top_dirs)
        del names, top_dirs  # raw paths never retained past categorization

        commits.append(
            CommitMetrics(
                sha=sha,
                timestamp=ts,
                model=model,
                commit_type=commit_type,
                files_changed=files,
                insertions=ins,
                deletions=dels,
                churn=ins + dels,
                dir_category=dir_category,
                developer=developer,
            )
        )

    commits.sort(key=lambda c: c.timestamp)
    return commits


def group_into_sessions(
    repo: str, commits: list[CommitMetrics], *, same_developer_only: bool = False
) -> list[SessionMetrics]:
    """Group consecutive commits into sessions by inter-commit gap.

    ``same_developer_only=True`` additionally splits a session whenever the
    developer changes, even within the gap window. AI sessions are always
    effectively single-developer (one person's assistant), so this defaults
    to False and is unused there; a long-lived branch with many independent
    human committers needs it to avoid fusing unrelated people's commits into
    one fabricated "session" purely because they landed within 4h of each
    other.
    """
    if not commits:
        return []

    sessions: list[SessionMetrics] = []
    bucket: list[CommitMetrics] = [commits[0]]

    for c in commits[1:]:
        gap_h = (c.timestamp - bucket[-1].timestamp).total_seconds() / 3600
        same_dev = not same_developer_only or c.developer == bucket[-1].developer
        if gap_h > SESSION_GAP_H or not same_dev:
            sessions.append(_summarise(repo, len(sessions), bucket))
            bucket = [c]
        else:
            bucket.append(c)

    sessions.append(_summarise(repo, len(sessions), bucket))
    return sessions


def _plurality(values: list[str]) -> str:
    """Most frequent value; ties break by the earliest-occurring value (deterministic)."""
    return max(values, key=values.count)


def _summarise(repo: str, idx: int, bucket: list[CommitMetrics]) -> SessionMetrics:
    duration = (bucket[-1].timestamp - bucket[0].timestamp).total_seconds() / 60
    models = sorted(set(c.model for c in bucket), key=lambda m: (m is None, m))
    type_counts: dict[str, int] = {}
    for c in bucket:
        type_counts[c.commit_type] = type_counts.get(c.commit_type, 0) + 1

    total_ins = sum(c.insertions for c in bucket)
    total_dels = sum(c.deletions for c in bucket)
    total_churn = sum(c.churn for c in bucket)
    net_lines = total_ins - total_dels
    survivor = abs(net_lines) / total_churn if total_churn > 0 else None

    return SessionMetrics(
        repo=repo,
        session_id=idx,
        start_ts=bucket[0].timestamp.isoformat(),
        end_ts=bucket[-1].timestamp.isoformat(),
        duration_minutes=round(duration, 1),
        n_commits=len(bucket),
        models=models,
        commit_types=type_counts,
        total_files_changed=sum(c.files_changed for c in bucket),
        total_insertions=total_ins,
        total_deletions=total_dels,
        total_churn=total_churn,
        mean_churn_per_commit=round(total_churn / len(bucket), 1),
        survivor_ratio=round(survivor, 3) if survivor is not None else None,
        task_type=_plurality([c.dir_category for c in bucket]),
        developer=_plurality([c.developer for c in bucket]),
    )


def _repo_label_map(repos: list[str], salt: str) -> dict[str, str]:
    """Map real repo paths to opaque ``repo_A``/``repo_B``/... labels.

    Ordered by HMAC(salt, canonical remote URL) digest, so label order leaks
    neither repo size nor identity (a plain size- or alphabetical-sort would).
    """
    digests = {}
    for repo in repos:
        remote_url = _git(repo, "config", "--get", "remote.origin.url")
        digests[repo] = hmac.new(salt.encode(), remote_url.encode(), hashlib.sha256).hexdigest()
    ordered = sorted(repos, key=lambda r: digests[r])
    labels = {}
    for i, repo in enumerate(ordered):
        letter = ""
        n = i
        while True:
            n, rem = divmod(n, 26)
            letter = chr(ord("A") + rem) + letter
            if n == 0:
                break
            n -= 1
        labels[repo] = f"repo_{letter}"
    return labels


def _within_era_config() -> tuple[str, str | None, str | None]:
    """Read and validate the within-era env vars: (side, since, until).

    Within-era mode puts both sides of an AI-versus-non-AI contrast inside the *same*
    window, for the same people. The pre-AI control removes neither the person nor the
    era, and its within-person pairing compares a person's entire pre-AI output against
    only their AI-assisted later output. Holding the window fixed and splitting on
    whether a commit carries a Claude trailer removes the person confound (same
    developer) and the era confound (same window, codebase, tooling, seniority),
    leaving task selection -- which work they chose to delegate.

    A single purpose-built mode rather than orthogonal filter/window/grouping knobs:
    the two sides must differ in exactly one respect, and separate knobs could be
    combined into a comparison that silently differs in three. That is why the window
    is validated as *required* here rather than defaulted.

    Returns ``("", ...)`` for the side when the mode is off.
    """
    side = os.environ.get("COMPANY_WITHIN_ERA_MODE", "")
    since = os.environ.get("COMPANY_WITHIN_ERA_SINCE") or None
    until = os.environ.get("COMPANY_WITHIN_ERA_UNTIL") or None
    if not side:
        return side, since, until
    if side not in ("ai", "nonai"):
        raise OSError(f"COMPANY_WITHIN_ERA_MODE={side!r} invalid — expected 'ai' or 'nonai'.")
    if not (since and until):
        raise OSError(
            "COMPANY_WITHIN_ERA_MODE requires COMPANY_WITHIN_ERA_SINCE and "
            "COMPANY_WITHIN_ERA_UNTIL (the AI-era window, identical for both sides)."
        )
    return side, since, until


def _control_config() -> tuple[bool, str | None, str | None]:
    """Read and validate the control-mode env vars: (enabled, since, until).

    Control mode is pre-AI-adoption human-only extraction for the Part C robustness
    check, instead of the default Claude-co-authored extraction. Same repos, salt and
    anonymization; different filter and session grouping.
    """
    enabled = os.environ.get("COMPANY_CONTROL_MODE", "") == "1"
    until: str | None = None
    if enabled:
        until = os.environ.get("COMPANY_CONTROL_UNTIL", "")
        if not until:
            raise OSError(
                "COMPANY_CONTROL_MODE=1 requires COMPANY_CONTROL_UNTIL "
                "(e.g. 2022-09-19 — see tasks/... for how this was derived)."
            )
    return enabled, os.environ.get("COMPANY_CONTROL_SINCE") or None, until


def _output_path(within_era: str, control_mode: bool) -> Path:
    """Destination for the extracted artifact; each mode has its own env override."""
    if within_era:
        default_out_path = f"data/company_within_era_{within_era}_sessions.json"
        out_env_var = "COMPANY_WITHIN_ERA_RAW_OUTPUT"
    elif control_mode:
        default_out_path = "data/company_control_sessions.json"
        out_env_var = "COMPANY_CONTROL_RAW_OUTPUT"
    else:
        default_out_path = "data/company_sessions.json"
        out_env_var = "COMPANY_RAW_OUTPUT"
    return Path(os.environ.get(out_env_var, default_out_path))


def _summary_json(all_sessions: list[dict], all_commits: list[dict], labels: dict[str, str]) -> str:
    """Anonymized per-repo counts for stdout — labels only, never repo paths."""
    return json.dumps(
        {
            "total_sessions": len(all_sessions),
            "total_claude_commits": len(all_commits),
            "by_repo": {
                label: {
                    "sessions": sum(1 for s in all_sessions if s["repo"] == label),
                    "commits": sum(1 for c in all_commits if c["repo"] == label),
                }
                for label in sorted(labels.values())
            },
        },
        indent=2,
    )


def main() -> None:
    within_era, within_since, within_until = _within_era_config()
    control_mode, control_since, control_until = _control_config()

    # Each cohort gets its own roster, because each is built over its own time
    # window: COMPANY_CONTROL_ALIAS_MAP for the pre-AI control window,
    # COMPANY_ALIAS_MAP for the AI-tier window. Applying one cohort's roster to
    # the other would drop every author who only appears in the other era.
    # Within-era mode uses the AI-era roster for both sides, since both draw from
    # the same window and the same people -- that shared roster is what makes the
    # two sides poolable by developer at all.
    identity_resolver: IdentityResolver | None = None
    alias_map_env = "COMPANY_CONTROL_ALIAS_MAP" if control_mode else "COMPANY_ALIAS_MAP"
    alias_map_path = os.environ.get(alias_map_env)
    if alias_map_path:
        identity_resolver = load_alias_resolver(alias_map_path)

    repos = _get_repos()
    salt = _get_salt()
    labels = _repo_label_map(repos, salt)
    all_sessions: list[dict] = []
    all_commits: list[dict] = []

    for repo in repos:
        label = labels[repo]
        print(f"Extracting {label} ...", file=sys.stderr)
        if within_era:
            # The two sides differ in exactly one respect: whether a Claude trailer
            # is required or excluded. Window, branch policy, and session grouping
            # are identical, so a per-developer contrast is like-for-like.
            commits = extract_claude_commits(
                repo,
                salt,
                since=within_since,
                until=within_until,
                require_ai_trailer=within_era == "ai",
                exclude_ai_trailer=within_era == "nonai",
                use_all_branches=True,
                identity_resolver=identity_resolver,
            )
            kind = "Claude-assisted" if within_era == "ai" else "non-AI"
            print(f"  {len(commits)} in-window {kind} commits", file=sys.stderr)
            sessions = group_into_sessions(label, commits, same_developer_only=True)
        elif control_mode:
            commits = extract_claude_commits(
                repo,
                salt,
                since=control_since,
                until=control_until,
                require_ai_trailer=False,
                use_all_branches=True,
                identity_resolver=identity_resolver,
            )
            print(f"  {len(commits)} pre-cutoff human commits", file=sys.stderr)
            sessions = group_into_sessions(label, commits, same_developer_only=True)
        else:
            commits = extract_claude_commits(repo, salt, identity_resolver=identity_resolver)
            print(f"  {len(commits)} Claude commits", file=sys.stderr)
            sessions = group_into_sessions(label, commits)
        print(f"  {len(sessions)} sessions (gap={SESSION_GAP_H}h)", file=sys.stderr)

        for c in commits:
            d = asdict(c)
            d["repo"] = label
            d["sha"] = d["sha"][:12]  # truncate SHA — no need for full hash
            d["timestamp"] = c.timestamp.isoformat()
            all_commits.append(d)

        all_sessions.extend(asdict(s) for s in sessions)

    out = {
        "session_gap_hours": SESSION_GAP_H,
        # Records which grouping rule built these sessions. Control-mode sessions
        # are split on developer change, so a commit may only belong to a session
        # carrying the same developer; AI-tier sessions are not, so one session
        # legitimately spans several developers and its `developer` field is the
        # plurality. Downstream commit<->session linkage needs to know which,
        # hence the flag travels with the data (see anonymize_sessions.py).
        "same_developer_only": bool(control_mode or within_era),
        "repos": sorted(labels.values()),
        "sessions": all_sessions,
        "commits": all_commits,
    }
    if control_mode:
        # Both bounds travel with the data: the window is a derived quantity (cutoff
        # minus a matched observation span), so a consumer that only knows the end
        # date cannot reconstruct it, and the paper reports both.
        out["control_until"] = control_until
        out["control_since"] = control_since
    if within_era:
        # The side label and window must travel with the data: the two artifacts are
        # only comparable if they share a window, and a consumer cannot tell which
        # side a file is from by inspecting it (both carry a mix of `model` values
        # only on the ai side, and none at all on the nonai side).
        out["within_era_side"] = within_era
        out["within_era_since"] = within_since
        out["within_era_until"] = within_until

    out_path = _output_path(within_era, control_mode)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(
        f"\nWrote {len(all_sessions)} sessions, {len(all_commits)} commits → {out_path}",
        file=sys.stderr,
    )

    print(_summary_json(all_sessions, all_commits, labels))


if __name__ == "__main__":
    main()
