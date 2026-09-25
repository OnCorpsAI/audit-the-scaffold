"""
discover_cohorts.py
-------------------
Derive the Part C repository cohorts from a directory of clones, so the
selection rule the paper states is reproducible rather than recorded.

Three things are computed per repository, all with the same regexes
``git_session_extractor`` uses, so the cohorts cannot drift from the extraction:

1. **AI-tier qualification** -- commits carrying a Claude co-author trailer
   (``CLAUDE_CO_AUTHOR_RE``). A repo qualifies at ``--min-ai-commits``.
2. **The adoption cutoff** -- the earliest commit anywhere in the scanned set
   carrying *any* known AI-coding-tool trailer (``AI_TRAILER_RE``). The control
   window ends this date minus ``--margin-years``.
3. **Control qualification** -- non-bot commits inside the control window. A
   repo qualifies at ``--min-control-commits`` *and* only if it is not an
   AI-tier repo (the cohorts are disjoint by construction).

Writes a JSON manifest of absolute repo paths plus the derived cutoff. The
manifest contains real repository paths and must live outside this repo; the
output path is checked for that.

Usage
-----
Both paths are environment-specific and deliberately not spelled here -- see
``.env.example`` for the variables this pipeline reads, and keep the manifest
outside the repository::

    python3 scripts/discover_cohorts.py \\
        --root "$COMPANY_REPO_ROOT" \\
        --out "$COMPANY_COHORTS_OUT"
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import git_session_extractor as gse  # noqa: E402

# git's --grep is applied to the whole message; the extractor's regexes are the
# authority on what counts, so git only narrows and Python decides.
_ANY_COAUTHOR = "Co-Authored-By:"


@dataclass
class RepoScan:
    path: str
    name: str
    total_commits: int = 0
    ai_commits: int = 0
    earliest_ai_trailer: str = ""
    error: str = ""


def _git(repo: str, *args: str, timeout: int = 120) -> str:
    result = subprocess.run(
        ["git", "-C", repo, *args],
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    return result.stdout if result.returncode == 0 else ""


def scan_repo(path: str) -> RepoScan:
    """Count Claude commits and find the earliest any-AI-tool trailer date."""
    scan = RepoScan(path=path, name=os.path.basename(path.rstrip("/")))
    try:
        head = _git(path, "rev-parse", "--verify", "HEAD", timeout=30).strip()
        if not head:
            scan.error = "no commits / unreadable"
            return scan
        scan.total_commits = len(
            _git(
                path,
                "log",
                # refs/notes/* commits are authored by note writers, not developers.
                "--exclude=refs/notes/*",
                "--all",
                "--format=%H",
                timeout=300,
            ).splitlines()
        )
        # One pass over every commit that has any co-author trailer at all, then
        # apply the extractor's own regexes to the body.
        raw = _git(
            path,
            "log",
            # refs/notes/* commits are authored by note writers, not developers.
            "--exclude=refs/notes/*",
            "--all",
            f"--grep={_ANY_COAUTHOR}",
            "-i",
            "--format=%ad\x1f%B\x1e",
            "--date=short",
            timeout=300,
        )
        for block in raw.split("\x1e"):
            block = block.strip("\n")
            if not block or "\x1f" not in block:
                continue
            when, body = block.split("\x1f", 1)
            if gse.CLAUDE_CO_AUTHOR_RE.search(body):
                scan.ai_commits += 1
            if gse.AI_TRAILER_RE.search(body) and (
                not scan.earliest_ai_trailer or when < scan.earliest_ai_trailer
            ):
                scan.earliest_ai_trailer = when
    except subprocess.TimeoutExpired:
        scan.error = "timeout"
    except Exception as exc:  # pragma: no cover -- operational robustness
        scan.error = f"{type(exc).__name__}: {exc}"
    return scan


def count_window_commits(repo: str, since: str, until: str) -> int:
    """Non-bot commits in the control window, matching the extractor's filter."""
    raw = _git(
        repo,
        "log",
        # refs/notes/* commits are authored by note writers, not developers.
        "--exclude=refs/notes/*",
        "--all",
        f"--since={since}",
        f"--until={until}",
        "--format=%an\x1f%ae\x1e",
        timeout=300,
    )
    kept = 0
    for block in raw.split("\x1e"):
        block = block.strip("\n")
        if not block or "\x1f" not in block:
            continue
        name, email = block.split("\x1f", 1)
        if gse.BOT_AUTHOR_RE.search(name) or gse.BOT_AUTHOR_RE.search(email):
            continue
        kept += 1
    return kept


def _count_summary(counts: list[int]) -> tuple[int, int, int]:
    """(min, median, max) over a commit-count list, or zeroes when it is empty.

    The empty guard lives here once. It used to be six inline ternaries spread across
    two f-strings, which is most of what made ``main`` rank E.

    "Median" is ``sorted[len // 2]`` -- the upper middle for even lengths, not a mean of
    the two central values. Preserved deliberately: the published cohort summary was
    produced with that definition.
    """
    if not counts:
        return 0, 0, 0
    ordered = sorted(counts)
    return ordered[0], ordered[len(ordered) // 2], ordered[-1]


def _derive_window(
    scans: list[RepoScan], margin_years: int, window_days: int
) -> tuple[str, date, date]:
    """Derive (earliest AI trailer, control cutoff, control window start).

    The cutoff is the earliest AI-tool trailer seen in *any* repo, minus a safety
    margin; the control window is the matched observation span ending at that cutoff.
    Exits non-zero when no trailer exists anywhere -- without one there is nothing to
    anchor the cutoff to, and falling through would emit a window derived from nothing.
    """
    trailer_dates = [s.earliest_ai_trailer for s in scans if s.earliest_ai_trailer]
    if not trailer_dates:
        print(
            "Error: no AI co-author trailer found anywhere — cannot derive a cutoff.",
            file=sys.stderr,
        )
        sys.exit(1)
    earliest = min(trailer_dates)
    cutoff = date.fromisoformat(earliest) - timedelta(days=365 * margin_years)
    return earliest, cutoff, cutoff - timedelta(days=window_days)


def _split_cohorts(
    scans: list[RepoScan], min_ai_commits: int
) -> tuple[list[RepoScan], list[RepoScan]]:
    """Split scans into (AI tier, control candidates).

    AI-tier membership is decided on the Claude-commit count alone. Control candidates
    are everything else that scanned cleanly -- an errored repo has no countable
    history, so it cannot be a control, but it can still clear the AI-tier floor on the
    commits that were counted before the error.
    """
    ai_tier = [s for s in scans if s.ai_commits >= min_ai_commits]
    ai_paths = {s.path for s in ai_tier}
    candidates = [s for s in scans if s.path not in ai_paths and not s.error]
    return ai_tier, candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__ or "")
    parser.add_argument("--root", required=True, help="Directory of repository clones")
    parser.add_argument("--out", required=True, help="Manifest path (must be outside this repo)")
    parser.add_argument("--min-ai-commits", type=int, default=15)
    parser.add_argument("--min-control-commits", type=int, default=15)
    parser.add_argument("--margin-years", type=int, default=3)
    parser.add_argument(
        "--window-days",
        type=int,
        default=315,
        help="Control window length, matched to the AI-tier observation span",
    )
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()

    out_path = Path(os.path.expanduser(args.out))
    gse.require_outside_repo(str(out_path), "the cohort manifest")

    repos = sorted(
        str(p.parent) for p in Path(args.root).glob("*/.git") if (p.parent / ".git").exists()
    )
    print(f"Scanning {len(repos)} repositories under {args.root} ...", file=sys.stderr)

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        scans = list(pool.map(scan_repo, repos))

    errors = [s for s in scans if s.error]
    for s in errors:
        print(f"  skipped {s.name}: {s.error}", file=sys.stderr)

    earliest, cutoff, since = _derive_window(scans, args.margin_years, args.window_days)
    ai_tier, candidates = _split_cohorts(scans, args.min_ai_commits)
    print(
        f"\nEarliest AI trailer anywhere: {earliest}\n"
        f"Cutoff (minus {args.margin_years}y): {cutoff}\n"
        f"Control window: {since} .. {cutoff} ({args.window_days} days)\n"
        f"AI-tier repos (>={args.min_ai_commits} Claude commits): {len(ai_tier)}\n"
        f"Counting in-window commits for {len(candidates)} candidate control repos ...",
        file=sys.stderr,
    )

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        window_counts = list(
            pool.map(
                lambda s: (s, count_window_commits(s.path, str(since), str(cutoff))),
                candidates,
            )
        )
    control = [(s, n) for s, n in window_counts if n >= args.min_control_commits]

    manifest = {
        "root": args.root,
        "earliest_ai_trailer": earliest,
        "margin_years": args.margin_years,
        "control_until": str(cutoff),
        "control_since": str(since),
        "window_days": args.window_days,
        "min_ai_commits": args.min_ai_commits,
        "min_control_commits": args.min_control_commits,
        "ai_tier_repos": [s.path for s in sorted(ai_tier, key=lambda s: s.name)],
        "control_repos": [s.path for s, _ in sorted(control, key=lambda t: t[0].name)],
        "ai_tier_detail": [asdict(s) for s in sorted(ai_tier, key=lambda s: s.name)],
        "control_detail": [
            {"name": s.name, "window_commits": n}
            for s, n in sorted(control, key=lambda t: t[0].name)
        ],
        "skipped": [{"name": s.name, "error": s.error} for s in errors],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2))

    ai_min, ai_med, ai_max = _count_summary([s.ai_commits for s in ai_tier])
    ctrl_min, ctrl_med, ctrl_max = _count_summary([n for _, n in control])
    print(
        f"\nAI-tier:  {len(ai_tier)} repos, Claude commits "
        f"min={ai_min} median={ai_med} max={ai_max}\n"
        f"Control:  {len(control)} repos, in-window commits "
        f"min={ctrl_min} median={ctrl_med} max={ctrl_max}\n"
        f"Wrote manifest -> {out_path}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
