# Data

Committed inputs for the paper's figures, tables, and statistics. Everything here is
either public benchmark output (Part A/B) or sanitized structural metadata (Part C) —
no code content, file paths, commit messages, or author identities.

## Part C — production session datasets

Four datasets, all written by the same sanitization pipeline
(`scripts/anonymize_sessions.py`):

| File | Cohort |
| --- | --- |
| `company_sessions.json` | AI-tier sessions (Findings 1–2) |
| `company_control_sessions.json` | pre-AI human baseline (Finding 3) |
| `company_within_era_ai_sessions.json` | within-era contrast, AI-assisted side |
| `company_within_era_nonai_sessions.json` | within-era contrast, non-AI side |

Each holds `sessions` and `commits` tables plus cohort metadata (`repos`,
`session_gap_hours`, `same_developer_only`, and window bounds on the cohorts that have
them — `control_since`/`control_until`, `within_era_since`/`within_era_until`). Session
fields are structural only: churn totals, insertions/deletions, files changed, commit
count, duration, survivor ratio, coarse `task_type`, the `models` seen in the session,
`repo`, `developer`. Commits carry the same shape one level down, with a single `model`
trailer plus `session_id`/`seq` linking them to their session.

Derived statistics live in `part_c_control_stats.json`, recomputed from these four files
by `scripts/part_c_control_stats.py`. That file is the sole source for every `\statPC*`
macro the paper prints; `scripts/generate_stats_macros.py --check` fails the build if a
quoted value drifts from the recomputed one.

### Sanitization

- Real repository names → synthetic labels (`repo_A`, `repo_B`, …).
- Commit SHAs → order-stable counters (`c0000`, `c0001`, …).
- Author identities → salted HMAC hashes (`dev_` prefix). The roster mapping real names
  to person labels is **not** released, so this cohort cannot be rebuilt from what is
  published here.
- Timestamps truncated to day granularity (`YYYY-MM-DD`), dropping time-of-day and zone.
- Directory paths → six coarse categories (`code`, `test`, `docs`, `config`, `infra`,
  `other`).

This makes the data IP-safe, not anonymous in the strong sense: day-granularity commit
timing for a named organization's contributors is in principle linkable by someone
holding the original repositories. The paper says so in its Responsible Use Statement.

### Gotcha: `repo` labels are positional, `developer` labels are not

The two label kinds look alike and behave differently. **Read this before joining any two
of these files.**

- `developer` labels are `HMAC(salt, identity)`. Under one salt and one roster the same
  person carries the same `dev_` label in every file, so pairing on `developer` across
  cohorts is meaningful — it is what the within-developer analyses rely on.
- `repo` labels are assigned **per extraction run**, by HMAC rank over that run's
  repository list (`git_session_extractor._repo_label_map`). A run over 23 repositories
  letters them `repo_A`..`repo_W`; a run over 122 letters them `repo_A`..`repo_DR`. So
  **`repo_A` denotes a different repository in each file**, and the label sets overlap
  without the repositories overlapping.

Concretely: all 23 AI-tier labels also appear in `company_control_sessions.json`, where
they carry 1,648 sessions — different repositories, colliding names. Joining or grouping
on raw `repo` across files fuses unrelated repositories into one random-effect level.

`company_simulations.namespace_repos(data, prefix)` is the supported fix and what
`part_c_control_stats.py` uses before fitting any pooled model. It prefixes `repo` and
deliberately leaves `developer` alone.

## Part B — SWE-bench Lite agent runs

- `llm_results_{tier}_{mode}.json` / `llm_metrics_{tier}_{mode}.json` — per-trajectory
  results and derived metrics for the 2×3 model-tier × feedback-mode ablation
  (`tier` ∈ {haiku, sonnet}, `mode` ∈ {independent, blind, diagnostic}).
- `*_t4_pinned.json` — the digest-pinned T=4 runs.
- `part_b_k55_anova_power.json`, `part_b_k55_paper_numbers.json`,
  `part_b_edge_overlap.json` — analysis outputs feeding the `\statPB*` macros.
- `swebench_image_digests.json` — pinned Docker image digests for the evaluator.
- `llm_cache/` — raw LLM response cache, keyed by request hash. Public SWE-bench task
  content and model-proposed diffs only.

## Part A — stylized simulations

`swe_exp{1,2,3}_results.json` — decision-stump results for the transferred-mechanics
checks. Regenerate with `python swe_bench_simulations.py`.

## Regenerating

```
uv run python company_simulations.py                  # -> company_params.json + figures
uv run python scripts/part_c_control_stats.py         # -> part_c_control_stats.json
uv run python scripts/generate_stats_macros.py        # -> stats_macros.tex
```

The Part C extraction itself needs the private config described in `.env.example`
(repository paths, salt, curated rosters) and cannot be re-run from this repository alone.
