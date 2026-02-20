# Agent Instructions

This project uses **bd** (beads) for issue tracking. Run `bd onboard` to get started. 

## Codex Setup and Memory (Project Standard)

- Primary project instruction file: `AGENTS.md` (this file).
- Synced policy/rules file: `CLAUDE.md`.
- Skill-specific instructions (only when using a skill): `SKILL.md` inside the skill directory.
- Codex runtime state is **not** markdown and lives under `~/.codex/` (for example `config.toml`, `history.jsonl`, `sessions/`).
- To persist project memory for future sessions, update `AGENTS.md` and/or `CLAUDE.md` in the repo.

## Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --status in_progress  # Claim work
bd close <id>         # Complete work
bd sync               # Sync with git
```

## Project Directives (Synced from `CLAUDE.md`)

### Scope and Environment
- Goal: replace `stitch_cost()` in `utils/utils_stitcher_cost.py` with learned models, compare architectures, and integrate the best model into the pipeline.
- Environment: run in `i24` conda environment:
  ```bash
  source activate i24
  ```

### Mandatory Working Rules
1. Before writing code, describe your approach and wait for approval. Ask clarifying questions first if requirements are ambiguous.
2. If a task requires changes to more than 3 files, stop and break it into smaller tasks.
3. After writing code, list what could break and suggest tests to cover it.
4. For bugs, start by writing a test that reproduces the issue, then fix until the test passes.
5. Every time the user corrects the agent, add a new rule to `CLAUDE.md`.
6. When issues, bugs, changes, or updates are found in the I-24 project, add them to `AGENTS.md`.

### Model, Data, and Evaluation Invariants
1. Logistic Regression production model is `Logistic Regression/model_artifacts/consensus_top10_full47.pkl` (not `combined_optimal_10features.pkl`).
2. This model corresponds to the 2,100-pair dataset in `Logistic Regression/training_dataset_advanced.npz` (47 features).
3. RAW naming uses `RAW_*_Bhat.json` and reconciled outputs include both `REC_*.json` (Bhattacharyya) and `REC_*_LR.json` (Logistic Regression).
4. `mot_i24.py` must handle all supported file naming conventions above.
5. Sw/GT must be computed as frame-level ID switches divided by GT object count.
6. Evaluation scripts should support both Bhattacharyya (`REC_*.json`) and LR (`REC_*_LR.json`) variants.

### Quick Validation Flow
```bash
python pp_lite.py i --config parameters_LR.json --tag LR
python diagnose_json.py REC_i_LR.json --fix   # if needed
python mot_i24.py i
```

## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd sync
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds

## Resolved Review Findings (2026-02-12)

- [P1] `Logistic Regression/optuna_search_lr.py` (`--keep-best` path): legacy best trials now use defaulted `use_logit=False` and `logit_offset=5.0` when fields are absent.
- [P1] `Logistic Regression/optuna_search_lr.py` (CSV append path): schema guard now writes to a versioned CSV when headers differ, preventing silent column drift.
- [P2] `run_experiments.py` (`--all-configs` path): `--evaluate` now supports per-run coverage and calls `hota_trackeval.py --gt-file --tracker-file --name` for each generated tagged output.

## Active Follow-Ups (2026-02-12)

- Leakage risk remains high because LR metrics on advanced/v4 variants are near-perfect; keep source-holdout diagnostics in the validation path before promoting new artifacts.
- `Logistic Regression/generate_evaluation_results.py` now supports `--eval-protocol {random,source_holdout,both}` and performs split-before-scale evaluation to avoid optimistic leakage-prone reporting.
- Complementarity audit (`models/evaluate_transformer.py --audit-complementarity`) can yield zero eligible anchors on small sampled runs when GT-labeled anchors do not contain both positive and negative candidates; prefer full-pair runs (`--audit-max-pairs 0`) for go/no-go decisions and monitor `eligible_anchor_count`.
