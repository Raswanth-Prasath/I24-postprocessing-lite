# Logistic Regression Workspace Organization

This document summarizes how to work with the reorganized LR folder safely.

## Goals

- Keep production behavior stable.
- Reduce clutter from mixed scripts/artifacts/checkpoints.
- Provide a clear place for code, data, models, outputs, and reports.

## Recommended Workflow

1. Use scripts from `Logistic Regression/` (legacy entrypoints still valid).
2. Prefer canonical data paths under `Logistic Regression/data/` when available.
3. Keep deployable model references in `model_artifacts/` until full migration is complete.
4. Store generated figures and CSV outputs in `outputs/` and summaries in `reports/`.

## Legacy Compatibility

- Existing command invocations remain valid.
- Older relative paths are intentionally still supported by key scripts.
- See `MANIFEST.md` for path and migration details.

## Cleanup Policy

- Do not commit runtime caches (`__pycache__`, `.pytest_cache`, `.ipynb_checkpoints`).
- Archive legacy one-off artifacts under `archive/` rather than deleting immediately.
