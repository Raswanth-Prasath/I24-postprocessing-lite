# Archive Notes

This directory holds legacy LR files that are no longer canonical but are retained for traceability.

Archive candidates include:
- old checkpoints and generated caches
- duplicated dataset copies
- superseded experiment artifacts

Rules:
- Prefer moving (not deleting) during cleanup.
- Record origin path and reason in commit messages or migration notes.

Current archived folders:
- `ipynb_checkpoints_root/`
- `pytest_cache_root/`
- `pycache_root/`
