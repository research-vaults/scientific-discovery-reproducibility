# Release validation — 2 October 2026

## Completed checks

- Exported an explicit allowlist into a new Git repository; no commits from the mixed internal project archive were imported.
- Reproduced the fixed analyses and manuscript in the export, then repeated source retrieval and the full build in a separate clean Git checkout.
- Both `python3 scripts/fetch_sources.py` and `python3 scripts/build_paper.py` exited successfully in that clean checkout. Source SHA-256 checks passed, including chemistry and METR inputs that are not distributed in Git.
- The clean replay left every tracked reference result, table, figure and manuscript source byte-identical (`git diff` empty). Generated output: a 30-page PDF. The final TeX log has no undefined citations/references or overfull boxes; underfull-box notices remain.
- Build checks include fixed-table replay, independent arithmetic/score checks, bootstrap/refit analyses, horizon specification and main-evidence reconstruction. This is computational reproduction of the supplied analyses, not a new experiment or independent scientific replication.
- Installed Python package versions match every pin in `requirements.txt`. The run used the existing Python 3.11 runtime and local TeX installation, not a newly provisioned operating system or dependency environment.
- Reviewed the allowlist, source attribution, tracked PDF text/metadata and reachable Git history for personal identifiers, local paths and common credential signatures. No matching sensitive findings remained. Third-party attribution in bibliography/style files is intentionally retained.

## Boundaries

Automated signature scans cannot prove absence of every possible secret or identifying clue. This is a public reproducibility repository, not a claim of perfect anonymity. Full scientific validity, data licenses beyond established upstream terms, future source availability and prospective forecasting performance are not certified by these checks. No new page-by-page visual review was performed for this release. Raw chemistry data, METR YAML inputs, private reviews, planning records, model-chat traces are excluded from Git.

`SHA256SUMS` records all tracked release files except itself. Source acquisition is checksum-locked; upstream changes must not be silently accepted.
