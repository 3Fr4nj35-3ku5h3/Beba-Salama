# Contributing

This is a team capstone project — a few ground rules so we don't step on each other.

## Branches
- `main` — always in a working, presentable state. This is what the mentor and judges see.
- `feature/<short-description>` — all new work happens on a branch, e.g. `feature/lightgbm-model`, `feature/shap-layer`.
- Never commit directly to `main`. Open a pull request, even if you're the only reviewer available at the time.

## Commits
- Write commit messages that describe *what changed and why*, not just "update file".
  Good: `Add road name standardisation for Layer 2 dataset`
  Avoid: `fix stuff`

## Pull Requests
- Keep PRs scoped to one thing — one model change, one dashboard fix, one doc update.
- Tag a teammate for review before merging where possible.
- Update the Roadmap table in `README.md` when a phase moves from 🔄 to ✅.

## Data
- Raw data goes in `data/raw/` and should not be modified in place.
- Any cleaning or feature engineering goes in `src/`, not ad hoc in a notebook that never gets committed.
- If you add a new data source, document it in `data/README.md` with its citation/license terms.

## Before your mentor session
- Make sure `main` reflects your actual current status — don't let uncommitted work sit on someone's laptop before a check-in.
