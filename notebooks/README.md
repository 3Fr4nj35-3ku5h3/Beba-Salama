# Notebooks

EDA and model-development notebooks go here.

Suggested sequence as the project progresses:

1. `01_eda.ipynb` — exploratory analysis of Layer 1 + Layer 2 (hourly/daily/geographic patterns)
2. `02_feature_engineering.ipynb` — validating and iterating on `src/features.py`
3. `03_model_training.ipynb` — LightGBM training, cross-validation, evaluation
4. `04_shap_explainability.ipynb` — SHAP value analysis for the explainability layer

Nothing has been committed here yet — the equivalent exploratory work so far
lives in the interactive dashboard (`docs/index.html`). Move it into notebooks
once the team starts formal model training so results are reproducible and
reviewable commit-by-commit.
