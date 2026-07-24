"""
explain_model.py
SHAP explainability for the Beba Salama high-risk classifier.

Run after train_model.py — loads the same data/feature pipeline, retrains
in-memory (LightGBM training is fast enough here that re-running beats
juggling a separate serialized-model-loading path for SHAP), and produces:
  - models/shap_importance.csv   (global feature importance by mean |SHAP|)
  - A worked example explaining one individual high-risk prediction
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import shap

from data_prep import load_layer1
from features import engineer_features, TARGET
from train_model import SAFE_FEATURES, DEFAULT_DATA_PATH, DEFAULT_OUT_DIR


def explain(data_path: str = DEFAULT_DATA_PATH, out_dir: str = DEFAULT_OUT_DIR,
            sample_size: int = 500):
    df = load_layer1(data_path)
    df = engineer_features(df)

    X = df[SAFE_FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = lgb.LGBMClassifier(
        random_state=42, verbose=-1, scale_pos_weight=scale_pos_weight,
        n_estimators=300, learning_rate=0.05, max_depth=6,
        num_leaves=31, min_child_samples=20,
    )
    model.fit(X_train, y_train)

    explainer = shap.TreeExplainer(model)
    sample = X_test.sample(min(sample_size, len(X_test)), random_state=42)
    shap_values = explainer.shap_values(sample)
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values

    importance = pd.DataFrame({
        "feature": SAFE_FEATURES,
        "mean_abs_shap": np.abs(sv).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)

    os.makedirs(out_dir, exist_ok=True)
    importance.to_csv(f"{out_dir}/shap_importance.csv", index=False)
    print("SHAP mean |importance| (test sample, n={}):".format(len(sample)))
    print(importance.to_string(index=False))

    # Worked example: explain one specific high-risk prediction
    high_risk_mask = model.predict_proba(X_test)[:, 1] > 0.5
    if high_risk_mask.sum() > 0:
        example_idx = X_test[high_risk_mask].index[0]
        example = X_test.loc[[example_idx]]
        example_shap = explainer.shap_values(example)
        example_sv = example_shap[1] if isinstance(example_shap, list) else example_shap

        print("\nWorked example — one specific high-risk prediction:")
        print(f"  Predicted probability: {model.predict_proba(example)[0,1]:.3f}")
        for feat, val in zip(SAFE_FEATURES, example_sv[0]):
            print(f"    {feat:15s} {val:+.4f}")

    return importance


if __name__ == "__main__":
    explain()
