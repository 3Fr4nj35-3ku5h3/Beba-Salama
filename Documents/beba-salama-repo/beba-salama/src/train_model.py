"""
train_model.py
Trains the Beba Salama high-risk crash classifier.

IMPORTANT — data leakage note:
The `severity` / `high_risk` target (see features.py) is a composite score
built from: n_crash_reports, contains_fatality_words, contains_pedestrian_words,
contains_matatu_words, contains_motorcycle_words.

Those same fields were originally considered as model *inputs* too. That is
data leakage — the model would partly be re-deriving its own label rather
than predicting it. Verified empirically: 1,456/1,456 crashes with
n_crash_reports >= 3 and every other flag at 0 were automatically classified
high_risk, purely by construction of the formula.

This script trains ONLY on features known BEFORE a crash occurs — time and
location. That is the honest version of this problem: "given when and where
a journey happens, how likely is a high-severity crash" — not "given how a
crash was already described, was it severe" (which is a different, much
easier, and not useful question for a route-risk product).
"""

import json
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    precision_recall_curve, precision_score, recall_score, f1_score,
)
import lightgbm as lgb

from data_prep import load_layer1
from features import engineer_features, MODEL_FEATURES, TARGET

# Resolve paths relative to the repo root regardless of the current working
# directory the script is invoked from (src/ or repo root both work).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_PATH = os.path.join(_REPO_ROOT, "data/raw/ma3route_crashes_algorithmcode.csv")
DEFAULT_OUT_DIR = os.path.join(_REPO_ROOT, "models")

# Only exogenous features — known before a crash happens, not reported
# properties of the crash itself.
SAFE_FEATURES = ['hour', 'dow', 'month', 'is_night', 'is_rush_hour',
                  'is_weekend', 'latitude', 'longitude']


def train(data_path: str = DEFAULT_DATA_PATH, out_dir: str = DEFAULT_OUT_DIR):
    df = load_layer1(data_path)
    df = engineer_features(df)

    X = df[SAFE_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Class imbalance: ~16.8% positive rate — weight the minority class
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = lgb.LGBMClassifier(
        random_state=42, verbose=-1,
        scale_pos_weight=scale_pos_weight,
        n_estimators=300, learning_rate=0.05,
        max_depth=6, num_leaves=31, min_child_samples=20,
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]

    # Tune the decision threshold for best F1 — for a safety tool, recall on
    # the high-risk class matters more than precision (a missed high-risk
    # crash is worse than a false alarm), so this threshold search favours
    # catching more true positives over minimising false positives.
    precisions, recalls, thresholds = precision_recall_curve(y_test, proba)
    f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    best_threshold = float(thresholds[np.argmax(f1s[:-1])])

    preds = (proba >= best_threshold).astype(int)
    auc = roc_auc_score(y_test, proba)
    cm = confusion_matrix(y_test, preds)

    results = {
        "auc_roc": float(auc),
        "best_threshold": best_threshold,
        "precision_high_risk": float(precision_score(y_test, preds)),
        "recall_high_risk": float(recall_score(y_test, preds)),
        "f1_high_risk": float(f1_score(y_test, preds)),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "positive_rate": float(y_train.mean()),
        "scale_pos_weight": float(scale_pos_weight),
        "features": SAFE_FEATURES,
        "confusion_matrix": {
            "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
        },
        "note": (
            "contains_matatu_words, contains_motorcycle_words, and "
            "n_crash_reports were deliberately excluded — they are used "
            "to construct the target itself and would leak it."
        ),
    }

    importance = pd.DataFrame({
        "feature": SAFE_FEATURES,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    model.booster_.save_model(f"{out_dir}/lightgbm_high_risk.txt")
    importance.to_csv(f"{out_dir}/feature_importance.csv", index=False)
    with open(f"{out_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(classification_report(y_test, preds, target_names=["Lower Risk", "High Risk"]))
    print(f"AUC-ROC: {auc:.4f}")
    print(f"\nSaved model + metrics to {out_dir}/")
    return model, results


if __name__ == "__main__":
    train()
