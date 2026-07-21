"""
features.py
Feature engineering for the Beba Salama risk model — ported from the
project's exploratory analysis. These are the functions that produced
data/processed/layer1_engineered.csv.
"""

import pandas as pd


def add_time_features(df: pd.DataFrame, datetime_col: str = "crash_datetime") -> pd.DataFrame:
    """
    Derive hour, year, month, day-of-week, and coarse time-window flags
    from a crash datetime column.
    """
    df = df.copy()
    dt = df[datetime_col]

    df["hour"] = dt.dt.hour
    df["year"] = dt.dt.year
    df["month"] = dt.dt.month
    df["dow"] = dt.dt.dayofweek          # 0 = Monday
    df["dow_name"] = dt.dt.day_name()

    df["is_night"] = df["hour"].apply(lambda h: 1 if (h >= 21 or h <= 4) else 0)
    df["is_rush_hour"] = df["hour"].apply(
        lambda h: 1 if (6 <= h <= 8 or 17 <= h <= 19) else 0
    )
    df["is_weekend"] = df["dow"].apply(lambda d: 1 if d >= 5 else 0)

    return df


def compute_severity(
    df: pd.DataFrame,
    high_risk_threshold: float = 5,
) -> pd.DataFrame:
    """
    Composite severity score and binary high-risk target.

    Weighting rationale (from project EDA):
      - n_crash_reports x2   : more independent reports ~ bigger/more visible incident
      - fatality flag  x5    : dominant weight — a death changes the category of event
      - pedestrian flag x3   : pedestrians are the most vulnerable road user group
      - matatu flag    x2    : multi-passenger vehicle, higher potential victim count
      - motorcycle flag x2   : boda riders carry high individual injury risk

    These weights are a starting hypothesis, not a validated ground truth —
    revisit once the classification model is trained and evaluated.
    """
    df = df.copy()
    df["severity"] = (
        df["n_crash_reports"] * 2
        + df["contains_fatality_words"] * 5
        + df["contains_pedestrian_words"] * 3
        + df["contains_matatu_words"] * 2
        + df["contains_motorcycle_words"] * 2
    )
    df["high_risk"] = (df["severity"] >= high_risk_threshold).astype(int)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience wrapper: run the full feature pipeline in one call."""
    df = add_time_features(df)
    df = compute_severity(df)
    return df


MODEL_FEATURES = [
    "hour",
    "dow",
    "month",
    "is_night",
    "is_rush_hour",
    "is_weekend",
    "latitude",
    "longitude",
    "n_crash_reports",
    "contains_matatu_words",
    "contains_motorcycle_words",
]

TARGET = "high_risk"
