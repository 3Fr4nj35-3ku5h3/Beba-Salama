"""
data_prep.py
Loading and cleaning functions for both Beba Salama data sources.

Layer 1: World Bank / Ma3Route geolocated crashes (2012-2023)
Layer 2: Kenya Police / HDX road accident records (2016-2017)
"""

import pandas as pd


def load_layer1(path: str) -> pd.DataFrame:
    """
    Load the World Bank / Ma3Route crash dataset.

    Parameters
    ----------
    path : str
        Path to ma3route_crashes_algorithmcode.csv (or the manualcode variant).

    Returns
    -------
    pd.DataFrame with crash_datetime parsed to datetime.
    """
    df = pd.read_csv(path)
    df["crash_datetime"] = pd.to_datetime(df["crash_datetime"])
    df["crash_date"] = pd.to_datetime(df["crash_date"])
    return df


def load_layer2(path: str) -> pd.DataFrame:
    """
    Load and combine both sheets of the Kenya Police / HDX accident workbook.

    Parameters
    ----------
    path : str
        Path to kenya-accidents-database.xlsx

    Returns
    -------
    pd.DataFrame combining the 2016 and 2017 sheets, with a `year` column added.
    """
    sheet_2016 = pd.read_excel(path, sheet_name="2016")
    sheet_2016["year"] = 2016
    sheet_2017 = pd.read_excel(path, sheet_name="2017")
    sheet_2017["year"] = 2017

    combined = pd.concat([sheet_2016, sheet_2017], ignore_index=True)
    return combined


# Canonical road name mapping — extend this as you find more variants in the data.
# Keys should be UPPERCASE to match after .str.upper().str.strip() normalisation.
ROAD_NAME_MAP = {
    "NAIROBI MOMBASA": "MOMBASA ROAD",
    "MOMBASA NAIROBI": "MOMBASA ROAD",
    "NAIROBI-MOMBASA": "MOMBASA ROAD",
    "MOMBASA-NAIROBI": "MOMBASA ROAD",
    "MOMBASA RD": "MOMBASA ROAD",
    "THIKA SUPERHIGHWAY": "THIKA ROAD",
    "THIKA SUPER HIGHWAY": "THIKA ROAD",
    "THIKA RD": "THIKA ROAD",
    "WAIYAKI WAY": "WAIYAKI WAY",
    "EASTERN BY-PASS": "EASTERN BYPASS",
    "EASTERN BY PASS": "EASTERN BYPASS",
    "SOUTHERN BY-PASS": "SOUTHERN BYPASS",
    "SOUTHERN BY PASS": "SOUTHERN BYPASS",
}


def standardize_road_names(df: pd.DataFrame, road_col: str = "ROAD") -> pd.DataFrame:
    """
    Normalise road name variants to a single canonical form.

    This is a starting map based on variants observed in the 2016/2017 sheets —
    extend ROAD_NAME_MAP above as you discover more during EDA.

    Parameters
    ----------
    df : pd.DataFrame
    road_col : str
        Name of the column containing raw road names.

    Returns
    -------
    pd.DataFrame with a new `ROAD_CLEAN` column.
    """
    df = df.copy()
    normalized = df[road_col].astype(str).str.upper().str.strip()
    df["ROAD_CLEAN"] = normalized.map(ROAD_NAME_MAP).fillna(normalized)
    return df
