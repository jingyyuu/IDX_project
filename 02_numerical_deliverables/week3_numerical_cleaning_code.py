
"""
Week 3 Numerical Data Preprocessing — Rebecca
This file can be copied into 02_preprocessing.ipynb.
It assumes df already contains 2025.01-2026.05 data filtered to:
PropertyType == "Residential" and PropertySubType == "SingleFamilyResidence".
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

KEY_NUMERIC_COLS = [
    "ClosePrice", "LivingArea", "BedroomsTotal", "BathroomsTotalInteger",
    "LotSizeSquareFeet", "YearBuilt", "DaysOnMarket", "GarageSpaces",
    "ParkingTotal", "Latitude", "Longitude"
]

EXCLUDE_NUMERIC_FEATURES = ["ListingKey", "ListingKeyNumeric", "StreetNumberNumeric", "ClosePrice"]


def get_numerical_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Create a summary table for all numerical columns."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    rows = []
    for col in num_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        q = s.quantile([0, .01, .05, .25, .5, .75, .95, .99, 1]).to_dict()
        rows.append({
            "column": col,
            "count": int(s.count()),
            "missing_count": int(s.isna().sum()),
            "missing_pct": round(float(s.isna().mean() * 100), 4),
            "zero_count": int((s == 0).sum()),
            "negative_count": int((s < 0).sum()),
            "mean": s.mean(),
            "std": s.std(),
            "min": q.get(0),
            "p1": q.get(.01),
            "p5": q.get(.05),
            "p25": q.get(.25),
            "median": q.get(.5),
            "p75": q.get(.75),
            "p95": q.get(.95),
            "p99": q.get(.99),
            "max": q.get(1),
        })
    return pd.DataFrame(rows)


def add_numerical_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add numerical issue flags without deleting rows."""
    out = df.copy()

    def s(col):
        return pd.to_numeric(out[col], errors="coerce") if col in out.columns else pd.Series(np.nan, index=out.index)

    close_price = s("ClosePrice")
    list_price = s("ListPrice")
    price_ratio = close_price / list_price.replace(0, np.nan)
    out["ClosePrice_to_ListPrice_ratio"] = price_ratio

    out["flag_closeprice_nonpositive"] = (close_price <= 0).fillna(False).astype(int)
    out["flag_price_ratio_extreme"] = ((price_ratio < 0.05) | (price_ratio > 5)).fillna(False).astype(int)
    out["flag_livingarea_extreme"] = ((s("LivingArea") <= 0) | (s("LivingArea") > 20000)).fillna(False).astype(int)
    out["flag_bedrooms_extreme"] = ((s("BedroomsTotal") < 0) | (s("BedroomsTotal") > 15)).fillna(False).astype(int)
    out["flag_bathrooms_extreme"] = ((s("BathroomsTotalInteger") < 0) | (s("BathroomsTotalInteger") > 15)).fillna(False).astype(int)
    out["flag_lotsize_extreme"] = ((s("LotSizeSquareFeet") <= 0) | (s("LotSizeSquareFeet") > 5_000_000)).fillna(False).astype(int)
    out["flag_yearbuilt_invalid"] = ((s("YearBuilt") < 1800) | (s("YearBuilt") > 2026)).fillna(False).astype(int)
    out["flag_daysonmarket_negative"] = (s("DaysOnMarket") < 0).fillna(False).astype(int)
    out["flag_garage_extreme"] = ((s("GarageSpaces") < 0) | (s("GarageSpaces") > 10)).fillna(False).astype(int)
    out["flag_parking_extreme"] = ((s("ParkingTotal") < 0) | (s("ParkingTotal") > 20)).fillna(False).astype(int)
    out["flag_latitude_outside_ca"] = ((s("Latitude") < 32) | (s("Latitude") > 42.5)).fillna(False).astype(int)
    out["flag_longitude_outside_ca"] = ((s("Longitude") < -125) | (s("Longitude") > -114)).fillna(False).astype(int)

    strict_flags = [
        "flag_closeprice_nonpositive", "flag_price_ratio_extreme",
        "flag_yearbuilt_invalid", "flag_daysonmarket_negative",
        "flag_latitude_outside_ca", "flag_longitude_outside_ca"
    ]
    review_flags = [
        "flag_livingarea_extreme", "flag_bedrooms_extreme", "flag_bathrooms_extreme",
        "flag_lotsize_extreme", "flag_garage_extreme", "flag_parking_extreme"
    ]
    out["numeric_strict_issue_flag"] = out[strict_flags].max(axis=1)
    out["numeric_review_flag"] = out[review_flags].max(axis=1)
    return out


def apply_numerical_strict_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove strict numerical issues only.
    Review outliers are kept and flagged because some luxury or rural properties may be valid.
    """
    flagged = add_numerical_flags(df)
    cleaned = flagged[flagged["numeric_strict_issue_flag"] == 0].copy()
    return cleaned


def get_numeric_feature_columns(df: pd.DataFrame) -> list:
    """Return numerical feature columns, excluding target and ID-like fields."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in num_cols if c not in EXCLUDE_NUMERIC_FEATURES and not c.startswith("flag_")]


def scale_numeric_features(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list):
    """
    Fit StandardScaler on train only, then transform both train and test.
    Use after time-based train/test split.
    """
    scaler = StandardScaler()
    train_scaled = train_df.copy()
    test_scaled = test_df.copy()

    available_cols = [c for c in feature_cols if c in train_df.columns and c in test_df.columns]
    train_scaled[available_cols] = scaler.fit_transform(train_scaled[available_cols])
    test_scaled[available_cols] = scaler.transform(test_scaled[available_cols])
    return train_scaled, test_scaled, scaler
