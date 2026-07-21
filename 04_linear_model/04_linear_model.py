# %% [markdown]
# # Linear Model - Week 4
#
# **Goal:** build a reproducible, leakage-safe Linear Regression baseline for
# California single-family home close prices.
#
# This notebook implements the IDX AVM Data Science Best Practices:
#
# - chronological development, validation, and test months;
# - rolling-origin backtesting;
# - invalid-record removal with logged row counts;
# - train-only ClosePrice outlier thresholds;
# - structural leakage prevention with a scikit-learn Pipeline;
# - deliberate missing-value handling;
# - intrinsic, locational, and temporal features; and
# - MdAPE as the headline metric alongside R2, MAPE, MAE, and RMSE.

# %%
from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

TARGET = "ClosePrice"
VALIDATION_MONTH = "2026-04"
TEST_MONTH = "2026-05"
TRAIN_FILE = "crmls_sfr_train_X12_2025-05_to_2026-04.csv"
TEST_FILE = "crmls_sfr_test_2026-05.csv"
OUTPUT_DIR = Path("outputs/week4_linear_model")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def find_data_file(file_name):
    """Find a team CSV locally or under a mounted Google Drive in Colab."""
    candidates = [
        Path("data/week3_drive") / file_name,
        Path("/content/drive/MyDrive") / file_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    drive_root = Path("/content/drive/MyDrive")
    if drive_root.exists():
        matches = list(drive_root.rglob(file_name))
        if matches:
            return matches[0]

    raise FileNotFoundError(
        f"Could not find {file_name}. In Colab, mount Google Drive and add a shortcut "
        "to the shared IDX Summer Intern folder under My Drive."
    )


try:
    from google.colab import drive
    drive.mount("/content/drive")
except ImportError:
    pass


TRAIN_PATH = find_data_file(TRAIN_FILE)
TEST_PATH = find_data_file(TEST_FILE)

# %% [markdown]
# ## 1. Load the versioned data snapshot
#
# The file names and chronological coverage are recorded so this result can be tied
# to a specific data snapshot. May 2026 is the untouched final test month. April
# 2026 is the model-selection month.

# %%
history_raw = pd.read_csv(TRAIN_PATH, low_memory=False)
test_raw = pd.read_csv(TEST_PATH, low_memory=False)

assert history_raw["close_month"].min() == "2025-05"
assert history_raw["close_month"].max() == VALIDATION_MONTH
assert test_raw["close_month"].eq(TEST_MONTH).all()

data_snapshot = pd.DataFrame([
    {
        "role": "history",
        "file": TRAIN_PATH.name,
        "rows": len(history_raw),
        "first_month": history_raw["close_month"].min(),
        "last_month": history_raw["close_month"].max(),
    },
    {
        "role": "test",
        "file": TEST_PATH.name,
        "rows": len(test_raw),
        "first_month": test_raw["close_month"].min(),
        "last_month": test_raw["close_month"].max(),
    },
])
display(data_snapshot)

# Restore numeric NaNs that Week 3 marked before producing its model-ready CSV.
# This lets every Week 4 fold fit its own imputer instead of reusing values that
# may have been filled using a broader historical window.
RESTORABLE_NUMERIC_FEATURES = [
    "LivingArea", "BedroomsTotal", "BathroomsTotalInteger", "LotSizeSquareFeet",
    "YearBuilt", "Latitude", "Longitude", "AssociationFee", "Stories", "GarageSpaces",
]


def restore_marked_missing_values(frame, dataset_name):
    restored = frame.copy()
    audit = []
    for feature in RESTORABLE_NUMERIC_FEATURES:
        flag = f"{feature}_was_missing"
        if flag in restored.columns:
            missing_mask = restored[flag].fillna(False).astype(bool)
            restored.loc[missing_mask, feature] = np.nan
            audit.append({
                "dataset": dataset_name,
                "feature": feature,
                "missing_values_restored": int(missing_mask.sum()),
                "missing_rate": float(missing_mask.mean()),
            })
    return restored, pd.DataFrame(audit)


history_raw, history_missing_restore = restore_marked_missing_values(history_raw, "history")
test_raw, test_missing_restore = restore_marked_missing_values(test_raw, "test")
missing_restore_audit = pd.concat(
    [history_missing_restore, test_missing_restore], ignore_index=True
)
display(missing_restore_audit.round(4))

# %% [markdown]
# ## 2. Remove invalid records and log every rule
#
# The target is never imputed. Rows with a missing/non-positive target, impossible
# physical values, reversed listing/close dates, or duplicate transaction IDs are
# removed. A missing predictor is retained for pipeline imputation.

# %%
def clean_invalid_records(frame, dataset_name):
    cleaned = frame.copy()
    audit_rows = []

    def apply_rule(rule_name, invalid_mask):
        nonlocal cleaned
        invalid_mask = invalid_mask.fillna(False)
        before = len(cleaned)
        removed = int(invalid_mask.sum())
        cleaned = cleaned.loc[~invalid_mask].copy()
        audit_rows.append({
            "dataset": dataset_name,
            "rule": rule_name,
            "rows_before": before,
            "rows_removed": removed,
            "removed_pct": removed / before if before else 0,
            "rows_after": len(cleaned),
        })

    target_numeric = pd.to_numeric(cleaned[TARGET], errors="coerce")
    apply_rule("ClosePrice missing or non-positive", target_numeric.isna() | target_numeric.le(0))

    living_area = pd.to_numeric(cleaned["LivingArea"], errors="coerce")
    apply_rule("LivingArea non-positive when reported", living_area.notna() & living_area.le(0))

    bedrooms = pd.to_numeric(cleaned["BedroomsTotal"], errors="coerce")
    apply_rule("BedroomsTotal negative", bedrooms.notna() & bedrooms.lt(0))

    bathrooms = pd.to_numeric(cleaned["BathroomsTotalInteger"], errors="coerce")
    apply_rule("BathroomsTotalInteger negative", bathrooms.notna() & bathrooms.lt(0))

    close_date = pd.to_datetime(cleaned["CloseDate"], errors="coerce")
    list_date = pd.to_datetime(cleaned["ListingContractDate"], errors="coerce")
    apply_rule(
        "CloseDate earlier than ListingContractDate",
        close_date.notna() & list_date.notna() & close_date.lt(list_date),
    )

    duplicate_subset = ["ListingKey"] if "ListingKey" in cleaned.columns else [
        "UnparsedAddress", "CloseDate"
    ]
    apply_rule(
        "Duplicate property transaction",
        cleaned.duplicated(subset=duplicate_subset, keep="first"),
    )

    return cleaned, pd.DataFrame(audit_rows)


history_clean, history_quality_audit = clean_invalid_records(history_raw, "history")
test_clean, test_quality_audit = clean_invalid_records(test_raw, "test")

# Remove any transaction appearing in both history and test.
history_keys = set(history_clean["ListingKey"].dropna())
overlap_mask = test_clean["ListingKey"].isin(history_keys)
overlap_audit = pd.DataFrame([{
    "dataset": "test",
    "rule": "ListingKey also appears in history",
    "rows_before": len(test_clean),
    "rows_removed": int(overlap_mask.sum()),
    "removed_pct": float(overlap_mask.mean()),
    "rows_after": int((~overlap_mask).sum()),
}])
test_clean = test_clean.loc[~overlap_mask].copy()

quality_audit = pd.concat(
    [history_quality_audit, test_quality_audit, overlap_audit],
    ignore_index=True,
)
display(quality_audit.round(4))

# %% [markdown]
# ## 3. Feature leakage audit
#
# Only information that could be known before a sale outcome is used. The model
# excludes `ListPrice`, `OriginalListPrice`, DOM, price-reduction information,
# contract/closing fields, brokerage/agent identities, identifiers, and all derived
# versions of those fields.

# %%
RAW_NUMERIC_FEATURES = [
    "LivingArea",
    "BedroomsTotal",
    "BathroomsTotalInteger",
    "LotSizeSquareFeet",
    "Latitude",
    "Longitude",
    "AssociationFee",
    "Stories",
    "GarageSpaces",
]

RAW_CATEGORICAL_FEATURES = [
    "PostalCode",
    "City",
    "PropertySubType",
]

ENGINEERED_NUMERIC_FEATURES = [
    "PropertyAge",
    "SaleMonthSin",
    "SaleMonthCos",
]

MODEL_FEATURES = RAW_NUMERIC_FEATURES + RAW_CATEGORICAL_FEATURES
FORBIDDEN_NAME_PATTERNS = (
    "listprice",
    "originallistprice",
    "closeprice_to_listprice",
    "daysonmarket",
    "purchasecontractdate",
    "contractstatuschangedate",
    "closedate",
    "mlsstatus",
    "listingagent",
    "listoffice",
    "buyeragent",
    "buyeroffice",
    "listingkey",
    "listingid",
    "price_reduction",
)

leaking_features = [
    feature
    for feature in MODEL_FEATURES
    if any(pattern in feature.lower() for pattern in FORBIDDEN_NAME_PATTERNS)
]
assert not leaking_features, f"Forbidden/leaky features selected: {leaking_features}"
assert set(MODEL_FEATURES).issubset(history_clean.columns)
assert set(MODEL_FEATURES).issubset(test_clean.columns)

leakage_audit = pd.DataFrame({
    "feature_family": [
        "ListPrice / OriginalListPrice and derivatives",
        "DaysOnMarket / price reductions",
        "Contract, closing, and status fields",
        "Agent, brokerage, and identifiers",
        "Intrinsic property characteristics",
        "Fixed location attributes",
        "Lag-safe temporal attributes",
    ],
    "verdict": ["Exclude", "Exclude", "Exclude", "Exclude", "Keep", "Keep", "Keep"],
    "rationale": [
        "Unavailable off market and too close to ClosePrice",
        "Sale-process information unavailable at valuation time",
        "Known during or after the sale process",
        "Potential pricing-strategy proxy or non-generalizable ID",
        "Physical facts known before sale",
        "Known before sale",
        "Derived only from the sale month and property year built",
    ],
})
display(leakage_audit)

missingness_report = pd.DataFrame({
    "feature": MODEL_FEATURES + ["YearBuilt"],
    "history_missing_rate": history_clean[MODEL_FEATURES + ["YearBuilt"]].isna().mean().to_numpy(),
    "handling": [
        "Training median + indicator" if feature in RESTORABLE_NUMERIC_FEATURES
        else "Unknown category + rare-category grouping"
        for feature in MODEL_FEATURES + ["YearBuilt"]
    ],
}).sort_values("history_missing_rate", ascending=False)
display(missingness_report.round(4))

# %% [markdown]
# ## 4. Structural preprocessing pipeline
#
# Deterministic temporal features are created inside the pipeline. Numeric medians,
# missing indicators, scaling parameters, rare-category grouping, and one-hot
# categories are all fit only when `Pipeline.fit()` receives training data.
#
# `OneHotEncoder(min_frequency=50)` groups rare categories instead of creating a
# separate sparse column for every low-frequency ZIP/city/subtype value. Unknown
# validation/test categories are ignored safely.

# %%
def engineer_features(frame):
    engineered = frame.copy()
    sale_month = pd.to_datetime(engineered["close_month"], format="%Y-%m", errors="coerce")
    year_built = pd.to_numeric(engineered["YearBuilt"], errors="coerce")
    engineered["PropertyAge"] = (sale_month.dt.year - year_built).where(
        year_built.gt(0) & year_built.le(sale_month.dt.year)
    )
    engineered["SaleMonthSin"] = np.sin(2 * np.pi * sale_month.dt.month / 12)
    engineered["SaleMonthCos"] = np.cos(2 * np.pi * sale_month.dt.month / 12)
    return engineered


NUMERIC_PIPELINE_FEATURES = RAW_NUMERIC_FEATURES + ENGINEERED_NUMERIC_FEATURES


def build_pipeline():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                min_frequency=50,
                sparse_output=True,
            ),
        ),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_PIPELINE_FEATURES),
        ("categorical", categorical_pipeline, RAW_CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("feature_engineering", FunctionTransformer(engineer_features, validate=False)),
        ("preprocessing", preprocessor),
        ("model", LinearRegression()),
    ])


def absolute_percentage_errors(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = y_true != 0
    return np.abs((y_true[valid] - y_pred[valid]) / y_true[valid])


def evaluate(y_true, y_pred, prefix=""):
    percentage_errors = absolute_percentage_errors(y_true, y_pred)
    return {
        f"{prefix}r2": r2_score(y_true, y_pred),
        f"{prefix}mape": percentage_errors.mean(),
        f"{prefix}mdape": np.median(percentage_errors),
        f"{prefix}mae": mean_absolute_error(y_true, y_pred),
        f"{prefix}rmse": mean_squared_error(y_true, y_pred) ** 0.5,
    }


def select_trailing_months(frame, n_months=None):
    available_months = sorted(frame["close_month"].unique())
    selected_months = available_months if n_months is None else available_months[-n_months:]
    return frame.loc[frame["close_month"].isin(selected_months)].copy(), selected_months


def fit_closeprice_bounds(train_frame):
    lower, upper = train_frame[TARGET].quantile([0.005, 0.995])
    return float(lower), float(upper)


def apply_closeprice_bounds(frame, lower, upper):
    keep = frame[TARGET].between(lower, upper, inclusive="both")
    return frame.loc[keep].copy(), int((~keep).sum())

# %% [markdown]
# ## 5. Select the training window without touching May 2026
#
# April 2026 is the validation month. For each candidate window, the 0.5th/99.5th
# ClosePrice thresholds are learned from that window and frozen before application
# to April. MdAPE is the primary selection metric.

# %%
development = history_clean.loc[history_clean["close_month"] < VALIDATION_MONTH].copy()
validation = history_clean.loc[history_clean["close_month"] == VALIDATION_MONTH].copy()
assert len(development) > 0 and len(validation) > 0

WINDOWS = {"6_months": 6, "9_months": 9, "all_available": None}
validation_results = []

for window_name, n_months in WINDOWS.items():
    window_raw, months = select_trailing_months(development, n_months)
    lower, upper = fit_closeprice_bounds(window_raw)
    window_train, train_outliers = apply_closeprice_bounds(window_raw, lower, upper)
    validation_eval, validation_outliers = apply_closeprice_bounds(validation, lower, upper)

    candidate = build_pipeline()
    candidate.fit(window_train, window_train[TARGET])
    validation_predictions = candidate.predict(validation_eval)
    validation_results.append({
        "window": window_name,
        "first_train_month": months[0],
        "last_train_month": months[-1],
        "train_rows": len(window_train),
        "train_outliers_removed": train_outliers,
        "validation_rows": len(validation_eval),
        "validation_outliers_removed": validation_outliers,
        "lower_closeprice": lower,
        "upper_closeprice": upper,
        **evaluate(validation_eval[TARGET], validation_predictions, "validation_"),
    })

validation_results_df = pd.DataFrame(validation_results).sort_values(
    ["validation_mdape", "validation_mape", "validation_r2"],
    ascending=[True, True, False],
).reset_index(drop=True)
display(validation_results_df.round(4))

selected_window = validation_results_df.iloc[0]
selected_window_name = selected_window["window"]
selected_n_months = WINDOWS[selected_window_name]
print("Selected training window:", selected_window_name)
print("Validation MdAPE:", f"{float(selected_window['validation_mdape']):.2%}")

# %% [markdown]
# ## 6. Rolling-origin backtest
#
# The selected design is re-fit at three earlier historical cutoffs. Each fold uses
# only the months preceding its evaluation month, including separately fitted
# preprocessing and outlier thresholds.

# %%
backtest_rows = []
for cutoff_month in ["2026-02", "2026-03", "2026-04"]:
    fold_history = history_clean.loc[history_clean["close_month"] < cutoff_month].copy()
    fold_eval_raw = history_clean.loc[history_clean["close_month"] == cutoff_month].copy()
    fold_train_raw, fold_months = select_trailing_months(fold_history, selected_n_months)

    lower, upper = fit_closeprice_bounds(fold_train_raw)
    fold_train, train_outliers = apply_closeprice_bounds(fold_train_raw, lower, upper)
    fold_eval, eval_outliers = apply_closeprice_bounds(fold_eval_raw, lower, upper)

    fold_pipeline = build_pipeline()
    fold_pipeline.fit(fold_train, fold_train[TARGET])
    fold_predictions = fold_pipeline.predict(fold_eval)
    backtest_rows.append({
        "cutoff_month": cutoff_month,
        "first_train_month": fold_months[0],
        "last_train_month": fold_months[-1],
        "train_rows": len(fold_train),
        "evaluation_rows": len(fold_eval),
        "train_outliers_removed": train_outliers,
        "evaluation_outliers_removed": eval_outliers,
        **evaluate(fold_eval[TARGET], fold_predictions),
    })

backtest_results_df = pd.DataFrame(backtest_rows)
display(backtest_results_df.round(4))

# %% [markdown]
# ## 7. Final fit and one-time May 2026 evaluation

# %%
final_train_raw, final_train_months = select_trailing_months(history_clean, selected_n_months)
lower_closeprice, upper_closeprice = fit_closeprice_bounds(final_train_raw)
final_train, final_train_outliers = apply_closeprice_bounds(
    final_train_raw, lower_closeprice, upper_closeprice
)
final_test, final_test_outliers = apply_closeprice_bounds(
    test_clean, lower_closeprice, upper_closeprice
)

baseline_pipeline = build_pipeline()
baseline_pipeline.fit(final_train, final_train[TARGET])
train_predictions = baseline_pipeline.predict(final_train)
test_predictions = baseline_pipeline.predict(final_test)

baseline_results = pd.DataFrame([{
    "model": "Linear Regression",
    "training_window": selected_window_name,
    "first_train_month": final_train_months[0],
    "last_train_month": final_train_months[-1],
    "test_month": TEST_MONTH,
    "train_rows": len(final_train),
    "test_rows": len(final_test),
    "train_outliers_removed": final_train_outliers,
    "test_outliers_removed": final_test_outliers,
    "lower_closeprice": lower_closeprice,
    "upper_closeprice": upper_closeprice,
    **evaluate(final_train[TARGET], train_predictions, "train_"),
    **evaluate(final_test[TARGET], test_predictions, "test_"),
}])

display(baseline_results.round(4))
print("Test R2:", round(float(baseline_results.loc[0, "test_r2"]), 4))
print("Test MAPE:", f"{float(baseline_results.loc[0, 'test_mape']):.2%}")
print("Test MdAPE (headline):", f"{float(baseline_results.loc[0, 'test_mdape']):.2%}")
print("Test MAE:", f"${float(baseline_results.loc[0, 'test_mae']):,.0f}")

# %% [markdown]
# ## 8. Error by actual-price quintile
#
# Price bands are derived from the final training target distribution and frozen
# before being applied to test. This reveals whether typical error changes across
# entry-level, mid-market, and higher-priced homes.

# %%
_, price_edges = pd.qcut(
    final_train[TARGET], q=5, retbins=True, duplicates="drop"
)
price_edges[0], price_edges[-1] = -np.inf, np.inf
test_diagnostics = pd.DataFrame({
    "actual": final_test[TARGET].to_numpy(),
    "predicted": test_predictions,
})
test_diagnostics["absolute_error"] = np.abs(
    test_diagnostics["actual"] - test_diagnostics["predicted"]
)
test_diagnostics["absolute_percentage_error"] = (
    test_diagnostics["absolute_error"] / test_diagnostics["actual"]
)
test_diagnostics["training_price_band"] = pd.cut(
    test_diagnostics["actual"], bins=price_edges, include_lowest=True
)

price_band_results = (
    test_diagnostics.groupby("training_price_band", observed=True)
    .agg(
        rows=("actual", "size"),
        actual_price_median=("actual", "median"),
        mae=("absolute_error", "mean"),
        mape=("absolute_percentage_error", "mean"),
        mdape=("absolute_percentage_error", "median"),
    )
    .reset_index()
)
display(price_band_results.round(4))

# %% [markdown]
# ## 9. Save the complete reproducibility package locally

# %%
quality_audit.to_csv(OUTPUT_DIR / "data_quality_audit.csv", index=False)
missing_restore_audit.to_csv(OUTPUT_DIR / "restored_missing_values.csv", index=False)
missingness_report.to_csv(OUTPUT_DIR / "feature_missingness.csv", index=False)
validation_results_df.to_csv(OUTPUT_DIR / "training_window_comparison.csv", index=False)
backtest_results_df.to_csv(OUTPUT_DIR / "rolling_origin_backtest.csv", index=False)
baseline_results.to_csv(OUTPUT_DIR / "linear_regression_baseline.csv", index=False)
price_band_results.to_csv(OUTPUT_DIR / "price_band_metrics.csv", index=False)

test_diagnostics.assign(
    residual=test_diagnostics["actual"] - test_diagnostics["predicted"]
).to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

joblib.dump(
    {
        "pipeline": baseline_pipeline,
        "raw_numeric_features": RAW_NUMERIC_FEATURES,
        "raw_categorical_features": RAW_CATEGORICAL_FEATURES,
        "engineered_numeric_features": ENGINEERED_NUMERIC_FEATURES,
        "closeprice_bounds": [lower_closeprice, upper_closeprice],
        "random_seed": RANDOM_SEED,
    },
    OUTPUT_DIR / "linear_regression_pipeline.joblib",
)

metric_keys = [
    "train_r2", "train_mape", "train_mdape", "train_mae", "train_rmse",
    "test_r2", "test_mape", "test_mdape", "test_mae", "test_rmse",
]
metadata = {
    "project_week": 4,
    "model": "Linear Regression",
    "data_snapshot": data_snapshot.to_dict(orient="records"),
    "validation_month": VALIDATION_MONTH,
    "test_month": TEST_MONTH,
    "selected_training_window": selected_window_name,
    "final_train_months": final_train_months,
    "random_seed": RANDOM_SEED,
    "closeprice_quantiles": [0.005, 0.995],
    "closeprice_bounds": [lower_closeprice, upper_closeprice],
    "raw_numeric_features": RAW_NUMERIC_FEATURES,
    "raw_categorical_features": RAW_CATEGORICAL_FEATURES,
    "engineered_numeric_features": ENGINEERED_NUMERIC_FEATURES,
    "forbidden_name_patterns": list(FORBIDDEN_NAME_PATTERNS),
    "missing_numeric": "training median plus missing indicator",
    "missing_categorical": "Unknown",
    "categorical_encoding": "OneHotEncoder min_frequency=50, handle_unknown=ignore",
    **{key: float(baseline_results.loc[0, key]) for key in metric_keys},
}
with open(OUTPUT_DIR / "run_metadata.json", "w") as file:
    json.dump(metadata, file, indent=2)

# %% [markdown]
# ## 10. Actual vs. predicted diagnostic

# %%
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(final_test[TARGET], test_predictions, alpha=0.18, s=10)
low = min(final_test[TARGET].min(), test_predictions.min())
high = max(final_test[TARGET].max(), test_predictions.max())
ax.plot([low, high], [low, high], "--", color="black", linewidth=1)
ax.set(
    xlabel="Actual ClosePrice",
    ylabel="Predicted ClosePrice",
    title=f"Linear Regression baseline - {TEST_MONTH} test set",
)
ax.ticklabel_format(style="plain", axis="both")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "actual_vs_predicted.png", dpi=160)
plt.show()

# %% [markdown]
# ## Assumptions, limitations, and next steps
#
# **Assumptions**
#
# - Missing predictor values are retained because missingness may be informative;
#   the target is never imputed.
# - The 0.5th/99.5th ClosePrice rule is treated as the agreed data-quality policy.
# - Month and property age are available at valuation time and do not use outcomes.
#
# **Limitations**
#
# - Linear Regression cannot model complex nonlinear location/property interactions.
# - City and ZIP are broad location proxies; rare values are grouped.
# - The data snapshot contains only twelve historical months, limiting longer-term
#   stability analysis.
#
# **Next steps**
#
# - Compare Decision Tree and Random Forest models with this exact evaluation setup.
# - Add a train-only price-per-square-foot quality check and justify its thresholds.
# - Repeat the rolling-origin backtest as each new monthly snapshot arrives.
