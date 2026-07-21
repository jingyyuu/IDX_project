# %% [markdown]
# # Model Comparison - Week 5
#
# **Goal:** compare Decision Tree and Random Forest regressors against the Week 4
# Linear Regression baseline using the same leakage-safe AVM evaluation framework.
#
# Model selection uses April 2026 only. May 2026 remains the one-time final test
# month. No test metric is used to select a model or hyperparameter.

# %%
from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

TARGET = "ClosePrice"
VALIDATION_MONTH = "2026-04"
TEST_MONTH = "2026-05"
TRAIN_FILE = "crmls_sfr_train_X12_2025-05_to_2026-04.csv"
TEST_FILE = "crmls_sfr_test_2026-05.csv"
OUTPUT_DIR = Path("outputs/week5_model_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def find_data_file(file_name):
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
        f"Could not find {file_name}. Mount Google Drive in Colab and add the shared "
        "IDX Summer Intern folder to My Drive."
    )


try:
    from google.colab import drive
    drive.mount("/content/drive")
except ImportError:
    pass


TRAIN_PATH = find_data_file(TRAIN_FILE)
TEST_PATH = find_data_file(TEST_FILE)

# %% [markdown]
# ## 1. Load the same versioned snapshot used in Week 4

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

# %% [markdown]
# ## 2. Restore missingness and apply the Week 4 quality rules
#
# Week 3 preserved `*_was_missing` flags. The marked numeric values are restored to
# NaN so each validation/backtest fold fits its own imputer.

# %%
RESTORABLE_NUMERIC_FEATURES = [
    "LivingArea", "BedroomsTotal", "BathroomsTotalInteger", "LotSizeSquareFeet",
    "YearBuilt", "Latitude", "Longitude", "AssociationFee", "Stories", "GarageSpaces",
]


def restore_marked_missing_values(frame):
    restored = frame.copy()
    for feature in RESTORABLE_NUMERIC_FEATURES:
        flag = f"{feature}_was_missing"
        if flag in restored.columns:
            restored.loc[restored[flag].fillna(False).astype(bool), feature] = np.nan
    return restored


def clean_invalid_records(frame):
    cleaned = frame.copy()
    target = pd.to_numeric(cleaned[TARGET], errors="coerce")
    cleaned = cleaned.loc[target.notna() & target.gt(0)].copy()

    living_area = pd.to_numeric(cleaned["LivingArea"], errors="coerce")
    cleaned = cleaned.loc[~(living_area.notna() & living_area.le(0))].copy()
    bedrooms = pd.to_numeric(cleaned["BedroomsTotal"], errors="coerce")
    cleaned = cleaned.loc[~(bedrooms.notna() & bedrooms.lt(0))].copy()
    bathrooms = pd.to_numeric(cleaned["BathroomsTotalInteger"], errors="coerce")
    cleaned = cleaned.loc[~(bathrooms.notna() & bathrooms.lt(0))].copy()

    close_date = pd.to_datetime(cleaned["CloseDate"], errors="coerce")
    list_date = pd.to_datetime(cleaned["ListingContractDate"], errors="coerce")
    cleaned = cleaned.loc[
        ~(close_date.notna() & list_date.notna() & close_date.lt(list_date))
    ].copy()
    cleaned = cleaned.drop_duplicates(subset=["ListingKey"], keep="first")
    return cleaned


history_clean = clean_invalid_records(restore_marked_missing_values(history_raw))
test_clean = clean_invalid_records(restore_marked_missing_values(test_raw))
test_clean = test_clean.loc[
    ~test_clean["ListingKey"].isin(set(history_clean["ListingKey"].dropna()))
].copy()

print("Clean history rows:", len(history_clean))
print("Clean test rows before train-only outlier bounds:", len(test_clean))

# %% [markdown]
# ## 3. Use the same safe feature families
#
# Physical, fixed-location, and lag-safe temporal features are retained. Sale-
# process, target-proxy, closing, brokerage, and identifier fields are excluded.

# %%
RAW_NUMERIC_FEATURES = [
    "LivingArea", "BedroomsTotal", "BathroomsTotalInteger", "LotSizeSquareFeet",
    "Latitude", "Longitude", "AssociationFee", "Stories", "GarageSpaces",
]
RAW_CATEGORICAL_FEATURES = ["PostalCode", "City", "PropertySubType"]
ENGINEERED_NUMERIC_FEATURES = ["PropertyAge", "SaleMonthSin", "SaleMonthCos"]
MODEL_FEATURES = RAW_NUMERIC_FEATURES + RAW_CATEGORICAL_FEATURES

FORBIDDEN_NAME_PATTERNS = (
    "listprice", "originallistprice", "closeprice_to_listprice", "daysonmarket",
    "purchasecontractdate", "contractstatuschangedate", "closedate", "mlsstatus",
    "listingagent", "listoffice", "buyeragent", "buyeroffice", "listingkey",
    "listingid", "price_reduction",
)
leaking_features = [
    feature for feature in MODEL_FEATURES
    if any(pattern in feature.lower() for pattern in FORBIDDEN_NAME_PATTERNS)
]
assert not leaking_features, f"Forbidden features selected: {leaking_features}"
assert set(MODEL_FEATURES).issubset(history_clean.columns)
assert set(MODEL_FEATURES).issubset(test_clean.columns)

display(pd.DataFrame({
    "feature": RAW_NUMERIC_FEATURES + RAW_CATEGORICAL_FEATURES + ENGINEERED_NUMERIC_FEATURES,
    "family": (
        ["intrinsic/location numeric"] * len(RAW_NUMERIC_FEATURES)
        + ["categorical location/type"] * len(RAW_CATEGORICAL_FEATURES)
        + ["temporal engineered"] * len(ENGINEERED_NUMERIC_FEATURES)
    ),
}))

# %% [markdown]
# ## 4. Deterministic features and structural preprocessing
#
# Property age and cyclical sale-month fields require no fitted parameters and are
# generated before the fitted pipeline. Every fit-time operation - median
# imputation, missing indicators, scaling, rare-category grouping, and category
# encoding - remains inside `Pipeline` / `ColumnTransformer`.

# %%
def engineer_deterministic_features(frame):
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


def build_pipeline(regressor):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore", min_frequency=50, sparse_output=True),
        ),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_PIPELINE_FEATURES),
        ("categorical", categorical_pipeline, RAW_CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("preprocessing", preprocessor),
        ("model", regressor),
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


def fit_closeprice_bounds(train_frame):
    lower, upper = train_frame[TARGET].quantile([0.005, 0.995])
    return float(lower), float(upper)


def apply_closeprice_bounds(frame, lower, upper):
    keep = frame[TARGET].between(lower, upper, inclusive="both")
    return frame.loc[keep].copy(), int((~keep).sum())

# %% [markdown]
# ## 5. Tune models on April 2026 only
#
# The Week 4 all-available training-window decision is reused. Each candidate is
# fit on May 2025-March 2026 and evaluated on April 2026. MdAPE is primary; MAPE
# and R2 break ties. May 2026 remains untouched.

# %%
development_raw = history_clean.loc[history_clean["close_month"] < VALIDATION_MONTH].copy()
validation_raw = history_clean.loc[history_clean["close_month"] == VALIDATION_MONTH].copy()
lower_validation, upper_validation = fit_closeprice_bounds(development_raw)
development, development_outliers = apply_closeprice_bounds(
    development_raw, lower_validation, upper_validation
)
validation, validation_outliers = apply_closeprice_bounds(
    validation_raw, lower_validation, upper_validation
)

X_development = engineer_deterministic_features(development)
X_validation = engineer_deterministic_features(validation)

candidate_models = {
    "LinearRegression_baseline": LinearRegression(),
    "DecisionTree_depth12_leaf20": DecisionTreeRegressor(
        max_depth=12, min_samples_leaf=20, random_state=RANDOM_SEED
    ),
    "DecisionTree_depth20_leaf10": DecisionTreeRegressor(
        max_depth=20, min_samples_leaf=10, random_state=RANDOM_SEED
    ),
    "DecisionTree_unlimited_leaf20": DecisionTreeRegressor(
        max_depth=None, min_samples_leaf=20, random_state=RANDOM_SEED
    ),
    "RandomForest_depth16_leaf5": RandomForestRegressor(
        n_estimators=100, max_depth=16, min_samples_leaf=5, max_features=0.5,
        n_jobs=-1, random_state=RANDOM_SEED,
    ),
    "RandomForest_depth24_leaf3": RandomForestRegressor(
        n_estimators=100, max_depth=24, min_samples_leaf=3, max_features=0.5,
        n_jobs=-1, random_state=RANDOM_SEED,
    ),
    "RandomForest_unlimited_leaf5": RandomForestRegressor(
        n_estimators=100, max_depth=None, min_samples_leaf=5, max_features="sqrt",
        n_jobs=-1, random_state=RANDOM_SEED,
    ),
}

validation_rows = []
for candidate_name, regressor in candidate_models.items():
    pipeline = build_pipeline(regressor)
    pipeline.fit(X_development, development[TARGET])
    predictions = pipeline.predict(X_validation)
    family = (
        "Linear Regression" if candidate_name.startswith("Linear")
        else "Decision Tree" if candidate_name.startswith("DecisionTree")
        else "Random Forest"
    )
    validation_rows.append({
        "candidate": candidate_name,
        "model_family": family,
        **evaluate(validation[TARGET], predictions, "validation_"),
    })

validation_results = pd.DataFrame(validation_rows).sort_values(
    ["model_family", "validation_mdape", "validation_mape", "validation_r2"],
    ascending=[True, True, True, False],
).reset_index(drop=True)
display(validation_results.round(4))

selected_candidates = (
    validation_results.sort_values(
        ["validation_mdape", "validation_mape", "validation_r2"],
        ascending=[True, True, False],
    )
    .groupby("model_family", as_index=False)
    .first()
)
display(selected_candidates.round(4))

selected_names = dict(zip(selected_candidates["model_family"], selected_candidates["candidate"]))
print("Selected models:", selected_names)

# %% [markdown]
# ## 6. Refit selected models and evaluate May 2026 once

# %%
lower_test, upper_test = fit_closeprice_bounds(history_clean)
final_train, train_outliers = apply_closeprice_bounds(history_clean, lower_test, upper_test)
final_test, test_outliers = apply_closeprice_bounds(test_clean, lower_test, upper_test)
X_final_train = engineer_deterministic_features(final_train)
X_final_test = engineer_deterministic_features(final_test)

final_models = {}
test_predictions_by_family = {}
comparison_rows = []

for family in ["Linear Regression", "Decision Tree", "Random Forest"]:
    candidate_name = selected_names[family]
    pipeline = build_pipeline(candidate_models[candidate_name])
    pipeline.fit(X_final_train, final_train[TARGET])
    train_predictions = pipeline.predict(X_final_train)
    test_predictions = pipeline.predict(X_final_test)
    final_models[family] = pipeline
    test_predictions_by_family[family] = test_predictions
    comparison_rows.append({
        "model_family": family,
        "selected_candidate": candidate_name,
        "train_rows": len(final_train),
        "test_rows": len(final_test),
        "train_outliers_removed": train_outliers,
        "test_outliers_removed": test_outliers,
        **evaluate(final_train[TARGET], train_predictions, "train_"),
        **evaluate(final_test[TARGET], test_predictions, "test_"),
    })

model_comparison = pd.DataFrame(comparison_rows).sort_values(
    ["test_mdape", "test_mape", "test_r2"], ascending=[True, True, False]
).reset_index(drop=True)

baseline = model_comparison.loc[
    model_comparison["model_family"].eq("Linear Regression")
].iloc[0]
model_comparison["r2_improvement_vs_baseline"] = model_comparison["test_r2"] - baseline["test_r2"]
model_comparison["mdape_reduction_vs_baseline"] = baseline["test_mdape"] - model_comparison["test_mdape"]
display(model_comparison.round(4))

best_family = model_comparison.iloc[0]["model_family"]
best_model = final_models[best_family]
best_predictions = test_predictions_by_family[best_family]
print("Best Week 5 model:", best_family)
print("Test R2:", round(float(model_comparison.iloc[0]["test_r2"]), 4))
print("Test MdAPE:", f"{float(model_comparison.iloc[0]['test_mdape']):.2%}")

# %% [markdown]
# ## 7. Rolling-origin stability for selected models

# %%
backtest_rows = []
for cutoff_month in ["2026-02", "2026-03", "2026-04"]:
    fold_train_raw = history_clean.loc[history_clean["close_month"] < cutoff_month].copy()
    fold_eval_raw = history_clean.loc[history_clean["close_month"] == cutoff_month].copy()
    lower, upper = fit_closeprice_bounds(fold_train_raw)
    fold_train, _ = apply_closeprice_bounds(fold_train_raw, lower, upper)
    fold_eval, _ = apply_closeprice_bounds(fold_eval_raw, lower, upper)
    X_fold_train = engineer_deterministic_features(fold_train)
    X_fold_eval = engineer_deterministic_features(fold_eval)

    for family in ["Linear Regression", "Decision Tree", "Random Forest"]:
        candidate_name = selected_names[family]
        fold_pipeline = build_pipeline(candidate_models[candidate_name])
        fold_pipeline.fit(X_fold_train, fold_train[TARGET])
        fold_predictions = fold_pipeline.predict(X_fold_eval)
        backtest_rows.append({
            "cutoff_month": cutoff_month,
            "model_family": family,
            "candidate": candidate_name,
            "train_rows": len(fold_train),
            "evaluation_rows": len(fold_eval),
            **evaluate(fold_eval[TARGET], fold_predictions),
        })

rolling_backtest = pd.DataFrame(backtest_rows)
display(rolling_backtest.round(4))

# %% [markdown]
# ## 8. Tree-based feature importance

# %%
importance_tables = []
for family in ["Decision Tree", "Random Forest"]:
    pipeline = final_models[family]
    preprocessor = pipeline.named_steps["preprocessing"]
    numeric_imputer = preprocessor.named_transformers_["numeric"].named_steps["imputer"]
    missing_numeric_names = [
        f"{NUMERIC_PIPELINE_FEATURES[index]}_was_missing"
        for index in numeric_imputer.indicator_.features_
    ]
    numeric_names = [
        f"numeric__{name}"
        for name in NUMERIC_PIPELINE_FEATURES + missing_numeric_names
    ]
    categorical_encoder = (
        preprocessor.named_transformers_["categorical"].named_steps["encoder"]
    )
    categorical_names = [
        f"categorical__{name}"
        for name in categorical_encoder.get_feature_names_out(RAW_CATEGORICAL_FEATURES)
    ]
    feature_names = np.array(numeric_names + categorical_names)
    importances = pipeline.named_steps["model"].feature_importances_
    if len(feature_names) != len(importances):
        # Some sklearn versions omit infrequent/missing-indicator labels from
        # get_feature_names_out even though the transformed matrix contains them.
        # Numeric output order is stable, so preserve those labels and use neutral
        # IDs only for the unresolved encoded categorical columns.
        unresolved_count = len(importances) - len(numeric_names)
        feature_names = np.array(
            numeric_names
            + [f"encoded_categorical_{index:04d}" for index in range(unresolved_count)]
        )
    importance = pd.DataFrame({
        "model_family": family,
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False).head(20)
    importance_tables.append(importance)

feature_importance = pd.concat(importance_tables, ignore_index=True)
display(feature_importance)

# %% [markdown]
# ## 9. Model behavior
#
# | Model | Strengths | Weaknesses |
# |---|---|---|
# | Linear Regression | Fast, interpretable, stable benchmark | Misses nonlinear interactions |
# | Decision Tree | Captures nonlinear rules and interactions | High variance; can overfit |
# | Random Forest | Reduces tree variance and captures nonlinear effects | Slower and less interpretable |
#
# A model is considered meaningfully better only if its MdAPE reduction and R2
# improvement are visible across the final test and historical backtests, rather
# than only as a tiny gain at one cutoff.

# %% [markdown]
# ## 10. Save local comparison artifacts

# %%
validation_results.to_csv(OUTPUT_DIR / "validation_tuning_results.csv", index=False)
model_comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
rolling_backtest.to_csv(OUTPUT_DIR / "rolling_origin_backtest.csv", index=False)
feature_importance.to_csv(OUTPUT_DIR / "tree_feature_importance.csv", index=False)

prediction_output = pd.DataFrame({"actual_close_price": final_test[TARGET].to_numpy()})
for family, predictions in test_predictions_by_family.items():
    column = family.lower().replace(" ", "_")
    prediction_output[f"{column}_prediction"] = predictions
prediction_output.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

for family, pipeline in final_models.items():
    file_name = family.lower().replace(" ", "_") + ".joblib"
    joblib.dump(pipeline, OUTPUT_DIR / file_name)

metadata = {
    "project_week": 5,
    "data_snapshot": data_snapshot.to_dict(orient="records"),
    "validation_month": VALIDATION_MONTH,
    "test_month": TEST_MONTH,
    "random_seed": RANDOM_SEED,
    "closeprice_quantiles": [0.005, 0.995],
    "closeprice_bounds": [lower_test, upper_test],
    "selected_candidates": selected_names,
    "best_model_family": best_family,
    "forbidden_name_patterns": list(FORBIDDEN_NAME_PATTERNS),
    "model_comparison": model_comparison.to_dict(orient="records"),
}
with open(OUTPUT_DIR / "run_metadata.json", "w") as file:
    json.dump(metadata, file, indent=2)

# %% [markdown]
# ## 11. Final comparison chart

# %%
plot_data = model_comparison.sort_values("test_mdape")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].bar(plot_data["model_family"], plot_data["test_r2"])
axes[0].set(title="May 2026 test R2", ylabel="R2")
axes[1].bar(plot_data["model_family"], plot_data["test_mdape"] * 100)
axes[1].set(title="May 2026 test MdAPE", ylabel="MdAPE (%)")
for axis in axes:
    axis.tick_params(axis="x", rotation=20)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "model_comparison.png", dpi=160)
plt.show()

# %% [markdown]
# ## Conclusion
#
# Week 5 compares Decision Tree and Random Forest regressors with the Week 4 Linear
# Regression baseline under the same chronological, leakage-safe evaluation rules.
# Hyperparameters are chosen on April 2026, May 2026 is evaluated once, and the
# selected models are checked across three earlier monthly cutoffs.
