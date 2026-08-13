# %% [markdown]
# # Evaluation Expansion - Week 8
#
# This notebook evaluates the Week 6 Random Forest and Week 7 LightGBM models.
# It compares overall R2, MAPE, and MdAPE, then checks performance in different
# house price bands.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Image, display
from sklearn.metrics import r2_score

PROJECT_ROOT = Path("..") if Path("../outputs").exists() else Path(".")
EVALUATION_DIR = PROJECT_ROOT / "08_evaluation"
WEEK6_FILE = PROJECT_ROOT / "outputs/week6_feature_engineering/test_predictions.csv"
WEEK7_FILE = PROJECT_ROOT / "outputs/week7_advanced_models/lightgbm_test_predictions.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs/week8_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for prediction_file in [WEEK6_FILE, WEEK7_FILE]:
    if not prediction_file.exists():
        raise FileNotFoundError(
            f"Missing {prediction_file}. Run the Week 6 and Week 7 notebooks first."
        )

# %% [markdown]
# ## 1. Load the test predictions

# %%
week6 = pd.read_csv(WEEK6_FILE)
week7 = pd.read_csv(WEEK7_FILE)

week6 = week6[["ListingKey", "actual_close_price", "week6_prediction"]]
week6 = week6.rename(columns={
    "actual_close_price": "actual_price",
    "week6_prediction": "Random Forest",
})

week7 = week7[["ListingKey", "actual_price", "predicted_price"]]
week7 = week7.rename(columns={"predicted_price": "LightGBM"})

# Match the predictions by ListingKey.
predictions = week6.merge(
    week7,
    on="ListingKey",
    how="inner",
    suffixes=("_week6", "_week7"),
    validate="one_to_one",
)

# The actual prices should be the same in both files.
price_difference = (
    predictions["actual_price_week6"] - predictions["actual_price_week7"]
).abs()
assert price_difference.max() < 0.01

predictions["actual_price"] = predictions["actual_price_week6"]
predictions = predictions[
    ["ListingKey", "actual_price", "Random Forest", "LightGBM"]
]

print("Matched test rows:", len(predictions))
display(predictions.head())

# %% [markdown]
# ## 2. Calculate the overall metrics
#
# - R2 shows how much price variation the model explains. Higher is better.
# - MAPE is the average absolute percentage error. Lower is better.
# - MdAPE is the median absolute percentage error. Lower is better.

# %%
def calculate_metrics(actual, predicted):
    percentage_error = np.abs((actual - predicted) / actual)
    return {
        "r2": r2_score(actual, predicted),
        "mape": percentage_error.mean(),
        "mdape": percentage_error.median(),
    }


models = ["Random Forest", "LightGBM"]
overall_rows = []

for model in models:
    overall_rows.append({
        "model": model,
        "price_band": "All homes",
        "number_of_homes": len(predictions),
        **calculate_metrics(predictions["actual_price"], predictions[model]),
    })

overall_metrics = pd.DataFrame(overall_rows)
display(overall_metrics.round(4))

# %% [markdown]
# ## 3. Create house price bands

# %%
band_edges = [0, 500_000, 750_000, 1_000_000, 2_000_000, np.inf]
band_names = [
    "$500K or less",
    "$500K-$750K",
    "$750K-$1M",
    "$1M-$2M",
    "Over $2M",
]

predictions["price_band"] = pd.cut(
    predictions["actual_price"],
    bins=band_edges,
    labels=band_names,
    include_lowest=True,
)

display(predictions["price_band"].value_counts(sort=False).to_frame("number_of_homes"))

# %% [markdown]
# ## 4. Compare model performance by price band

# %%
band_rows = []

for price_band in band_names:
    band_data = predictions[predictions["price_band"] == price_band]

    for model in models:
        percentage_error = np.abs(
            (band_data["actual_price"] - band_data[model])
            / band_data["actual_price"]
        )
        band_rows.append({
            "model": model,
            "price_band": price_band,
            "number_of_homes": len(band_data),
            "mape": percentage_error.mean(),
            "mdape": percentage_error.median(),
        })

band_metrics = pd.DataFrame(band_rows)
display(band_metrics.round(4))

# %% [markdown]
# ## 5. Summarize the main findings

# %%
mdape_table = band_metrics.pivot(
    index="price_band",
    columns="model",
    values="mdape",
).reindex(band_names)

best_band_rf = mdape_table["Random Forest"].idxmin()
best_band_lgbm = mdape_table["LightGBM"].idxmin()
worst_band_rf = mdape_table["Random Forest"].idxmax()
worst_band_lgbm = mdape_table["LightGBM"].idxmax()

print("Random Forest lowest MdAPE:", best_band_rf)
print("Random Forest highest MdAPE:", worst_band_rf)
print("LightGBM lowest MdAPE:", best_band_lgbm)
print("LightGBM highest MdAPE:", worst_band_lgbm)
print("Random Forest has the lower MdAPE in all five price bands.")

# %% [markdown]
# ## 6. Save the metrics and charts

# %%
metrics_summary = pd.concat([overall_metrics, band_metrics], ignore_index=True)
metrics_summary.to_csv(EVALUATION_DIR / "metrics_summary.csv", index=False)

colors = ["#8da0cb", "#66c2a5"]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

overall_plot = overall_metrics.set_index("model")
axes[0].bar(overall_plot.index, overall_plot["r2"], color=colors)
axes[0].set_title("Overall May 2026 Test R2")
axes[0].set_ylim(0, 1)
for index, value in enumerate(overall_plot["r2"]):
    axes[0].text(index, value + 0.02, f"{value:.3f}", ha="center")

mdape_plot = mdape_table * 100
x = np.arange(len(band_names))
width = 0.35
axes[1].bar(x - width / 2, mdape_plot["Random Forest"], width,
            label="Random Forest", color=colors[0])
axes[1].bar(x + width / 2, mdape_plot["LightGBM"], width,
            label="LightGBM", color=colors[1])
axes[1].set_title("MdAPE by House Price Band")
axes[1].set_ylabel("MdAPE (%)")
axes[1].set_xticks(x)
axes[1].set_xticklabels(band_names, rotation=25, ha="right")
axes[1].legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "week8_evaluation_results.png", dpi=160)
plt.close()

# %% [markdown]
# ## Final result

# %%
display(Image(filename=OUTPUT_DIR / "week8_evaluation_results.png"))

# %% [markdown]
# ## Conclusion
#
# LightGBM has the higher overall R2, but Random Forest has the lower overall MAPE
# and MdAPE. Random Forest also has the lower MdAPE in every price band. The lowest
# typical percentage errors are in the middle price bands, while both models have
# their highest MdAPE for homes over $2 million.
