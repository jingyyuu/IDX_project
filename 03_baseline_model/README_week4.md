# Week 4 - Baseline Model

## Data source

This work uses the team model-ready files in Google Drive:

`IDX Summer Intern / outputs / crmls_sfr_train_X12_2025-05_to_2026-04.csv`

`IDX Summer Intern / outputs / crmls_sfr_test_2026-05.csv`

The training set contains 129,745 rows from May 2025 through April 2026. The held-out test set contains 12,012 rows from May 2026.

## Method

Five Linear Regression feature bundles were compared. All use the Week 3 scaled and frequency-encoded fields. `ListPrice`, `OriginalListPrice`, `ClosePrice_to_ListPrice_ratio`, identifiers, and date/split fields were excluded to avoid target leakage and preserve the intended off-market valuation use case.

## Results

| Model | Features | Train R2 | Test R2 | Test MAPE | Test MdAPE |
|---|---:|---:|---:|---:|---:|
| M5 expanded non-leaky | 14 | 0.508 | 0.537 | 49.3% | 32.8% |
| M4 structure + location | 9 | 0.496 | 0.529 | 50.5% | 33.8% |
| M3 structure + bathrooms | 5 | 0.458 | 0.479 | 53.1% | 36.1% |
| M2 basic structure | 4 | 0.442 | 0.465 | 53.6% | 35.4% |
| M1 size only | 1 | 0.385 | 0.402 | 59.7% | 41.7% |

## Decision

M5 is the Week 4 baseline because it has the highest May 2026 test R2 (0.537) and the lowest test MAPE and MdAPE. It should be treated as a benchmark rather than a production valuation model. Week 5 should compare Decision Tree and Random Forest models using the same train/test files.

## Files

- `03_baseline_model.ipynb`: executed Week 4 deliverable
- `03_baseline_model.py`: paired source for easier version control
- `../outputs/week4_baseline/baseline_model_comparison.csv`: complete metrics table
- `../outputs/week4_baseline/baseline_test_predictions.csv`: selected-model test predictions
- `../outputs/week4_baseline/linear_regression_baseline.joblib`: fitted selected model and feature list
- `../outputs/week4_baseline/actual_vs_predicted.png`: diagnostic plot
- `../outputs/week4_baseline/run_metadata.json`: reproducibility metadata
