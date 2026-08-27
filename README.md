# California Property Close-Price Prediction

This project predicts the final close price of single-family homes in
California. Over the course of the internship, I cleaned the CRMLS sales data,
tested several regression models, added new property and location features, and
built a small Streamlit app for making predictions.

The target variable is `ClosePrice`. I only used listings where:

- `PropertyType = Residential`
- `PropertySubType = SingleFamilyResidence`

## Data and time split

The model-ready data covers May 2025 through May 2026. I used May 2025 through
March 2026 for model development, April 2026 for model selection, and May 2026
as the final test month. Keeping the split in time order makes the evaluation
closer to a real prediction task and prevents future sales from leaking into
the training data.

The CRMLS source files are not included in GitHub because they are large and
contain licensed listing data. To rerun the project, place the two prepared
files here:

```text
data/week3_drive/crmls_sfr_train_X12_2025-05_to_2026-04.csv
data/week3_drive/crmls_sfr_test_2026-05.csv
```

## What I did

### Cleaning and preprocessing

I removed records with invalid prices or clearly invalid property values,
checked for duplicate listings, and made sure the same `ListingKey` did not
appear in both training and test data. Missing numeric values were imputed,
numeric features were scaled where needed, and categorical features were
one-hot encoded. These preprocessing steps were fitted on the training data
only.

### Feature engineering

The main features include living area, bedrooms, bathrooms, lot size, location,
HOA fee, stories, garage spaces, city, and ZIP code. I also created property age,
the bedroom-to-bathroom ratio, and seasonal month features.

For a more detailed geographic layer, I matched property coordinates to the
California school-district boundary data. This added elementary, high-school,
and unified school-district features.

### Models

I tested four model types:

- Linear Regression
- Decision Tree
- Random Forest
- LightGBM

I used April 2026 to compare settings and kept May 2026 untouched until the
final evaluation.

## Results

The table below shows the results on the May 2026 test set.

| Model | R² | MAPE | MdAPE |
|---|---:|---:|---:|
| Linear Regression | 0.8145 | 23.57% | 15.91% |
| Decision Tree | 0.8236 | 15.64% | 10.49% |
| Random Forest with Week 6 features | 0.8669 | **13.15%** | **8.29%** |
| LightGBM | **0.8758** | 14.70% | 10.42% |

LightGBM had the highest R², but Random Forest had the better MAPE and MdAPE.
Since I wanted the app to have a lower typical percentage error, I chose Random
Forest as the final app model.

The Random Forest performed best for homes in the `$500K-$750K` range, where
its MdAPE was about 6.41%. Error was higher for homes over `$2M`, with a MdAPE of
about 13.39%.

## Streamlit prediction app

The Week 9 app lets a user enter four main values: living area, bedrooms,
bathrooms, and lot size. Optional fields include city, ZIP code, year built,
coordinates, HOA fee, stories, and garage spaces.

The app prepares the features, loads the saved Random Forest pipeline, and
returns an estimated close price.

To run it, use Python 3.9 or newer and enter these commands from the project
folder:

```bash
python -m pip install -r 09_prediction_app/requirements-week9.txt
python -m streamlit run 09_prediction_app/app.py
```

If the browser does not open automatically, go to `http://localhost:8501`.

The app needs this locally generated model file:

```text
outputs/week6_feature_engineering/random_forest_week6.joblib
```

If it is missing, generate it by running Week 6 first:

```bash
python -m pip install -r 06_feature_engineering/requirements-week6.txt
python 06_feature_engineering/06_feature_engineering.py
```

The model file is about 214 MB, so it is not stored in GitHub.

## Running the full analysis

After adding the two prepared CSV files, run each stage from the project folder:

```bash
python 04_linear_model/04_linear_model.py
python 05_model_comparison/05_model_comparison.py
python 06_feature_engineering/06_feature_engineering.py
python 07_advanced_models/07_advanced_models.py
python 08_evaluation/08_evaluation.py
```

Install the requirements file for each stage before running it. The Week 6 code
downloads the public California school-district boundary file if it is not
already available locally.

## Project folders

```text
01_exploration.ipynb.py              Initial data exploration
02_numerical_deliverables/           Numerical checks and cleaning
03_baseline_model/                    Early baseline work
04_linear_model/                      Linear Regression baseline
05_model_comparison/                  Decision Tree and Random Forest
06_feature_engineering/               Property and school-district features
07_advanced_models/                   LightGBM model and tuning
08_evaluation/                        Overall and price-band evaluation
09_prediction_app/                    Streamlit application
```

## Limitations and next steps

The prediction is an estimate, not a formal appraisal. Results depend on the
quality of the historical data and may change as the housing market changes.
The model also has higher error in some price ranges, especially for luxury
homes.

A useful next step would be to add address geocoding so the app can identify a
property's school districts automatically. I would also add prediction
intervals, monitor performance on newer sales, and deploy the app with external
storage for the model file.
