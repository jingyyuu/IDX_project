# Week 9 - Simple Prediction App

Week 9 turns the trained model into a small Streamlit app. A user enters the
living area, bedrooms, bathrooms, and lot size, and the app returns an estimated
close price for a California single-family residence. Optional location and
property fields can improve the information supplied to the model.

The app uses the Week 6 Random Forest pipeline because Week 8 showed that it had
the lower overall MAPE (13.1%) and MdAPE (8.3%). LightGBM had a slightly higher
R2, but it was not saved as a reusable model pipeline.

## Run the app

From the project folder:

```bash
python -m pip install -r 09_prediction_app/requirements-week9.txt
streamlit run 09_prediction_app/app.py
```

The app expects this locally generated model file:

```text
outputs/week6_feature_engineering/random_forest_week6.joblib
```

If it is missing, first run:

```bash
python 06_feature_engineering/06_feature_engineering.py
```

Model and source-data files remain outside Git because they are large. The app
is intended as an educational demo, not a formal real-estate appraisal.

## Deliverable

- `app.py`: Streamlit prediction interface
- `requirements-week9.txt`: packages required to launch the interface
- `README_week9.md`: setup, model-selection rationale, and run instructions
