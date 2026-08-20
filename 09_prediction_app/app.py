"""Week 9 Streamlit app for California home close-price prediction."""

from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "week6_feature_engineering"
    / "random_forest_week6.joblib"
)
MODEL_MDAPE = 0.0829


@st.cache_resource
def load_model():
    """Load the fitted Week 6 preprocessing-and-model pipeline once."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "The Week 6 model was not found. Run "
            "06_feature_engineering/06_feature_engineering.py first."
        )
    return joblib.load(MODEL_PATH)


def optional_text(value: str):
    """Convert an empty optional text input to a missing value."""
    cleaned = value.strip()
    return cleaned if cleaned else np.nan


def build_property_row(
    living_area: float,
    bedrooms: int,
    bathrooms: int,
    lot_size: float,
    postal_code: str = "",
    city: str = "",
    year_built: int | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    association_fee: float | None = None,
    stories: float | None = None,
    garage_spaces: float | None = None,
) -> pd.DataFrame:
    """Create the single-row feature frame expected by the Week 6 pipeline."""
    prediction_date = date.today()
    property_age = (
        prediction_date.year - year_built
        if year_built and 1800 <= year_built <= prediction_date.year
        else np.nan
    )
    bed_bath_ratio = bedrooms / bathrooms if bathrooms > 0 else np.nan

    return pd.DataFrame(
        [
            {
                "LivingArea": living_area,
                "BedroomsTotal": bedrooms,
                "BathroomsTotalInteger": bathrooms,
                "LotSizeSquareFeet": lot_size,
                "Latitude": latitude if latitude is not None else np.nan,
                "Longitude": longitude if longitude is not None else np.nan,
                "AssociationFee": (
                    association_fee if association_fee is not None else np.nan
                ),
                "Stories": stories if stories is not None else np.nan,
                "GarageSpaces": garage_spaces if garage_spaces is not None else np.nan,
                "PropertyAge": property_age,
                "SaleMonthSin": np.sin(2 * np.pi * prediction_date.month / 12),
                "SaleMonthCos": np.cos(2 * np.pi * prediction_date.month / 12),
                "BedBathRatio": bed_bath_ratio,
                "PostalCode": optional_text(postal_code),
                "City": optional_text(city),
                "PropertySubType": "SingleFamilyResidence",
                # The saved pipeline imputes unknown district values. A future version
                # can spatially join these fields when an address/geocoder is available.
                "SchoolDistrictElementary": np.nan,
                "SchoolDistrictHigh": np.nan,
                "SchoolDistrictUnified": np.nan,
            }
        ]
    )


st.set_page_config(page_title="California Home Price Estimator", page_icon="🏠")
st.title("California Home Close-Price Estimator")
st.write(
    "Enter the characteristics of a California single-family home to estimate "
    "its final sale price."
)

with st.form("property_form"):
    col1, col2 = st.columns(2)
    with col1:
        living_area = st.number_input(
            "Living area (sq ft)", min_value=100, max_value=30_000, value=1_800
        )
        bedrooms = st.number_input(
            "Bedrooms", min_value=0, max_value=20, value=3, step=1
        )
    with col2:
        bathrooms = st.number_input(
            "Bathrooms", min_value=0, max_value=20, value=2, step=1
        )
        lot_size = st.number_input(
            "Lot size (sq ft)", min_value=0, max_value=5_000_000, value=6_000
        )

    with st.expander("Optional property details"):
        city = st.text_input("City", placeholder="Los Angeles")
        postal_code = st.text_input("ZIP code", placeholder="90001")
        year_built_text = st.text_input("Year built", placeholder="1995")
        latitude_text = st.text_input("Latitude", placeholder="34.0522")
        longitude_text = st.text_input("Longitude", placeholder="-118.2437")
        association_fee_text = st.text_input("Monthly HOA fee ($)", placeholder="0")
        stories_text = st.text_input("Stories", placeholder="1")
        garage_spaces_text = st.text_input("Garage spaces", placeholder="2")

    submitted = st.form_submit_button("Estimate close price", type="primary")


def optional_number(value: str, field_name: str, cast=float):
    """Parse an optional numeric form value and show a friendly validation error."""
    if not value.strip():
        return None
    try:
        return cast(value)
    except ValueError:
        st.error(f"{field_name} must be a number.")
        st.stop()


if submitted:
    year_built = optional_number(year_built_text, "Year built", int)
    latitude = optional_number(latitude_text, "Latitude")
    longitude = optional_number(longitude_text, "Longitude")
    association_fee = optional_number(association_fee_text, "Monthly HOA fee")
    stories = optional_number(stories_text, "Stories")
    garage_spaces = optional_number(garage_spaces_text, "Garage spaces")

    if year_built is not None and not 1800 <= year_built <= date.today().year:
        st.error(f"Year built must be between 1800 and {date.today().year}.")
        st.stop()
    if latitude is not None and not 32 <= latitude <= 43:
        st.error("Latitude should be between 32 and 43 for a California property.")
        st.stop()
    if longitude is not None and not -125 <= longitude <= -113:
        st.error("Longitude should be between -125 and -113 for a California property.")
        st.stop()

    property_row = build_property_row(
        living_area=living_area,
        bedrooms=bedrooms,
        bathrooms=bathrooms,
        lot_size=lot_size,
        postal_code=postal_code,
        city=city,
        year_built=year_built,
        latitude=latitude,
        longitude=longitude,
        association_fee=association_fee,
        stories=stories,
        garage_spaces=garage_spaces,
    )

    try:
        prediction = max(0.0, float(load_model().predict(property_row)[0]))
    except FileNotFoundError as error:
        st.error(str(error))
        st.stop()

    st.success("Estimated close price")
    st.metric("Predicted price", f"${prediction:,.0f}")
    st.caption(
        f"The selected model's median absolute percentage error on the May 2026 "
        f"test set was {MODEL_MDAPE:.1%}. This is historical model performance, "
        "not a confidence interval for this home."
    )

st.divider()
st.caption(
    "Educational estimate only. The model was trained on historical CRMLS sales "
    "of California single-family residences and is not an appraisal or financial advice."
)
