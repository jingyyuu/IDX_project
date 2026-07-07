# Rebecca — Week 3 Numerical Data

## What I worked on

For my Week 3 part, I focused on the numerical columns in our California property dataset. The goal was to check whether the main numeric variables look reasonable before we use the data for modeling.

I used the monthly CRMLSSold files from **2025.01 to 2026.05**. After applying the required filter:

- `PropertyType = Residential`
- `PropertySubType = SingleFamilyResidence`

we had **181,482 rows** to work with.

## Columns I checked

I mainly looked at these numerical variables:

- `ClosePrice`
- `LivingArea`
- `BedroomsTotal`
- `BathroomsTotalInteger`
- `LotSizeSquareFeet`
- `YearBuilt`
- `DaysOnMarket`
- `GarageSpaces`
- `ParkingTotal`
- `Latitude`
- `Longitude`

I also checked the full list of numerical columns, but some columns, like `ListingKey`, `ListingKeyNumeric`, and `StreetNumberNumeric`, should not be used for modeling because they are more like IDs or address-related fields.

## What I found

Overall, most of the main numerical columns look usable, but there are some values we should clean or at least flag before modeling.

A few examples:

- `ClosePrice` has a small number of impossible or suspicious values, such as prices less than or equal to 0.
- `LivingArea` is mostly complete, but there are some 0 values and some very large values.
- `BedroomsTotal` and `BathroomsTotalInteger` look mostly reasonable, with a few extreme cases.
- `LotSizeSquareFeet` is very skewed, which makes sense for property data, but there are some very large values.
- Some latitude and longitude values are outside a reasonable California range, including a few invalid coordinates.

## How I suggest handling them

I separated the issues into two groups.

### 1. Values we should probably remove

These look like clear data errors:

- `ClosePrice <= 0`
- `ClosePrice / ListPrice < 0.05` or `> 5`
- `YearBuilt < 1800` or `> 2026`
- `DaysOnMarket < 0`
- Latitude/longitude outside a rough California range

### 2. Values we should flag and review

These are unusual, but they might still be real properties, so I would not delete them automatically:

- `LivingArea <= 0` or `> 20,000`
- `BedroomsTotal > 15`
- `BathroomsTotalInteger > 15`
- `LotSizeSquareFeet <= 0` or `> 5,000,000`
- `GarageSpaces > 10`
- `ParkingTotal > 20`

For example, a very expensive house or a very large lot could still be a real listing, so I think it is better to flag those rows first instead of removing them right away.

## Scaling note

Some numerical features may need scaling later, especially for Linear Regression. But scaling should happen **after** the train/test split.

The scaler should be fitted on the training set only, then applied to both the train and test sets. This avoids data leakage.

Also, `ClosePrice` is our target variable, so it should not be used as a feature.

## Files included

- `week3_numerical_summary_all_columns.csv` — summary for all numerical columns
- `week3_key_numerical_summary.csv` — summary for the main numerical variables
- `week3_numerical_outlier_rules_and_counts.csv` — rules and counts for flagged values
- `week3_numerical_flagged_rows_sample.csv` — sample rows with numerical flags
- `week3_numerical_feature_recommendations.csv` — which numerical columns to keep, drop, or review
- `week3_numerical_cleaning_code.py` — code that can be added to the final preprocessing notebook
- `week3_numerical_distribution_plots.pdf` — distribution plots for the main numerical variables
- `week3_after_numeric_strict_cleaning.csv.gz` — dataset after removing strict numerical errors
