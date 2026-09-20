"""
EMIPredict AI - Data Preprocessing Module
===========================================
Handles data loading, cleaning, quality assessment and validation for the
400K-record EMI dataset. The raw data contains realistic data-quality issues:
  - Numeric columns stored as text with corrupted suffixes (e.g. "38.0.0")
  - Inconsistent categorical casing (e.g. "Male", "MALE", "M", "female")
  - Missing values in several numeric/categorical columns
  - Out-of-range values (credit_score > 850)
This module fixes all of the above deterministically and is reused by both
the training pipeline and the Streamlit app (for single-record inference).
"""

import re
import numpy as np
import pandas as pd

NUMERIC_TEXT_COLS = ["age", "monthly_salary", "bank_balance"]

GENDER_MAP = {
    "male": "Male", "m": "Male", "MALE".lower(): "Male",
    "female": "Female", "f": "Female",
}

NUMERIC_COLS = [
    "age", "years_of_employment", "monthly_salary", "monthly_rent",
    "family_size", "dependents", "school_fees", "college_fees",
    "travel_expenses", "groceries_utilities", "other_monthly_expenses",
    "current_emi_amount", "credit_score", "bank_balance", "emergency_fund",
    "requested_amount", "requested_tenure",
]

CATEGORICAL_COLS = [
    "gender", "marital_status", "education", "employment_type",
    "company_type", "house_type", "existing_loans", "emi_scenario",
]

CREDIT_SCORE_RANGE = (300, 850)
AGE_RANGE = (18, 75)


def _clean_numeric_text(series: pd.Series) -> pd.Series:
    """Fix corrupted numeric strings like '38.0.0' -> 38.0, strip stray chars."""
    def fix(v):
        if pd.isna(v):
            return np.nan
        s = str(v).strip()
        # collapse repeated trailing '.0' artifacts e.g. '23400.0.0.0' -> '23400.0'
        s = re.sub(r"(\.0){2,}$", ".0", s)
        s = re.sub(r"[^0-9.\-]", "", s)
        try:
            return float(s)
        except ValueError:
            return np.nan
    return series.apply(fix)


def load_raw_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def clean_data(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Full data-quality pass: fix types, normalize categories, clip outliers,
    impute missing values, drop duplicates."""
    df = df.copy()
    report = {}

    report["rows_in"] = len(df)

    # 1. Duplicates
    dupes = df.duplicated().sum()
    df = df.drop_duplicates()
    report["duplicates_removed"] = int(dupes)

    # 2. Fix numeric columns that were loaded as corrupted text
    for col in NUMERIC_TEXT_COLS:
        if col in df.columns:
            df[col] = _clean_numeric_text(df[col])

    # 3. Coerce all numeric columns properly
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 4. Normalize categorical casing/spacing
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    if "gender" in df.columns:
        df["gender"] = df["gender"].str.lower().map(GENDER_MAP).fillna(df["gender"])
        df["gender"] = df["gender"].replace({"nan": np.nan})

    # 5. Fix out-of-range / physically impossible values -> treat as missing then impute
    if "credit_score" in df.columns:
        mask = (df["credit_score"] < CREDIT_SCORE_RANGE[0]) | (df["credit_score"] > CREDIT_SCORE_RANGE[1])
        report["credit_score_out_of_range"] = int(mask.sum())
        df.loc[mask, "credit_score"] = np.nan

    if "age" in df.columns:
        mask = (df["age"] < AGE_RANGE[0]) | (df["age"] > AGE_RANGE[1])
        report["age_out_of_range"] = int(mask.sum())
        df.loc[mask, "age"] = np.nan

    # negative values are impossible for these financial fields
    non_negative_cols = [
        "monthly_salary", "monthly_rent", "school_fees", "college_fees",
        "travel_expenses", "groceries_utilities", "other_monthly_expenses",
        "current_emi_amount", "bank_balance", "emergency_fund",
        "requested_amount", "requested_tenure",
    ]
    for col in non_negative_cols:
        if col in df.columns:
            df.loc[df[col] < 0, col] = np.nan

    # 6. Missing value imputation
    missing_before = int(df.isna().sum().sum())
    for col in NUMERIC_COLS:
        if col in df.columns and df[col].isna().any():
            median_by_scenario = df.groupby("emi_scenario")[col].transform("median")
            df[col] = df[col].fillna(median_by_scenario)
            df[col] = df[col].fillna(df[col].median())

    for col in CATEGORICAL_COLS:
        if col in df.columns and df[col].isna().any():
            mode_val = df[col].mode(dropna=True)
            df[col] = df[col].fillna(mode_val.iloc[0] if len(mode_val) else "Unknown")

    report["missing_values_imputed"] = missing_before

    # 7. Final target sanity: drop rows with missing target (should be ~0)
    for tgt in ["emi_eligibility", "max_monthly_emi"]:
        if tgt in df.columns:
            df = df[df[tgt].notna()]

    df = df.reset_index(drop=True)
    report["rows_out"] = len(df)

    if verbose:
        print("=== Data Cleaning Report ===")
        for k, v in report.items():
            print(f"  {k}: {v}")

    df.attrs["cleaning_report"] = report
    return df


def train_val_test_split(df: pd.DataFrame, target_col: str, test_size=0.15,
                          val_size=0.15, random_state=42, stratify=True):
    from sklearn.model_selection import train_test_split
    strat = df[target_col] if stratify else None
    train_df, temp_df = train_test_split(
        df, test_size=(test_size + val_size), random_state=random_state, stratify=strat
    )
    strat2 = temp_df[target_col] if stratify else None
    rel_test = test_size / (test_size + val_size)
    val_df, test_df = train_test_split(
        temp_df, test_size=rel_test, random_state=random_state, stratify=strat2
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


if __name__ == "__main__":
    raw = load_raw_data("data/emi_prediction_dataset.csv")
    print("Raw shape:", raw.shape)
    clean = clean_data(raw)
    print("Clean shape:", clean.shape)
    clean.to_csv("data/emi_dataset_clean.csv", index=False)
    print("Saved cleaned dataset to data/emi_dataset_clean.csv")
