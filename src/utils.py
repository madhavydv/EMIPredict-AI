"""Shared utilities: feature lists and sklearn ColumnTransformer builder."""
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.feature_engineering import ENGINEERED_FEATURE_NAMES

RAW_NUMERIC_FEATURES = [
    "age", "years_of_employment", "monthly_salary", "monthly_rent",
    "family_size", "dependents", "school_fees", "college_fees",
    "travel_expenses", "groceries_utilities", "other_monthly_expenses",
    "current_emi_amount", "credit_score", "bank_balance", "emergency_fund",
    "requested_amount", "requested_tenure",
]

CATEGORICAL_FEATURES = [
    "gender", "marital_status", "education", "employment_type",
    "company_type", "house_type", "existing_loans", "emi_scenario",
]

NUMERIC_FEATURES = RAW_NUMERIC_FEATURES + ENGINEERED_FEATURE_NAMES
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

CLASSIFICATION_TARGET = "emi_eligibility"
REGRESSION_TARGET = "max_monthly_emi"

CLASS_LABELS = ["Eligible", "High_Risk", "Not_Eligible"]


def build_preprocessor():
    """ColumnTransformer: scale numeric, one-hot encode categorical."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
