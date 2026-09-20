"""
EMIPredict AI - Feature Engineering Module
=============================================
Builds derived financial ratios, risk scores and interaction features on top
of the cleaned dataset. Shared by the training pipeline and the Streamlit app.
"""

import numpy as np
import pandas as pd

EPS = 1.0  # avoids divide-by-zero


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Total monthly obligations / expenses
    df["total_monthly_expenses"] = (
        df["school_fees"] + df["college_fees"] + df["travel_expenses"] +
        df["groceries_utilities"] + df["other_monthly_expenses"] + df["monthly_rent"]
    )
    df["total_monthly_obligations"] = df["total_monthly_expenses"] + df["current_emi_amount"]

    # Core financial ratios
    df["debt_to_income_ratio"] = df["current_emi_amount"] / (df["monthly_salary"] + EPS)
    df["expense_to_income_ratio"] = df["total_monthly_expenses"] / (df["monthly_salary"] + EPS)
    df["obligation_to_income_ratio"] = df["total_monthly_obligations"] / (df["monthly_salary"] + EPS)

    # Disposable income & affordability
    df["disposable_income"] = df["monthly_salary"] - df["total_monthly_obligations"]
    df["disposable_income"] = df["disposable_income"].clip(lower=0)
    df["affordability_ratio"] = df["disposable_income"] / (df["monthly_salary"] + EPS)

    # Requested EMI (simple amortization-free estimate) vs disposable income
    df["estimated_new_emi"] = df["requested_amount"] / (df["requested_tenure"] + EPS)
    df["new_emi_to_disposable_ratio"] = df["estimated_new_emi"] / (df["disposable_income"] + EPS)
    df["new_emi_to_income_ratio"] = df["estimated_new_emi"] / (df["monthly_salary"] + EPS)

    # Savings & liquidity
    df["savings_to_income_ratio"] = df["bank_balance"] / (df["monthly_salary"] * 12 + EPS)
    df["emergency_fund_months"] = df["emergency_fund"] / (df["total_monthly_expenses"] + EPS)

    # Household burden
    df["dependents_ratio"] = df["dependents"] / (df["family_size"] + EPS)
    df["per_capita_income"] = df["monthly_salary"] / (df["family_size"] + EPS)

    # Employment stability score (0-1)
    df["employment_stability"] = np.clip(df["years_of_employment"] / 15.0, 0, 1)

    # Credit risk score (0-100, higher = safer) blending credit score, DTI, stability
    credit_norm = (df["credit_score"] - 300) / (850 - 300)
    dti_penalty = np.clip(1 - df["debt_to_income_ratio"], 0, 1)
    df["risk_score"] = (
        0.5 * credit_norm.clip(0, 1) +
        0.3 * dti_penalty +
        0.2 * df["employment_stability"]
    ) * 100

    # Existing loan flag as numeric
    df["has_existing_loans"] = (df["existing_loans"].astype(str).str.lower() == "yes").astype(int)

    # Interaction: income x employment stability (earning power proxy)
    df["earning_power"] = df["monthly_salary"] * df["employment_stability"]

    # Replace any inf/nan produced by ratios
    df = df.replace([np.inf, -np.inf], np.nan)
    ratio_cols = [c for c in df.columns if "ratio" in c or "score" in c or c.startswith("estimated")
                  or c in ("disposable_income", "emergency_fund_months", "per_capita_income", "earning_power")]
    for c in ratio_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())

    return df


ENGINEERED_FEATURE_NAMES = [
    "total_monthly_expenses", "total_monthly_obligations", "debt_to_income_ratio",
    "expense_to_income_ratio", "obligation_to_income_ratio", "disposable_income",
    "affordability_ratio", "estimated_new_emi", "new_emi_to_disposable_ratio",
    "new_emi_to_income_ratio", "savings_to_income_ratio", "emergency_fund_months",
    "dependents_ratio", "per_capita_income", "employment_stability", "risk_score",
    "has_existing_loans", "earning_power",
]

if __name__ == "__main__":
    df = pd.read_csv("data/emi_dataset_clean.csv")
    df = add_features(df)
    df.to_csv("data/emi_dataset_features.csv", index=False)
    print("Feature-engineered dataset saved. Shape:", df.shape)
    print("New columns:", ENGINEERED_FEATURE_NAMES)
