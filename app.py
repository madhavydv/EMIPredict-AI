"""
EMIPredict AI - Intelligent Financial Risk Assessment Platform
==================================================================
Multi-page Streamlit application for EMI eligibility classification and
maximum EMI regression, with MLflow experiment visibility and a simple
data-management (CRUD) console. Deployable to Streamlit Cloud as-is.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from src.preprocessing import clean_data
from src.feature_engineering import add_features
from src.utils import NUMERIC_FEATURES, CATEGORICAL_FEATURES, RAW_NUMERIC_FEATURES

st.set_page_config(
    page_title="EMIPredict AI",
    page_icon="\U0001F4B0",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Global style
# ----------------------------------------------------------------------------
st.markdown("""
<style>
:root {
    --brand-navy: #0F2038;
    --brand-teal: #0E7C7B;
    --brand-gold: #C9A24B;
}
.main-header {
    font-size: 2.1rem; font-weight: 700; color: var(--brand-navy);
    margin-bottom: 0;
}
.sub-header { color: #55677A; font-size: 1.0rem; margin-top: 0.2rem; }
.metric-card {
    background: #F7F9FB; border: 1px solid #E4E9EF; border-radius: 10px;
    padding: 1rem 1.2rem; text-align: center;
}
div[data-testid="stMetricValue"] { color: var(--brand-navy); }
.eligible-badge { background:#E4F6EE; color:#137A4C; padding:.4rem .9rem; border-radius:8px; font-weight:600; }
.highrisk-badge { background:#FFF4E0; color:#9A6300; padding:.4rem .9rem; border-radius:8px; font-weight:600; }
.notel-badge   { background:#FDEAEA; color:#B23434; padding:.4rem .9rem; border-radius:8px; font-weight:600; }
</style>
""", unsafe_allow_html=True)

DATA_PATH = "data/emi_prediction_dataset.csv"
DATA_FEAT_PATH = "data/emi_dataset_features.csv"
MODELS_DIR = "models"


# ----------------------------------------------------------------------------
# Cached loaders
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading trained models...")
def load_models():
    clf = joblib.load(os.path.join(MODELS_DIR, "best_classifier.pkl"))
    reg = joblib.load(os.path.join(MODELS_DIR, "best_regressor.pkl"))
    le = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
    with open(os.path.join(MODELS_DIR, "classifier_meta.json")) as f:
        clf_meta = json.load(f)
    with open(os.path.join(MODELS_DIR, "regressor_meta.json")) as f:
        reg_meta = json.load(f)
    return clf, reg, le, clf_meta, reg_meta


@st.cache_data(show_spinner="Loading dataset...")
def load_feature_dataset():
    if os.path.exists(DATA_FEAT_PATH):
        return pd.read_csv(DATA_FEAT_PATH)
    raw = pd.read_csv(DATA_PATH, low_memory=False)
    clean = clean_data(raw, verbose=False)
    return add_features(clean)


def models_available():
    required = ["best_classifier.pkl", "best_regressor.pkl", "label_encoder.pkl",
                "classifier_meta.json", "regressor_meta.json"]
    return all(os.path.exists(os.path.join(MODELS_DIR, f)) for f in required)


def build_single_row(inputs: dict) -> pd.DataFrame:
    """Turn a form's raw inputs into a cleaned + feature-engineered 1-row df."""
    df = pd.DataFrame([inputs])
    df = clean_data(df, verbose=False)
    df = add_features(df)
    return df


# ----------------------------------------------------------------------------
# Sidebar navigation
# ----------------------------------------------------------------------------
st.sidebar.markdown("## \U0001F3E6 EMIPredict AI")
st.sidebar.caption("Intelligent Financial Risk Assessment Platform")
page = st.sidebar.radio(
    "Navigate",
    ["\U0001F3E0 Home", "\U0001F4CA Data Exploration", "\U0001F3AF Predict EMI Eligibility",
     "\U0001F4C8 Model Performance (MLflow)", "\U0001F5C4\uFE0F Data Management"],
    label_visibility="collapsed",
)
st.sidebar.divider()
st.sidebar.caption("Built with Python, scikit-learn, XGBoost, MLflow & Streamlit")

# ============================================================================
# PAGE: HOME
# ============================================================================
if page.endswith("Home"):
    st.markdown('<p class="main-header">EMIPredict AI</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Intelligent Financial Risk Assessment Platform for EMI Lending</p>', unsafe_allow_html=True)
    st.write("")

    df = load_feature_dataset()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Financial Records", f"{len(df):,}")
    c2.metric("Input Variables", "22")
    c3.metric("EMI Scenarios", df["emi_scenario"].nunique())
    c4.metric("Eligible Rate", f"{(df['emi_eligibility']=='Eligible').mean()*100:.1f}%")

    st.write("")
    left, right = st.columns([1.3, 1])
    with left:
        st.subheader("What this platform does")
        st.markdown("""
- **Classification** — predicts EMI eligibility (`Eligible` / `High_Risk` / `Not_Eligible`) for a loan applicant.
- **Regression** — estimates the **maximum safe monthly EMI** an applicant can afford.
- **MLflow tracking** — every model (4 classifiers, 4 regressors) is logged with parameters, metrics and artifacts, with the best of each registered for production use.
- **Data management** — browse, filter, add and export financial records.
        """)
        st.info("Use the sidebar to explore the data, run a live prediction, inspect model performance, or manage records.")
    with right:
        if models_available():
            _, _, _, clf_meta, reg_meta = load_models()
            st.subheader("Deployed Models")
            st.success(f"**Classifier:** {clf_meta['best_model']}  \nTest accuracy: **{clf_meta['test_accuracy']*100:.2f}%**  ·  F1 (macro): **{clf_meta['test_f1_macro']:.3f}**")
            st.success(f"**Regressor:** {reg_meta['best_model']}  \nTest RMSE: **₹{reg_meta['test_rmse']:.0f}**  ·  R²: **{reg_meta['test_r2']:.3f}**")
        else:
            st.warning("No trained models found yet. Run `python src/train_models.py` first.")

    st.divider()
    st.subheader("EMI Scenario Coverage")
    scen = df["emi_scenario"].value_counts().reset_index()
    scen.columns = ["Scenario", "Records"]
    fig = px.bar(scen, x="Scenario", y="Records", color="Scenario",
                 color_discrete_sequence=px.colors.sequential.Teal)
    fig.update_layout(showlegend=False, height=380)
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# PAGE: DATA EXPLORATION (EDA)
# ============================================================================
elif page.endswith("Data Exploration"):
    st.markdown('<p class="main-header">Data Exploration</p>', unsafe_allow_html=True)
    st.caption("Interactive exploratory analysis of the 400K-record EMI dataset")

    df = load_feature_dataset()

    with st.expander("Dataset preview & summary statistics", expanded=False):
        st.dataframe(df.head(50), use_container_width=True)
        st.write(df.describe().T)

    tab1, tab2, tab3, tab4 = st.tabs(["Eligibility Patterns", "Financial Ratios", "Correlations", "By Scenario"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            counts = df["emi_eligibility"].value_counts().reset_index()
            counts.columns = ["Class", "Count"]
            fig = px.pie(counts, names="Class", values="Count", hole=0.45,
                         color="Class",
                         color_discrete_map={"Eligible": "#0E7C7B", "High_Risk": "#C9A24B", "Not_Eligible": "#B23434"})
            fig.update_layout(title="EMI Eligibility Distribution")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.box(df, x="emi_eligibility", y="credit_score", color="emi_eligibility",
                         color_discrete_map={"Eligible": "#0E7C7B", "High_Risk": "#C9A24B", "Not_Eligible": "#B23434"})
            fig.update_layout(title="Credit Score by Eligibility", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(df, x="debt_to_income_ratio", nbins=60, color="emi_eligibility",
                                barmode="overlay", opacity=0.6, range_x=[0, 1.5])
            fig.update_layout(title="Debt-to-Income Ratio Distribution")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.scatter(df.sample(min(5000, len(df)), random_state=1), x="monthly_salary",
                              y="max_monthly_emi", color="emi_eligibility", opacity=0.5,
                              color_discrete_map={"Eligible": "#0E7C7B", "High_Risk": "#C9A24B", "Not_Eligible": "#B23434"})
            fig.update_layout(title="Salary vs Max Monthly EMI")
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        key_cols = ["monthly_salary", "credit_score", "debt_to_income_ratio",
                    "expense_to_income_ratio", "disposable_income", "risk_score",
                    "current_emi_amount", "bank_balance", "emergency_fund", "max_monthly_emi"]
        corr = df[key_cols].corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
        fig.update_layout(title="Correlation Heatmap — Key Financial Variables", height=550)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.box(df, x="emi_scenario", y="max_monthly_emi", color="emi_scenario")
            fig.update_layout(title="Max EMI by Scenario", showlegend=False)
            fig.update_xaxes(tickangle=25)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            ct = pd.crosstab(df["emi_scenario"], df["emi_eligibility"], normalize="index") * 100
            ct = ct.reset_index().melt(id_vars="emi_scenario", var_name="Eligibility", value_name="Percent")
            fig = px.bar(ct, x="emi_scenario", y="Percent", color="Eligibility", barmode="stack",
                         color_discrete_map={"Eligible": "#0E7C7B", "High_Risk": "#C9A24B", "Not_Eligible": "#B23434"})
            fig.update_layout(title="Eligibility Rate by Scenario")
            fig.update_xaxes(tickangle=25)
            st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# PAGE: PREDICTION
# ============================================================================
elif page.endswith("Predict EMI Eligibility"):
    st.markdown('<p class="main-header">Real-Time EMI Prediction</p>', unsafe_allow_html=True)
    st.caption("Enter an applicant's profile to get instant eligibility and max-EMI predictions")

    if not models_available():
        st.error("No trained models found. Run `python src/train_models.py` first to generate `models/*.pkl`.")
        st.stop()

    clf, reg, le, clf_meta, reg_meta = load_models()

    with st.form("prediction_form"):
        st.subheader("Applicant Profile")
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", 18, 75, 32)
            gender = st.selectbox("Gender", ["Male", "Female"])
            marital_status = st.selectbox("Marital Status", ["Single", "Married"])
            education = st.selectbox("Education", ["High School", "Graduate", "Post Graduate", "Professional"])
            employment_type = st.selectbox("Employment Type", ["Private", "Government", "Self-employed"])
            company_type = st.selectbox("Company Type", ["MNC", "Mid-size", "Startup", "Government", "Small Business"])
            years_of_employment = st.number_input("Years of Employment", 0.0, 40.0, 4.0)
        with c2:
            monthly_salary = st.number_input("Monthly Salary (₹)", 15000, 200000, 45000, step=1000)
            house_type = st.selectbox("House Type", ["Rented", "Own", "Family"])
            monthly_rent = st.number_input("Monthly Rent (₹)", 0, 80000, 10000, step=500)
            family_size = st.number_input("Family Size", 1, 5, 3)
            dependents = st.number_input("Dependents", 0, 4, 1)
            existing_loans = st.selectbox("Existing Loans", ["No", "Yes"])
            current_emi_amount = st.number_input("Current EMI Amount (₹)", 0, 56300, 0, step=500)
        with c3:
            credit_score = st.number_input("Credit Score", 300, 850, 700)
            bank_balance = st.number_input("Bank Balance (₹)", 0, 900000, 150000, step=1000)
            emergency_fund = st.number_input("Emergency Fund (₹)", 0, 900000, 60000, step=1000)
            emi_scenario = st.selectbox("EMI Scenario", [
                "E-commerce Shopping EMI", "Home Appliances EMI", "Vehicle EMI",
                "Personal Loan EMI", "Education EMI"])
            requested_amount = st.number_input("Requested Amount (₹)", 10000, 1500000, 200000, step=5000)
            requested_tenure = st.number_input("Requested Tenure (months)", 3, 84, 24)

        st.markdown("**Monthly Expenses**")
        e1, e2, e3, e4 = st.columns(4)
        school_fees = e1.number_input("School Fees (₹)", 0, 15000, 0, step=500)
        college_fees = e2.number_input("College Fees (₹)", 0, 25000, 0, step=500)
        travel_expenses = e3.number_input("Travel Expenses (₹)", 0, 30300, 3000, step=500)
        groceries_utilities = e4.number_input("Groceries & Utilities (₹)", 0, 71200, 8000, step=500)
        other_monthly_expenses = st.number_input("Other Monthly Expenses (₹)", 0, 42900, 3000, step=500)

        submitted = st.form_submit_button("Run Prediction", type="primary", use_container_width=True)

    if submitted:
        inputs = dict(
            age=age, gender=gender, marital_status=marital_status, education=education,
            monthly_salary=monthly_salary, employment_type=employment_type,
            years_of_employment=years_of_employment, company_type=company_type,
            house_type=house_type, monthly_rent=monthly_rent, family_size=family_size,
            dependents=dependents, school_fees=school_fees, college_fees=college_fees,
            travel_expenses=travel_expenses, groceries_utilities=groceries_utilities,
            other_monthly_expenses=other_monthly_expenses, existing_loans=existing_loans,
            current_emi_amount=current_emi_amount, credit_score=credit_score,
            bank_balance=bank_balance, emergency_fund=emergency_fund,
            emi_scenario=emi_scenario, requested_amount=requested_amount,
            requested_tenure=requested_tenure,
        )
        row = build_single_row(inputs)
        X = row[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

        pred_class_idx = clf.predict(X)[0]
        pred_class = le.inverse_transform([pred_class_idx])[0]
        pred_proba = clf.predict_proba(X)[0]
        pred_emi = float(reg.predict(X)[0])

        st.divider()
        st.subheader("Prediction Results")
        r1, r2 = st.columns([1, 1.4])
        with r1:
            badge_class = {"Eligible": "eligible-badge", "High_Risk": "highrisk-badge", "Not_Eligible": "notel-badge"}[pred_class]
            st.markdown(f'<span class="{badge_class}">EMI Eligibility: {pred_class.replace("_"," ")}</span>', unsafe_allow_html=True)
            st.metric("Maximum Safe Monthly EMI", f"₹{pred_emi:,.0f}")
            est_new_emi = float(row["estimated_new_emi"].iloc[0])
            if est_new_emi > pred_emi:
                st.warning(f"Requested EMI (~₹{est_new_emi:,.0f}/mo) exceeds the recommended safe max — consider a longer tenure or lower amount.")
            else:
                st.success(f"Requested EMI (~₹{est_new_emi:,.0f}/mo) is within the recommended safe max.")
        with r2:
            proba_df = pd.DataFrame({"Class": le.classes_, "Probability": pred_proba})
            fig = px.bar(proba_df, x="Class", y="Probability", color="Class",
                         color_discrete_map={"Eligible": "#0E7C7B", "High_Risk": "#C9A24B", "Not_Eligible": "#B23434"},
                         range_y=[0, 1], text_auto=".1%")
            fig.update_layout(title="Class Probabilities", showlegend=False, height=320)
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("View computed financial ratios for this applicant"):
            ratio_cols = ["debt_to_income_ratio", "expense_to_income_ratio", "affordability_ratio",
                          "disposable_income", "risk_score", "emergency_fund_months"]
            st.dataframe(row[ratio_cols].T.rename(columns={0: "Value"}), use_container_width=True)


# ============================================================================
# PAGE: MODEL PERFORMANCE
# ============================================================================
elif page.endswith("Model Performance (MLflow)"):
    st.markdown('<p class="main-header">Model Performance & MLflow Tracking</p>', unsafe_allow_html=True)
    st.caption("Comparison across all trained models — logged via MLflow experiment tracking")

    if not os.path.exists("reports/classification_model_comparison.csv"):
        st.warning("No training reports found yet. Run `python src/train_models.py` first.")
        st.stop()

    clf_df = pd.read_csv("reports/classification_model_comparison.csv")
    reg_df = pd.read_csv("reports/regression_model_comparison.csv")

    tab1, tab2 = st.tabs(["Classification Models", "Regression Models"])

    with tab1:
        st.dataframe(clf_df.style.highlight_max(subset=["accuracy", "f1_macro", "roc_auc_macro"], color="#D7F3E5"),
                     use_container_width=True)
        fig = px.bar(clf_df.melt(id_vars="model", value_vars=["accuracy", "precision_macro", "recall_macro", "f1_macro"]),
                     x="model", y="value", color="variable", barmode="group")
        fig.update_layout(title="Classification Metrics by Model", yaxis_title="Score")
        st.plotly_chart(fig, use_container_width=True)
        best = clf_df.sort_values("accuracy", ascending=False).iloc[0]
        st.success(f"**Selected for production:** {best['model']} — accuracy {best['accuracy']*100:.2f}%, F1 (macro) {best['f1_macro']:.3f}, ROC-AUC {best['roc_auc_macro']:.3f}")

    with tab2:
        st.dataframe(reg_df.style.highlight_min(subset=["rmse", "mae", "mape"], color="#D7F3E5")
                              .highlight_max(subset=["r2"], color="#D7F3E5"),
                     use_container_width=True)
        fig = px.bar(reg_df, x="model", y="rmse", color="model", text_auto=".0f")
        fig.update_layout(title="RMSE by Model (lower is better)", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        best = reg_df.sort_values("rmse").iloc[0]
        st.success(f"**Selected for production:** {best['model']} — RMSE ₹{best['rmse']:.0f}, MAE ₹{best['mae']:.0f}, R² {best['r2']:.3f}")

    st.divider()
    st.subheader("MLflow Experiment Tracking")
    st.markdown("""
All 8 model runs (4 classification + 4 regression) are logged to MLflow with full parameters,
metrics and model artifacts, organized under two experiments: `EMI_Eligibility_Classification`
and `Max_EMI_Regression`. The best model of each type is saved to `models/` for this app and can
be promoted through the MLflow Model Registry.

To open the full interactive MLflow UI (run locally, not on Streamlit Cloud):
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
    """)


# ============================================================================
# PAGE: DATA MANAGEMENT (CRUD)
# ============================================================================
elif page.endswith("Data Management"):
    st.markdown('<p class="main-header">Data Management Console</p>', unsafe_allow_html=True)
    st.caption("Browse, filter, add and export financial records (session-scoped)")

    if "working_df" not in st.session_state:
        st.session_state.working_df = load_feature_dataset().copy()

    df = st.session_state.working_df

    tab1, tab2, tab3 = st.tabs(["Browse & Filter", "Add Record", "Export"])

    with tab1:
        c1, c2, c3 = st.columns(3)
        scen_filter = c1.multiselect("EMI Scenario", sorted(df["emi_scenario"].unique()))
        elig_filter = c2.multiselect("Eligibility", sorted(df["emi_eligibility"].unique()))
        min_salary = c3.number_input("Min Monthly Salary (₹)", 0, 200000, 0, step=5000)

        view = df.copy()
        if scen_filter:
            view = view[view["emi_scenario"].isin(scen_filter)]
        if elig_filter:
            view = view[view["emi_eligibility"].isin(elig_filter)]
        view = view[view["monthly_salary"] >= min_salary]

        st.caption(f"{len(view):,} records match your filters")
        st.dataframe(view.head(500), use_container_width=True, height=400)

    with tab2:
        st.write("Add a new applicant record to the working dataset (session-only, does not modify the source CSV).")
        with st.form("add_record"):
            c1, c2, c3 = st.columns(3)
            new_age = c1.number_input("Age", 18, 75, 30)
            new_salary = c2.number_input("Monthly Salary", 15000, 200000, 40000, step=1000)
            new_scenario = c3.selectbox("EMI Scenario", sorted(df["emi_scenario"].unique()))
            add_submit = st.form_submit_button("Add Record")
        if add_submit:
            new_row = df.iloc[[0]].copy()
            new_row["age"] = new_age
            new_row["monthly_salary"] = new_salary
            new_row["emi_scenario"] = new_scenario
            st.session_state.working_df = pd.concat([df, new_row], ignore_index=True)
            st.success("Record added to the working dataset for this session.")

    with tab3:
        csv = st.session_state.working_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download current working dataset as CSV", csv,
                            file_name="emi_dataset_export.csv", mime="text/csv",
                            use_container_width=True)
        if st.button("Reset working dataset to original", use_container_width=True):
            st.session_state.working_df = load_feature_dataset().copy()
            st.rerun()
