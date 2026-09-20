# Exploratory Data Analysis Report — EMIPredict AI

## Dataset Overview
- **404,800** raw records, **27** columns (22 input features + 3 loan-application
  fields + 2 targets), spanning **5 EMI scenarios** (~81K records each).
- Data-quality issues found and fixed by `src/preprocessing.py`:
  - `age`, `monthly_salary`, `bank_balance` were loaded as corrupted text
    (e.g. `"38.0.0"`, `"23400.0.0.0"`) — fixed with regex normalization.
  - `gender` had 8 inconsistent variants (`Male`, `MALE`, `M`, `female`, …) —
    normalized to `Male` / `Female`.
  - `credit_score` had **4,776** values outside the valid 300–850 range
    (up to 1200) — treated as missing and imputed.
  - **16,803** missing values across `education`, `monthly_rent`,
    `credit_score`, `bank_balance`, `emergency_fund` — imputed using
    per-scenario medians (numeric) / mode (categorical).
  - No duplicate rows were found.

## Target Distributions
| Class | Records | % |
|---|---|---|
| Not_Eligible | 312,868 | 77.3% |
| Eligible | 74,444 | 18.4% |
| High_Risk | 17,488 | 4.3% |

The classes are **imbalanced** (≈77% Not_Eligible), which is realistic for a
lending population and is handled in training via `class_weight="balanced"`
for tree models and macro-averaged metrics (precision/recall/F1) rather than
raw accuracy alone.

`max_monthly_emi` (regression target) ranges 500–91,040 INR with a right
skew typical of income-driven variables.

## Eligibility by EMI Scenario
| Scenario | Eligible % | High Risk % | Not Eligible % |
|---|---|---|---|
| Home Appliances EMI | 26.0% | 5.2% | 68.8% |
| E-commerce Shopping EMI | 26.3% | 4.9% | 68.7% |
| Education EMI | 17.7% | 4.8% | 77.5% |
| Personal Loan EMI | 11.3% | 3.4% | 85.2% |
| Vehicle EMI | 10.6% | 3.3% | 86.2% |

**Insight:** small-ticket, short-tenure products (appliances, e-commerce)
have roughly 2.5x higher eligibility rates than large-ticket, long-tenure
products (personal loans, vehicles) — consistent with affordability being
driven primarily by the *size and tenure* of the requested EMI relative to
income, not just the applicant's income level alone.

## Credit & Risk Patterns
| Eligibility | Mean Credit Score | Mean Debt-to-Income Ratio |
|---|---|---|
| Eligible | 725.2 | 0.031 |
| High_Risk | 715.7 | 0.043 |
| Not_Eligible | 693.6 | 0.098 |

Both credit score and existing debt burden move in the expected direction:
eligible applicants carry roughly **3x lower existing debt-to-income** than
rejected applicants, while credit score differences are more modest — this
is why the engineered `debt_to_income_ratio` and `risk_score` features add
meaningfully more separating power than credit score alone (confirmed by
feature importance in the trained tree-based models).

## Correlation with Maximum Safe Monthly EMI
| Feature | Correlation |
|---|---|
| Disposable income | 0.398 |
| Monthly salary | 0.379 |
| Risk score | 0.311 |
| Credit score | 0.299 |
| Debt-to-income ratio | −0.289 |

**Insight:** disposable income (salary minus obligations) is a stronger
predictor of affordable EMI than raw salary — validating the engineered
`disposable_income` and `affordability_ratio` features used in the
regression models.

## Visualizations
See `reports/eda_plots/` for: eligibility distribution, eligibility-by-scenario
stacked bars, correlation heatmap, max-EMI boxplots by scenario, credit-score
by eligibility, and applicant age distribution. All charts are also available
interactively in the Streamlit app's **Data Exploration** page.

## Business Recommendations
1. **Scenario-aware underwriting:** apply different risk thresholds for
   short-tenure/small-ticket vs. long-tenure/large-ticket products rather
   than a single blanket policy.
2. **Prioritize DTI and disposable income** over raw salary or credit score
   alone when setting EMI limits — they carry more predictive signal in this
   population.
3. **High_Risk segment (4.3% of applicants)** is a candidate for risk-based
   pricing (higher interest rate) rather than outright rejection, recovering
   revenue currently left on the table by a binary approve/reject policy.
