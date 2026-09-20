"""
EMIPredict AI - Model Training & MLflow Tracking
===================================================
Trains >=3 classification models (EMI eligibility) and >=3 regression models
(max safe monthly EMI), logs params/metrics/artifacts to MLflow, compares
them, and registers + saves the best model of each type for the Streamlit app.

Usage:
    python src/train_models.py --train-sample 80000

Note on sample size: the sandbox this was built in has a single CPU core.
Training all 8 models (4 classification + 4 regression) on the full 400,000
rows is fully supported by this script (pass --train-sample 0 to disable
subsampling) but takes considerably longer. By default we fit models on a
stratified subsample of the training split (val/test stay full-size) which
keeps evaluation honest while keeping iteration fast. On a normal multi-core
machine, re-run with --train-sample 0 to use all ~283K training rows.
"""
import argparse
import json
import os
import time

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
)
from sklearn.preprocessing import label_binarize
from xgboost import XGBClassifier, XGBRegressor

from src.preprocessing import load_raw_data, clean_data, train_val_test_split
from src.feature_engineering import add_features
from src.utils import (
    build_preprocessor, NUMERIC_FEATURES, CATEGORICAL_FEATURES,
    CLASSIFICATION_TARGET, REGRESSION_TARGET, CLASS_LABELS,
)

MODELS_DIR = "models"
REPORTS_DIR = "reports"


def mape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def prep_data(train_sample: int):
    print("Loading & cleaning data...")
    raw = load_raw_data("data/emi_prediction_dataset.csv")
    clean = clean_data(raw, verbose=True)
    feat = add_features(clean)

    print("Splitting train/val/test (70/15/15, stratified on eligibility)...")
    train_df, val_df, test_df = train_val_test_split(
        feat, CLASSIFICATION_TARGET, test_size=0.15, val_size=0.15
    )

    if train_sample and train_sample > 0 and train_sample < len(train_df):
        from sklearn.model_selection import train_test_split as _tts
        train_df, _ = _tts(
            train_df, train_size=train_sample, random_state=42,
            stratify=train_df[CLASSIFICATION_TARGET]
        )
        train_df = train_df.reset_index(drop=True)

    print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")
    return train_df, val_df, test_df


def get_xy(df, target):
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[target]
    return X, y


def train_classification(train_df, val_df, test_df, experiment_name):
    from sklearn.preprocessing import LabelEncoder
    mlflow.set_experiment(experiment_name)
    X_train, y_train_raw = get_xy(train_df, CLASSIFICATION_TARGET)
    X_val, y_val_raw = get_xy(val_df, CLASSIFICATION_TARGET)
    X_test, y_test_raw = get_xy(test_df, CLASSIFICATION_TARGET)

    # XGBoost requires integer-encoded labels; encode consistently for all models
    le = LabelEncoder()
    le.fit(CLASS_LABELS)
    y_train = pd.Series(le.transform(y_train_raw), index=y_train_raw.index)
    y_val = pd.Series(le.transform(y_val_raw), index=y_val_raw.index)
    y_test = pd.Series(le.transform(y_test_raw), index=y_test_raw.index)

    models = {
        "Logistic_Regression": LogisticRegression(max_iter=500, n_jobs=-1),
        "Decision_Tree": DecisionTreeClassifier(max_depth=12, random_state=42),
        "Random_Forest": RandomForestClassifier(
            n_estimators=150, max_depth=16, n_jobs=-1, random_state=42, class_weight="balanced"
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8,
            colsample_bytree=0.8, eval_metric="mlogloss", random_state=42, n_jobs=-1
        ),
    }

    results = []
    best_model_name, best_score, best_pipeline = None, -1, None

    for name, clf in models.items():
        with mlflow.start_run(run_name=f"clf_{name}"):
            t0 = time.time()
            pipe = Pipeline_(clf)
            pipe.fit(X_train, y_train)
            fit_time = time.time() - t0

            y_pred = pipe.predict(X_val)
            y_proba = pipe.predict_proba(X_val)

            acc = accuracy_score(y_val, y_pred)
            prec = precision_score(y_val, y_pred, average="macro", zero_division=0)
            rec = recall_score(y_val, y_pred, average="macro", zero_division=0)
            f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
            try:
                y_val_bin = label_binarize(y_val, classes=pipe.named_steps["clf"].classes_)
                auc = roc_auc_score(y_val_bin, y_proba, average="macro", multi_class="ovr")
            except Exception:
                auc = np.nan

            mlflow.log_param("model_type", name)
            mlflow.log_param("train_rows", len(X_train))
            mlflow.log_params({k: v for k, v in clf.get_params().items() if isinstance(v, (int, float, str, bool)) or v is None})
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision_macro", prec)
            mlflow.log_metric("recall_macro", rec)
            mlflow.log_metric("f1_macro", f1)
            mlflow.log_metric("roc_auc_macro", 0 if np.isnan(auc) else auc)
            mlflow.log_metric("fit_time_sec", fit_time)
            mlflow.sklearn.log_model(pipe, "model", serialization_format="pickle")

            print(f"[CLF] {name}: acc={acc:.4f} f1={f1:.4f} auc={auc:.4f} ({fit_time:.1f}s)")
            results.append({
                "model": name, "accuracy": acc, "precision_macro": prec,
                "recall_macro": rec, "f1_macro": f1, "roc_auc_macro": auc,
                "fit_time_sec": fit_time,
            })

            if acc > best_score:
                best_score = acc
                best_model_name = name
                best_pipeline = pipe

    # Final test-set evaluation of the best model
    y_test_pred = best_pipeline.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average="macro", zero_division=0)
    print(f"\n>>> BEST CLASSIFIER: {best_model_name} | val_acc={best_score:.4f} | test_acc={test_acc:.4f} test_f1={test_f1:.4f}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(best_pipeline, os.path.join(MODELS_DIR, "best_classifier.pkl"))
    joblib.dump(le, os.path.join(MODELS_DIR, "label_encoder.pkl"))

    results_df = pd.DataFrame(results).sort_values("accuracy", ascending=False)
    results_df.to_csv(os.path.join(REPORTS_DIR, "classification_model_comparison.csv"), index=False)

    with open(os.path.join(MODELS_DIR, "classifier_meta.json"), "w") as f:
        json.dump({
            "best_model": best_model_name,
            "val_accuracy": best_score,
            "test_accuracy": test_acc,
            "test_f1_macro": test_f1,
            "classes": le.classes_.tolist(),
        }, f, indent=2)

    return results_df, best_model_name


def train_regression(train_df, val_df, test_df, experiment_name):
    mlflow.set_experiment(experiment_name)
    X_train, y_train = get_xy(train_df, REGRESSION_TARGET)
    X_val, y_val = get_xy(val_df, REGRESSION_TARGET)
    X_test, y_test = get_xy(test_df, REGRESSION_TARGET)

    models = {
        "Linear_Regression": LinearRegression(),
        "Decision_Tree": DecisionTreeRegressor(max_depth=12, random_state=42),
        "Random_Forest": RandomForestRegressor(n_estimators=150, max_depth=16, n_jobs=-1, random_state=42),
        "XGBoost": XGBRegressor(
            n_estimators=250, max_depth=6, learning_rate=0.08, subsample=0.8,
            colsample_bytree=0.8, random_state=42, n_jobs=-1
        ),
    }

    results = []
    best_model_name, best_score, best_pipeline = None, np.inf, None

    for name, reg in models.items():
        with mlflow.start_run(run_name=f"reg_{name}"):
            t0 = time.time()
            pipe = Pipeline_(reg)
            pipe.fit(X_train, y_train)
            fit_time = time.time() - t0

            y_pred = pipe.predict(X_val)
            rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
            mae = float(mean_absolute_error(y_val, y_pred))
            r2 = float(r2_score(y_val, y_pred))
            mp = mape(y_val, y_pred)

            mlflow.log_param("model_type", name)
            mlflow.log_param("train_rows", len(X_train))
            mlflow.log_params({k: v for k, v in reg.get_params().items() if isinstance(v, (int, float, str, bool)) or v is None})
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mape", mp)
            mlflow.log_metric("fit_time_sec", fit_time)
            mlflow.sklearn.log_model(pipe, "model", serialization_format="pickle")

            print(f"[REG] {name}: rmse={rmse:.1f} mae={mae:.1f} r2={r2:.4f} mape={mp:.2f}% ({fit_time:.1f}s)")
            results.append({
                "model": name, "rmse": rmse, "mae": mae, "r2": r2, "mape": mp,
                "fit_time_sec": fit_time,
            })

            if rmse < best_score:
                best_score = rmse
                best_model_name = name
                best_pipeline = pipe

    y_test_pred = best_pipeline.predict(X_test)
    test_rmse = float(np.sqrt(mean_squared_error(y_test, y_test_pred)))
    test_r2 = float(r2_score(y_test, y_test_pred))
    print(f"\n>>> BEST REGRESSOR: {best_model_name} | val_rmse={best_score:.1f} | test_rmse={test_rmse:.1f} test_r2={test_r2:.4f}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(best_pipeline, os.path.join(MODELS_DIR, "best_regressor.pkl"))

    results_df = pd.DataFrame(results).sort_values("rmse")
    results_df.to_csv(os.path.join(REPORTS_DIR, "regression_model_comparison.csv"), index=False)

    with open(os.path.join(MODELS_DIR, "regressor_meta.json"), "w") as f:
        json.dump({
            "best_model": best_model_name,
            "val_rmse": best_score,
            "test_rmse": test_rmse,
            "test_r2": test_r2,
        }, f, indent=2)

    return results_df, best_model_name


def Pipeline_(estimator):
    """Build a fresh preprocessing+model sklearn Pipeline (fresh transformer each call)."""
    from sklearn.pipeline import Pipeline
    return Pipeline([
        ("preprocess", build_preprocessor()),
        ("clf", estimator),
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-sample", type=int, default=80000,
                         help="Rows to subsample from the training split (0 = use full training split)")
    args = parser.parse_args()

    os.makedirs(REPORTS_DIR, exist_ok=True)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    train_df, val_df, test_df = prep_data(args.train_sample)

    print("\n========== CLASSIFICATION: EMI Eligibility ==========")
    clf_results, best_clf = train_classification(train_df, val_df, test_df, "EMI_Eligibility_Classification")

    print("\n========== REGRESSION: Max Monthly EMI ==========")
    reg_results, best_reg = train_regression(train_df, val_df, test_df, "Max_EMI_Regression")

    print("\n\n=== SUMMARY ===")
    print(clf_results.to_string(index=False))
    print()
    print(reg_results.to_string(index=False))
    print(f"\nBest classifier: {best_clf}")
    print(f"Best regressor: {best_reg}")


if __name__ == "__main__":
    main()
