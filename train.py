"""
train.py
--------
Defines, trains, and evaluates three classification models:
  - Logistic Regression
  - Random Forest Classifier
  - Support Vector Machine

Uses GridSearchCV for hyperparameter tuning and cross-validation.
Supports manual single-param selection from the UI.
Saves the best model and all metrics for the Flask UI.
"""

import os
import json
import numpy as np
import joblib

from sklearn.linear_model  import LogisticRegression
from sklearn.ensemble      import RandomForestClassifier
from sklearn.svm           import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)

from app.preprocessing import load_data, preprocess

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ── Model catalogue ───────────────────────────────────────────────────────────
MODEL_CATALOGUE = {
    "logistic_regression": {
        "label":     "Logistic Regression",
        "estimator": LogisticRegression(max_iter=1000, random_state=42),
        "param_grid": {
            "C":       [0.01, 0.1, 1, 10],
            "solver":  ["lbfgs", "liblinear"],
            "penalty": ["l2"],
        },
        "ui_params": {
            "C":      {"label": "C (Regularisation)", "options": ["0.01", "0.1", "1", "10"]},
            "solver": {"label": "Solver",             "options": ["lbfgs", "liblinear"]},
        },
    },
    "random_forest": {
        "label":     "Random Forest",
        "estimator": RandomForestClassifier(random_state=42),
        "param_grid": {
            "n_estimators":      [50, 100, 200],
            "max_depth":         [None, 10, 20],
            "min_samples_split": [2, 5],
        },
        "ui_params": {
            "n_estimators":      {"label": "Trees (n_estimators)",   "options": ["50", "100", "200"]},
            "max_depth":         {"label": "Max Depth",              "options": ["None", "10", "20"]},
            "min_samples_split": {"label": "Min Samples Split",      "options": ["2", "5"]},
        },
    },
    "svm": {
        "label":     "Support Vector Machine",
        "estimator": SVC(probability=True, random_state=42),
        "param_grid": {
            "C":      [0.1, 1, 10],
            "kernel": ["rbf", "linear"],
            "gamma":  ["scale", "auto"],
        },
        "ui_params": {
            "C":      {"label": "C (Regularisation)", "options": ["0.1", "1", "10"]},
            "kernel": {"label": "Kernel",             "options": ["rbf", "linear"]},
            "gamma":  {"label": "Gamma",              "options": ["scale", "auto"]},
        },
    },
}


def _get_feature_importance(model, model_key, feature_names):
    """Extract and normalise feature importances for any supported model."""
    importances = None
    if model_key == "random_forest":
        importances = model.feature_importances_
    elif model_key == "logistic_regression":
        importances = np.abs(model.coef_[0])
    elif model_key == "svm" and hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])

    if importances is None:
        return []

    total = importances.sum()
    if total > 0:
        importances = importances / total

    pairs = sorted(zip(feature_names, importances.tolist()), key=lambda x: x[1], reverse=True)
    return [{"feature": f, "importance": round(imp * 100, 2)} for f, imp in pairs]


def _coerce_param(v):
    """Convert a string form value to the appropriate Python type."""
    if v == "None":
        return None
    try:
        return int(v)
    except (ValueError, TypeError):
        try:
            return float(v)
        except (ValueError, TypeError):
            return v


def train_model(model_key, custom_params=None, manual_single_params=None):
    """
    Train a model.

    Parameters
    ----------
    model_key            : str  — key from MODEL_CATALOGUE
    custom_params        : dict — optional param_grid override (multi-value GridSearch)
    manual_single_params : dict — optional single fixed values chosen from UI dropdowns;
                                  when provided, GridSearch is skipped entirely.
    Returns dict with all metrics.
    """
    if model_key not in MODEL_CATALOGUE:
        raise ValueError(f"Unknown model: {model_key}")

    df = load_data()
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess(df)

    catalogue_entry = MODEL_CATALOGUE[model_key]

    # Fresh estimator to avoid state pollution between calls
    if model_key == "logistic_regression":
        estimator = LogisticRegression(max_iter=1000, random_state=42)
    elif model_key == "random_forest":
        estimator = RandomForestClassifier(random_state=42)
    else:
        estimator = SVC(probability=True, random_state=42)

    if manual_single_params:
        coerced = {k: _coerce_param(v) for k, v in manual_single_params.items()}
        estimator.set_params(**coerced)
        estimator.fit(X_train, y_train)
        best_model  = estimator
        best_params = coerced
        mode_label  = "Manual (fixed parameters)"
    else:
        param_grid = custom_params if custom_params else catalogue_entry["param_grid"]
        grid_search = GridSearchCV(
            estimator  = estimator,
            param_grid = param_grid,
            cv         = 5,
            scoring    = "accuracy",
            n_jobs     = -1,
            verbose    = 0,
        )
        grid_search.fit(X_train, y_train)
        best_model  = grid_search.best_estimator_
        best_params = grid_search.best_params_
        mode_label  = "Auto (GridSearchCV — 5-fold CV)"

    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring="accuracy")
    y_pred    = best_model.predict(X_test)

    accuracy  = round(accuracy_score(y_test, y_pred)  * 100, 2)
    precision = round(precision_score(y_test, y_pred, zero_division=0) * 100, 2)
    recall    = round(recall_score(y_test, y_pred,    zero_division=0) * 100, 2)
    f1        = round(f1_score(y_test, y_pred,        zero_division=0) * 100, 2)
    cm        = confusion_matrix(y_test, y_pred).tolist()
    report    = classification_report(
        y_test, y_pred,
        target_names=["No Disease", "Heart Disease"],
        output_dict=True
    )
    feat_importance = _get_feature_importance(best_model, model_key, feature_names)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, os.path.join(MODEL_DIR, f"{model_key}.pkl"))

    results = {
        "model_key":             model_key,
        "model_label":           catalogue_entry["label"],
        "mode":                  mode_label,
        "best_params":           best_params,
        "cv_mean":               round(float(cv_scores.mean()) * 100, 2),
        "cv_std":                round(float(cv_scores.std())  * 100, 2),
        "accuracy":              accuracy,
        "precision":             precision,
        "recall":                recall,
        "f1":                    f1,
        "confusion_matrix":      cm,
        "classification_report": report,
        "feature_names":         feature_names,
        "feature_importance":    feat_importance,
    }

    with open(os.path.join(MODEL_DIR, f"{model_key}_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


def compare_all_models():
    comparison = []
    for key in MODEL_CATALOGUE:
        try:
            res = train_model(key)
            comparison.append({
                "model":     res["model_label"],
                "accuracy":  res["accuracy"],
                "precision": res["precision"],
                "recall":    res["recall"],
                "f1":        res["f1"],
                "cv_mean":   res["cv_mean"],
            })
        except Exception as e:
            comparison.append({"model": key, "error": str(e)})
    return comparison


if __name__ == "__main__":
    print("Training all models...")
    for row in compare_all_models():
        print(row)
