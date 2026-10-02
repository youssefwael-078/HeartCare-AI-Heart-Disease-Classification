"""
predict.py
----------
Loads a saved model + scaler and runs inference on a
single patient's feature vector.
"""

import os
import json
import numpy as np
import joblib

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

TARGET_LABELS = {0: "No Disease — Healthy", 1: "Heart Disease Detected"}
TARGET_COLORS = {0: "success",              1: "danger"}


def load_model(model_key):
    """Load a persisted model from the models/ directory."""
    path = os.path.join(MODEL_DIR, f"{model_key}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model '{model_key}' not found. Please train it first."
        )
    return joblib.load(path)


def load_scaler():
    """Load the StandardScaler saved during preprocessing."""
    path = os.path.join(MODEL_DIR, "scaler.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError("Scaler not found. Please train a model first.")
    return joblib.load(path)


def predict(model_key, feature_values: list):
    """
    Run prediction for a single patient.

    Parameters
    ----------
    model_key      : str  — which trained model to use
    feature_values : list — raw (unscaled) feature values in correct order

    Returns
    -------
    dict with prediction, probability, label, and colour tag
    """
    model  = load_model(model_key)
    scaler = load_scaler()

    # Scale the input the same way training data was scaled
    X = np.array(feature_values, dtype=float).reshape(1, -1)
    X_scaled = scaler.transform(X)

    # Predict class and probability
    pred_class = int(model.predict(X_scaled)[0])
    probas     = model.predict_proba(X_scaled)[0]  # [prob_class0, prob_class1]

    confidence = round(float(probas[pred_class]) * 100, 1)

    return {
        "prediction":  pred_class,
        "label":       TARGET_LABELS[pred_class],
        "color":       TARGET_COLORS[pred_class],
        "confidence":  confidence,
        "prob_healthy": round(float(probas[0]) * 100, 1),
        "prob_disease": round(float(probas[1]) * 100, 1),
    }


def load_results(model_key):
    """Load the saved training results JSON for a model."""
    path = os.path.join(MODEL_DIR, f"{model_key}_results.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def get_best_params(model_key):
    """Return best hyperparameters found by GridSearchCV."""
    results = load_results(model_key)
    if results is None:
        return None
    return results.get("best_params", {})
