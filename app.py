"""
app.py  — Flask entry point
Routes:
  /             Home / Dataset overview
  /train        Model training page
  /train/run    AJAX: train model, return JSON metrics
  /predict      Prediction form
  /predict/run  AJAX: run inference, return JSON
  /best-params  Best hyperparameters comparison page
"""

import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, request, jsonify

from app.preprocessing import load_data, get_dataset_summary, FEATURE_INFO, TARGET_CLASSES
from app.train         import train_model, MODEL_CATALOGUE
from app.predict       import predict, load_results, get_best_params

app = Flask(__name__)
app.secret_key = "heartcare_academic_2024"
os.makedirs("models", exist_ok=True)


# ── HOME ──────────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    df      = load_data()
    summary = get_dataset_summary(df)
    sample  = df.head(6).to_dict(orient="records")
    return render_template("home.html",
        summary=summary, sample=sample,
        feature_info=FEATURE_INFO, target_classes=TARGET_CLASSES)


# ── TRAIN ─────────────────────────────────────────────────────────────────────
@app.route("/train")
def train_page():
    saved = {}
    for key in MODEL_CATALOGUE:
        res = load_results(key)
        if res:
            saved[key] = res
    return render_template("train.html", models=MODEL_CATALOGUE, saved=saved)


@app.route("/train/run", methods=["POST"])
def train_run():
    model_key = request.form.get("model_key", "random_forest")
    mode      = request.form.get("mode", "auto")   # "auto" | "manual"

    manual_params = None
    if mode == "manual":
        catalogue_entry = MODEL_CATALOGUE.get(model_key, {})
        ui_params = catalogue_entry.get("ui_params", {})
        manual_params = {}
        for param_key in ui_params:
            val = request.form.get(f"param_{param_key}")
            if val is not None and val != "":
                manual_params[param_key] = val

    try:
        results = train_model(model_key, manual_single_params=manual_params if mode == "manual" else None)
        return jsonify({"status": "success", "results": results})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ── PREDICT ───────────────────────────────────────────────────────────────────
@app.route("/predict")
def predict_page():
    return render_template("predict.html", feature_info=FEATURE_INFO, models=MODEL_CATALOGUE)


@app.route("/predict/run", methods=["POST"])
def predict_run():
    model_key = request.form.get("model_key", "random_forest")
    try:
        feature_order = list(FEATURE_INFO.keys())
        values = [float(request.form.get(f, 0)) for f in feature_order]
        result = predict(model_key, values)
        return jsonify({"status": "success", "result": result})
    except FileNotFoundError as e:
        return jsonify({"status": "error", "message": str(e)}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ── BEST PARAMS ───────────────────────────────────────────────────────────────
@app.route("/best-params")
def best_params_page():
    all_params  = {}
    all_results = {}
    for key in MODEL_CATALOGUE:
        params  = get_best_params(key)
        results = load_results(key)
        if params:
            all_params[key]  = params
        if results:
            all_results[key] = results
    return render_template("best_params.html",
        models=MODEL_CATALOGUE, all_params=all_params, all_results=all_results)


if __name__ == "__main__":
    print("\n🫀  HeartCare AI — Academic ML Project")
    print("    Running on http://127.0.0.1:5000\n")
    app.run(debug=True, port=5000)
