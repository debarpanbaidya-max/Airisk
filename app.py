# app.py
from flask import Flask, render_template, jsonify, request, send_file
import pandas as pd
import json
import os
import joblib
# IMPORTANT: You must run train_model.py and explainability.py first!
# This backend assumes the following files exist in the same directory:
# - engineered_features.csv
# - risk_prediction_model.joblib
# - model_metrics.json
# - clinician_friendly_explanations.csv
# - cm_uncalibrated.png
# - calibration_curve.png
# - feature_importance.png
# - global_feature_importance_bar.png
# - detailed_feature_impact_beeswarm.png
app = Flask(__name__, template_folder='.', static_folder='.')
# === Global Data Caching (Efficient for a small app) ===
# Load the data once when the server starts to avoid I/O on every request
try:
    patient_df = pd.read_csv("engineered_features.csv")
    explanations_df = pd.read_csv("clinician_friendly_explanations.csv")
    model = joblib.load('risk_prediction_model.joblib')
    with open("model_metrics.json") as f:
        model_metrics = json.load(f)
    print("✅ All required data and model files loaded successfully.")
except FileNotFoundError as e:
    print(f"❌ Startup Error: Missing file - {e}")
    print("Please ensure you have run 'train_model.py' and 'explainability.py' first.")
    exit()
# === Routes ===
@app.route("/")
def dashboard():
    """Serves the main dashboard HTML page."""
    return render_template("dashboard.html")
@app.route("/model_metrics")
def get_model_metrics():
    """Returns the pre-calculated model metrics as JSON."""
    return jsonify(model_metrics)
@app.route("/confusion_matrix")
def get_confusion_matrix():
    """Returns the confusion matrix image."""
    return send_file("cm_uncalibrated.png", mimetype='image/png')
@app.route("/calibration_curve")
def get_calibration_curve():
    """Returns the calibration curve image."""
    return send_file("calibration_curve.png", mimetype='image/png')
    
@app.route("/feature_importance")
def get_feature_importance():
    """Returns the feature importance image."""
    return send_file("feature_importance.png", mimetype='image/png')

# New routes for additional images
@app.route("/global_feature_importance_bar")
def get_global_feature_importance_bar():
    """Returns the global feature importance bar chart image."""
    return send_file("global_feature_importance_bar.png", mimetype='image/png')

@app.route("/detailed_feature_impact_beeswarm")
def get_detailed_feature_impact_beeswarm():
    """Returns the detailed feature impact beeswarm plot image."""
    return send_file("detailed_feature_impact_beeswarm.png", mimetype='image/png')

@app.route("/patients")
def get_patient_ids():
    """Returns a list of all patient IDs."""
    patient_ids = sorted(patient_df["patient_id"].unique().tolist())
    return jsonify(patient_ids)
# Change the route to accept a string
@app.route("/explanation/<string:patient_id>")
def get_explanation(patient_id):
    # Now you can directly search for the string format in your DataFrame
    patient_row = explanations_df[explanations_df["patient_id"] == patient_id]
    if patient_row.empty:
        return jsonify({"error": "Patient explanation not found"}), 404
    row_dict = patient_row.iloc[0].to_dict()
    return jsonify({
        "score": row_dict["predicted_risk_score"],
        "category": row_dict["risk_category"],
        "summary": row_dict["summary"]
    })
if __name__ == "__main__":
    app.run(debug=True)