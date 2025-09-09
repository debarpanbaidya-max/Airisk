import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import joblib
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split

# Set a style for better plot visuals
plt.style.use('seaborn-v0_8-whitegrid')

# --- 1. Load Data and Trained Model ---
try:
    df = pd.read_csv('engineered_features.csv')
    print("✅ Successfully loaded 'engineered_features.csv'")
except FileNotFoundError:
    print("❌ Error: 'engineered_features.csv' not found. Please run the feature engineering script.")
    exit()

try:
    model = joblib.load('risk_prediction_model.joblib')
    print("✅ Successfully loaded 'risk_prediction_model.joblib'")
except FileNotFoundError:
    print("❌ Error: 'risk_prediction_model.joblib' not found. Please run the model training script.")
    exit()

# Define features and target
X = df.drop(['patient_id', 'date', 'deteriorated_in_90_days'], axis=1)
y = df['deteriorated_in_90_days']

# Split data for consistency
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- 2. Generate or Load SHAP Values ---
shap_file = "shap_values.pkl"
explainer = shap.TreeExplainer(model.calibrated_classifiers_[0].estimator)

if os.path.exists(shap_file):
    shap_values = joblib.load(shap_file)
    print("✅ Loaded SHAP values from 'shap_values.pkl'")
else:
    print("⏳ Computing SHAP values (this may take a while)...")
    shap_values = explainer.shap_values(X_test)
    joblib.dump(shap_values, shap_file)
    print("✅ SHAP values computed and saved to 'shap_values.pkl'")

# --- 3. Global Feature Importance Plots ---
print("\n--- Plotting Global Feature Importance ---")
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.title("Global Feature Importance (Bar)")
plt.tight_layout()
plt.show()

shap.summary_plot(shap_values, X_test, show=False)
plt.title("Detailed Feature Impact (Beeswarm)")
plt.tight_layout()
plt.show()

# --- 4. Local Explanation for a Single High-Risk Patient ---
try:
    high_risk_index = np.where(model.predict(X_test) == 1)[0][0]
    print(f"\n--- Local Explanation for High-Risk Patient (ID: {df.loc[X_test.index[high_risk_index], 'patient_id']}) ---")
    
    force_plot = shap.force_plot(
        explainer.expected_value, 
        shap_values[high_risk_index, :], 
        X_test.iloc[high_risk_index, :]
    )
    print("Displaying SHAP Force Plot. Best viewed in Jupyter.")
    print(force_plot)

except IndexError:
    print("❌ No high-risk patients found in the test set.")
except Exception as e:
    print(f"❌ Error generating local explanation: {e}")

# --- 5. Clinician-Friendly Explanations ---
print("\n--- Creating Clinician-Friendly Explanations ---")

def get_risk_category(score):
    if score < 0.3:
        return 'Low'
    elif score < 0.7:
        return 'Medium'
    else:
        return 'High'

feature_aliases = {
    'systolic_bp_mean_60D': 'Systolic BP (60-day Avg)',
    'systolic_bp_std_60D': 'Systolic BP (60-day Std Dev)',
    'systolic_bp_trend_60D': 'Systolic BP (60-day Trend)',
    'diastolic_bp_mean_60D': 'Diastolic BP (60-day Avg)',
    'diastolic_bp_std_60D': 'Diastolic BP (60-day Std Dev)',
    'diastolic_bp_trend_60D': 'Diastolic BP (60-day Trend)',
    'glucose_mean_60D': 'Glucose (60-day Avg)',
    'glucose_std_60D': 'Glucose (60-day Std Dev)',
    'glucose_trend_60D': 'Glucose (60-day Trend)',
    'hba1c_mean_60D': 'HbA1c (60-day Avg)',
    'hba1c_std_60D': 'HbA1c (60-day Std Dev)',
    'hba1c_trend_60D': 'HbA1c (60-day Trend)',
    'cholesterol_mean_60D': 'Cholesterol (60-day Avg)',
    'cholesterol_std_60D': 'Cholesterol (60-day Std Dev)',
    'cholesterol_trend_60D': 'Cholesterol (60-day Trend)',
    'weight_mean_60D': 'Weight (60-day Avg)',
    'weight_std_60D': 'Weight (60-day Std Dev)',
    'weight_trend_60D': 'Weight (60-day Trend)',
    'sleep_duration_mean_60D': 'Sleep Duration (60-day Avg)',
    'sleep_duration_std_60D': 'Sleep Duration (60-day Std Dev)',
    'sleep_duration_trend_60D': 'Sleep Duration (60-day Trend)',
    'steps_mean_60D': 'Steps (60-day Avg)',
    'steps_std_60D': 'Steps (60-day Std Dev)',
    'steps_trend_60D': 'Steps (60-day Trend)',
    'med_adherence_mean_60D': 'Med Adherence (60-day Avg)',
    'med_adherence_std_60D': 'Med Adherence (60-day Std Dev)',
    'med_adherence_trend_60D': 'Med Adherence (60-day Trend)',
}

summary_data = []

for i in range(len(X_test)):
    patient_id = df.loc[X_test.index[i], 'patient_id']
    score = model.predict_proba(X_test.iloc[[i]])[0][1]
    category = get_risk_category(score)

    shap_row = shap_values[i]
    top_indices = np.argsort(np.abs(shap_row))[-3:][::-1]

    top_features = []
    for idx in top_indices:
        feat = X_test.columns[idx]
        alias = feature_aliases.get(feat, feat)
        value = X_test.iloc[i, idx]
        direction = 'raising' if shap_row[idx] > 0 else 'lowering'
        top_features.append(f"{alias} ({value:.2f}) is {direction} the risk.")

    narrative_summary = f"Patient {patient_id} has a {category} risk score ({score:.2f}). "
    narrative_summary += "The primary risk drivers are: " + " and ".join(top_features) + "."

    summary_data.append({
        "patient_id": patient_id,
        "predicted_risk_score": round(score, 4),
        "risk_category": category,
        "summary": narrative_summary
    })

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv("clinician_friendly_explanations.csv", index=False)

print("\n✅ Clinician-friendly explanations saved as 'clinician_friendly_explanations.csv'.")
print("\n✅ All explainability tasks completed successfully!")
