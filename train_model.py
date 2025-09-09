import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import joblib
import warnings
import json
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, brier_score_loss
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

# Ignore the specific matplotlib warning about 'tight_layout' with mixed plots.
warnings.filterwarnings("ignore", category=UserWarning)

# --- 1. Load Data ---
try:
    df = pd.read_csv('engineered_features.csv')
    print("✅ Successfully loaded 'engineered_features.csv'")
except FileNotFoundError:
    print("❌ Error: 'engineered_features.csv' not found. Please run the feature engineering script first.")
    exit()

X = df.drop(['patient_id', 'date', 'deteriorated_in_90_days'], axis=1)
y = df['deteriorated_in_90_days']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"\nTraining shape: {X_train.shape} | Testing shape: {X_test.shape}")
print(f"Target distribution in training set:\n{y_train.value_counts(normalize=True)}")

# --- 2. Hyperparameter Tuning using Randomized Search ---
# This is a more efficient way to find good parameters for a hackathon
print("\n--- Starting Hyperparameter Tuning with Randomized Search ---")
# Define the parameter space to search
param_dist = {
    'n_estimators': [100, 200, 300, 400],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5, 6, 7],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
}

# The `scale_pos_weight` helps handle imbalanced classes
scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)

xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1
)

# Use StratifiedKFold for cross-validation on the imbalanced data
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_dist,
    n_iter=20, # Number of parameter settings that are sampled
    scoring='roc_auc',
    cv=cv,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

print("Tuning complete.")
print(f"Best hyperparameters found: {random_search.best_params_}")
best_model = random_search.best_estimator_

# --- 3. Evaluate the Best Uncalibrated Model ---
y_pred_proba = best_model.predict_proba(X_test)[:, 1]
y_pred = best_model.predict(X_test)

print("\n--- Best Uncalibrated Model Evaluation ---")
print(f"AUROC   : {roc_auc_score(y_test, y_pred_proba):.4f}")
print(f"AUPRC   : {average_precision_score(y_test, y_pred_proba):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=[0, 1]).plot(cmap=plt.cm.Blues)
plt.title("Uncalibrated Confusion Matrix")
plt.tight_layout()
plt.savefig("cm_uncalibrated.png")  # ✅ save to file
plt.close()


# --- 4. Plot Feature Importance ---
print("\n--- Plotting Global Feature Importance ---")
feature_importances = best_model.get_booster().get_score(importance_type='gain')
sorted_importances = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
top_features = sorted_importances[:20] # Plot top 20 features
x_values = [item[0] for item in top_features]
y_values = [item[1] for item in top_features]

plt.figure(figsize=(12, 8))
plt.barh(x_values, y_values, color='skyblue')
plt.xlabel("Importance (Gain)")
plt.title("XGBoost Feature Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# --- 5. Calibrate the Best Model ---
print("\n--- Applying Isotonic Calibration ---")
calibrated_model = CalibratedClassifierCV(best_model, method='isotonic', cv='prefit')
calibrated_model.fit(X_train, y_train)

# --- 6. Evaluate Calibrated Model ---
cal_probs = calibrated_model.predict_proba(X_test)[:, 1]
cal_preds = calibrated_model.predict(X_test)

print("\n--- Calibrated Model Evaluation ---")
print(f"AUROC   : {roc_auc_score(y_test, cal_probs):.4f}")
print(f"AUPRC   : {average_precision_score(y_test, cal_probs):.4f}")
print(f"Brier Score: {brier_score_loss(y_test, cal_probs):.4f}")
print("\nClassification Report:\n", classification_report(y_test, cal_preds))

# Calibration Plot
print("\n--- Calibration Plot ---")
prob_true_cal, prob_pred_cal = calibration_curve(y_test, cal_probs, n_bins=10)
prob_true_uncal, prob_pred_uncal = calibration_curve(y_test, y_pred_proba, n_bins=10)

plt.figure(figsize=(8, 6))
plt.plot(prob_pred_uncal, prob_true_uncal, marker='o', label='Uncalibrated')
plt.plot(prob_pred_cal, prob_true_cal, marker='o', label='Isotonic Calibrated')
plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
plt.title('Calibration Curve')
plt.xlabel('Predicted Probability')
plt.ylabel('True Probability')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("calibration_curve.png")  # ✅ save to file
plt.close()
plt.figure(figsize=(12, 8))
plt.barh(x_values, y_values, color='skyblue')
plt.xlabel("Importance (Gain)")
plt.title("XGBoost Feature Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("feature_importance.png")  # ✅ optional
plt.close()


# --- 7. Save the Calibrated Model ---
model_filename = 'risk_prediction_model.joblib'
joblib.dump(calibrated_model, model_filename)
print(f"\n✅ Final calibrated model saved as '{model_filename}'")
metrics = {
    "auroc": roc_auc_score(y_test, cal_probs),
    "auprc": average_precision_score(y_test, cal_probs),
    "brier": brier_score_loss(y_test, cal_probs)
}

# Save JSON for API
with open("model_metrics.json", "w") as f_json:
    json.dump(metrics, f_json, indent=4)

# Save CSV for download
pd.DataFrame([metrics]).to_csv("model_metrics.csv", index=False)

print("✅ Model metrics saved to 'model_metrics.json' and 'model_metrics.csv'")