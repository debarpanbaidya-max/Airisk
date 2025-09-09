# 🩺 PredictivePulse – AI-Driven Risk Prediction Engine for Chronic Care Patients

**PredictivePulse** is a powerful AI system designed to predict whether a chronic care patient is at risk of clinical deterioration within the next 90 days using the past 30–180 days of health data.  
Built for clinicians, optimized for explainability, and trained on data from **1,000+ real-world patient records**, PredictivePulse empowers healthcare teams to act *before it's too late*.

> 🚀 Built to make chronic care proactive, not reactive.

---

## 💡 Why This Matters

Chronic care patients often deteriorate silently — small spikes in blood pressure, missed medications, or abnormal labs are easy to miss in routine monitoring.  
**PredictivePulse** captures these patterns and delivers **personalized risk scores** + **clinician-friendly explanations**, enabling early intervention and better outcomes.

---

## 🧪 Model Overview

Our AI model processes time-series health data and outputs a calibrated risk score:

- 🔢 **Inputs**: Vitals, lab results, medication adherence, lifestyle logs (30–180 days per patient)
- 🧠 **Features**: Rolling averages, clinical deviation, volatility, adherence gaps
- 🤖 **Model**: XGBoost with isotonic calibration
- 📊 **Performance on 1000-patient dataset**:
  - **AUROC**: 0.988
  - **AUPRC**: 0.982
  - **Brier Score**: 0.0021

Each prediction is powered by **SHAP explainability**, ensuring clinical trust and transparency.

---

## 🔍 Explainability That Matters

We use **SHAP (SHapley Additive exPlanations)** to answer:

- **Globally**: What are the most important factors across all patients?
- **Locally**: Why is *this* patient at high risk?

Visualizations clearly highlight risk drivers (e.g., *"consistently elevated BP over past 30 days"*) so that healthcare providers don’t just get a number — they get context.

---

## 🖥️ Clinician Dashboard Features

Built with **TailwindCSS + Chart.js**, our dashboard delivers clarity and control:

- 📊 **Cohort View**: All patients with risk categories (Low / Medium / High)
- 👤 **Patient View**: Individual trends, vitals, lab graphs, SHAP-based explanations
- 📁 **Export**: One-click download of patient summaries (PDF & CSV)
- 📈 **Model Visuals**: AUROC, AUPRC, Calibration Plot, Confusion Matrix

> Designed with clinicians in mind — no ML knowledge needed to act on insights.

---

## 🛠️ Tech Stack

| Layer            | Tools & Libraries                       |
|------------------|-----------------------------------------|
| **Backend**      | Python, Flask, XGBoost, SHAP            |
| **Frontend**     | HTML, TailwindCSS, Chart.js             |
| **Visualization**| Matplotlib, SHAP, ReportLab             |
| **Data Handling**| Pandas, NumPy                           |

---

## 📂 Repository Structure

```bash
backend/          # Flask APIs, model logic, SHAP generation
frontend/         # Clinician dashboard (HTML + JS)
models/           # Trained model + calibration files
plots/            # Visuals: SHAP, confusion matrix, calibration
data/             # Sample (sanitized) patient input
