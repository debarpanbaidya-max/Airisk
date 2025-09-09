import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# --- 1. Configuration ---
NUM_PATIENTS = 1000
START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2023, 12, 31)
DETERIORATION_WINDOW = 90 # Days to look ahead for the target label

# --- 2. Generate Base Data for All Patients ---
date_range = pd.to_datetime(pd.date_range(start=START_DATE, end=END_DATE))
patient_ids = [f'patient_{i+1}' for i in range(NUM_PATIENTS)]

# Create the full data grid
df = pd.DataFrame({
    'date': np.tile(date_range, NUM_PATIENTS),
    'patient_id': np.repeat(patient_ids, len(date_range))
})

# --- 3. Simulate Health Metrics ---

# Set a random seed for reproducibility
np.random.seed(42)

# Generate baseline characteristics for each patient
patient_characteristics = {}
for pid in patient_ids:
    patient_characteristics[pid] = {
        'base_systolic_bp': np.random.normal(125, 10),
        'base_diastolic_bp': np.random.normal(80, 8),
        'base_glucose': np.random.normal(100, 15),
        'base_hba1c': np.random.normal(5.7, 0.5), # New lab result
        'base_cholesterol': np.random.normal(190, 20),
        'base_weight': np.random.normal(180, 30), # New vital sign
        'base_sleep_duration': np.random.normal(7.5, 1.0), # New lifestyle metric
        'base_steps': np.random.normal(6000, 1500),
        'base_adherence': np.random.uniform(0.75, 0.99),
        'risk_factor': np.random.rand() # A latent risk factor
    }

# Function to generate daily data based on characteristics
def generate_daily_metrics(row):
    pid = row['patient_id']
    char = patient_characteristics[pid]
    
    # Add some noise and trends
    systolic_bp_noise = np.random.normal(0, 5)
    diastolic_bp_noise = np.random.normal(0, 4)
    glucose_noise = np.random.normal(0, 10)
    hba1c_noise = np.random.normal(0, 0.1)
    cholesterol_noise = np.random.normal(0, 10)
    weight_noise = np.random.normal(0, 1)
    sleep_duration_noise = np.random.normal(0, 0.5)
    steps_noise = np.random.normal(0, 500)
    adherence_noise = np.random.normal(0, 0.05)
    
    # Simulate a slow increase in risk over time for some patients
    time_factor = (row['date'] - START_DATE).days / 365.0
    
    systolic_bp = char['base_systolic_bp'] + systolic_bp_noise + (char['risk_factor'] * 30 * time_factor)
    diastolic_bp = char['base_diastolic_bp'] + diastolic_bp_noise + (char['risk_factor'] * 15 * time_factor)
    glucose = char['base_glucose'] + glucose_noise + (char['risk_factor'] * 40 * time_factor)
    hba1c = char['base_hba1c'] + hba1c_noise + (char['risk_factor'] * 1.5 * time_factor)
    cholesterol = char['base_cholesterol'] + cholesterol_noise + (char['risk_factor'] * 30 * time_factor)
    weight = char['base_weight'] + weight_noise + (char['risk_factor'] * 20 * time_factor)
    sleep_duration = char['base_sleep_duration'] + sleep_duration_noise - (char['risk_factor'] * 2.0 * time_factor)
    steps = max(0, char['base_steps'] + steps_noise - (char['risk_factor'] * 2000 * time_factor))
    med_adherence = max(0, min(1, char['base_adherence'] + adherence_noise - (char['risk_factor'] * 0.3 * time_factor)))
    
    return pd.Series([systolic_bp, diastolic_bp, glucose, hba1c, cholesterol, weight, sleep_duration, steps, med_adherence])

print("Generating synthetic patient data... (This may take a moment)")
# Apply the function to generate metrics
df[['systolic_bp', 'diastolic_bp', 'glucose', 'hba1c', 'cholesterol', 'weight', 'sleep_duration', 'steps', 'med_adherence']] = df.apply(generate_daily_metrics, axis=1)


# --- 4. Create the Target Variable ('deteriorated_in_90_days') ---

# Simulate deterioration events for higher-risk patients
deterioration_dates = {}
for pid in patient_ids:
    char = patient_characteristics[pid]
    if char['risk_factor'] > 0.7: # Only higher-risk patients will have an event
        # Event happens in the latter half of the year
        event_day = np.random.randint(180, 360)
        deterioration_dates[pid] = START_DATE + timedelta(days=event_day)

# Function to create the target label
def assign_target_label(row):
    pid = row['patient_id']
    if pid in deterioration_dates:
        event_date = deterioration_dates[pid]
        # If the current date is within the 90-day window BEFORE the event, label is 1
        if 0 < (event_date - row['date']).days <= DETERIORATION_WINDOW:
            return 1
    return 0

df['deteriorated_in_90_days'] = df.apply(assign_target_label, axis=1)


# --- 5. Save to CSV ---
output_filename = 'synthetic_patient_data.csv'
df.to_csv(output_filename, index=False)

print(f"\n--- Success! ---")
print(f"Dataset has been generated and saved as '{output_filename}'")
print("\n--- First 5 rows of your new data: ---")
print(df.head())
print("\n--- Data Info: ---")
df.info()
