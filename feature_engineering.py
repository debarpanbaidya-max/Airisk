import pandas as pd
import numpy as np

# --- 1. Load and Prepare Data ---
try:
    df = pd.read_csv('cleaned_synthetic_data.csv')
    print("✅ Successfully loaded 'cleaned_synthetic_data.csv'")
except FileNotFoundError:
    print("❌ Error: 'cleaned_synthetic_data.csv' not found. Please run the data cleaning script first.")
    exit()

df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by=['patient_id', 'date'])

# *** KEY CHANGE: Set 'date' as the index for time-based operations ***
df.set_index('date', inplace=True)

# --- 2. Feature Engineering ---
# We will use a 60-day rolling window as an example, but this can be adjusted.
WINDOW_SIZE = '60D'
vitals_to_engineer = [
    'systolic_bp', 
    'diastolic_bp', 
    'glucose', 
    'hba1c', 
    'cholesterol', 
    'weight', 
    'sleep_duration', 
    'steps', 
    'med_adherence'
]

# Create a list to hold the feature DataFrames for each patient
all_features_list = []
total_patients = df['patient_id'].nunique()
current_patient = 0

print("\nStarting feature engineering...")

# Group by patient and calculate features for each one
for patient_id, patient_data in df.groupby('patient_id'):
    current_patient += 1
    # print(f"Processing patient {current_patient}/{total_patients}...")

    # Create a new DataFrame for this patient's features, using their date index
    features = pd.DataFrame(index=patient_data.index)
    features['patient_id'] = patient_id
    features['deteriorated_in_90_days'] = patient_data['deteriorated_in_90_days']

    # --- Calculate Rolling Mean and Standard Deviation ---
    for vital in vitals_to_engineer:
        # The .rolling() function now automatically uses the DatetimeIndex
        features[f'{vital}_mean_{WINDOW_SIZE}'] = patient_data[vital].rolling(window=WINDOW_SIZE).mean()
        features[f'{vital}_std_{WINDOW_SIZE}'] = patient_data[vital].rolling(window=WINDOW_SIZE).std()

    # --- Calculate Rolling Trend (Slope) ---
    def calculate_slope(series):
        y = series.values
        x = np.arange(len(y))
        # Ensure there's enough data to calculate a slope
        if len(y[~np.isnan(y)]) < 2:
            return np.nan
        slope, _ = np.polyfit(x[~np.isnan(y)], y[~np.isnan(y)], 1)
        return slope

    for vital in vitals_to_engineer:
        # Use .apply() with the custom slope function
        features[f'{vital}_trend_{WINDOW_SIZE}'] = patient_data[vital].rolling(window=WINDOW_SIZE).apply(calculate_slope, raw=False)

    all_features_list.append(features)

print("\n--- Feature Engineering Complete! ---")

# --- 3. Combine and Finalize Dataset ---
# Concatenate all patient feature DataFrames into one
final_df = pd.concat(all_features_list)

# The first few days for each patient will have NaN values from the rolling window
final_df.dropna(inplace=True)

# Bring the 'date' index back as a column
final_df.reset_index(inplace=True)

print("\n--- First 5 rows of the final dataset with engineered features: ---")
print(final_df.head())

print("\n--- Final dataset info: ---")
final_df.info()

# Save the engineered features to a new file
final_df.to_csv('engineered_features.csv', index=False)
print("\nFinal dataset saved to 'engineered_features.csv'")
