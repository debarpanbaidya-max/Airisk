import pandas as pd
import numpy as np

# --- 1. Configuration ---
input_filename = "synthetic_patient_data.csv"
output_filename = "cleaned_synthetic_data.csv"

# --- 2. Load the dataset ---
try:
    df = pd.read_csv(input_filename)
    print(f"✅ Successfully loaded '{input_filename}'")
except FileNotFoundError:
    print(f"❌ Error: The file '{input_filename}' was not found. Please run 'dataset_maker.py' first.")
    # Exit if the file is not found
    exit()

print(f"📊 Original Shape: {df.shape}")
print("\n🔍 Checking for Missing Values:")
print(df.isnull().sum())

# --- 3. Drop duplicate rows ---
initial_rows = len(df)
df.drop_duplicates(inplace=True)
dropped_rows = initial_rows - len(df)
print(f"\n🗑️ Dropped {dropped_rows} duplicate rows.")

# --- 4. Validate and clean numeric columns ---
# Since the data is synthetically generated, it is unlikely to have negative values,
# but it's good practice to add a check for robustness.
numeric_cols = ['systolic_bp', 'diastolic_bp', 'glucose', 'hba1c', 'cholesterol', 'weight', 'sleep_duration', 'steps', 'med_adherence']
for col in numeric_cols:
    invalid_count = (df[col] < 0).sum()
    if invalid_count > 0:
        print(f"⚠️ Warning: Found {invalid_count} negative values in '{col}'. Correcting...")
        df[col] = df[col].apply(lambda x: max(0, x)) # Set any negative values to 0

# --- 5. Convert data types ---
# The 'date' column needs to be a datetime object for time-series analysis
df['date'] = pd.to_datetime(df['date'])
print("\n🔄 Converted 'date' column to datetime type.")

# --- 6. Final check and save cleaned dataset ---
print("\n✅ Data cleaning complete.")
df.to_csv(output_filename, index=False)
print(f"✅ Cleaned data saved to '{output_filename}'")
print(f"🆕 Final Shape: {df.shape}")
print("\nFirst 5 rows of the cleaned data:")
print(df.head())
