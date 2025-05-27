import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load data
data = pd.read_csv('C:/RPM/data/eicu-crd-demo/vitalPeriodic.csv')

# Define vital sign columns
vital_columns = ['heartrate', 'sao2', 'respiration', 'systemicsystolic', 'systemicdiastolic']

# Check for alternative BP columns
available_columns = data.columns.tolist()
if 'systemicsystolic' not in available_columns:
    if 'noninvasivesystolic' in available_columns:
        vital_columns[3] = 'noninvasivesystolic'
    elif 'systolic' in available_columns:
        vital_columns[3] = 'systolic'
if 'systemicdiastolic' not in available_columns:
    if 'noninvasivediastolic' in available_columns:
        vital_columns[4] = 'noninvasivediastolic'
    elif 'diastolic' in available_columns:
        vital_columns[4] = 'diastolic'

print("Using columns:", vital_columns)

# Filter: Average vital signs per patient
data_filtered = data.groupby('patientunitstayid')[vital_columns + ['observationoffset']].mean().reset_index()

# Select vital signs
data_vitals = data_filtered[vital_columns]

# Impute missing values
data_vitals = data_vitals.interpolate(method='linear').fillna(method='ffill').fillna(method='bfill')

# Normalize data
scaler = MinMaxScaler()
data_vitals_scaled = scaler.fit_transform(data_vitals)
data_vitals_scaled = pd.DataFrame(data_vitals_scaled, columns=vital_columns)

# Combine with non-vital columns
data_processed = data_filtered[['patientunitstayid', 'observationoffset']].join(data_vitals_scaled)

# Save
data_processed.to_csv('C:/RPM/data/preprocessed_vitals.csv', index=False)

# Verify
print("\nPreprocessed rows:", len(data_processed))
print("Columns:", data_processed.columns.tolist())
print("Missing values:\n", data_processed[vital_columns].isnull().sum())
print("Sample data:\n", data_processed.head())