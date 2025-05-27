import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load data
data = pd.read_csv('C:/RPM/data/eicu-crd-demo/vitalPeriodic.csv')

# Define vital sign columns (adjust based on check_raw_data.py output)
vital_columns = ['heartrate', 'sao2', 'respiration']
bp_columns = ['systemicsystolic', 'systemicdiastolic']

# Check for alternative BP column names
available_columns = data.columns.tolist()
if 'systemicsystolic' not in available_columns:
    if 'noninvasivesystolic' in available_columns:
        bp_columns[0] = 'noninvasivesystolic'
    elif 'systolic' in available_columns:
        bp_columns[0] = 'systolic'
if 'systemicdiastolic' not in available_columns:
    if 'noninvasivediastolic' in available_columns:
        bp_columns[1] = 'noninvasivediastolic'
    elif 'diastolic' in available_columns:
        bp_columns[1] = 'diastolic'

vital_columns.extend(bp_columns)
print("Using columns:", vital_columns)

# Select vital signs
data_vitals = data[vital_columns]

# Impute missing values: try interpolation, then forward-fill
data_vitals = data_vitals.interpolate(method='linear').fillna(method='ffill').fillna(method='bfill')

# Normalize data
scaler = MinMaxScaler()
data_vitals_scaled = scaler.fit_transform(data_vitals)
data_vitals_scaled = pd.DataFrame(data_vitals_scaled, columns=vital_columns)

# Add back non-vital columns
data_processed = data[['patientunitstayid', 'observationoffset']].join(data_vitals_scaled)

# Save preprocessed data
data_processed.to_csv('C:/RPM/data/preprocessed_vitals.csv', index=False)

# Verify
print("\nPreprocessed data head:")
print(data_processed.head())
print("\nMissing values:")
print(data_processed.isnull().sum())