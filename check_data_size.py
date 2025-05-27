import pandas as pd

# Load data
preprocessed = pd.read_csv('C:/RPM/data/preprocessed_vitals.csv')
raw = pd.read_csv('C:/RPM/data/eicu-crd-demo/vitalPeriodic.csv')

# Print sizes
print("Preprocessed rows:", len(preprocessed))
print("Unique patientunitstayid:", preprocessed['patientunitstayid'].nunique())
print("Columns:", preprocessed.columns.tolist())
print("Missing values:\n", preprocessed[['systemicsystolic', 'systemicdiastolic']].isnull().sum())

print("\nRaw data rows:", len(raw))
print("Unique patientunitstayid:", raw['patientunitstayid'].nunique())