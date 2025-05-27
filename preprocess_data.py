import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load data
data = pd.read_csv('C:/RPM/data/eicu-crd-demo/vitalPeriodic.csv')

# Select vital sign columns (adjust based on actual columns)
vital_columns = ['heartrate', 'sao2', 'respiration', 'systemicsystolic', 'systemicdiastolic']
data_vitals = data[vital_columns]

# Impute missing values with linear interpolation
data_vitals = data_vitals.interpolate(method='linear')

# Normalize data
scaler = MinMaxScaler()
data_vitals_scaled = scaler.fit_transform(data_vitals)
data_vitals_scaled = pd.DataFrame(data_vitals_scaled, columns=vital_columns)

# Add back non-vital columns (e.g., patient ID, timestamp)
data_processed = data[['patientunitstayid', 'observationoffset']].join(data_vitals_scaled)

# Save preprocessed data
data_processed.to_csv('C:/RPM/data/preprocessed_vitals.csv', index=False)

# Verify
print(data_processed.head())
print(data_processed.isnull().sum())